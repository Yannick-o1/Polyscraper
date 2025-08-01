#!/usr/bin/env python3
"""
Polymarket Continuous Trading Bot - Clean Version
Cycles through BTC -> ETH -> SOL -> XRP -> BTC... as fast as possible
Dynamic position management based on model predictions
"""

# === IMPORTS ===
import requests
import pandas as pd
import sqlite3
import numpy as np
import lightgbm as lgb
import os
import time
import threading
from datetime import datetime, UTC, timedelta
from collections import defaultdict, deque
from zoneinfo import ZoneInfo

# Polymarket imports
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType, BalanceAllowanceParams, AssetType
    from py_clob_client.order_builder.constants import BUY, SELL
    POLYMARKET_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: py-clob-client not installed. Trading disabled.")
    POLYMARKET_AVAILABLE = False

# === CONFIGURATION ===
CURRENCY_CONFIG = {
    'btc': {
        'name': 'Bitcoin', 'asset_symbol': 'BTCUSDT', 'db_column': 'btc_usdt_spot',
        'market_pattern': r'bitcoin-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et',
        'market_question_pattern': 'bitcoin up or down', 'error_context': 'Bitcoin'
    },
    'eth': {
        'name': 'Ethereum', 'asset_symbol': 'ETHUSDT', 'db_column': 'eth_usdt_spot',
        'market_pattern': r'ethereum-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et',
        'market_question_pattern': 'ethereum up or down', 'error_context': 'Ethereum'
    },
    'sol': {
        'name': 'Solana', 'asset_symbol': 'SOLUSDT', 'db_column': 'sol_usdt_spot',
        'market_pattern': r'solana-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et',
        'market_question_pattern': 'solana up or down', 'error_context': 'Solana'
    },
    'xrp': {
        'name': 'XRP', 'asset_symbol': 'XRPUSDT', 'db_column': 'xrp_usdt_spot',
        'market_pattern': r'xrp-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et',
        'market_question_pattern': 'xrp up or down', 'error_context': 'XRP'
    }
}

# Trading parameters
BANKROLL = 0.30
BANKROLL_FRACTION = 0.8
PROBABILITY_DELTA_THRESHOLD = 3.0
MINIMUM_ORDER_SIZE_SHARES = 5.0
CYCLE_DELAY_SECONDS = 2.0
# DB_WRITE_INTERVAL_SECONDS = 60  # Removed - now writing every cycle
API_CALLS_PER_MINUTE = 80

# Mock trading parameters
MOCK_TRADING_ENABLED = True
MOCK_INITIAL_BANKROLL = 100.0
MOCK_KELLY_FRACTION = 0.5
MOCK_THETA = 0.03  # 3% minimum delta threshold

# Environment
TRADING_ENABLED = True
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
POLYMARKET_PROXY_ADDRESS = os.getenv("POLYMARKET_PROXY_ADDRESS")
CLOB_API_URL = "https://clob.polymarket.com"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === GLOBAL STATE ===
class TradingState:
    def __init__(self):
        self.models = {}
        self.polymarket_client = None
        self.data_cache = defaultdict(lambda: deque(maxlen=100))
# self.last_db_write = defaultdict(float)  # Removed - writing every cycle now
        self.api_call_times = deque(maxlen=1000)
        self.running = True
        self.hour_start_cache = {}  # Cache p_start prices
        
        # Mock trading state
        if MOCK_TRADING_ENABLED:
            self.mock_available_cash = MOCK_INITIAL_BANKROLL
            self.mock_positions = {}  # {currency: {'direction': 'UP'/'DOWN', 'shares': float, 'avg_cost': float}}
            self.mock_initialized = False

state = TradingState()

# === CORE FUNCTIONS ===
def log_startup():
    """Clean startup logging."""
    print("🚀 Polymarket Continuous Trading Bot")
    print("📊 Loading models and connecting to markets...")

def wait_for_rate_limit():
    """Efficient rate limiting."""
    now = time.time()
    # Remove old calls
    while state.api_call_times and now - state.api_call_times[0] > 60:
        state.api_call_times.popleft()
    
    # Smart delay only when needed
    if len(state.api_call_times) >= API_CALLS_PER_MINUTE * 0.85:
        time.sleep(0.8)  # Brief pause near limit
    
    state.api_call_times.append(now)

def get_hour_start_price(currency, current_crypto_price):
    """Get exact price from Binance at the precise start of the current hour."""
    hour_key = datetime.now(UTC).strftime('%Y-%m-%d_%H')
    
    # Check if we already have p_start cached for this hour
    if hour_key in state.hour_start_cache:
        if currency in state.hour_start_cache[hour_key]:
            return state.hour_start_cache[hour_key][currency]
    else:
        # Initialize cache for this hour
        state.hour_start_cache[hour_key] = {}
    
    # Get exact hour start timestamp
    hour_start = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    hour_start_ms = int(hour_start.timestamp() * 1000)
    
    try:
        config = CURRENCY_CONFIG[currency]
        wait_for_rate_limit()
        
        # Get price from Binance at exact hour start using klines
        response = requests.get("https://api.binance.com/api/v3/klines", 
                              params={
                                  "symbol": config['asset_symbol'], 
                                  "interval": "1m",
                                  "startTime": hour_start_ms,
                                  "limit": 1
                              }, 
                              timeout=5)
        response.raise_for_status()
        
        klines = response.json()
        if klines:
            # Kline format: [open_time, open, high, low, close, volume, close_time, ...]
            # Use opening price of the first minute of the hour
            p_start = float(klines[0][1])  # open price
            state.hour_start_cache[hour_key][currency] = p_start
            print(f"🔒 {currency.upper()}: Fetched exact hour start price ${p_start:.2f} from Binance for {hour_key}")
            return p_start
            
    except Exception as e:
        print(f"⚠️ Failed to get hour start price from Binance for {currency}: {e}")
    
    # Fallback to current price if Binance call fails
    state.hour_start_cache[hour_key][currency] = current_crypto_price
    print(f"🔒 {currency.upper()}: Using current price ${current_crypto_price:.2f} as fallback for {hour_key}")
    return current_crypto_price

def get_market_data(currency):
    """Get current market token IDs for the current hour."""
    try:
        markets_file = f"{currency}_polymarkets.csv"
        
        if not os.path.exists(markets_file):
            return None, None, None
            
        df = pd.read_csv(markets_file)
        if df.empty:
            return None, None, None
        
        # Get current hour and convert to ET timezone for market matching
        current_hour_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        et_tz = ZoneInfo("America/New_York")
        target_hour_dt_et = current_hour_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        # Find market matching current hour
        matching_rows = df[df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            print(f"⚠️ No market found for current hour: {target_datetime_str}")
            # Fall back to most recent market as backup
            latest_market = df.iloc[-1]
            print(f"   Using fallback market: {latest_market['market_name']}")
        else:
            latest_market = matching_rows.iloc[0]
            # Market found - no need to log this every time
        
        token_id_yes = str(int(latest_market['token_id_yes']))
        token_id_no = str(int(latest_market['token_id_no']))
        market_name = latest_market['market_name']
        
        return token_id_yes, token_id_no, market_name
        
    except Exception as e:
        print(f"❌ Market data error for {currency}: {e}")
        return None, None, None

def get_live_price_and_ofi(currency):
    """Get current price and order flow imbalance."""
    try:
        wait_for_rate_limit()
        config = CURRENCY_CONFIG[currency]
        
        response = requests.get("https://api.binance.com/api/v3/trades", 
                              params={"symbol": config['asset_symbol'], "limit": 500}, 
                              timeout=5)
        response.raise_for_status()
        trades = response.json()
        
        if not trades:
            return None, None
            
        df = pd.DataFrame(trades)
        latest_price = float(df.iloc[-1]["price"])
        df["qty"] = df["qty"].astype(float)
        
        # Calculate OFI
        buy_vol = df[~df["isBuyerMaker"]]["qty"].sum()
        sell_vol = df[df["isBuyerMaker"]]["qty"].sum()
        total_vol = buy_vol + sell_vol
        ofi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        
        return latest_price, ofi
        
    except Exception as e:
        print(f"❌ Binance error for {currency}: {e}")
        return None, None

def get_order_book_prices(token_id):
    """Get best bid and ask prices using Polymarket's price API."""
    try:
        wait_for_rate_limit()
        
        # Make both calls in parallel for speed
        results = {}
        
        def get_price(side, results_dict):
            try:
                response = requests.get(f"{CLOB_API_URL}/price", 
                                      params={"token_id": token_id, "side": side}, 
                                      timeout=5)
                response.raise_for_status()
                results_dict[side] = float(response.json()['price'])
            except Exception as e:
                results_dict[side] = None
        
        # Launch both requests simultaneously
        bid_thread = threading.Thread(target=get_price, args=("buy", results))
        ask_thread = threading.Thread(target=get_price, args=("sell", results))
        
        bid_thread.start()
        ask_thread.start()
        
        bid_thread.join()
        ask_thread.join()
        
        best_bid = results.get("buy")
        best_ask = results.get("sell")
        
        if best_bid is not None and best_ask is not None:
            return best_bid, best_ask
        
        return None, None
        
    except Exception as e:
        print(f"❌ Price API error for token {token_id}: {e}")
        return None, None

def calculate_model_prediction(currency, current_price, ofi, market_price):
    """Calculate model prediction with proper features."""
    if not state.models.get(currency) or not current_price:
        return None
        
    try:
        # Get p_start and calculate features using CRYPTO prices
        p_start = get_hour_start_price(currency, current_price)
        r = (current_price / p_start - 1) if p_start > 0 else 0
        
        current_minute = datetime.now(UTC).minute
        tau = max(1 - (current_minute / 60), 0.01)
        
        # Get volatility from last 20 minutes of live price cache
        if currency in state.data_cache and len(state.data_cache[currency]) >= 5:
            # Filter cache to last 20 minutes
            current_time = datetime.now(UTC)
            twenty_minutes_ago = current_time - timedelta(minutes=20)
            
            # Extract data points from last 20 minutes: (timestamp, market_name, token_id, best_bid, best_ask, spot_price, ofi, prediction)
            recent_data = []
            for data_point in state.data_cache[currency]:
                # Parse timestamp from data point
                try:
                    data_timestamp = datetime.strptime(data_point[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
                    if data_timestamp >= twenty_minutes_ago:
                        recent_data.append(data_point)
                except:
                    # If timestamp parsing fails, include the point (better safe than sorry)
                    recent_data.append(data_point)
            
            if len(recent_data) >= 5:
                # Sample data to ~1-minute intervals to get proper volatility calculation
                # Group by minute and take the last price in each minute
                minute_prices = {}
                for data_point in recent_data:
                    try:
                        data_timestamp = datetime.strptime(data_point[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
                        minute_key = data_timestamp.strftime('%Y-%m-%d %H:%M')  # Group by minute
                        minute_prices[minute_key] = data_point[5]  # Keep latest price for each minute
                    except:
                        continue
                
                if len(minute_prices) >= 3:
                    # Use minute-level prices for volatility calculation
                    prices_by_minute = [minute_prices[key] for key in sorted(minute_prices.keys())]
                    price_series = pd.Series(prices_by_minute)
                    log_returns = np.log(price_series).diff()
                    rolling_vol = log_returns.std()
                    vol = rolling_vol * np.sqrt(60) if not pd.isna(rolling_vol) and rolling_vol > 0 else 0.01
                else:
                    vol = 0.01
            else:
                vol = 0.01
        else:
            # Fallback: use a reasonable default volatility
            vol = 0.02  # 2% hourly vol as default
            
        if pd.isna(vol) or vol <= 0:
            vol = 0.01
        
        # Model features
        r_scaled = r / np.sqrt(tau)
        ofi = ofi if ofi is not None else 0.0
        
        X_live = [[r_scaled, tau, vol, ofi]]
        prediction = float(state.models[currency].predict(X_live)[0])
        
        return prediction
        
    except Exception as e:
        print(f"❌ Prediction error for {currency}: {e}")
        return None

def execute_dynamic_position_management(currency, prediction, market_price, token_yes, token_no, best_bid, best_ask):
    """Execute position adjustment based on model prediction with position tracking."""
    if not prediction or not TRADING_ENABLED:
        return {"executed": False, "reason": "no_prediction_or_disabled"}
    
    # Calculate delta and check threshold
    delta = (prediction - market_price) * 100
    
    if abs(delta) < PROBABILITY_DELTA_THRESHOLD:
        return {"executed": False, "reason": "delta_too_small", "delta": delta}
    
    # Get current positions
    current_yes, current_no = get_current_position(token_yes, token_no)
    current_net = current_yes - current_no  # Net position (positive = bullish)
    
    # Calculate target position
    target_exposure = BANKROLL_FRACTION * BANKROLL * abs(delta) / 100
    
    if delta > 0:
        # Bullish: want long YES position
        target_yes = target_exposure / market_price if market_price > 0 else 0
        target_no = 0
        target_net = target_yes
    else:
        # Bearish: want long NO position  
        target_yes = 0
        target_no = target_exposure / (1 - market_price) if market_price < 1 else 0
        target_net = -target_no
    
    # Calculate adjustment needed
    position_adjustment = target_net - current_net
    
    # Check if we can afford minimum trade
    min_cost = MINIMUM_ORDER_SIZE_SHARES * max(market_price, 1-market_price)
    
    if abs(position_adjustment) < 0.001:
        return {
            "executed": False,
            "reason": "position_aligned", 
            "delta": delta,
            "current_yes": current_yes,
            "current_no": current_no,
            "target_yes": target_yes,
            "target_no": target_no,
            "adjustment": f"{position_adjustment:+.2f}"
        }
    
    if abs(position_adjustment) < MINIMUM_ORDER_SIZE_SHARES:
        return {
            "executed": False,
            "reason": "adjustment_too_small",
            "delta": delta,
            "current_yes": current_yes,
            "current_no": current_no,
            "target_yes": target_yes,
            "target_no": target_no,
            "adjustment": f"{position_adjustment:+.2f}"
        }
    
    if min_cost > BANKROLL * 0.9:
        return {
            "executed": False,
            "reason": "insufficient_funds",
            "delta": delta,
            "current_yes": current_yes,
            "current_no": current_no,
            "target_yes": target_yes,
            "target_no": target_no,
            "adjustment": f"{position_adjustment:+.2f}",
            "needed": min_cost,
            "available": BANKROLL
        }
    
    # For now, return simulated trade (would execute real trade here)
    return {
        "executed": True,
        "reason": "simulated_trade",
        "delta": delta,
        "direction": "UP" if delta > 0 else "DOWN", 
        "target_exposure": target_exposure,
        "current_yes": current_yes,
        "current_no": current_no,
        "target_yes": target_yes,
        "target_no": target_no,
        "adjustment": f"{position_adjustment:+.2f}"
    }

def save_data_point(currency, timestamp, market_name, token_id_yes, best_bid, best_ask, spot_price, ofi, prediction):
    """Save data point to cache."""
    data_point = (timestamp, market_name, token_id_yes, best_bid, best_ask, spot_price, ofi, prediction)
    state.data_cache[currency].append(data_point)

def write_to_database(currency):
    """Write latest data point to database every cycle."""
    if state.data_cache[currency]:
        try:
            config = CURRENCY_CONFIG[currency]
            latest_data = state.data_cache[currency][-1]
            timestamp, market_name, token_id, best_bid, best_ask, spot_price, ofi, prediction = latest_data
            
            with sqlite3.connect(f"{currency}_polyscraper.db") as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    INSERT INTO polydata (timestamp, market_name, token_id, best_bid, best_ask, {config['db_column']}, ofi, p_up_prediction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', latest_data)
            
            # Calculate and display model features using CRYPTO prices
            p_start = get_hour_start_price(currency, spot_price)
            r = (spot_price / p_start - 1) if p_start > 0 else 0
            current_minute = datetime.now(UTC).minute
            tau = max(1 - (current_minute / 60), 0.01)
            
            # Get volatility from last 20 minutes (consistent with prediction calculation)
            if currency in state.data_cache and len(state.data_cache[currency]) >= 5:
                # Filter cache to last 20 minutes
                current_time = datetime.now(UTC)
                twenty_minutes_ago = current_time - timedelta(minutes=20)
                
                # Extract data points from last 20 minutes: (timestamp, market_name, token_id, best_bid, best_ask, spot_price, ofi, prediction)
                recent_data = []
                for data_point in state.data_cache[currency]:
                    # Parse timestamp from data point
                    try:
                        data_timestamp = datetime.strptime(data_point[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
                        if data_timestamp >= twenty_minutes_ago:
                            recent_data.append(data_point)
                    except:
                        # If timestamp parsing fails, include the point (better safe than sorry)
                        recent_data.append(data_point)
                
                if len(recent_data) >= 5:
                    # Sample data to ~1-minute intervals to get proper volatility calculation
                    # Group by minute and take the last price in each minute
                    minute_prices = {}
                    for data_point in recent_data:
                        try:
                            data_timestamp = datetime.strptime(data_point[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
                            minute_key = data_timestamp.strftime('%Y-%m-%d %H:%M')  # Group by minute
                            minute_prices[minute_key] = data_point[5]  # Keep latest price for each minute
                        except:
                            continue
                    
                    if len(minute_prices) >= 3:
                        # Use minute-level prices for volatility calculation
                        prices_by_minute = [minute_prices[key] for key in sorted(minute_prices.keys())]
                        price_series = pd.Series(prices_by_minute)
                        log_returns = np.log(price_series).diff()
                        rolling_vol = log_returns.std()
                        vol = rolling_vol * np.sqrt(60) if not pd.isna(rolling_vol) and rolling_vol > 0 else 0.01
                    else:
                        vol = 0.01
                else:
                    vol = 0.01
            else:
                # Fallback: use a reasonable default volatility
                vol = 0.02  # 2% hourly vol as default
                
            if pd.isna(vol) or vol <= 0:
                vol = 0.01
            
            print(f"    ⚙️ Features: ofi={ofi:.4f} | r={r:.4f} | p_start=${p_start:.2f} | vol={vol:.4f} | tau={tau:.3f}")
            
        except Exception as e:
            print(f"❌ DB write error for {currency}: {e}")

def trade_currency_cycle(currency):
    """Execute one complete trading cycle for a currency."""
    start_time = time.time()
    timings = {}
    
    try:
        # Step 1: Get market data
        start = time.time()
        token_yes, token_no, market_name = get_market_data(currency)
        timings['market'] = time.time() - start
        
        if not token_yes:
            return False
        
        # Step 2: Get live price and OFI
        start = time.time()
        spot_price, ofi = get_live_price_and_ofi(currency)
        timings['binance'] = time.time() - start
        
        if not spot_price:
            return False
        
        # Step 3: Get order book prices
        start = time.time()
        best_bid, best_ask = get_order_book_prices(token_yes)
        timings['orderbook'] = time.time() - start
        
        if not best_bid or not best_ask:
            return False
        
        market_price = (best_bid + best_ask) / 2
        
        # Step 4: Calculate prediction
        start = time.time()
        prediction = calculate_model_prediction(currency, spot_price, ofi, market_price)
        timings['prediction'] = time.time() - start
        
        # Step 5: Execute trading logic
        start = time.time()
        trade_result = execute_dynamic_position_management(
            currency, prediction, market_price, token_yes, token_no, best_bid, best_ask
        )
        timings['trading'] = time.time() - start
        
        # Step 6: Execute mock trading
        if MOCK_TRADING_ENABLED:
            execute_mock_trading(currency, prediction, market_price, spot_price)
        
        # Log results
        
        if prediction:
            delta = (prediction - market_price) * 100
            action = "BUY UP" if delta > PROBABILITY_DELTA_THRESHOLD else "BUY DOWN" if delta < -PROBABILITY_DELTA_THRESHOLD else "HOLD"
            print(f"  📈 {currency.upper()}: {prediction*100:.1f}% vs {market_price*100:.1f}% (Δ{delta:+.1f}pp) → {action}")
        else:
            print(f"  📈 {currency.upper()}: P=N/A M={market_price*100:.1f}%")
        
        # Display trade result with position info
        if trade_result["executed"]:
            print(f"    🟢 SIMULATED: {trade_result['direction']} exposure ${trade_result['target_exposure']:.2f}")
            print(f"    ▶️ YES: {trade_result['current_yes']:.2f}→{trade_result['target_yes']:.2f} | NO: {trade_result['current_no']:.2f}→{trade_result['target_no']:.2f}")
        elif trade_result["reason"] == "delta_too_small":
            print(f"    ⚪ FLAT: Delta {trade_result['delta']:.1f}pp below threshold ({PROBABILITY_DELTA_THRESHOLD}pp)")
        elif trade_result["reason"] == "insufficient_funds":
            print(f"    🔴 TOO EXPENSIVE: Need ${trade_result['needed']:.2f}, have ${trade_result['available']:.2f}")
            print(f"    ▶️ YES: {trade_result['current_yes']:.2f}→{trade_result['target_yes']:.2f} | NO: {trade_result['current_no']:.2f}→{trade_result['target_no']:.2f}")
        elif trade_result["reason"] == "adjustment_too_small":
            print(f"    🟡 TOO SMALL: Adjustment {trade_result['adjustment']} below minimum")
            print(f"    ▶️ YES: {trade_result['current_yes']:.2f}→{trade_result['target_yes']:.2f} | NO: {trade_result['current_no']:.2f}→{trade_result['target_no']:.2f}")
        elif trade_result["reason"] == "position_aligned":
            print(f"    🟢 ALIGNED: Position already optimal")
            print(f"    ▶️ YES: {trade_result['current_yes']:.2f}→{trade_result['target_yes']:.2f} | NO: {trade_result['current_no']:.2f}→{trade_result['target_no']:.2f}")
        
        # Save data
        timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        save_data_point(currency, timestamp, market_name, token_yes, best_bid, best_ask, spot_price, ofi, prediction)
        
        # Write to database every cycle
        write_to_database(currency)
        
        return True
        
    except Exception as e:
        print(f"  ❌ {currency.upper()} error: {e}")
        return False

def initialize_system():
    """Initialize models and setup."""
    log_startup()
    
    # Load models
    for currency in CURRENCY_CONFIG:
        try:
            model_file = f"{currency}_lgbm.txt"
            if os.path.exists(model_file):
                state.models[currency] = lgb.Booster(model_file=model_file)
                print(f"✅ {currency.upper()} model loaded")
            else:
                print(f"⚠️ {currency.upper()} model file not found")
        except Exception as e:
            print(f"❌ Error loading {currency.upper()} model: {e}")
    
    # Initialize mock trading
    if MOCK_TRADING_ENABLED:
        print("💰 Initializing mock trading system...")
        initialize_mock_trading()
        state.mock_initialized = True
        print(f"✅ Mock trading ready with ${MOCK_INITIAL_BANKROLL} bankroll")
    
    print("🎯 System ready for continuous trading!")

def continuous_trading_loop():
    """Main continuous trading loop."""
    currencies = list(CURRENCY_CONFIG.keys())
    cycle_count = 0
    currency_index = 0  # Track which currency to trade this cycle
    
    print(f"\n🔄 Starting continuous loop: {' → '.join(c.upper() for c in currencies)}")
    print(f"⏱️ Cycle delay: {CYCLE_DELAY_SECONDS}s | 💾 DB writes: Every cycle")
    if MOCK_TRADING_ENABLED:
        print(f"💰 Mock trading: ${MOCK_INITIAL_BANKROLL} bankroll | Kelly: {MOCK_KELLY_FRACTION*100:.0f}% | Theta: {MOCK_THETA*100:.0f}%")
    print("-" * 60)
    
    try:
        while state.running:
            cycle_start = time.time()
            cycle_count += 1
            
            timestamp = datetime.now(UTC).strftime('%H:%M:%S')
            current_currency = currencies[currency_index]
            print(f"\n\n⚡ {timestamp} [Cycle {cycle_count}] - {current_currency.upper()}")
            
            # Check for hour change and calculate outcomes
            check_hour_outcomes()
            
            # Cancel all open orders from previous cycles for maximum dynamism
            if TRADING_ENABLED:
                cancel_all_open_orders()
            
            # Trade current currency
            trade_currency_cycle(current_currency)
            
            # Move to next currency for next cycle
            currency_index = (currency_index + 1) % len(currencies)
            
            # Show portfolio summary every 20 cycles
            if MOCK_TRADING_ENABLED and cycle_count % 20 == 0:
                display_mock_portfolio_summary()
            
            print("\n" + "─"*50)  # Clear separator between cycles
            
            # Sleep until next cycle
            cycle_time = time.time() - cycle_start
            sleep_time = max(0, CYCLE_DELAY_SECONDS - cycle_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested")
        state.running = False

def cancel_all_open_orders():
    """Cancel all open orders to prevent overlapping and increase dynamism."""
    if not state.polymarket_client:
        return 0
    
    try:
        wait_for_rate_limit()
        
        # Get all open orders
        response = state.polymarket_client.get_orders()
        
        if not response or 'data' not in response:
            return 0
        
        open_orders = [order for order in response['data'] if order.get('status') == 'LIVE']
        
        if not open_orders:
            return 0
        
        cancelled_count = 0
        
        # Cancel each open order
        for order in open_orders:
            try:
                wait_for_rate_limit()
                cancel_response = state.polymarket_client.cancel_order(order['id'])
                
                if cancel_response.get('success', False):
                    cancelled_count += 1
                    
            except Exception:
                # Log individual order cancellation errors but continue
                pass
        
        if cancelled_count > 0:
            print(f"  🗑️ Cancelled {cancelled_count} open orders")
        
        return cancelled_count
        
    except Exception:
        # Silently handle API errors - don't let order cancellation stop trading
        return 0

def get_current_position(token_id_yes, token_id_no):
    """Get current position in YES and NO tokens for a market."""
    if not state.polymarket_client:
        return 0, 0
    
    try:
        wait_for_rate_limit()
        
        # Get balances for both YES and NO tokens
        balance_params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL,
            token_ids=[token_id_yes, token_id_no]
        )
        
        response = state.polymarket_client.get_balances(balance_params)
        
        if not response or 'balances' not in response:
            return 0, 0
        
        # Extract positions
        yes_position = 0
        no_position = 0
        
        for balance in response['balances']:
            if balance['token_id'] == token_id_yes:
                yes_position = float(balance.get('balance', 0))
            elif balance['token_id'] == token_id_no:
                no_position = float(balance.get('balance', 0))
        
        return yes_position, no_position
        
    except Exception:
        # Don't let position lookup errors stop trading
        return 0, 0

def initialize_mock_trading():
    """Initialize mock trading database tables."""
    if not MOCK_TRADING_ENABLED:
        return
        
    for currency in CURRENCY_CONFIG.keys():
        try:
            with sqlite3.connect(f"{currency}_polyscraper.db") as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS mock_trading (
                        timestamp TEXT,
                        total_bankroll REAL,
                        return_pct REAL,
                        currency TEXT,
                        direction TEXT,
                        amount REAL,
                        action TEXT,
                        delta REAL,
                        market_price REAL,
                        prediction REAL
                    )
                ''')
        except Exception as e:
            print(f"❌ Error initializing mock trading DB for {currency}: {e}")

def execute_mock_trading(currency, prediction, market_price, spot_price):
    """Execute mock trading based on model prediction vs market price."""
    if not MOCK_TRADING_ENABLED or not prediction:
        return
        
    try:
        # Calculate delta (edge)
        delta = prediction - market_price
        
        # Get current position for this currency
        old_position = state.mock_positions.get(currency, {'direction': None, 'shares': 0, 'avg_cost': 0}).copy()
        position = state.mock_positions.get(currency, {'direction': None, 'shares': 0, 'avg_cost': 0}).copy()
        
        # Calculate current portfolio value (mark-to-market)
        total_pos_value = 0
        for curr, pos in state.mock_positions.items():
            if pos['shares'] > 0:
                # Use current market prices for mark-to-market
                if curr == currency:
                    curr_market_price = market_price
                else:
                    # For other currencies, try to get recent market price from cache
                    curr_market_price = 0.5  # Default fallback
                    if curr in state.data_cache and state.data_cache[curr]:
                        recent_data = state.data_cache[curr][-1]  # Most recent data
                        if len(recent_data) >= 6:  # best_bid, best_ask are at indices 3,4
                            best_bid, best_ask = recent_data[3], recent_data[4]
                            if best_bid is not None and best_ask is not None:
                                curr_market_price = (best_bid + best_ask) / 2
                    
                if pos['direction'] == 'UP':
                    total_pos_value += pos['shares'] * curr_market_price
                elif pos['direction'] == 'DOWN':
                    total_pos_value += pos['shares'] * (1 - curr_market_price)
        
        portfolio_value = state.mock_available_cash + total_pos_value
        
        # Debug: Show current state
        if old_position['shares'] > 0:
            old_pos_value = old_position['shares'] * (market_price if old_position['direction'] == 'UP' else (1 - market_price))
            print(f"    💰 Mock {currency.upper()}: Delta={delta:+.1%} | Current: {old_position['shares']:.2f} {old_position['direction']} = ${old_pos_value:.2f}")
        else:
            print(f"    💰 Mock {currency.upper()}: Delta={delta:+.1%} | Current: No position")
        
        # Determine target position using Kelly criterion
        target_direction = None
        target_value = 0
        
        if abs(delta) >= MOCK_THETA:
            target_direction = 'UP' if delta > 0 else 'DOWN'
            target_value = abs(delta) * MOCK_KELLY_FRACTION * portfolio_value
            target_value = min(target_value, portfolio_value * 0.1)  # Cap at 10% of portfolio
            print(f"       🎯 Target: {target_direction} | Value: ${target_value:.2f} | Portfolio: ${portfolio_value:.2f}")
        else:
            print(f"       🎯 Target: FLAT (delta {delta:+.1%} < threshold {MOCK_THETA:.1%})")
        
        # Calculate position adjustment needed
        trade_pnl = 0
        action = "HOLD"
        amount = 0
        
        # Calculate current position value and target position value
        current_pos_value = position['shares'] * (market_price if position['direction'] == 'UP' else (1 - market_price)) if position['shares'] > 0 else 0
        
        # Determine if we need to adjust position
        should_adjust = False
        
        # Case 1: Direction changed (e.g., UP -> DOWN or DOWN -> UP)
        if position['direction'] and position['direction'] != target_direction:
            should_adjust = True
            print(f"       🔄 Direction change: {position['direction']} → {target_direction}")
        
        # Case 2: Going flat (target is None but we have position)
        elif position['direction'] and target_direction is None:
            should_adjust = True
            print(f"       🔄 Going flat: {position['direction']} → FLAT")
        
        # Case 3: Position size mismatch (current vs target value)
        elif position['shares'] > 0 and target_direction == position['direction']:
            size_diff = abs(target_value - current_pos_value)
            if size_diff > 0.5:  # If difference is more than $0.50
                should_adjust = True
                print(f"       🔄 Size adjustment: ${current_pos_value:.2f} → ${target_value:.2f}")
        
        # Case 4: New position when we have none
        elif not position['direction'] and target_direction:
            should_adjust = True
            print(f"       🔄 New position: FLAT → {target_direction}")
        
        # Execute position adjustment
        if should_adjust:
            # Case 1: Direction change or going flat - close entire position
            if position['direction'] and position['direction'] != target_direction:
                price = market_price if position['direction'] == 'UP' else (1 - market_price)
                cash_received = position['shares'] * price
                cost_of_shares = position['shares'] * position['avg_cost']
                trade_pnl = cash_received - cost_of_shares
                
                state.mock_available_cash += cash_received
                print(f"       💰 Sold {position['shares']:.2f} {position['direction']} @ ${price:.3f} = ${cash_received:.2f}")
                
                # Clear position
                position = {'direction': None, 'shares': 0, 'avg_cost': 0}
                action = f"SELL {old_position['direction']}"
                amount = old_position['shares']
                
                # Then buy new position if target direction is set
                if target_direction and target_value > 0.01:
                    price = market_price if target_direction == 'UP' else (1 - market_price)
                    shares_to_buy = target_value / price if price > 0 else 0
                    cost = shares_to_buy * price
                    
                    if cost <= state.mock_available_cash and shares_to_buy > 0.01:
                        state.mock_available_cash -= cost
                        position = {'direction': target_direction, 'shares': shares_to_buy, 'avg_cost': price}
                        action = f"BUY {target_direction}"
                        amount = shares_to_buy
                        print(f"       💰 Bought {shares_to_buy:.2f} {target_direction} @ ${price:.3f} = ${cost:.2f}")
            
            # Case 2: Size adjustment - only adjust the difference
            elif position['shares'] > 0 and target_direction == position['direction']:
                current_value = position['shares'] * (market_price if position['direction'] == 'UP' else (1 - market_price))
                value_diff = target_value - current_value
                
                if abs(value_diff) > 0.5:  # Only adjust if difference is significant
                    price = market_price if position['direction'] == 'UP' else (1 - market_price)
                    
                    if value_diff > 0:  # Need to buy more
                        shares_to_add = value_diff / price if price > 0 else 0
                        cost = shares_to_add * price
                        
                        if cost <= state.mock_available_cash and shares_to_add > 0.01:
                            state.mock_available_cash -= cost
                            
                            # Add to existing position
                            total_cost = (position['shares'] * position['avg_cost']) + cost
                            position['shares'] += shares_to_add
                            position['avg_cost'] = total_cost / position['shares']
                            action = f"ADD {target_direction}"
                            amount = shares_to_add
                            print(f"       💰 Added {shares_to_add:.2f} {target_direction} @ ${price:.3f} = ${cost:.2f}")
                    
                    else:  # Need to sell some
                        shares_to_sell = abs(value_diff) / price if price > 0 else 0
                        cash_received = shares_to_sell * price
                        
                        state.mock_available_cash += cash_received
                        position['shares'] -= shares_to_sell
                        action = f"REDUCE {target_direction}"
                        amount = shares_to_sell
                        print(f"       💰 Sold {shares_to_sell:.2f} {target_direction} @ ${price:.3f} = ${cash_received:.2f}")
            
            # Case 3: New position when we have none
            elif not position['direction'] and target_direction:
                price = market_price if target_direction == 'UP' else (1 - market_price)
                shares_to_buy = target_value / price if price > 0 else 0
                cost = shares_to_buy * price
                
                if cost <= state.mock_available_cash and shares_to_buy > 0.01:
                    state.mock_available_cash -= cost
                    position = {'direction': target_direction, 'shares': shares_to_buy, 'avg_cost': price}
                    action = f"BUY {target_direction}"
                    amount = shares_to_buy
                    print(f"       💰 Bought {shares_to_buy:.2f} {target_direction} @ ${price:.3f} = ${cost:.2f}")
        else:
            print(f"       ✅ Position optimal: {position['shares']:.2f} {position['direction']} = ${current_pos_value:.2f}")
        
        # Update position
        state.mock_positions[currency] = position
        
        # Calculate final portfolio value
        final_pos_value = 0
        for curr, pos in state.mock_positions.items():
            if pos['shares'] > 0:
                if curr == currency:
                    curr_market_price = market_price
                else:
                    # For other currencies, try to get recent market price from cache
                    curr_market_price = 0.5  # Default fallback
                    if curr in state.data_cache and state.data_cache[curr]:
                        recent_data = state.data_cache[curr][-1]  # Most recent data
                        if len(recent_data) >= 6:  # best_bid, best_ask are at indices 3,4
                            best_bid, best_ask = recent_data[3], recent_data[4]
                            if best_bid is not None and best_ask is not None:
                                curr_market_price = (best_bid + best_ask) / 2
                    
                if pos['direction'] == 'UP':
                    final_pos_value += pos['shares'] * curr_market_price
                elif pos['direction'] == 'DOWN':
                    final_pos_value += pos['shares'] * (1 - curr_market_price)
        
        final_bankroll = state.mock_available_cash + final_pos_value
        return_pct = ((final_bankroll - MOCK_INITIAL_BANKROLL) / MOCK_INITIAL_BANKROLL) * 100
        
        # Save to database
        save_mock_trading_result(currency, final_bankroll, return_pct, 
                                target_direction if target_direction else 'FLAT', 
                                amount, action, delta, market_price, prediction)
        
        # Show the trade result with detailed before/after
        if action != "HOLD":
            new_pos = state.mock_positions.get(currency, {'direction': None, 'shares': 0, 'avg_cost': 0})
            
            print(f"       📋 {action} | Amount: ${amount:.2f} @ {market_price:.1%}")
            print(f"       📊 Before: {old_position['shares']:.2f} {old_position['direction'] or 'FLAT'} @ ${old_position['avg_cost']:.3f}")
            print(f"       📊 After:  {new_pos['shares']:.2f} {new_pos['direction'] or 'FLAT'} @ ${new_pos['avg_cost']:.3f}")
            print(f"       💵 Cash: ${state.mock_available_cash:.2f} | Portfolio: ${final_bankroll:.2f} ({return_pct:+.1f}%)")
        else:
            # For HOLD, show current position
            pos = state.mock_positions.get(currency, {'direction': None, 'shares': 0, 'avg_cost': 0})
            if pos['shares'] > 0:
                pos_value = pos['shares'] * (market_price if pos['direction'] == 'UP' else (1 - market_price))
                print(f"       💤 HOLD | {pos['shares']:.2f} {pos['direction']} @ ${pos['avg_cost']:.3f} = ${pos_value:.2f}")
            else:
                print(f"       💤 HOLD | No position | Portfolio: ${final_bankroll:.2f} ({return_pct:+.1f}%)")
        
    except Exception as e:
        print(f"❌ Mock trading error for {currency}: {e}")

def save_mock_trading_result(currency, total_bankroll, return_pct, direction, amount, action, delta, market_price, prediction):
    """Save mock trading result to database."""
    if not MOCK_TRADING_ENABLED:
        return
        
    try:
        timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        
        with sqlite3.connect(f"{currency}_polyscraper.db") as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO mock_trading 
                (timestamp, total_bankroll, return_pct, currency, direction, amount, action, delta, market_price, prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, total_bankroll, return_pct, currency, direction, amount, action, delta, market_price, prediction))
            
    except Exception as e:
        print(f"❌ Error saving mock trading result for {currency}: {e}")

def display_mock_portfolio_summary():
    """Display comprehensive portfolio summary."""
    if not MOCK_TRADING_ENABLED:
        return
        
    try:
        print(f"\n💼 MOCK PORTFOLIO SUMMARY")
        print(f"   💵 Available Cash: ${state.mock_available_cash:.2f}")
        
        total_position_value = 0
        active_positions = 0
        
        for currency, pos in state.mock_positions.items():
            if pos['shares'] > 0:
                active_positions += 1
                # Get current market price for this currency
                market_price = 0.5  # Default fallback
                if currency in state.data_cache and state.data_cache[currency]:
                    recent_data = state.data_cache[currency][-1]
                    if len(recent_data) >= 6:
                        best_bid, best_ask = recent_data[3], recent_data[4]
                        if best_bid is not None and best_ask is not None:
                            market_price = (best_bid + best_ask) / 2
                
                pos_value = pos['shares'] * (market_price if pos['direction'] == 'UP' else (1 - market_price))
                total_position_value += pos_value
                
                cost_basis = pos['shares'] * pos['avg_cost'] 
                unrealized_pnl = pos_value - cost_basis
                
                print(f"   📈 {currency.upper()}: {pos['shares']:.2f} {pos['direction']} @ ${pos['avg_cost']:.3f} | "
                      f"Value: ${pos_value:.2f} | P&L: ${unrealized_pnl:+.2f}")
        
        total_portfolio = state.mock_available_cash + total_position_value
        total_return_pct = ((total_portfolio - MOCK_INITIAL_BANKROLL) / MOCK_INITIAL_BANKROLL) * 100
        
        if active_positions == 0:
            print("   📭 No active positions")
        
        print(f"   🏦 Total Portfolio: ${total_portfolio:.2f} ({total_return_pct:+.1f}%) | Positions: {active_positions}")
        print("━" * 60)
        
    except Exception as e:
        print(f"❌ Error displaying portfolio summary: {e}")

def check_hour_outcomes():
    """Check if hour has changed and calculate UP/DOWN outcomes for previous hour."""
    current_hour_key = datetime.now(UTC).strftime('%Y-%m-%d_%H')
    
    # Initialize last_hour tracking if not exists
    if not hasattr(state, 'last_hour_key'):
        state.last_hour_key = current_hour_key
        return
    
    # Check if hour has changed
    if current_hour_key != state.last_hour_key:
        print(f"\n🕒 HOUR CHANGE DETECTED: {state.last_hour_key} → {current_hour_key}")
        print("📊 Calculating outcomes for previous hour...")
        
        # Calculate outcomes for each currency for the previous hour
        for currency in CURRENCY_CONFIG.keys():
            calculate_currency_outcome(currency, state.last_hour_key)
        
        # Update last hour tracker
        state.last_hour_key = current_hour_key
        
        # Show portfolio summary on hour change
        if MOCK_TRADING_ENABLED:
            display_mock_portfolio_summary()
        
        print("━" * 60)

def calculate_currency_outcome(currency, previous_hour_key):
    """Calculate UP/DOWN outcome for a specific currency and hour."""
    try:
        # Check if we have p_start for this hour
        if previous_hour_key not in state.hour_start_cache:
            print(f"  ⚠️ {currency.upper()}: No p_start data for {previous_hour_key}")
            return
        
        if currency not in state.hour_start_cache[previous_hour_key]:
            print(f"  ⚠️ {currency.upper()}: No p_start data for {previous_hour_key}")
            return
        
        p_start = state.hour_start_cache[previous_hour_key][currency]
        
        # Get exact end-of-hour price from Binance
        # Parse the hour to get end time
        hour_parts = previous_hour_key.split('_')
        date_part = hour_parts[0]  # '2025-08-01'
        hour_part = int(hour_parts[1])  # 13
        
        # Calculate end of hour timestamp
        end_hour = datetime.strptime(f"{date_part} {hour_part:02d}:59:59", '%Y-%m-%d %H:%M:%S')
        end_hour = end_hour.replace(tzinfo=UTC)
        end_hour_ms = int(end_hour.timestamp() * 1000)
        
        # Get price from Binance at exact hour end using klines
        config = CURRENCY_CONFIG[currency]
        wait_for_rate_limit()
        
        response = requests.get("https://api.binance.com/api/v3/klines", 
                              params={
                                  "symbol": config['asset_symbol'], 
                                  "interval": "1m",
                                  "endTime": end_hour_ms,
                                  "limit": 1
                              }, 
                              timeout=5)
        response.raise_for_status()
        
        klines = response.json()
        if klines:
            # Use closing price of the last minute of the hour
            p_end = float(klines[0][4])  # close price
            
            # Calculate outcome
            price_change = p_end - p_start
            outcome = "UP" if price_change > 0 else "DOWN"
            change_pct = (price_change / p_start) * 100
            
            print(f"  🎯 {currency.upper()}: ${p_start:.2f} → ${p_end:.2f} = {outcome} ({change_pct:+.2f}%)")
            
            # Store outcome in database if needed
            store_outcome_in_db(currency, previous_hour_key, p_start, p_end, outcome, change_pct)
            
        else:
            print(f"  ❌ {currency.upper()}: No kline data available for {previous_hour_key}")
            
    except Exception as e:
        print(f"  ❌ {currency.upper()}: Error calculating outcome - {e}")

def store_outcome_in_db(currency, hour_key, p_start, p_end, outcome, change_pct):
    """Store the hour outcome in the database."""
    try:
        with sqlite3.connect(f"{currency}_polyscraper.db") as conn:
            cursor = conn.cursor()
            
            # Create outcomes table if it doesn't exist
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS hour_outcomes (
                    hour_key TEXT PRIMARY KEY,
                    p_start REAL,
                    p_end REAL,
                    outcome TEXT,
                    change_pct REAL,
                    timestamp TEXT
                )
            ''')
            
            # Insert outcome
            cursor.execute(f'''
                INSERT OR REPLACE INTO hour_outcomes 
                (hour_key, p_start, p_end, outcome, change_pct, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (hour_key, p_start, p_end, outcome, change_pct, datetime.now(UTC).isoformat()))
            
    except Exception as e:
        print(f"    ⚠️ DB error storing outcome: {e}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🧪 Test mode - all systems check")
        initialize_system()
        print("✅ Test completed successfully!")
    else:
        initialize_system()
        continuous_trading_loop() 