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
DB_WRITE_INTERVAL_SECONDS = 60
API_CALLS_PER_MINUTE = 80

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
        self.last_db_write = defaultdict(float)
        self.api_call_times = deque(maxlen=1000)
        self.running = True
        self.hour_start_cache = {}  # Cache p_start prices

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
            print(f"📌 {currency.upper()}: Fetched exact hour start price ${p_start:.2f} from Binance for {hour_key}")
            return p_start
            
    except Exception as e:
        print(f"⚠️ Failed to get hour start price from Binance for {currency}: {e}")
    
    # Fallback to current price if Binance call fails
    state.hour_start_cache[hour_key][currency] = current_crypto_price
    print(f"📌 {currency.upper()}: Using current price ${current_crypto_price:.2f} as fallback for {hour_key}")
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
        
        # Get volatility from recent data
        config = CURRENCY_CONFIG[currency]
        with sqlite3.connect(f"{currency}_polyscraper.db") as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT {config['db_column']} FROM polydata 
                WHERE {config['db_column']} IS NOT NULL 
                ORDER BY timestamp DESC LIMIT 20
            """)
            recent_prices = [row[0] for row in cursor.fetchall()]
        
        if len(recent_prices) >= 20:
            price_series = pd.Series(recent_prices[::-1])
            log_returns = np.log(price_series).diff()
            rolling_vol = log_returns.rolling(window=20, min_periods=2).std()
            vol = rolling_vol.iloc[-1] * np.sqrt(60) if not rolling_vol.empty else 0.01
        else:
            vol = 0.01
            
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

def write_cached_data_if_needed(currency):
    """Write cached data to database if interval has passed."""
    now = time.time()
    if now - state.last_db_write[currency] >= DB_WRITE_INTERVAL_SECONDS:
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
                
                # Get volatility from recent data
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT {config['db_column']} FROM polydata 
                    WHERE {config['db_column']} IS NOT NULL 
                    ORDER BY timestamp DESC LIMIT 20
                """)
                recent_prices = [row[0] for row in cursor.fetchall()]
                
                if len(recent_prices) >= 20:
                    price_series = pd.Series(recent_prices[::-1])
                    log_returns = np.log(price_series).diff()
                    rolling_vol = log_returns.rolling(window=20, min_periods=2).std()
                    vol = rolling_vol.iloc[-1] * np.sqrt(60) if not rolling_vol.empty else 0.01
                else:
                    vol = 0.01
                    
                if pd.isna(vol) or vol <= 0:
                    vol = 0.01
                
                print(f"📝 {currency.upper()}: DB written ({len(state.data_cache[currency])} points)")
                print(f"    📊 Features: ofi={ofi:.4f} | r={r:.4f} | p_start=${p_start:.2f} | vol={vol:.4f} | tau={tau:.3f}")
                
                state.last_db_write[currency] = now
                
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
        
        # Log results
        
        if prediction:
            delta = (prediction - market_price) * 100
            action = "BUY UP" if delta > PROBABILITY_DELTA_THRESHOLD else "BUY DOWN" if delta < -PROBABILITY_DELTA_THRESHOLD else "HOLD"
            print(f"  🔸 {currency.upper()}: {prediction*100:.1f}% vs {market_price*100:.1f}% (Δ{delta:+.1f}pp) → {action}")
        else:
            print(f"  🔸 {currency.upper()}: P=N/A M={market_price*100:.1f}%")
        
        # Display trade result with position info
        if trade_result["executed"]:
            print(f"    💰 SIMULATED: {trade_result['direction']} exposure ${trade_result['target_exposure']:.2f}")
            print(f"    📊 YES: {trade_result['current_yes']:.2f}→{trade_result['target_yes']:.2f} | NO: {trade_result['current_no']:.2f}→{trade_result['target_no']:.2f}")
        elif trade_result["reason"] == "delta_too_small":
            print(f"    ⏸️ FLAT: Delta {trade_result['delta']:.1f}pp below threshold ({PROBABILITY_DELTA_THRESHOLD}pp)")
        elif trade_result["reason"] == "insufficient_funds":
            print(f"    💸 TOO EXPENSIVE: Need ${trade_result['needed']:.2f}, have ${trade_result['available']:.2f}")
            print(f"    📊 YES: {trade_result['current_yes']:.2f}→{trade_result['target_yes']:.2f} | NO: {trade_result['current_no']:.2f}→{trade_result['target_no']:.2f}")
        elif trade_result["reason"] == "adjustment_too_small":
            print(f"    📏 TOO SMALL: Adjustment {trade_result['adjustment']} below minimum")
            print(f"    📊 YES: {trade_result['current_yes']:.2f}→{trade_result['target_yes']:.2f} | NO: {trade_result['current_no']:.2f}→{trade_result['target_no']:.2f}")
        elif trade_result["reason"] == "position_aligned":
            print(f"    ✅ ALIGNED: Position already optimal")
            print(f"    📊 YES: {trade_result['current_yes']:.2f}→{trade_result['target_yes']:.2f} | NO: {trade_result['current_no']:.2f}→{trade_result['target_no']:.2f}")
        
        # Save data
        timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        save_data_point(currency, timestamp, market_name, token_yes, best_bid, best_ask, spot_price, ofi, prediction)
        
        # Write to DB if needed
        write_cached_data_if_needed(currency)
        
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
    
    print("🎯 System ready for continuous trading!")

def continuous_trading_loop():
    """Main continuous trading loop."""
    currencies = list(CURRENCY_CONFIG.keys())
    cycle_count = 0
    
    print(f"\n🔄 Starting continuous loop: {' → '.join(c.upper() for c in currencies)}")
    print(f"⏱️ Cycle delay: {CYCLE_DELAY_SECONDS}s | 💾 DB writes: {DB_WRITE_INTERVAL_SECONDS}s")
    print("-" * 60)
    
    try:
        while state.running:
            cycle_start = time.time()
            cycle_count += 1
            
            timestamp = datetime.now(UTC).strftime('%H:%M:%S')
            print(f"\n\n⚡ {timestamp} [Cycle {cycle_count}]")
            
            # Cancel all open orders from previous cycles for maximum dynamism
            if TRADING_ENABLED:
                cancel_all_open_orders()
            
            # Trade each currency
            for currency in currencies:
                if not state.running:
                    break
                trade_currency_cycle(currency)
            
            print("\n" + "="*50)  # Clear separator between cycles
            
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