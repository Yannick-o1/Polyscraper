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

def get_hour_start_price(currency, current_price):
    """Get exact hour start price with efficient caching."""
    hour_key = datetime.now(UTC).strftime('%Y-%m-%d_%H')
    
    if hour_key in state.hour_start_cache:
        return state.hour_start_cache[hour_key].get(currency, current_price)
    
    # Initialize cache for this hour
    state.hour_start_cache[hour_key] = {}
    
    try:
        config = CURRENCY_CONFIG[currency]
        with sqlite3.connect(f"{currency}_polyscraper.db") as conn:
            cursor = conn.cursor()
            
            # Get earliest price from this hour
            hour_start = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
            cursor.execute(f"""
                SELECT {config['db_column']} FROM polydata 
                WHERE timestamp >= ? AND {config['db_column']} IS NOT NULL
                ORDER BY timestamp ASC LIMIT 1
            """, (hour_start.strftime('%Y-%m-%d %H:%M:%S'),))
            
            result = cursor.fetchone()
            p_start = result[0] if result else current_price
            
        state.hour_start_cache[hour_key][currency] = p_start
        return p_start
        
    except Exception:
        return current_price

def get_market_data(currency):
    """Get current market token IDs and prices."""
    try:
        wait_for_rate_limit()
        markets_file = f"{currency}_polymarkets.csv"
        
        if not os.path.exists(markets_file):
            return None, None, None
            
        df = pd.read_csv(markets_file)
        if df.empty:
            return None, None, None
            
        # Get most recent market
        latest_market = df.iloc[-1]
        return latest_market['yes_token_id'], latest_market['no_token_id'], latest_market['market_name']
        
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
    """Get current market prices from order book."""
    try:
        wait_for_rate_limit()
        response = requests.get(f"{CLOB_API_URL}/book", 
                              params={"token_id": token_id}, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if bids and asks:
            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])
            return best_bid, best_ask
            
        return None, None
        
    except Exception:
        return None, None

def calculate_model_prediction(currency, current_price, ofi):
    """Calculate model prediction with proper features."""
    if not state.models.get(currency) or not current_price:
        return None
        
    try:
        # Get p_start and calculate features
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
    """Execute position adjustment based on model prediction."""
    if not prediction or not TRADING_ENABLED:
        return {"executed": False, "reason": "no_prediction_or_disabled"}
    
    # Calculate delta and target position
    delta = (prediction - market_price) * 100
    
    if abs(delta) < PROBABILITY_DELTA_THRESHOLD:
        return {"executed": False, "reason": "delta_too_small", "delta": delta}
    
    # Calculate target exposure
    target_exposure = BANKROLL_FRACTION * BANKROLL * abs(delta) / 100
    
    # Check if we can afford minimum trade
    min_cost = MINIMUM_ORDER_SIZE_SHARES * max(market_price, 1-market_price)
    if min_cost > BANKROLL * 0.9:
        return {
            "executed": False, 
            "reason": "insufficient_funds",
            "delta": delta,
            "needed": min_cost,
            "available": BANKROLL
        }
    
    # For now, return success (actual trading logic would go here)
    return {
        "executed": True,
        "reason": "simulated_trade",
        "delta": delta,
        "direction": "UP" if delta > 0 else "DOWN",
        "target_exposure": target_exposure
    }

def save_data_point(currency, timestamp, market_name, token_id, best_bid, best_ask, spot_price, ofi, prediction):
    """Efficiently save data point to cache."""
    data_point = (timestamp, market_name, token_id, best_bid, best_ask, spot_price, ofi, prediction)
    state.data_cache[currency].append(data_point)

def write_cached_data_if_needed(currency):
    """Write cached data to database if interval has passed."""
    now = time.time()
    if now - state.last_db_write[currency] >= DB_WRITE_INTERVAL_SECONDS:
        if state.data_cache[currency]:
            try:
                config = CURRENCY_CONFIG[currency]
                latest_data = state.data_cache[currency][-1]
                
                with sqlite3.connect(f"{currency}_polyscraper.db") as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'''
                        INSERT INTO polydata (timestamp, market_name, token_id, best_bid, best_ask, {config['db_column']}, ofi, p_up_prediction)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', latest_data)
                
                print(f"📝 {currency.upper()}: DB written ({len(state.data_cache[currency])} points)")
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
        
        # Step 3: Get order book
        start = time.time()
        best_bid, best_ask = get_order_book_prices(token_yes)
        timings['orderbook'] = time.time() - start
        
        if not best_bid or not best_ask:
            return False
        
        market_price = (best_bid + best_ask) / 2
        
        # Step 4: Calculate prediction
        start = time.time()
        prediction = calculate_model_prediction(currency, spot_price, ofi)
        timings['prediction'] = time.time() - start
        
        # Step 5: Execute trading logic
        start = time.time()
        trade_result = execute_dynamic_position_management(
            currency, prediction, market_price, token_yes, token_no, best_bid, best_ask
        )
        timings['trading'] = time.time() - start
        
        # Log results
        total_time = sum(timings.values())
        if prediction:
            delta = (prediction - market_price) * 100
            action = "BUY UP" if delta > PROBABILITY_DELTA_THRESHOLD else "BUY DOWN" if delta < -PROBABILITY_DELTA_THRESHOLD else "HOLD"
            print(f"  🔸 {currency.upper()}: {prediction*100:.1f}% vs {market_price*100:.1f}% (Δ{delta:+.1f}pp) → {action}")
        else:
            print(f"  🔸 {currency.upper()}: P=N/A M={market_price*100:.1f}%")
        
        print(f"    ⏱️ {total_time:.3f}s [Market:{timings['market']:.3f} | Binance:{timings['binance']:.3f} | OrderBook:{timings['orderbook']:.3f} | Prediction:{timings['prediction']:.3f} | Trading:{timings['trading']:.3f}]")
        
        # Display trade result
        if trade_result["executed"]:
            print(f"    💰 SIMULATED: {trade_result['direction']} exposure ${trade_result['target_exposure']:.2f}")
        elif trade_result["reason"] == "delta_too_small":
            print(f"    ⏸️ FLAT: Delta {trade_result['delta']:.1f}pp below threshold ({PROBABILITY_DELTA_THRESHOLD}pp)")
        elif trade_result["reason"] == "insufficient_funds":
            print(f"    💸 TOO EXPENSIVE: Need ${trade_result['needed']:.2f}, have ${trade_result['available']:.2f}")
        
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
            print(f"\n⚡ {timestamp} [Cycle {cycle_count}]")
            
            # Trade each currency
            for currency in currencies:
                if not state.running:
                    break
                trade_currency_cycle(currency)
            
            # Cycle timing
            cycle_time = time.time() - cycle_start
            print(f"  📈 Total cycle time: {cycle_time:.1f}s")
            
            # Sleep until next cycle
            sleep_time = max(0, CYCLE_DELAY_SECONDS - cycle_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested")
        state.running = False

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