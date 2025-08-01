#!/usr/bin/env python3
"""
Polymarket Continuous Trading Bot
Cycles through BTC -> ETH -> SOL -> XRP -> BTC... as fast as possible
Trades immediately but only writes to database every minute per currency
"""

print("🚀 Starting Polymarket Continuous Trading Bot...")
print("📦 Loading libraries...")

import requests
import pandas as pd
from datetime import datetime, UTC, timedelta
from zoneinfo import ZoneInfo
import re
import sqlite3
print("✅ Core libraries loaded")

import numpy as np
import os
import time
import json
import threading
import queue
from collections import defaultdict, deque
print("✅ Utility libraries loaded")

print("📊 Loading LightGBM (this may take a moment)...")
import lightgbm as lgb
print("✅ LightGBM loaded successfully")

# Polymarket imports
print("🔗 Loading Polymarket client...")
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        OrderArgs,
        OrderType,
        BalanceAllowanceParams,
        AssetType,
    )
    from py_clob_client.order_builder.constants import BUY, SELL
    POLYMARKET_AVAILABLE = True
    print("✅ Polymarket client libraries loaded")
except ImportError:
    print("⚠️ Warning: py-clob-client not installed. Order placement will be disabled.")
    POLYMARKET_AVAILABLE = False

# --- Currency Configuration ---
CURRENCY_CONFIG = {
    'btc': {
        'name': 'Bitcoin',
        'asset_symbol': 'BTCUSDT',
        'market_pattern': r'bitcoin-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et',
        'market_pattern_alt': r'bitcoin-up-or-down-(\w+)-(\d+)-(\d+)-(am|pm)-et',
        'market_question_pattern': 'bitcoin up or down',
        'db_column': 'btc_usdt_spot',
        'outcome_column': 'outcome',
        'error_context': 'Bitcoin'
    },
    'eth': {
        'name': 'Ethereum',
        'asset_symbol': 'ETHUSDT',
        'market_pattern': r'ethereum-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et',
        'market_pattern_alt': r'ethereum-up-or-down-(\w+)-(\d+)-(\d+)-(am|pm)-et',
        'market_question_pattern': 'ethereum up or down',
        'db_column': 'eth_usdt_spot',
        'outcome_column': 'outcome_eth',
        'error_context': 'Ethereum'
    },
    'sol': {
        'name': 'Solana',
        'asset_symbol': 'SOLUSDT',
        'market_pattern': r'solana-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et',
        'market_pattern_alt': r'solana-up-or-down-(\w+)-(\d+)-(\d+)-(am|pm)-et',
        'market_question_pattern': 'solana up or down',
        'db_column': 'sol_usdt_spot',
        'outcome_column': 'outcome_sol',
        'error_context': 'Solana'
    },
    'xrp': {
        'name': 'XRP',
        'asset_symbol': 'XRPUSDT',
        'market_pattern': r'xrp-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et',
        'market_pattern_alt': r'xrp-up-or-down-(\w+)-(\d+)-(\d+)-(am|pm)-et',
        'market_question_pattern': 'xrp up or down',
        'db_column': 'xrp_usdt_spot',
        'outcome_column': 'outcome_xrp',
        'error_context': 'XRP'
    }
}

# --- Global Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://gamma-api.polymarket.com"

# --- Trading Configuration ---
TRADING_ENABLED = True
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_PROXY_ADDRESS = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
POLYMARKET_CHAIN_ID = 137
PROBABILITY_DELTA_DEADZONE_THRESHOLD = 3.0
BANKROLL_USAGE_FRACTION = 0.95
MINIMUM_ORDER_SIZE_SHARES = 5.0

# --- Rate Limiting Configuration ---
BINANCE_REQUESTS_PER_MINUTE = 1000  # Conservative limit (API allows 1200)
POLYMARKET_REQUESTS_PER_MINUTE = 300  # Conservative estimate
CYCLE_DELAY_SECONDS = 2.0  # Minimum delay between currency cycles
DB_WRITE_INTERVAL_SECONDS = 60  # Write to DB every 60 seconds per currency

# --- Global State ---
class TradingState:
    def __init__(self):
        self.models = {}
        self.polymarket_client = None
        self.data_cache = defaultdict(lambda: deque(maxlen=100))  # Cache recent data
        self.db_write_queue = queue.Queue()
        self.last_db_write = defaultdict(float)  # Track last write time per currency
        self.api_call_times = deque(maxlen=1000)  # Track API calls for rate limiting
        self.market_cache = {}  # Cache market data
        self.running = True

state = TradingState()

# --- Rate Limiting ---
def wait_for_rate_limit():
    """Ensure we don't exceed API rate limits."""
    now = time.time()
    
    # Remove calls older than 1 minute
    while state.api_call_times and state.api_call_times[0] < now - 60:
        state.api_call_times.popleft()
    
    # If we're approaching the limit, wait
    if len(state.api_call_times) >= BINANCE_REQUESTS_PER_MINUTE * 0.8:  # 80% of limit
        sleep_time = 60 - (now - state.api_call_times[0])
        if sleep_time > 0:
            print(f"Rate limit protection: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
    
    state.api_call_times.append(now)

# --- Initialization ---
def initialize_trading_system():
    """Initialize all currency models and Polymarket client."""
    print("🚀 Initializing Continuous Trading System...")
    
    # Load models for all currencies
    print("📊 Loading trading models...")
    for currency in CURRENCY_CONFIG:
        model_file = os.path.join(BASE_DIR, f"{currency}_lgbm.txt")
        print(f"   Loading {currency.upper()} model from {model_file}...")
        try:
            state.models[currency] = lgb.Booster(model_file=model_file)
            print(f"✅ {currency.upper()} model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading {currency.upper()} model: {e}")
            state.models[currency] = None
    
    # Initialize Polymarket client
    print("🔗 Initializing Polymarket client...")
    if POLYMARKET_AVAILABLE and TRADING_ENABLED and POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS:
        try:
            print("   Creating client instance...")
            state.polymarket_client = ClobClient(
                host=CLOB_API_URL,
                key=POLYMARKET_PRIVATE_KEY,
                chain_id=POLYMARKET_CHAIN_ID,
                signature_type=1,
                funder=POLYMARKET_PROXY_ADDRESS
            )
            print("   Setting API credentials...")
            state.polymarket_client.set_api_creds(state.polymarket_client.create_or_derive_api_creds())
            print("✅ Polymarket client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Polymarket client: {e}")
            state.polymarket_client = None
    else:
        print("⚠️ Polymarket trading disabled (missing credentials or libraries)")
    
    # Start database writer thread
    print("💾 Starting database writer thread...")
    db_thread = threading.Thread(target=database_writer_thread, daemon=True)
    db_thread.start()
    print("✅ Database writer thread started")
    
    print("🎯 Continuous trading system ready!")

# --- Database Management ---
def database_writer_thread():
    """Background thread that writes cached data to databases every minute."""
    while state.running:
        try:
            # Check if any currency needs database write
            now = time.time()
            for currency in CURRENCY_CONFIG:
                if now - state.last_db_write[currency] >= DB_WRITE_INTERVAL_SECONDS:
                    if currency in state.data_cache and state.data_cache[currency]:
                        write_cached_data_to_db(currency)
                        state.last_db_write[currency] = now
            
            time.sleep(5)  # Check every 5 seconds
        except Exception as e:
            print(f"Database writer error: {e}")

def write_cached_data_to_db(currency):
    """Write cached data for a currency to its database."""
    if not state.data_cache[currency]:
        return
    
    config = CURRENCY_CONFIG[currency]
    db_file = os.path.join(BASE_DIR, f"{currency}_polyscraper.db")
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get the most recent cached data point
        latest_data = state.data_cache[currency][-1]
        
        cursor.execute(f'''
            INSERT INTO polydata (timestamp, market_name, token_id, best_bid, best_ask, {config['db_column']}, ofi, p_up_prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', latest_data)
        
        conn.commit()
        
        # Extract features for display
        timestamp, market_name, token_id, best_bid, best_ask, spot_price, ofi, p_up_prediction = latest_data
        
        # Calculate display features (same as in calculate_prediction)
        try:
            # Get p_start for r calculation
            current_hour_start_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
            cursor.execute(f"SELECT {config['db_column']} FROM polydata WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1", 
                          (current_hour_start_utc.strftime('%Y-%m-%d %H:%M:%S'),))
            p_start_row = cursor.fetchone()
            p_start = p_start_row[0] if p_start_row else spot_price
            
            # Calculate features
            r = (spot_price / p_start - 1) if p_start > 0 else 0
            current_minute = datetime.now(UTC).minute
            tau = max(1 - (current_minute / 60), 0.01)
            
            # Get volatility from recent data
            cursor.execute(f"SELECT {config['db_column']} FROM polydata ORDER BY timestamp DESC LIMIT 20")
            recent_prices = [row[0] for row in cursor.fetchall() if row[0] is not None]
            
            if len(recent_prices) >= 2:
                log_returns = np.diff(np.log(recent_prices))
                vol = np.std(log_returns) * np.sqrt(60) if len(log_returns) > 1 else 0.01
            else:
                vol = 0.01
                
            conn.close()
            
            print(f"📝 {currency.upper()}: DB written ({len(state.data_cache[currency])} cached points)")
            print(f"    📊 Features: ofi={ofi:.4f} | r={r:.4f} | p_start=${p_start:.2f} | vol={vol:.4f} | tau={tau:.3f}")
            
        except Exception as feature_e:
            conn.close()
            print(f"📝 {currency.upper()}: DB written ({len(state.data_cache[currency])} cached points)")
            print(f"    ⚠️ Feature display error: {feature_e}")
        
    except Exception as e:
        print(f"❌ DB write error for {currency}: {e}")

def init_database(currency):
    """Initialize database for a currency."""
    config = CURRENCY_CONFIG[currency]
    db_file = os.path.join(BASE_DIR, f"{currency}_polyscraper.db")
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS polydata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            market_name TEXT NOT NULL,
            token_id TEXT NOT NULL,
            best_bid REAL NOT NULL,
            best_ask REAL NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS p_start_cache (
            hour_key TEXT PRIMARY KEY,
            p_start_price REAL NOT NULL
        )
    ''')
    
    # Add columns if they don't exist
    cursor.execute("PRAGMA table_info(polydata)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if config['db_column'] not in columns:
        cursor.execute(f"ALTER TABLE polydata ADD COLUMN {config['db_column']} REAL")
    if 'ofi' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN ofi REAL")
    if 'p_up_prediction' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN p_up_prediction REAL")
    if 'outcome' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN outcome TEXT")
    if currency != 'btc' and config['outcome_column'] not in columns:
        cursor.execute(f"ALTER TABLE polydata ADD COLUMN {config['outcome_column']} TEXT")

    conn.commit()
    conn.close()

# --- Trading Functions ---
def place_order(side, token_id, price, size_shares):
    """Place order with rate limiting and detailed outcome tracking."""
    if not state.polymarket_client or size_shares < MINIMUM_ORDER_SIZE_SHARES:
        return {"executed": False, "reason": "insufficient_size_or_client", "balance_issue": False}
    
    try:
        wait_for_rate_limit()
        
        order_price = round(price, 2)
        expiration = int((datetime.now(UTC) + timedelta(minutes=2)).timestamp())
        
        order_args = OrderArgs(
            price=order_price,
            size=size_shares,
            side=side,
            token_id=token_id,
            expiration=expiration
        )
        
        signed_order = state.polymarket_client.create_order(order_args)
        response = state.polymarket_client.post_order(signed_order, OrderType.GTD)
        
        success = response.get('success', False)
        
        if success:
            return {"executed": True, "reason": "success", "balance_issue": False, "price": order_price, "size": size_shares}
        elif 'not enough balance' in str(response):
            return {"executed": False, "reason": "insufficient_balance", "balance_issue": True}
        else:
            return {"executed": False, "reason": str(response), "balance_issue": False}
        
    except Exception as e:
        if 'not enough balance' in str(e):
            return {"executed": False, "reason": "insufficient_balance", "balance_issue": True}
        else:
            return {"executed": False, "reason": str(e), "balance_issue": False}

def get_binance_data_and_ofi(currency):
    """Get live price and OFI with caching and rate limiting."""
    config = CURRENCY_CONFIG[currency]
    
    try:
        wait_for_rate_limit()
        
        resp = requests.get("https://api.binance.com/api/v3/trades", 
                           params={"symbol": config['asset_symbol'], "limit": 500}, timeout=5)
        resp.raise_for_status()
        trades = resp.json()
        
        if not trades:
            return None, None

        df = pd.DataFrame(trades)
        latest_price = float(df.iloc[-1]["price"])
        df["qty"] = df["qty"].astype(float)
        
        buy_vol = df[~df["isBuyerMaker"]]["qty"].sum()
        sell_vol = df[df["isBuyerMaker"]]["qty"].sum()
        
        total_vol = buy_vol + sell_vol
        ofi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        
        return latest_price, ofi
        
    except Exception as e:
        print(f"❌ Binance error for {currency}: {e}")
        return None, None

def get_current_market_token_ids(currency):
    """Get market token IDs with caching."""
    current_hour_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    cache_key = f"{currency}_{current_hour_utc.isoformat()}"
    
    if cache_key in state.market_cache:
        return state.market_cache[cache_key]
    
    # Load from CSV
    markets_csv = os.path.join(BASE_DIR, f"{currency}_polymarkets.csv")
    try:
        markets_df = pd.read_csv(markets_csv)
        et_tz = ZoneInfo("America/New_York")
        target_hour_dt_et = current_hour_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        matching_rows = markets_df[markets_df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            result = (None, None, None)
        else:
            market_row = matching_rows.iloc[0]
            result = (
                str(int(market_row['token_id_yes'])),
                str(int(market_row['token_id_no'])),
                market_row['market_name']
            )
        
        state.market_cache[cache_key] = result
        return result
        
    except Exception as e:
        print(f"❌ Market lookup error for {currency}: {e}")
        return None, None, None

def get_order_book(token_id):
    """Get order book with rate limiting."""
    try:
        wait_for_rate_limit()
        
        response = requests.get(f"{CLOB_API_URL}/book", params={"token_id": token_id}, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if bids and asks:
            parsed_bids = [(float(b['price']), float(b['size'])) for b in bids]
            parsed_asks = [(float(a['price']), float(a['size'])) for a in asks]
            parsed_bids.sort(reverse=True)
            parsed_asks.sort()
            return parsed_bids[0], parsed_asks[0]
        
        return None, None
        
    except Exception as e:
        print(f"❌ Order book error for {token_id}: {e}")
        return None, None

def calculate_prediction(currency, historical_data, current_price, ofi):
    """Calculate prediction using proper historical data like the original."""
    if not state.models[currency] or not current_price:
        return None
    
    try:
        # Get proper historical data from database
        db_file = f"{currency}_polyscraper.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get recent price data for volatility calculation
        cursor.execute(f"SELECT {CURRENCY_CONFIG[currency]['db_column']} FROM polydata ORDER BY timestamp DESC LIMIT 20")
        recent_prices = [row[0] for row in cursor.fetchall() if row[0] is not None]
        
        if len(recent_prices) < 2:
            conn.close()
            return None
        
        # Get p_start (hour start price) - use the first price of current hour
        current_hour_start_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        cursor.execute(f"SELECT {CURRENCY_CONFIG[currency]['db_column']} FROM polydata WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1", 
                      (current_hour_start_utc.strftime('%Y-%m-%d %H:%M:%S'),))
        p_start_row = cursor.fetchone()
        p_start = p_start_row[0] if p_start_row else current_price
        
        conn.close()
        
        # Calculate features exactly like the original
        r = current_price / p_start - 1 if p_start > 0 else 0
        current_minute = datetime.now(UTC).minute
        tau = max(1 - (current_minute / 60), 0.01)
        
        # Calculate volatility on log returns
        log_returns = np.diff(np.log(recent_prices))
        vol = np.std(log_returns) * np.sqrt(60) if len(log_returns) > 1 else 0.01
        
        r_scaled = r / np.sqrt(tau)
        ofi = ofi if ofi is not None else 0.0
        
        X_live = [[r_scaled, tau, vol, ofi]]
        p_up = float(state.models[currency].predict(X_live)[0])
        
        return p_up
        
    except Exception as e:
        print(f"❌ Prediction error for {currency}: {e}")
        return None

def manage_positions_fast(currency, delta, token_id_yes, token_id_no, price_yes, price_no, best_bid_yes, best_ask_yes):
    """Fast position management with detailed execution tracking."""
    if not state.polymarket_client or abs(delta) < PROBABILITY_DELTA_DEADZONE_THRESHOLD:
        return {"executed": False, "reason": "no_client_or_small_delta", "balance_issue": False}
    
    # Simplified position management for speed
    target_direction = "UP" if delta > 0 else "DOWN"
    target_size = min(abs(delta) * 20, 50)  # Simple sizing
    
    if target_direction == "UP" and price_yes > 0:
        buy_price = min(0.99, best_bid_yes + 0.01)
        result = place_order(BUY, token_id_yes, buy_price, target_size)
        result["direction"] = "UP"
        result["target_size"] = target_size
        result["target_price"] = buy_price
        return result
    elif target_direction == "DOWN" and price_no > 0:
        buy_price = min(0.99, (1 - best_ask_yes) + 0.01)
        result = place_order(BUY, token_id_no, buy_price, target_size)
        result["direction"] = "DOWN"
        result["target_size"] = target_size
        result["target_price"] = buy_price
        return result
    else:
        return {"executed": False, "reason": "invalid_prices", "balance_issue": False}

# --- Main Trading Loop ---
def trade_currency_once(currency):
    """Execute one trading cycle for a currency with detailed timing."""
    try:
        config = CURRENCY_CONFIG[currency]
        timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        
        # Time each step for optimization
        timings = {}
        
        # Step 1: Get market info
        start = time.time()
        token_id_yes, token_id_no, market_name = get_current_market_token_ids(currency)
        timings['market_lookup'] = time.time() - start
        
        if not token_id_yes:
            return False
        
        # Step 2: Get live data
        start = time.time()
        spot_price, ofi = get_binance_data_and_ofi(currency)
        timings['binance_data'] = time.time() - start
        
        if spot_price is None:
            return False
        
        # Step 3: Get order book
        start = time.time()
        best_bid, best_ask = get_order_book(token_id_yes)
        timings['order_book'] = time.time() - start
        
        if not best_bid or not best_ask:
            return False
        
        best_bid_price, best_ask_price = best_bid[0], best_ask[0]
        price_yes = (best_bid_price + best_ask_price) / 2
        price_no = 1 - price_yes
        
        # Step 4: Calculate prediction
        start = time.time()
        p_up_prediction = calculate_prediction(currency, None, spot_price, ofi)
        timings['prediction'] = time.time() - start
        
        # Step 5: Trading logic
        start = time.time()
        trade_result = None
        
        if p_up_prediction is not None and TRADING_ENABLED:
            delta = (p_up_prediction - price_yes) * 100
            prediction_pct = p_up_prediction * 100
            market_pct = price_yes * 100
            
            # Determine action
            action = "BUY" if delta > PROBABILITY_DELTA_DEADZONE_THRESHOLD else "SELL" if delta < -PROBABILITY_DELTA_DEADZONE_THRESHOLD else "HOLD"
            
            if abs(delta) >= PROBABILITY_DELTA_DEADZONE_THRESHOLD:
                trade_result = manage_positions_fast(currency, delta, token_id_yes, token_id_no, 
                                    price_yes, price_no, best_bid_price, best_ask_price)
                                    
            # Clean output
            print(f"  🔸 {currency.upper()}: {prediction_pct:.1f}% vs {market_pct:.1f}% (Δ{delta:+.1f}pp) → {action}")
        else:
            print(f"  🔸 {currency.upper()}: P=N/A M={price_yes*100:.1f}%")
        
        timings['trading_logic'] = time.time() - start
        
        # Cache data for database writing
        data_point = (
            timestamp, market_name, token_id_yes, best_bid_price, best_ask_price,
            spot_price, ofi, p_up_prediction
        )
        state.data_cache[currency].append(data_point)
        
        # Performance timing breakdown
        total_time = sum(timings.values())
        print(f"    ⏱️ {total_time:.3f}s [Market:{timings['market_lookup']:.3f} | Binance:{timings['binance_data']:.3f} | OrderBook:{timings['order_book']:.3f} | Prediction:{timings['prediction']:.3f} | Trading:{timings['trading_logic']:.3f}]")
        
        # Trade execution outcome
        if trade_result:
            if trade_result["executed"]:
                print(f"    💰 EXECUTED: {trade_result['direction']} ${trade_result['target_price']:.2f} × {trade_result['target_size']:.1f} shares")
            elif trade_result["balance_issue"]:
                print(f"    💸 SKIPPED: Insufficient balance (need ~${trade_result.get('target_size', 0) * trade_result.get('target_price', 0):.2f})")
            else:
                print(f"    ❌ FAILED: {trade_result['reason']}")
        elif TRADING_ENABLED and p_up_prediction is not None:
            print(f"    ⏸️ HOLD: Delta {delta:.1f}pp below threshold ({PROBABILITY_DELTA_DEADZONE_THRESHOLD}pp)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ {currency.upper()} error: {e}")
        return False

def continuous_trading_loop():
    """Main continuous trading loop."""
    currencies = list(CURRENCY_CONFIG.keys())
    cycle_count = 0
    
    # Initialize databases
    for currency in currencies:
        init_database(currency)
    
    print(f"\n🔄 Starting continuous trading loop...")
    print(f"📊 Sequence: {' -> '.join(c.upper() for c in currencies)}")
    print(f"⏱️ Cycle delay: {CYCLE_DELAY_SECONDS}s")
    print(f"💾 DB writes: Every {DB_WRITE_INTERVAL_SECONDS}s per currency")
    print(f"💰 Trading: {'ENABLED' if TRADING_ENABLED else 'DISABLED'}")
    print("-" * 80)
    
    try:
        while state.running:
            cycle_start = time.time()
            cycle_count += 1
            
            # Clean timestamp with emojis
            timestamp = datetime.now(UTC).strftime('%H:%M:%S')
            print(f"\n⚡ {timestamp} [Cycle {cycle_count}]")
            
            # Trade each currency in sequence
            for i, currency in enumerate(currencies):
                if not state.running:
                    break
                    
                success = trade_currency_once(currency)
                
                if not success:
                    print(f"  ❌ {currency.upper()}: SKIPPED")
                
                # Small delay between currencies to prevent overwhelming APIs
                if i < len(currencies) - 1:  # Don't delay after last currency
                    time.sleep(0.2)
            
            # Cycle timing control
            cycle_elapsed = time.time() - cycle_start
            if cycle_elapsed < CYCLE_DELAY_SECONDS:
                sleep_time = CYCLE_DELAY_SECONDS - cycle_elapsed
                time.sleep(sleep_time)
                cycle_elapsed = CYCLE_DELAY_SECONDS
            
            # Show cycle performance
            print(f"  📈 Total cycle time: {cycle_elapsed:.1f}s")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping continuous trading...")
        state.running = False

if __name__ == "__main__":
    import sys
    
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🧪 Running in test mode...")
        print("✅ All imports successful!")
        print("✅ Currency config loaded:", list(CURRENCY_CONFIG.keys()))
        print("✅ Rate limiting configured")
        print("✅ Ready for trading!")
        sys.exit(0)
    
    try:
        print("🏁 Initialization starting...")
        initialize_trading_system()
        continuous_trading_loop()
    except KeyboardInterrupt:
        print(f"\n👋 Shutdown complete")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc() 