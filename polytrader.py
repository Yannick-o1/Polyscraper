import requests
import pandas as pd
from datetime import datetime, UTC, timedelta
from zoneinfo import ZoneInfo
import re
import csv
import sqlite3
import lightgbm as lgb
import numpy as np
import os
import time
import json
import argparse

# Polymarket imports
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        OrderArgs,
        OrderType,
        BalanceAllowanceParams,
        AssetType,
        TradeParams,
    )
    from py_clob_client.order_builder.constants import BUY, SELL
    POLYMARKET_AVAILABLE = True
except ImportError:
    print("Warning: py-clob-client not installed. Order placement will be disabled.")
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
        'outcome_column': 'outcome',  # BTC uses generic outcome column
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

# --- Global Variables (set by initialize_currency) ---
CURRENCY = None
CONFIG = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://gamma-api.polymarket.com"
MARKETS_CSV_FILE = None
DB_FILE = None
MODEL_FILE = None
ASSET_SYMBOL = None

# --- Polymarket Configuration ---
TRADING_ENABLED = True
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_PROXY_ADDRESS = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
POLYMARKET_CHAIN_ID = 137
PROBABILITY_DELTA_DEADZONE_THRESHOLD = 3.0
BANKROLL_USAGE_FRACTION = 0.95
ORDER_SIZE_USD = 4.0  # Deprecated in favor of dynamic sizing
MINIMUM_ORDER_SIZE_SHARES = 5.0

# Make sure trading is disabled if the client is not available
if not POLYMARKET_AVAILABLE:
    TRADING_ENABLED = False

# Initialize Polymarket client
polymarket_client = None
model = None

def initialize_currency(currency):
    """Initialize global variables based on the selected currency."""
    global CURRENCY, CONFIG, MARKETS_CSV_FILE, DB_FILE, MODEL_FILE, ASSET_SYMBOL
    global polymarket_client, model
    
    if currency not in CURRENCY_CONFIG:
        raise ValueError(f"Unsupported currency: {currency}. Supported: {list(CURRENCY_CONFIG.keys())}")
    
    CURRENCY = currency
    CONFIG = CURRENCY_CONFIG[currency]
    MARKETS_CSV_FILE = os.path.join(BASE_DIR, f"{currency}_polymarkets.csv")
    DB_FILE = os.path.join(BASE_DIR, f"{currency}_polyscraper.db")
    MODEL_FILE = os.path.join(BASE_DIR, f"{currency}_lgbm.txt")
    ASSET_SYMBOL = CONFIG['asset_symbol']
    
    # Initialize Polymarket client
    if POLYMARKET_AVAILABLE and TRADING_ENABLED and POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS:
        try:
            polymarket_client = ClobClient(
                host=CLOB_API_URL,
                key=POLYMARKET_PRIVATE_KEY,
                chain_id=POLYMARKET_CHAIN_ID,
                signature_type=1,
                funder=POLYMARKET_PROXY_ADDRESS
            )
            polymarket_client.set_api_creds(polymarket_client.create_or_derive_api_creds())
            print(f"Polymarket client initialized successfully for {CONFIG['name']}")
        except Exception as e:
            print(f"Failed to initialize Polymarket client for {CONFIG['name']}: {e}")
            polymarket_client = None
    elif POLYMARKET_AVAILABLE:
        if TRADING_ENABLED:
            print(f"Polymarket credentials not provided for {CONFIG['name']}. Set environment variables to enable order placement.")
    else:
        print(f"Polymarket functionality disabled for {CONFIG['name']}.")
    
    # Load the LightGBM model
    try:
        model = lgb.Booster(model_file=MODEL_FILE)
    except Exception as e:
        print(f"Error loading model file '{MODEL_FILE}': {e}")
        print("Predictions will be disabled.")
        model = None

def place_order(side, token_id, price, size_shares):
    """Places a BUY or SELL order on Polymarket."""
    order_price = round(price, 2)
    size_usd = size_shares * order_price
    print(f"  [TRADE] => Placing Order: {side} {size_shares:.2f} shares @ ${order_price:.2f} (${size_usd:.2f} total)")
    
    if not polymarket_client:
        print("      -> Client not available.")
        return False
    
    if size_shares < 0.1:
        print(f"      -> SKIPPED: Order size {size_shares:.2f} shares too small.")
        return True

    try:
        if size_shares < MINIMUM_ORDER_SIZE_SHARES:
            print(f"      -> ❌ SKIPPED: Order size of {size_shares:.2f} shares is below the minimum of {MINIMUM_ORDER_SIZE_SHARES}.")
            return True

        expiration = int((datetime.now(UTC) + timedelta(minutes=2)).timestamp())
        
        order_args = OrderArgs(
            price=order_price,
            size=size_shares,
            side=side,
            token_id=token_id,
            expiration=expiration
        )
        
        signed_order = polymarket_client.create_order(order_args)
        response = polymarket_client.post_order(signed_order, OrderType.GTD)
        
        if response.get('success', False):
            print(f"      -> ✅ SUCCESS: Order placed.")
            return True
        else:
            print(f"      -> ❌ FAILED: {response.get('errorMsg', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"      -> ❌ ERROR placing order: {e}")
        return False

def get_p_start_from_binance(hour_start_utc):
    """Fetches the opening price for a specific hour from Binance 1-minute kline data with caching."""
    hour_key = f"{ASSET_SYMBOL}_{hour_start_utc.strftime('%Y-%m-%d %H')}"
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check cache first
        cursor.execute("SELECT p_start_price FROM p_start_cache WHERE hour_key = ?", (hour_key,))
        result = cursor.fetchone()
        if result:
            return result[0]

        # If miss, fetch new price from API
        print(f"Cache miss for p_start. Fetching from Binance for hour: {hour_key}")
        p_start_new = get_p_start_from_binance_api_call(hour_start_utc)

        # Check for previous hour's price in cache to resolve outcome
        previous_hour_utc = hour_start_utc - timedelta(hours=1)
        previous_hour_key = f"{ASSET_SYMBOL}_{previous_hour_utc.strftime('%Y-%m-%d %H')}"
        cursor.execute("SELECT p_start_price FROM p_start_cache WHERE hour_key = ?", (previous_hour_key,))
        prev_result = cursor.fetchone()

        if prev_result:
            p_start_previous = prev_result[0]
            p_start_current = p_start_new

            outcome = None
            if p_start_previous is not None and p_start_current is not None:
                if p_start_current > p_start_previous:
                    outcome = "UP"
                elif p_start_current < p_start_previous:
                    outcome = "DOWN"
                else:
                    outcome = "FLAT"
            
            if outcome:
                update_outcome_in_db(previous_hour_utc, outcome, p_start_previous, p_start_current)
        
        # Cache the new price in the database
        if p_start_new is not None:
            cursor.execute("INSERT OR REPLACE INTO p_start_cache (hour_key, p_start_price) VALUES (?, ?)", (hour_key, p_start_new))
            conn.commit()
        
        return p_start_new

    except sqlite3.Error as e:
        print(f"Database error in get_p_start_from_binance: {e}")
        return get_p_start_from_binance_api_call(hour_start_utc)
    finally:
        if conn:
            conn.close()

def get_p_start_from_binance_api_call(hour_start_utc):
    """Performs the actual API call to Binance for kline data."""
    try:
        start_time_ms = int(hour_start_utc.timestamp() * 1000)
        
        params = {
            "symbol": ASSET_SYMBOL,
            "interval": "1m",
            "startTime": start_time_ms,
            "limit": 1
        }
        resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=5)
        resp.raise_for_status()
        kline_data = resp.json()

        if not kline_data:
            print("Could not fetch kline data from Binance for p_start.")
            return None

        return float(kline_data[0][1])

    except Exception as e:
        print(f"Error getting p_start from Binance: {e}")
        return None

def update_outcome_in_db(previous_hour_utc, outcome, p_start_previous, p_start_current):
    """Updates all rows for a given hour with the resolved outcome."""
    try:
        _, _, market_name = get_market_token_ids_for_hour(previous_hour_utc, auto_update=False)
        print(f"Outcome for '{market_name}' was '{outcome}' (p_start={p_start_previous}, p_end={p_start_current}). Updating database...")
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        hour_start_str = previous_hour_utc.strftime('%Y-%m-%d %H:%M:%S')
        hour_end_str = (previous_hour_utc + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

        # Update both generic outcome and asset-specific outcome columns
        if CURRENCY == 'btc':
            cursor.execute("""
                UPDATE polydata 
                SET outcome = ? 
                WHERE timestamp >= ? AND timestamp < ?
            """, (outcome, hour_start_str, hour_end_str))
        else:
            cursor.execute(f"""
                UPDATE polydata 
                SET outcome = ?, {CONFIG['outcome_column']} = ?
                WHERE timestamp >= ? AND timestamp < ?
            """, (outcome, outcome, hour_start_str, hour_end_str))
        
        conn.commit()
        conn.close()
        print(f"Successfully updated {cursor.rowcount} rows for the previous hour.")
    except Exception as e:
        print(f"Error updating outcome in database: {e}")

def get_binance_data_and_ofi():
    """Fetch live price and calculate order flow imbalance from recent trades."""
    try:
        resp = requests.get("https://api.binance.com/api/v3/trades", 
                           params={"symbol": ASSET_SYMBOL, "limit": 1000}, timeout=5)
        resp.raise_for_status()
        trades = resp.json()
        
        if not trades:
            price_resp = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": ASSET_SYMBOL})
            price_resp.raise_for_status()
            latest_price = float(price_resp.json()['price'])
            return latest_price, 0.0

        df = pd.DataFrame(trades)
        latest_price = float(df.iloc[-1]["price"])
        df["qty"] = df["qty"].astype(float)
        
        buy_vol = df[~df["isBuyerMaker"]]["qty"].sum()
        sell_vol = df[df["isBuyerMaker"]]["qty"].sum()
        
        total_vol = buy_vol + sell_vol
        ofi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        
        return latest_price, ofi
        
    except Exception as e:
        print(f"\nError getting live {CONFIG['error_context']} data: {e}")
        return None, None

def calculate_live_prediction(historical_df, current_timestamp, current_ofi, p_start, current_price):
    """Calculate prediction using live data instead of historical lookup."""
    try:
        if p_start is None or current_price is None:
            return None
            
        df = historical_df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        
        df = df.sort_index()
        
        # Remove rows with missing prices
        df = df.dropna(subset=[CONFIG['db_column']])
        if len(df) < 2:
            return None
        
        r = current_price / p_start - 1
        
        current_minute = pd.to_datetime(current_timestamp).minute
        tau = 1 - (current_minute / 60)
        
        if tau <= 0:
            tau = 0.01
        
        # Calculate 20-minute rolling volatility on log returns
        df['lret'] = np.log(df[CONFIG['db_column']]).diff()
        
        rolling_vol = df['lret'].rolling(window=20, min_periods=2).std()
        vol = rolling_vol.iloc[-1] * np.sqrt(60) if not rolling_vol.empty else 0.0
            
        if pd.isna(vol) or vol == 0:
            vol = 0.01
            
        r_scaled = r / np.sqrt(tau)
        ofi = current_ofi if current_ofi is not None else 0.0
        
        # Diagnostic Logging
        print(f"\n  [PREDICT] Features:")
        print(f"      Spot: {current_price:<12.2f} | p_start: {p_start:<12.2f}")
        print(f"      r: {r:<10.6f} | tau: {tau:<7.4f} | vol: {vol:<10.6f} | ofi: {ofi:<7.4f}")
        print(f"      r_scaled: {r_scaled:<10.6f}")

        if pd.isna([r_scaled, tau, vol, ofi]).any():
            print("One of the features is NaN, skipping prediction.")
            return None
            
        X_live = [[r_scaled, tau, vol, ofi]]
        p_up = float(model.predict(X_live)[0])
        
        return p_up
        
    except Exception as e:
        print(f"\nError calculating live prediction: {e}")
        return None

def init_database():
    """Initializes the database and creates/updates the table if needed."""
    conn = sqlite3.connect(DB_FILE)
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
    
    # Check if columns exist and add them if they don't
    cursor.execute("PRAGMA table_info(polydata)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if CONFIG['db_column'] not in columns:
        cursor.execute(f"ALTER TABLE polydata ADD COLUMN {CONFIG['db_column']} REAL")
    if 'ofi' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN ofi REAL")
    if 'p_up_prediction' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN p_up_prediction REAL")
    if 'outcome' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN outcome TEXT")
    if CURRENCY != 'btc' and CONFIG['outcome_column'] not in columns:
        cursor.execute(f"ALTER TABLE polydata ADD COLUMN {CONFIG['outcome_column']} TEXT")

    conn.commit()
    conn.close()

def get_market_token_ids_for_hour(target_hour_dt_utc, auto_update=True):
    """Find the token IDs for the market corresponding to a specific hour."""
    try:
        markets_df = pd.read_csv(MARKETS_CSV_FILE)
        et_tz = ZoneInfo("America/New_York")
        target_hour_dt_et = target_hour_dt_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        matching_rows = markets_df[markets_df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            if auto_update:
                print(f"No market found for {target_datetime_str}. Auto-updating markets...")
                update_markets_csv()
                return get_market_token_ids_for_hour(target_hour_dt_utc, auto_update=False)
            return None, None, None
        
        market_row = matching_rows.iloc[0]
        token_id_yes = str(int(market_row['token_id_yes']))
        token_id_no = str(int(market_row['token_id_no']))
        market_name = market_row['market_name']
        
        return token_id_yes, token_id_no, market_name
        
    except FileNotFoundError:
        if auto_update:
            print(f"Error: '{MARKETS_CSV_FILE}' not found. Auto-updating markets...")
            update_markets_csv()
            return get_market_token_ids_for_hour(target_hour_dt_utc, auto_update=False)
        print(f"Error: '{MARKETS_CSV_FILE}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading or processing CSV for token ID lookup: {e}")
        return None, None, None

def parse_market_datetime(market_slug, market_question):
    """Extracts the market's date and time from its slug or question."""
    # Try primary pattern
    match = re.match(CONFIG['market_pattern'], market_slug)
    # Try alternative pattern
    if not match:
        match = re.match(CONFIG['market_pattern_alt'], market_slug)

    et_tz = ZoneInfo("America/New_York")
    now_in_et = datetime.now(et_tz)
    
    if match:
        month_name, day, hour, ampm = match.groups()
        try:
            month = list(map(lambda x: x.lower(), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])).index(month_name.lower()) + 1
            hour = int(hour)
            if ampm.lower() == 'pm' and hour != 12:
                hour += 12
            elif ampm.lower() == 'am' and hour == 12:
                hour = 0
            
            year = now_in_et.year
            market_dt = datetime(year, month, int(day), hour, 0, 0, tzinfo=et_tz)
            
            if market_dt < (now_in_et - timedelta(days=180)):
                market_dt = market_dt.replace(year=year + 1)
                
            return market_dt.strftime("%Y-%m-%d %H:%M EDT")
        except (ValueError, IndexError):
            pass

    # Fallback: Parse date from question text
    q_match = re.search(r'(\w+)\s+(\d+),?\s+(\d+)\s+(am|pm)\s+et', market_question, re.IGNORECASE)
    if q_match:
        month_name, day, hour, ampm = q_match.groups()
        try:
            month = list(map(lambda x: x.lower(), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])).index(month_name.lower()) + 1
            hour = int(hour)
            if ampm.lower() == 'pm' and hour != 12:
                hour += 12
            elif ampm.lower() == 'am' and hour == 12:
                hour = 0

            year = now_in_et.year
            market_dt = datetime(year, month, int(day), hour, 0, 0, tzinfo=et_tz)

            if market_dt < (now_in_et - timedelta(days=180)):
                market_dt = market_dt.replace(year=year + 1)

            return market_dt.strftime("%Y-%m-%d %H:%M EDT")
        except (ValueError, IndexError):
            pass

    return "Could not parse date"

def get_all_markets():
    """Fetches all markets from the Polymarket CLOB API using pagination."""
    url = f"{CLOB_API_URL}/markets"
    cursor = ""
    while True:
        try:
            resp = requests.get(url, params={"next_cursor": cursor} if cursor else {}).json()
            for m in resp["data"]:
                yield m
            cursor = resp["next_cursor"]
            if cursor in ("", "LTE="):
                break
        except requests.RequestException as e:
            print(f"Error fetching markets: {e}")
            break

def is_hourly_market(market):
    """Checks if a market is an hourly market for the current currency."""
    q = market["question"].lower()
    return CONFIG['market_question_pattern'] in q and ("pm et" in q or "am et" in q)

def update_markets_csv():
    """Fetches all hourly markets for the current currency and saves them to a CSV file."""
    market_data = []
    print("--- Updating market list ---")
    print(f"Searching for {CONFIG['name']} hourly markets...")
    market_count = 0

    for m in get_all_markets():
        if not is_hourly_market(m):
            continue
        
        market_count += 1
        slug = m.get("market_slug")
        question = m.get("question")
        condition_id = m.get("condition_id")
        
        token_id_yes = None
        token_id_no = None
        if m.get("tokens") and len(m["tokens"]) == 2:
            token_id_yes = m["tokens"][0].get("token_id")
            token_id_no = m["tokens"][1].get("token_id")
        else:
            token_id_yes, token_id_no = None, None

        parsed_date = parse_market_datetime(slug, question)
        
        print(f"Found market {market_count}: {slug}")
        print(f"  Question: {question}")
        print(f"  Token YES ID: {token_id_yes}")
        print(f"  Token NO ID: {token_id_no}")
        print(f"  Parsed Date: {parsed_date}")
        
        if not all([slug, question, token_id_yes, token_id_no, parsed_date != "Could not parse date"]):
            print(f"  -> REJECTED: Missing essential data")
            print()
            continue
        
        # Reject markets before July 18, 2025
        try:
            market_date = datetime.strptime(parsed_date.split(' ')[0], "%Y-%m-%d")
            cutoff_date = datetime(2025, 7, 18)
            if market_date < cutoff_date:
                print(f"  -> REJECTED: Before July 18, 2025")
                print()
                continue
        except (ValueError, IndexError):
            print(f"  -> REJECTED: Invalid date format")
            print()
            continue
        
        print(f"  -> ACCEPTED")
        print()
        
        market_data.append({
            "market_name": question,
            "token_id_yes": token_id_yes,
            "token_id_no": token_id_no,
            "date_time": parsed_date,
            "market_slug": slug,
            "condition_id": condition_id
        })
        
        time.sleep(0.1)

    if not market_data:
        print(f"No {CONFIG['name']} hourly markets found.")
        return

    market_data.sort(key=lambda x: x['date_time'])

    try:
        with open(MARKETS_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["market_name", "token_id_yes", "token_id_no", "date_time", "market_slug", "condition_id"])
            writer.writeheader()
            writer.writerows(market_data)
        print(f"Found {market_count} {CONFIG['name']} hourly markets")
        print(f"Successfully wrote {len(market_data)} valid markets to {MARKETS_CSV_FILE}")
    except IOError as e:
        print(f"Error writing to file {MARKETS_CSV_FILE}: {e}")

def get_current_market_token_ids():
    """Find the token IDs for the market that corresponds to the current date/time."""
    current_hour_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    return get_market_token_ids_for_hour(current_hour_utc)

def get_order_book(token_id):
    """Get the current order book for a given token ID with fallback logic."""
    try:
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

        # Fallback Logic
        print(f"Incomplete order book for token {token_id}. Falling back to local database.")
        conn = None
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT best_bid, best_ask FROM polydata WHERE token_id = ? ORDER BY timestamp DESC LIMIT 1",
                (token_id,)
            )
            last_reading = cursor.fetchone()
            
            if last_reading and last_reading[0] is not None and last_reading[1] is not None:
                last_bid, last_ask = last_reading
                last_midpoint = (last_bid + last_ask) / 2
                print(f"Fallback SUCCESS: Last DB midpoint was {last_midpoint:.4f}.")
                
                if last_midpoint > 0.5:
                    print("  -> Assuming market resolves UP. Setting price to 1.0.")
                    return (1.0, 0.0), (1.0, 0.0)
                else:
                    print("  -> Assuming market resolves DOWN. Setting price to 0.0.")
                    return (0.0, 0.0), (0.0, 0.0)
            else:
                print(f"Fallback FAILED: No valid previous data in DB for token {token_id}.")
                
        except sqlite3.Error as db_e:
            print(f"Fallback FAILED: Database error: {db_e}")
        finally:
            if conn:
                conn.close()
        
        print("Fallback failed. Cannot determine price.")
        return None, None
        
    except Exception as e:
        print(f"Error fetching order data for token {token_id}: {e}")
        return None, None

def get_user_state(token_id_yes, token_id_no):
    """Fetches user's USDC balance and positions in the given market."""
    if not polymarket_client:
        return None, None, None
    try:
        # Fetch USDC Balance
        usdc_balance_params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL, signature_type=-1
        )
        usdc_account_info = polymarket_client.get_balance_allowance(usdc_balance_params)
        usdc_balance = float(usdc_account_info["balance"]) / 1_000_000.0

        # Fetch YES Token Balance
        yes_balance_params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL, token_id=token_id_yes, signature_type=-1
        )
        yes_account_info = polymarket_client.get_balance_allowance(yes_balance_params)
        position_yes = float(yes_account_info["balance"]) / 1_000_000.0

        # Fetch NO Token Balance
        no_balance_params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL, token_id=token_id_no, signature_type=-1
        )
        no_account_info = polymarket_client.get_balance_allowance(no_balance_params)
        position_no = float(no_account_info["balance"]) / 1_000_000.0

        return usdc_balance, position_yes, position_no
    except Exception as e:
        print(f"Error getting user state: {e}")
        return None, None, None

def calculate_position_value(token_id, position_shares):
    """Calculates the current market value of a position."""
    if position_shares == 0:
        return 0.0
    
    try:
        best_bid, best_ask = get_order_book(token_id)
        if best_bid and best_ask:
            return position_shares * best_bid[0]
        return 0.0
    except Exception as e:
        print(f"Could not calculate position value for token {token_id}: {e}")
        return 0.0

def manage_positions(delta, token_id_yes, token_id_no, price_yes, price_no, best_bid_yes, best_ask_yes):
    """Adjusts user's position to match the target size based on delta."""
    print(f"\n--- Position Management ({CURRENCY.upper()}) ---")
    if not polymarket_client:
        print("-> Trading disabled, skipping position management.")
        return

    usdc_balance, position_yes, position_no = get_user_state(token_id_yes, token_id_no)

    if usdc_balance is None or position_yes is None or position_no is None:
        print("-> Could not retrieve full user state (balance or positions). Aborting position management.")
        return

    print(f"  [TRADE] Current State:")
    print(f"      YES: {position_yes:<8.4f} shares | NO: {position_no:<8.4f} shares | USDC: ${usdc_balance:<.2f}")
    
    # Value of current holdings for bankroll calculation
    value_yes_liquidatable = calculate_position_value(token_id_yes, position_yes)
    value_no_liquidatable = calculate_position_value(token_id_no, position_no)
    
    total_bankroll = usdc_balance + value_yes_liquidatable + value_no_liquidatable
    print(f"      Bankroll: ${total_bankroll:.2f}")

    # Target Calculation
    print(f"  [TRADE] Target Calculation:")
    target_shares_yes = 0.0
    target_shares_no = 0.0

    if abs(delta) < PROBABILITY_DELTA_DEADZONE_THRESHOLD:
        print(f"      -> Verdict: Delta |{delta:.2f}| is within dead zone. Goal is to close all positions.")
    else:
        target_value_usd = (abs(delta) / 100.0) * total_bankroll * BANKROLL_USAGE_FRACTION
        target_direction = "UP" if delta > 0 else "DOWN"
        print(f"      -> Verdict: Target {target_direction} position of ${target_value_usd:.2f}")

        if target_direction == "UP":
            if price_yes > 0:
                target_shares_yes = target_value_usd / price_yes
        else:
            if price_no > 0:
                target_shares_no = target_value_usd / price_no
    
    print(f"      Target Shares -> YES: {target_shares_yes:.4f} | NO: {target_shares_no:.4f}")

    # Position Adjustment
    print(f"  [TRADE] Position Adjustment:")

    # Adjust YES position
    adjustment_shares_yes = target_shares_yes - position_yes
    if abs(adjustment_shares_yes) < 0.1:
        print(f"      YES: No adjustment needed.")
    elif adjustment_shares_yes > 0.1:
        buy_price = min(0.99, best_bid_yes + 0.01)
        place_order(BUY, token_id_yes, buy_price, adjustment_shares_yes)
    elif adjustment_shares_yes < -0.1:
        sell_price = max(0.01, best_ask_yes - 0.01)
        place_order(SELL, token_id_yes, sell_price, abs(adjustment_shares_yes))
    
    # Adjust NO position
    adjustment_shares_no = target_shares_no - position_no
    if abs(adjustment_shares_no) < 0.1:
        print(f"      NO: No adjustment needed.")
    elif adjustment_shares_no > 0.1:
        buy_price = min(0.99, (1 - best_ask_yes) + 0.01)
        place_order(BUY, token_id_no, buy_price, adjustment_shares_no)
    elif adjustment_shares_no < -0.1:
        sell_price = max(0.01, (1 - best_bid_yes) - 0.01)
        place_order(SELL, token_id_no, sell_price, abs(adjustment_shares_no))
            
    print(f"--- End Position Management ({CURRENCY.upper()}) ---")

def collect_data_once():
    """Collect data for one minute and log it to the SQLite database."""
    init_database()
    
    try:
        t0 = datetime.now(UTC)
        asset_name = CURRENCY.upper()
        print("\n" + "="*80)
        print(f"[{t0.strftime('%Y-%m-%d %H:%M:%S')}] | {asset_name}USDT | Polyscraper Run")
        print("="*80)
        
        token_id_yes, token_id_no, market_name = get_current_market_token_ids()

        if not token_id_yes:
            print(f"  [INFO] No active market found for the current hour.")
            return
        
        print(f"\n  [INFO] Market: '{market_name}'")

        best_bid, best_ask = get_order_book(token_id_yes)
        t1 = datetime.now(UTC)
        print(f"  [INFO] Fetched Order Book in {(t1-t0).total_seconds():.4f}s")
        
        # Fetch the spot price and OFI from Binance
        spot_price, ofi = get_binance_data_and_ofi()
        
        # Prediction Logic
        p_up_prediction = None
        if model is not None and spot_price is not None:
            current_hour_start_utc = t0.replace(minute=0, second=0, microsecond=0)
            p_start = get_p_start_from_binance(current_hour_start_utc)

            # Fetch last 30 mins of data for volatility calculation
            conn = sqlite3.connect(DB_FILE)
            thirty_mins_ago = (t0 - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
            historical_df = pd.read_sql_query(
                f"SELECT timestamp, {CONFIG['db_column']} FROM polydata WHERE timestamp >= '{thirty_mins_ago}'", conn)
            conn.close()
            
            # Append current price to historical data
            if not historical_df.empty:
                current_data_df = pd.DataFrame([{'timestamp': t0.strftime('%Y-%m-%d %H:%M:%S'), CONFIG['db_column']: spot_price}])
                historical_df = pd.concat([historical_df, current_data_df], ignore_index=True)
            
            p_up_prediction = calculate_live_prediction(historical_df, t0, ofi, p_start, spot_price)

        if best_bid and best_ask:
            best_bid_price = best_bid[0]
            best_ask_price = best_ask[0]

            # Order Placement Logic
            if TRADING_ENABLED:
                if p_up_prediction is not None and best_bid_price is not None and best_ask_price is not None:
                    price_yes = (best_bid_price + best_ask_price) / 2
                    price_no = 1 - price_yes
                    delta = (p_up_prediction - price_yes) * 100
                    
                    prediction_pct = p_up_prediction * 100
                    market_pct = price_yes * 100
                    print(f"  [PREDICT] Prediction: {prediction_pct:.2f}% | Market: {market_pct:.2f}% | Delta: {delta:.2f} pp")
                    
                    manage_positions(delta, token_id_yes, token_id_no, price_yes, price_no, best_bid_price, best_ask_price)
            else:
                print(f"\n  [TRADE] Trading is disabled for {asset_name}.")

            # Database Logging
            data_row = {
                'timestamp': t0.strftime('%Y-%m-%d %H:%M:%S'),
                'market_name': market_name,
                CONFIG['db_column']: spot_price,
                'ofi': ofi,
                'p_up_prediction': p_up_prediction,
                'token_id': token_id_yes,
                'best_bid': best_bid_price,
                'best_ask': best_ask_price
            }

            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO polydata (timestamp, market_name, token_id, best_bid, best_ask, {CONFIG['db_column']}, ofi, p_up_prediction)
                VALUES (:timestamp, :market_name, :token_id, :best_bid, :best_ask, :{CONFIG['db_column']}, :ofi, :p_up_prediction)
            ''', data_row)
            conn.commit()
            conn.close()

            t3 = datetime.now(UTC)
            spot_price_str = f"{spot_price:.2f}" if spot_price is not None else "N/A"
            prediction_str = f"{p_up_prediction:.4f}" if p_up_prediction is not None else "N/A"
            print(f"\n  [LOG]   DB record created for {asset_name}={spot_price_str}, P(Up)={prediction_str}, Bid={best_bid_price:.2f}, Ask={best_ask_price:.2f}")
            print(f"  [END]   Run finished in {(t3-t0).total_seconds():.4f}s.\n")

        else:
            print(f"  [INFO] Could not retrieve valid order book for '{market_name}'")

    except Exception as e:
        print(f"An unexpected error occurred during data collection for {asset_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Polymarket trading bot for cryptocurrency hourly markets.",
        epilog="Example: python polytrader.py btc --run-once"
    )
    parser.add_argument('currency', choices=['btc', 'eth', 'sol', 'xrp'], 
                       help='Currency to trade (btc, eth, sol, xrp)')
    parser.add_argument('--update-markets', action='store_true', 
                       help='Only update the markets CSV and exit.')
    parser.add_argument('--run-once', action='store_true', 
                       help='Run a single data collection and exit.')
    args = parser.parse_args()

    # Initialize the currency-specific configuration
    initialize_currency(args.currency)

    if args.update_markets:
        update_markets_csv()
    elif args.run_once:
        collect_data_once()
    else:
        parser.print_help() 