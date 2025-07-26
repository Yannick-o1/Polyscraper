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
import json  # Import json for state persistence

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

# --- Globals ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://gamma-api.polymarket.com"  # Correct, verified Data API URL
MARKETS_CSV_FILE = os.path.join(BASE_DIR, "sol_polymarkets.csv")
DB_FILE = os.path.join(BASE_DIR, "sol_polyscraper.db")
MODEL_FILE = os.path.join(BASE_DIR, "sol_lgbm.txt")
ASSET_SYMBOL = "SOLUSDT"

# --- Polymarket Configuration ---
TRADING_ENABLED = True  # SET TO True TO ENABLE LIVE TRADING
# Set these environment variables or modify directly:
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")  # Your private key
POLYMARKET_PROXY_ADDRESS = os.getenv("POLYMARKET_PROXY_ADDRESS", "")  # Your proxy wallet address
POLYMARKET_CHAIN_ID = 137  # Polygon mainnet
PROBABILITY_DELTA_DEADZONE_THRESHOLD = 3.0 # If abs(delta) is less than this, close positions
BANKROLL_USAGE_FRACTION = 0.7 # Fraction of total bankroll to use for sizing
# TOTAL_BANKROLL_USD = 100.0 # No longer needed, will be fetched dynamically
ORDER_SIZE_USD = 4.0  # Deprecated in favor of dynamic sizing
MINIMUM_ORDER_SIZE_SHARES = 5.0 # Minimum shares per order, based on API error

# Make sure trading is disabled if the client is not available
if not POLYMARKET_AVAILABLE:
    TRADING_ENABLED = False

# Initialize Polymarket client
polymarket_client = None
if POLYMARKET_AVAILABLE and TRADING_ENABLED and POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS:
    try:
        polymarket_client = ClobClient(
            host=CLOB_API_URL,
            key=POLYMARKET_PRIVATE_KEY,
            chain_id=POLYMARKET_CHAIN_ID,
            signature_type=1,  # For email/magic accounts, use 2 for browser wallets
            funder=POLYMARKET_PROXY_ADDRESS
        )
        polymarket_client.set_api_creds(polymarket_client.create_or_derive_api_creds())
        print("Polymarket client initialized successfully for SOL")
    except Exception as e:
        print(f"Failed to initialize Polymarket client for SOL: {e}")
        polymarket_client = None
elif POLYMARKET_AVAILABLE:
    if TRADING_ENABLED:
        print("Polymarket credentials not provided for SOL. Set POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS environment variables to enable order placement.")
else:
    print("Polymarket functionality disabled for SOL.")

# Load the LightGBM model once when the script starts
try:
    model = lgb.Booster(model_file=MODEL_FILE)
except Exception as e:
    print(f"Error loading model file '{MODEL_FILE}': {e}")
    print("Predictions will be disabled.")
    model = None

def place_order(side, token_id, price, size_usd):
    """
    Places a BUY or SELL order on Polymarket.
    
    Args:
        side (str): "BUY" or "SELL"
        token_id (str): The token ID for the order.
        price (float): The price for the order (0-1 range).
        size_usd (float): Order size in USD.
    
    Returns:
        bool: True if order placed successfully, False otherwise
    """
    print(f"  --> Attempting to place order: {side} ${size_usd:.2f} of token {token_id} at price {price:.2f}")
    if not polymarket_client:
        print("  --> Polymarket client not available")
        return False
    
    if size_usd <= 0.1: # Minimum order size to avoid dust
        print(f"  --> Order size ${size_usd:.2f} too small, skipping.")
        return True # Return true to not halt any batch processing

    try:
        # Calculate order size in shares
        order_price = round(price, 2)
        if order_price <= 0.01: # Prevent division by zero or tiny price
            order_price = 0.01
            
        size_shares = size_usd / order_price

        # Check against the minimum share size required by the API
        if size_shares < MINIMUM_ORDER_SIZE_SHARES:
            print(f"  --> Order size of {size_shares:.2f} shares is below the minimum of {MINIMUM_ORDER_SIZE_SHARES}, skipping.")
            return True

        # Create order with a 2-minute expiration
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
            print(f"  --> ✅ Order placed: {side} {size_shares:.2f} shares of token {token_id} at ${order_price:.2f} (${size_usd:.2f} total)")
            return True
        else:
            print(f"  --> ❌ Order failed: {response.get('errorMsg', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"  --> Error placing order: {e}")
        return False

def get_p_start_from_binance(hour_start_utc):
    """
    Fetches the opening price for a specific hour from Binance 1-minute kline data.
    Uses a database cache to avoid redundant API calls.
    """
    hour_key = f"{ASSET_SYMBOL}_{hour_start_utc.strftime('%Y-%m-%d %H')}"
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 1. Check cache first
        cursor.execute("SELECT p_start_price FROM p_start_cache WHERE hour_key = ?", (hour_key,))
        result = cursor.fetchone()
        if result:
            return result[0]

        # 2. If miss, fetch new price from API
        print(f"Cache miss for p_start. Fetching from Binance for hour: {hour_key}")
        p_start_new = get_p_start_from_binance_api_call(hour_start_utc)

        # 3. Check for previous hour's price in cache to resolve outcome
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
        
        # 4. Cache the new price in the database
        if p_start_new is not None:
            cursor.execute("INSERT OR REPLACE INTO p_start_cache (hour_key, p_start_price) VALUES (?, ?)", (hour_key, p_start_new))
            conn.commit()
        
        return p_start_new

    except sqlite3.Error as e:
        print(f"Database error in get_p_start_from_binance: {e}")
        # Fallback to API call without caching if DB fails
        return get_p_start_from_binance_api_call(hour_start_utc)
    finally:
        if conn:
            conn.close()

def get_p_start_from_binance_api_call(hour_start_utc):
    """Performs the actual API call to Binance for kline data."""
    try:
        # Binance klines use milliseconds timestamps
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

        # kline format: [open_time, open, high, low, close, ...]
        return float(kline_data[0][1])

    except Exception as e:
        print(f"Error getting p_start from Binance: {e}")
        return None

def update_outcome_in_db(previous_hour_utc, outcome, p_start_previous, p_start_current):
    """Updates all rows for a given hour with the resolved outcome."""
    try:
        _, market_name = get_market_token_id_for_hour(previous_hour_utc, auto_update=False)
        print(f"Outcome for '{market_name}' was '{outcome}' (p_start={p_start_previous}, p_end={p_start_current}). Updating database...")
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        hour_start_str = previous_hour_utc.strftime('%Y-%m-%d %H:%M:%S')
        hour_end_str = (previous_hour_utc + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute("""
            UPDATE polydata 
            SET outcome = ? 
            WHERE timestamp >= ? AND timestamp < ?
        """, (outcome, hour_start_str, hour_end_str))
        
        conn.commit()
        conn.close()
        print(f"Successfully updated {cursor.rowcount} rows for the previous hour.")
    except Exception as e:
        print(f"Error updating outcome in database: {e}")

def get_binance_data_and_ofi():
    """Fetch live SOL/USDT price and calculate order flow imbalance from recent trades."""
    try:
        # Fetch the last 1000 trades to ensure we have recent data
        resp = requests.get("https://api.binance.com/api/v3/trades", 
                           params={"symbol": ASSET_SYMBOL, "limit": 1000}, timeout=5)
        resp.raise_for_status()
        trades = resp.json()
        
        if not trades:
            # Fallback to ticker price if no trades are returned
            price_resp = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": ASSET_SYMBOL})
            price_resp.raise_for_status()
            latest_price = float(price_resp.json()['price'])
            return latest_price, 0.0

        # Create a DataFrame from the trades
        df = pd.DataFrame(trades)
        latest_price = float(df.iloc[-1]["price"])
        df["qty"] = df["qty"].astype(float)
        
        # In Binance trade data, isBuyerMaker=False means the taker was a buyer (a market buy)
        buy_vol = df[~df["isBuyerMaker"]]["qty"].sum()
        sell_vol = df[df["isBuyerMaker"]]["qty"].sum()
        
        total_vol = buy_vol + sell_vol
        ofi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        
        return latest_price, ofi
        
    except Exception as e:
        print(f"\nError getting live Solana data: {e}")
        return None, None

def calculate_live_prediction(historical_df, current_timestamp, current_ofi, p_start, current_price):
    """Calculate prediction using live data instead of historical lookup"""
    try:
        if p_start is None or current_price is None:
            return None
            
        df = historical_df.copy()
        
        # Ensure 'timestamp' is the index and it's a datetime object
        if 'timestamp' in df.columns:
            # Tell pandas to interpret the timestamps as UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        
        df = df.sort_index()
        
        # Remove rows with missing prices
        df = df.dropna(subset=['sol_usdt_spot'])
        if len(df) < 2:
            return None
        
        r = current_price / p_start - 1
        
        current_minute = pd.to_datetime(current_timestamp).minute
        tau = 1 - (current_minute / 60)
        
        if tau <= 0:
            tau = 0.01
        
        # Calculate 20-minute rolling volatility on log returns
        df['lret'] = np.log(df['sol_usdt_spot']).diff()
        
        # Use a rolling window of 20 periods (minutes) for std deviation
        rolling_vol = df['lret'].rolling(window=20, min_periods=2).std()
        
        # Get the most recent volatility value and scale it to the hour
        vol = rolling_vol.iloc[-1] * np.sqrt(60) if not rolling_vol.empty else 0.0
            
        if pd.isna(vol) or vol == 0:
            vol = 0.01
            
        r_scaled = r / np.sqrt(tau)
        
        ofi = current_ofi if current_ofi is not None else 0.0
        
        # --- Diagnostic Logging ---
        print("--- Prediction Feature Calculation ---")
        print(f"p_start: {p_start}, current_price: {current_price}")
        print(f"r: {r:.6f}, tau: {tau:.4f}, vol: {vol:.6f}, ofi: {ofi:.4f}")
        print(f"r_scaled: {r_scaled:.6f}")
        # --- End Diagnostic Logging ---

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

    # Create the cache table for p_start values
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS p_start_cache (
            hour_key TEXT PRIMARY KEY,
            p_start_price REAL NOT NULL
        )
    ''')
    
    # Check if the columns exist and add them if they don't
    cursor.execute("PRAGMA table_info(polydata)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'sol_usdt_spot' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN sol_usdt_spot REAL")
    if 'ofi' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN ofi REAL")
    if 'p_up_prediction' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN p_up_prediction REAL")
    if 'outcome' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN outcome TEXT")
    if 'outcome_sol' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN outcome_sol TEXT")

    conn.commit()
    conn.close()

def get_market_token_id_for_hour(target_hour_dt_utc, auto_update=True):
    """Find the token ID for the market corresponding to a specific hour."""
    try:
        markets_df = pd.read_csv(MARKETS_CSV_FILE)
        et_tz = ZoneInfo("America/New_York")
        # Convert the target UTC datetime to America/New_York timezone for lookup
        target_hour_dt_et = target_hour_dt_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        matching_rows = markets_df[markets_df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            if auto_update:
                print(f"No market found for {target_datetime_str}. Auto-updating markets...")
                update_markets_csv()
                # Retry once after updating markets
                return get_market_token_id_for_hour(target_hour_dt_utc, auto_update=False)
            return None, None
        
        market_row = matching_rows.iloc[0]
        token_id = str(int(market_row['token_id']))
        market_name = market_row['market_name']
        
        return token_id, market_name
        
    except FileNotFoundError:
        if auto_update:
            print(f"Error: '{MARKETS_CSV_FILE}' not found. Auto-updating markets...")
            update_markets_csv()
            # Retry once after updating markets
            return get_market_token_id_for_hour(target_hour_dt_utc, auto_update=False)
        print(f"Error: '{MARKETS_CSV_FILE}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading or processing CSV for token ID lookup: {e}")
        return None, None

def get_market_token_ids_for_hour(target_hour_dt_utc, auto_update=True):
    """Find the token IDs for the market corresponding to a specific hour."""
    try:
        markets_df = pd.read_csv(MARKETS_CSV_FILE)
        et_tz = ZoneInfo("America/New_York")
        # Convert the target UTC datetime to America/New_York timezone for lookup
        target_hour_dt_et = target_hour_dt_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        matching_rows = markets_df[markets_df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            if auto_update:
                print(f"No market found for {target_datetime_str}. Auto-updating markets...")
                update_markets_csv()
                # Retry once after updating markets
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
            # Retry once after updating markets
            return get_market_token_ids_for_hour(target_hour_dt_utc, auto_update=False)
        print(f"Error: '{MARKETS_CSV_FILE}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading or processing CSV for token ID lookup: {e}")
        return None, None, None

def parse_market_datetime(market_slug, market_question):
    """
    Extracts the market's date and time from its slug or question.
    Handles different slug formats and timezones, and corrects for year-end rollovers.
    """
    # Pattern 1: solana-up-or-down-{month}-{day}-{hour}{am|pm}-et
    match = re.match(r'solana-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et', market_slug)
    # Pattern 2: solana-up-or-down-{month}-{day}-{hour}-{am|pm}-et
    if not match:
        match = re.match(r'solana-up-or-down-(\w+)-(\d+)-(\d+)-(am|pm)-et', market_slug)

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
            
            # Only bump to next year if it's really far in the past (more than 6 months)
            # This prevents 2025 dates from being incorrectly set to 2026
            if market_dt < (now_in_et - timedelta(days=180)):
                market_dt = market_dt.replace(year=year + 1)
                
            return market_dt.strftime("%Y-%m-%d %H:%M EDT")
        except (ValueError, IndexError):
            pass  # Fallback to question parsing

    # Fallback: Parse date from question text, e.g., "June 10, 5 PM ET"
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

            # Only bump to next year if it's really far in the past (more than 6 months)
            # This prevents 2025 dates from being incorrectly set to 2026
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
            if cursor in ("", "LTE="):  # End of pagination
                break
        except requests.RequestException as e:
            print(f"Error fetching markets: {e}")
            break

def is_solana_hourly_market(market):
    """Checks if a market is a Solana hourly market based on its question."""
    q = market["question"].lower()
    return "solana up or down" in q and ("pm et" in q or "am et" in q)

def update_markets_csv():
    """
    Fetches all Solana hourly markets and saves them to a CSV file.
    """
    market_data = []
    print("--- Updating market list ---")
    print("Searching for Solana hourly markets...")
    solana_market_count = 0

    for m in get_all_markets():
        if not is_solana_hourly_market(m):
            continue
        
        solana_market_count += 1
        slug = m.get("market_slug")
        question = m.get("question")
        condition_id = m.get("condition_id")
        
        # Find the YES (Up) and NO (Down) tokens
        token_id_yes = None
        token_id_no = None
        if m.get("tokens") and len(m["tokens"]) == 2:
            # Polymarket's first token (index 0) is typically the "YES" outcome
            token_id_yes = m["tokens"][0].get("token_id")
            token_id_no = m["tokens"][1].get("token_id")
        else:
            token_id_yes, token_id_no = None, None

        # Parse the date from the market
        parsed_date = parse_market_datetime(slug, question)
        
        print(f"Found market {solana_market_count}: {slug}")
        print(f"  Question: {question}")
        print(f"  Token YES ID: {token_id_yes}")
        print(f"  Token NO ID: {token_id_no}")
        print(f"  Parsed Date: {parsed_date}")
        
        # Reject markets with missing essential data
        if not all([slug, question, token_id_yes, token_id_no, parsed_date != "Could not parse date"]):
            print(f"  -> REJECTED: Missing essential data")
            print()
            continue
        
        # Reject markets before July 18, 2025
        try:
            from datetime import datetime
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
        
        # Add to market_data
        market_data.append({
            "market_name": question,
            "token_id_yes": token_id_yes,
            "token_id_no": token_id_no,
            "date_time": parsed_date,
            "market_slug": slug,
            "condition_id": condition_id
        })
        
        time.sleep(0.1)  # Small delay to be respectful to the API

    if not market_data:
        print("No Solana hourly markets found.")
        return

    # Sort by date_time - only valid dates will be here now
    market_data.sort(key=lambda x: x['date_time'])

    try:
        with open(MARKETS_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["market_name", "token_id_yes", "token_id_no", "date_time", "market_slug", "condition_id"])
            writer.writeheader()
            writer.writerows(market_data)
        print(f"Found {solana_market_count} Solana hourly markets")
        print(f"Successfully wrote {len(market_data)} valid markets to {MARKETS_CSV_FILE}")
    except IOError as e:
        print(f"Error writing to file {MARKETS_CSV_FILE}: {e}")


def get_current_market_token_ids():
    """Find the token IDs for the Solana market that corresponds to the current date/time"""
    current_hour_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    return get_market_token_ids_for_hour(current_hour_utc)

def get_order_book(token_id):
    """
    Get the current order book for a given token ID.
    If the order book is thin or empty, it uses a fallback to the last recorded 
    price in our own database to infer the outcome.
    """
    try:
        # First, try to get the full order book
        response = requests.get(f"{CLOB_API_URL}/book", params={"token_id": token_id}, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if bids and asks:
            # Happy path: parse bids and asks, sort them, and return the best
            parsed_bids = [(float(b['price']), float(b['size'])) for b in bids]
            parsed_asks = [(float(a['price']), float(a['size'])) for a in asks]
            parsed_bids.sort(reverse=True)
            parsed_asks.sort()
            return parsed_bids[0], parsed_asks[0]

        # --- Fallback Logic ---
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

def get_all_trades_for_token(token_id):
    """Fetches all pages of trades for a given token ID."""
    try:
        # It seems the `get_trades` method in the library *does* handle pagination itself, returning a single list.
        # Therefore, the error must be in the processing logic. Let's correct it.
        return polymarket_client.get_trades(params=TradeParams(asset_id=token_id))

    except Exception as e:
        print(f"Error fetching trades for token {token_id}: {e}")
        return [] # Return empty list on error


def get_user_state(token_id_yes, token_id_no):
    """Fetches user's USDC balance and positions in the given market."""
    if not polymarket_client or not POLYMARKET_PROXY_ADDRESS:
        return None, None, None
    try:
        # 1. Fetch USDC Balance (this part is confirmed working)
        balance_params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL, signature_type=-1
        )
        account_info = polymarket_client.get_balance_allowance(balance_params)
        usdc_balance = float(account_info["balance"]) / 1_000_000.0

        # 2. Calculate Net Position from Trade History
        user_address = POLYMARKET_PROXY_ADDRESS.lower()
        
        # --- YES Token Position ---
        position_yes = 0.0
        trades_yes = get_all_trades_for_token(token_id_yes)
        if trades_yes is None: # Defensive check in case the API returns None
            trades_yes = []
        for trade in trades_yes:
            size = float(trade.get("size", 0.0))
            maker_address = trade.get("maker_address", "").lower()
            taker_address = trade.get("taker_address", "").lower()

            if taker_address == user_address:
                # If we were the TAKER: our position changes according to the side.
                if trade.get("side") == "buy":
                    position_yes += size
                else: # sell
                    position_yes -= size
            elif maker_address == user_address:
                # If we were the MAKER: our position changes according to the side of our original resting order.
                if trade.get("side") == "buy": # A taker bought from our sell order
                    position_yes -= size
                else: # sell, a taker sold to our buy order
                    position_yes += size
        
        # --- NO Token Position ---
        position_no = 0.0
        trades_no = get_all_trades_for_token(token_id_no)
        if trades_no is None: # Defensive check in case the API returns None
            trades_no = []
        for trade in trades_no:
            size = float(trade.get("size", 0.0))
            maker_address = trade.get("maker_address", "").lower()
            taker_address = trade.get("taker_address", "").lower()

            if taker_address == user_address:
                if trade.get("side") == "buy":
                    position_no += size
                else: # sell
                    position_no -= size
            elif maker_address == user_address:
                if trade.get("side") == "buy": # A taker bought from our sell order
                    position_no -= size
                else: # sell, a taker sold to our buy order
                    position_no += size

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
            # We can sell at the best bid price
            return position_shares * best_bid[0]
        return 0.0
    except Exception as e:
        print(f"Could not calculate position value for token {token_id}: {e}")
        return 0.0

def manage_positions(delta, token_id_yes, token_id_no, price_yes, price_no):
    """
    Adjusts user's position to match the target size based on delta.
    - Sells positions if delta is in the dead zone.
    - Adjusts UP/DOWN positions based on delta magnitude and sign.
    """
    print("\n--- Position Management (SOL) ---")
    if not polymarket_client:
        print("-> Trading disabled, skipping position management.")
        return

    usdc_balance, position_yes, position_no = get_user_state(token_id_yes, token_id_no)

    if usdc_balance is None or position_yes is None or position_no is None:
        print("-> Could not retrieve full user state (balance or positions). Aborting position management.")
        return

    print(f"-> State: YES shares={position_yes:.4f}, NO shares={position_no:.4f}, USDC=${usdc_balance:.2f}")
    
    # Value of current holdings
    value_yes = calculate_position_value(token_id_yes, position_yes)
    value_no = calculate_position_value(token_id_no, position_no)
    
    print(f"-> Value: YES=${value_yes:.2f}, NO=${value_no:.2f}")

    # Dynamically calculate total bankroll
    total_bankroll = usdc_balance + value_yes + value_no
    print(f"-> Total Bankroll (USDC + Positions) = ${total_bankroll:.2f}")

    # --- Target Calculation ---
    print("-> Calculating Target...")
    target_value = 0
    target_direction = None

    if abs(delta) < PROBABILITY_DELTA_DEADZONE_THRESHOLD:
        print(f"-> Verdict: Delta |{delta:.2f}| is within dead zone (< {PROBABILITY_DELTA_DEADZONE_THRESHOLD}). Goal is to close all positions.")
        target_value = 0
    else:
        target_value = (abs(delta) / 100.0) * total_bankroll * BANKROLL_USAGE_FRACTION
        target_direction = "UP" if delta > 0 else "DOWN"
        print(f"-> Verdict: Target is a {target_direction} position of ${target_value:.2f}")

    # --- Position Adjustment ---
    print("-> Adjusting Positions...")
    # 1. Close unwanted positions first
    if target_direction != "UP" and position_yes > 0:
        print(f"-> Action: Target is not UP. Closing YES position of {position_yes:.4f} shares (value ${value_yes:.2f}).")
        place_order(SELL, token_id_yes, price_yes, value_yes)
    
    if target_direction != "DOWN" and position_no > 0:
        print(f"-> Action: Target is not DOWN. Closing NO position of {position_no:.4f} shares (value ${value_no:.2f}).")
        place_order(SELL, token_id_no, price_no, value_no)

    # 2. Adjust target position
    if target_direction == "UP":
        adjustment_usd = target_value - value_yes
        print(f"-> UP Adjustment: Target=${target_value:.2f}, Current=${value_yes:.2f}, Diff=${adjustment_usd:.2f}")
        if adjustment_usd > 0.1: # Threshold to avoid tiny orders
            print(f"-> Action: BUY ${adjustment_usd:.2f} of YES token.")
            place_order(BUY, token_id_yes, price_yes, adjustment_usd)
        elif adjustment_usd < -0.1: # Threshold to avoid tiny orders
            print(f"-> Action: SELL ${abs(adjustment_usd):.2f} of YES token.")
            place_order(SELL, token_id_yes, price_yes, abs(adjustment_usd))
        else:
            print("-> Action: No significant adjustment needed for UP position.")

    elif target_direction == "DOWN":
        adjustment_usd = target_value - value_no
        print(f"-> DOWN Adjustment: Target=${target_value:.2f}, Current=${value_no:.2f}, Diff=${adjustment_usd:.2f}")
        if adjustment_usd > 0.1: # Threshold to avoid tiny orders
            print(f"-> Action: BUY ${adjustment_usd:.2f} of NO token.")
            place_order(BUY, token_id_no, price_no, adjustment_usd)
        elif adjustment_usd < -0.1:
            print(f"-> Action: SELL ${abs(adjustment_usd):.2f} of NO token.")
            place_order(SELL, token_id_no, price_no, abs(adjustment_usd))
        else:
            print("-> Action: No significant adjustment needed for DOWN position.")
            
    print("--- End Position Management (SOL) ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Polymarket scraper for Solana hourly markets.",
        epilog="Default action is to do nothing. Use --run-once to collect data or --update-markets to refresh the market list."
    )
    parser.add_argument('--update-markets', action='store_true', help='Only update the markets CSV and exit.')
    parser.add_argument('--run-once', action='store_true', help='Run a single data collection and exit.')
    args = parser.parse_args()

    if args.update_markets:
        update_markets_csv()
    elif args.run_once:
        collect_data_once()
    else:
        parser.print_help() 