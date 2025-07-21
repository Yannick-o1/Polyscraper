import requests
import pandas as pd
from datetime import datetime, UTC, timedelta
from zoneinfo import ZoneInfo
import time
import os
import re
import csv
import sqlite3
import lightgbm as lgb
import numpy as np

# --- Globals ---
CLOB_API_URL = "https://clob.polymarket.com"
MARKETS_CSV_FILE = "btc_polymarkets.csv"
DB_FILE = "polyscraper.db"
MODEL_FILE = "btc_lgbm3.txt"
P_START_CACHE = {} # Simple cache: { "YYYY-MM-DD HH": price }

# Load the LightGBM model once when the script starts
try:
    model = lgb.Booster(model_file=MODEL_FILE)
except lgb.LGBMError as e:
    print(f"Error loading model file '{MODEL_FILE}': {e}")
    print("Predictions will be disabled.")
    model = None

def get_p_start_from_binance(hour_start_utc):
    """
    Fetches the opening price for a specific hour from Binance 1-minute kline data.
    Uses a cache to avoid redundant API calls.
    """
    global P_START_CACHE
    hour_key = hour_start_utc.strftime('%Y-%m-%d %H')
    
    # Check cache first
    if hour_key in P_START_CACHE:
        return P_START_CACHE[hour_key]

    print(f"Cache miss for p_start. Fetching from Binance for hour: {hour_key}")
    try:
        # Binance klines use milliseconds timestamps
        start_time_ms = int(hour_start_utc.timestamp() * 1000)
        
        params = {
            "symbol": "BTCUSDT",
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
        p_start = float(kline_data[0][1])
        
        # Store in cache and return
        P_START_CACHE[hour_key] = p_start
        return p_start

    except Exception as e:
        print(f"Error getting p_start from Binance: {e}")
        return None

def get_binance_data_and_ofi():
    """Fetch live BTC/USDT price and calculate order flow imbalance from recent trades."""
    try:
        # Fetch the last 1000 trades to ensure we have recent data
        resp = requests.get("https://api.binance.com/api/v3/trades", 
                           params={"symbol": "BTCUSDT", "limit": 1000}, timeout=5)
        resp.raise_for_status()
        trades = resp.json()
        
        if not trades:
            # Fallback to ticker price if no trades are returned
            price_resp = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
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
        print(f"\nError getting live Bitcoin data: {e}")
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
        df = df.dropna(subset=['btc_usdt_spot'])
        if len(df) < 2:
            return None
        
        r = current_price / p_start - 1
        
        current_minute = pd.to_datetime(current_timestamp).minute
        tau = 1 - (current_minute / 60)
        
        if tau <= 0:
            tau = 0.01
        
        # Calculate 20-minute rolling volatility on log returns
        df['lret'] = np.log(df['btc_usdt_spot']).diff()
        
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
        p_up = model.predict(X_live)[0]
        
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
    
    # Check if the btc_usdt_spot column exists and add it if it doesn't
    cursor.execute("PRAGMA table_info(polydata)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'btc_usdt_spot' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN btc_usdt_spot REAL")
    if 'ofi' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN ofi REAL")
    if 'p_up_prediction' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN p_up_prediction REAL")
    if 'outcome' not in columns:
        cursor.execute("ALTER TABLE polydata ADD COLUMN outcome TEXT")

    conn.commit()
    conn.close()

def get_market_token_id_for_hour(target_hour_dt_utc):
    """Find the token ID for the market corresponding to a specific hour."""
    try:
        markets_df = pd.read_csv(MARKETS_CSV_FILE)
        et_tz = ZoneInfo("America/New_York")
        # Convert the target UTC datetime to America/New_York timezone for lookup
        target_hour_dt_et = target_hour_dt_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        matching_rows = markets_df[markets_df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            return None, None
        
        market_row = matching_rows.iloc[0]
        token_id = str(int(market_row['token_id']))
        market_name = market_row['market_name']
        
        return token_id, market_name
        
    except FileNotFoundError:
        print(f"Error: '{MARKETS_CSV_FILE}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading or processing CSV for token ID lookup: {e}")
        return None, None

def check_and_update_outcome(current_time_utc):
    """At the top of the hour, check the price of the market that just closed and update its outcome."""
    if current_time_utc.minute != 0:
        return # Only run this logic at the top of the hour (e.g., 16:00)

    print(f"({current_time_utc.strftime('%H:%M:%S')}) Checking outcome for market that just resolved...")
    
    # Determine the start of the previous hour (the market that just ended)
    previous_hour_utc = current_time_utc - timedelta(hours=1)
    previous_hour_start_utc = previous_hour_utc.replace(minute=0, second=0, microsecond=0)
    
    token_id, market_name = get_market_token_id_for_hour(previous_hour_start_utc)

    if not token_id:
        print(f"Could not find market for resolved hour: {previous_hour_start_utc.strftime('%Y-%m-%d %H:%M')} UTC")
        return

    # Fetch the last known price for the resolved market
    outcome = None
    try:
        price_response = requests.get(f"{CLOB_API_URL}/price", params={"token_id": token_id})
        price_response.raise_for_status()
        price_data = price_response.json()
        last_price = float(price_data.get('price'))

        if last_price > 0.95:
            outcome = "UP"
        elif last_price < 0.05:
            outcome = "DOWN"
    except Exception as e:
        print(f"Error fetching final price for outcome: {e}")

    if outcome:
        print(f"Outcome for '{market_name}' was '{outcome}'. Updating database...")
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Define the time range for the update
            hour_end_utc = previous_hour_start_utc + timedelta(hours=1)
            start_str = previous_hour_start_utc.strftime('%Y-%m-%d %H:%M:%S')
            end_str = hour_end_utc.strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute("""
                UPDATE polydata 
                SET outcome = ? 
                WHERE timestamp >= ? AND timestamp < ?
            """, (outcome, start_str, end_str))
            
            conn.commit()
            conn.close()
            print(f"Successfully updated {cursor.rowcount} rows for the previous hour.")
        except Exception as e:
            print(f"Error updating outcome in database: {e}")
    else:
        print(f"Could not determine definitive outcome for '{market_name}'.")

def parse_market_datetime(market_slug, market_question):
    """
    Extracts the market's date and time from its slug or question.
    Handles different slug formats and timezones, and corrects for year-end rollovers.
    """
    # Pattern 1: bitcoin-up-or-down-{month}-{day}-{hour}{am|pm}-et
    match = re.match(r'bitcoin-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et', market_slug)
    # Pattern 2: bitcoin-up-or-down-{month}-{day}-{hour}-{am|pm}-et
    if not match:
        match = re.match(r'bitcoin-up-or-down-(\w+)-(\d+)-(\d+)-(am|pm)-et', market_slug)

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

def is_bitcoin_hourly_market(market):
    """Checks if a market is a Bitcoin hourly market based on its question."""
    q = market["question"].lower()
    return "bitcoin up or down" in q and ("pm et" in q or "am et" in q)

def update_markets_csv():
    """
    Fetches all Bitcoin hourly markets and saves them to a CSV file.
    """
    market_data = []
    print("--- Updating market list ---")
    print("Searching for Bitcoin hourly markets...")
    bitcoin_market_count = 0

    for m in get_all_markets():
        if not is_bitcoin_hourly_market(m):
            continue
        
        bitcoin_market_count += 1
        slug = m.get("market_slug")
        question = m.get("question")
        condition_id = m.get("condition_id") # We need this for outcome checking
        
        # Get the first token (assuming there are typically 2 tokens for yes/no)
        if m.get("tokens") and len(m["tokens"]) > 0:
            token_id = m["tokens"][0].get("token_id")
        else:
            token_id = None
        
        # Parse the date from the market
        parsed_date = parse_market_datetime(slug, question)
        
        print(f"Found market {bitcoin_market_count}: {slug}")
        print(f"  Question: {question}")
        print(f"  Token ID: {token_id}")
        print(f"  Parsed Date: {parsed_date}")
        
        # Reject markets with missing essential data
        if not slug or not question or not token_id or parsed_date == "Could not parse date":
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
            "token_id": token_id,
            "date_time": parsed_date,
            "market_slug": slug,
            "condition_id": condition_id
        })
        
        time.sleep(0.1)  # Small delay to be respectful to the API

    if not market_data:
        print("No Bitcoin hourly markets found.")
        return

    # Sort by date_time - only valid dates will be here now
    market_data.sort(key=lambda x: x['date_time'])

    try:
        with open(MARKETS_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["market_name", "token_id", "date_time", "market_slug", "condition_id"])
            writer.writeheader()
            writer.writerows(market_data)
        print(f"Found {bitcoin_market_count} Bitcoin hourly markets")
        print(f"Successfully wrote {len(market_data)} valid markets to {MARKETS_CSV_FILE}")
    except IOError as e:
        print(f"Error writing to file {MARKETS_CSV_FILE}: {e}")


def get_current_market_token_id():
    """Find the token ID for the Bitcoin market that corresponds to the current date/time"""
    current_hour_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    return get_market_token_id_for_hour(current_hour_utc)

def get_order_book(token_id):
    """
    Get the current order book for a given token ID.
    If the order book is thin or empty, fall back to fetching the last trade price.
    """
    try:
        # First, try to get the full order book
        response = requests.get(f"{CLOB_API_URL}/book", params={"token_id": token_id})
        response.raise_for_status()
        
        data = response.json()
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if bids and asks:
            # Parse bids and asks, sort them, and return the best
            parsed_bids = [(float(b['price']), float(b['size'])) for b in bids]
            parsed_asks = [(float(a['price']), float(a['size'])) for a in asks]
            parsed_bids.sort(reverse=True)
            parsed_asks.sort()
            return parsed_bids[0], parsed_asks[0]

        # --- Fallback Logic ---
        # If order book is incomplete, fetch the last trade price
        print(f"Incomplete order book for token {token_id}. Falling back to last price.")
        price_response = requests.get(f"{CLOB_API_URL}/price", params={"token_id": token_id})
        price_response.raise_for_status()
        price_data = price_response.json()
        last_price = float(price_data.get('price'))

        if last_price is not None:
            # Return a synthetic order book entry using the last price
            # We use a size of 0 as it's not a real, fillable order
            return (last_price, 0.0), (last_price, 0.0)
        
        return None, None
        
    except Exception as e:
        print(f"Error fetching order data for token {token_id}: {e}")
        return None, None

def collect_data_once():
    """Collect data for one minute and log it to the SQLite database."""
    try:
        t0 = datetime.now(UTC)
        print(f"({t0.strftime('%H:%M:%S.%f')}) --- Starting run ---")
        
        # At XX:01, check and update the outcome for the previous hour
        check_and_update_outcome(t0)
        
        # Fetch the BTC spot price and OFI from Binance first
        btc_price, ofi = get_binance_data_and_ofi()
        
        # --- Prediction Logic ---
        p_up_prediction = None
        if model is not None and btc_price is not None:
            # Determine the start of the current hour
            current_hour_start_utc = t0.replace(minute=0, second=0, microsecond=0)
            
            # Get p_start from cache or Binance Klines
            p_start = get_p_start_from_binance(current_hour_start_utc)

            # Fetch last 30 mins of data for volatility calculation
            conn = sqlite3.connect(DB_FILE)
            thirty_mins_ago = (t0 - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
            historical_df = pd.read_sql_query(
                f"SELECT timestamp, btc_usdt_spot FROM polydata WHERE timestamp >= '{thirty_mins_ago}'", conn)
            conn.close()
            
            # Append current price to historical data to get the most up-to-date volatility
            if not historical_df.empty:
                current_data_df = pd.DataFrame([{'timestamp': t0.strftime('%Y-%m-%d %H:%M:%S'), 'btc_usdt_spot': btc_price}])
                historical_df = pd.concat([historical_df, current_data_df], ignore_index=True)
            
            p_up_prediction = calculate_live_prediction(historical_df, t0, ofi, p_start, btc_price)
        # --- End Prediction Logic ---

        token_id, market_name = get_current_market_token_id()

        t1 = datetime.now(UTC)
        print(f"({t1.strftime('%H:%M:%S.%f')}) Found token ID. Elapsed: {(t1-t0).total_seconds():.4f}s")

        if not token_id:
            print(f"({datetime.now(UTC).strftime('%H:%M:%S.%f')}) No active market found for the current hour.")
            return

        best_bid, best_ask = get_order_book(token_id)

        t2 = datetime.now(UTC)
        print(f"({t2.strftime('%H:%M:%S.%f')}) Got order book. API call took: {(t2-t1).total_seconds():.4f}s")

        if best_bid and best_ask:
            best_bid_price = best_bid[0]
            best_ask_price = best_ask[0]

            data_row = {
                'timestamp': t0.strftime('%Y-%m-%d %H:%M:%S'),
                'market_name': market_name,
                'btc_usdt_spot': btc_price,
                'ofi': ofi,
                'p_up_prediction': p_up_prediction,
                'token_id': token_id,
                'best_bid': best_bid_price,
                'best_ask': best_ask_price
            }

            # Ensure the database and table exist, then insert the new row
            init_database()
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO polydata (timestamp, market_name, token_id, best_bid, best_ask, btc_usdt_spot, ofi, p_up_prediction)
                VALUES (:timestamp, :market_name, :token_id, :best_bid, :best_ask, :btc_usdt_spot, :ofi, :p_up_prediction)
            ''', data_row)
            conn.commit()
            conn.close()

            t3 = datetime.now(UTC)
            btc_price_str = f"{btc_price:.2f}" if btc_price is not None else "N/A"
            ofi_str = f"{ofi:.4f}" if ofi is not None else "N/A"
            prediction_str = f"{p_up_prediction:.4f}" if p_up_prediction is not None else "N/A"
            print(f"({t3.strftime('%H:%M:%S.%f')}) Logged to DB: BTC={btc_price_str}, OFI={ofi_str}, P(Up)={prediction_str}, Bid={best_bid_price:.2f}, Ask={best_ask_price:.2f} for '{market_name}'")
            print(f"({t3.strftime('%H:%M:%S.%f')}) --- Total run time: {(t3-t0).total_seconds():.4f}s ---")

        else:
            print(f"({datetime.now(UTC).strftime('%H:%M:%S.%f')}) Could not retrieve valid order book for '{market_name}'")

    except Exception as e:
        print(f"An unexpected error occurred during data collection: {e}")

def main():
    """Main function to run the scraping loop."""
    output_csv = 'btc_polydata.csv'

    # Update the markets first
    update_markets_csv()

    print("\n--- Running Polymarket Scraper ---")
    print(f"Data will be saved to {output_csv}")

    # Run data collection once
    collect_data_once()

def main_multi_run(num_runs=5):
    """Run the scraper multiple times with 1-minute intervals - ONLY collects bid/ask data."""
    output_csv = 'btc_polydata.csv'

    print(f"\n--- Running Polymarket Scraper ({num_runs} runs with 1-minute intervals) ---")
    print(f"Data will be saved to {output_csv}")
    print("*** NOT updating markets - just collecting bid/ask data ***")

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        collect_data_once()
        
        # Sleep for 60 seconds between runs (except after the last run)
        if i < num_runs - 1:
            print(f"Waiting 60 seconds until next run...")
            time.sleep(60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket scraper for Bitcoin hourly markets.")
    parser.add_argument('--update-markets', action='store_true', help='Only update the markets CSV and exit.')
    parser.add_argument('--multi-run', type=int, default=0, help='Run scraper multiple times with 1-minute intervals (e.g., --multi-run 5)')
    parser.add_argument('--run-once', action='store_true', help='Run a single data collection and exit.')
    args = parser.parse_args()

    if args.update_markets:
        update_markets_csv()
    elif args.multi_run > 0:
        main_multi_run(args.multi_run)
    elif args.run_once:
        collect_data_once()
    else:
        main() 