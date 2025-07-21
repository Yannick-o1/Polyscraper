import requests
import pandas as pd
from datetime import datetime, UTC, timedelta
from zoneinfo import ZoneInfo
import time
import os
import re
import csv
import sqlite3

CLOB_API_URL = "https://clob.polymarket.com"
MARKETS_CSV_FILE = "btc_polymarkets.csv"
DB_FILE = "polyscraper.db"

def get_binance_btc_price():
    """Fetches the current BTC/USDT spot price from Binance."""
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": "BTCUSDT"}
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except requests.RequestException as e:
        print(f"Error fetching Binance price: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error parsing Binance price data: {e}")
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

    conn.commit()
    conn.close()

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
            "market_slug": slug
        })
        
        time.sleep(0.1)  # Small delay to be respectful to the API

    if not market_data:
        print("No Bitcoin hourly markets found.")
        return

    # Sort by date_time - only valid dates will be here now
    market_data.sort(key=lambda x: x['date_time'])

    try:
        with open(MARKETS_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["market_name", "token_id", "date_time", "market_slug"])
            writer.writeheader()
            writer.writerows(market_data)
        print(f"Found {bitcoin_market_count} Bitcoin hourly markets")
        print(f"Successfully wrote {len(market_data)} valid markets to {MARKETS_CSV_FILE}")
    except IOError as e:
        print(f"Error writing to file {MARKETS_CSV_FILE}: {e}")


def get_current_market_token_id():
    """Find the token ID for the Bitcoin market that corresponds to the current date/time"""
    try:
        markets_df = pd.read_csv(MARKETS_CSV_FILE)
        et_tz = ZoneInfo("America/New_York")
        current_time = datetime.now(et_tz)
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        target_datetime = current_hour.strftime('%Y-%m-%d %H:%M EDT')
        
        matching_rows = markets_df[markets_df['date_time'] == target_datetime]
        
        if matching_rows.empty:
            return None, None
        
        market_row = matching_rows.iloc[0]
        token_id = str(int(market_row['token_id']))
        market_name = market_row['market_name']
        
        return token_id, market_name
        
    except FileNotFoundError:
        print(f"Error: '{MARKETS_CSV_FILE}' not found. Please run with --update-markets first.")
        return None, None
    except Exception as e:
        print(f"Error reading or processing CSV: {e}")
        return None, None

def get_order_book(token_id):
    """Get the current order book for a given token ID"""
    try:
        response = requests.get("https://clob.polymarket.com/book", params={"token_id": token_id})
        response.raise_for_status()
        
        data = response.json()
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if not bids or not asks:
            return None, None
        
        # Parse bids and asks, sort them, and return the best (highest bid, lowest ask)
        parsed_bids = [(float(b['price']), float(b['size'])) for b in bids]
        parsed_asks = [(float(a['price']), float(a['size'])) for a in asks]
        parsed_bids.sort(reverse=True)
        parsed_asks.sort()
        
        return parsed_bids[0], parsed_asks[0]
        
    except Exception as e:
        print(f"Error fetching order book for token {token_id}: {e}")
        return None, None

def collect_data_once():
    """Collect data for one minute and log it to the SQLite database."""
    try:
        t0 = datetime.now(UTC)
        print(f"({t0.strftime('%H:%M:%S.%f')}) --- Starting run ---")
        
        # Fetch the BTC spot price from Binance first
        btc_price = get_binance_btc_price()
        
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
                'token_id': token_id,
                'best_bid': best_bid_price,
                'best_ask': best_ask_price
            }

            # Ensure the database and table exist, then insert the new row
            init_database()
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO polydata (timestamp, market_name, token_id, best_bid, best_ask, btc_usdt_spot)
                VALUES (:timestamp, :market_name, :token_id, :best_bid, :best_ask, :btc_usdt_spot)
            ''', data_row)
            conn.commit()
            conn.close()

            t3 = datetime.now(UTC)
            btc_price_str = f"{btc_price:.2f}" if btc_price is not None else "N/A"
            print(f"({t3.strftime('%H:%M:%S.%f')}) Logged to DB: BTC={btc_price_str}, Bid={best_bid_price:.2f}, Ask={best_ask_price:.2f} for '{market_name}'")
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