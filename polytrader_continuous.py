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
import csv
import re
from datetime import datetime, UTC, timedelta
from collections import defaultdict, deque
from zoneinfo import ZoneInfo

# Polymarket imports
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType, BalanceAllowanceParams, AssetType, OpenOrderParams
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
BANKROLL_FRACTION = 0.8
PROBABILITY_DELTA_THRESHOLD = 3.2
MINIMUM_ORDER_SIZE_SHARES = 5.0
CYCLE_DELAY_SECONDS = 2.0
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
        self.api_call_times = deque(maxlen=1000)
        self.running = True
        self.hour_start_cache = {}  # Cache p_start prices
        self.current_bankroll = None  # Will be fetched from Polymarket
        self.last_bankroll_check = 0  # Track when we last checked bankroll
        self.last_outcome_update = {}  # Track last outcome update per currency to prevent duplicates
        self.last_cache_minute = {}  # Track last minute when cache was updated per currency
        self.last_api_report_minute = None  # Track last minute when API usage was reported
        # Cache of Binance aggregated trades (aggTrades) per currency for last ~60s
        self.agg_trades_cache = defaultdict(lambda: deque())  # currency -> deque of trades (dicts with keys including 'T','q','p','m')
        self.agg_trades_last_time = {}  # currency -> last fetched trade time (ms)


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

def report_api_usage_if_new_minute():
    """Report API usage statistics once per minute."""
    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    if state.last_api_report_minute != current_minute:
        state.last_api_report_minute = current_minute
        
        # Count API calls in the last 60 seconds
        now = time.time()
        recent_calls = sum(1 for call_time in state.api_call_times if now - call_time <= 60)
        
        # Calculate usage percentage
        usage_percent = (recent_calls / API_CALLS_PER_MINUTE) * 100
        
        # Choose emoji based on usage level
        if usage_percent >= 90:
            emoji = "🔴"  # Red - very high usage
        elif usage_percent >= 70:
            emoji = "🟡"  # Yellow - high usage
        elif usage_percent >= 50:
            emoji = "🟢"  # Green - moderate usage
        else:
            emoji = "🔵"  # Blue - low usage
        
        print(f"{emoji} API USAGE: {recent_calls}/{API_CALLS_PER_MINUTE} calls/min ({usage_percent:.1f}%)")

def get_hour_start_price(currency, current_crypto_price):
    """Get exact price from Binance at the precise start of the current hour with maximum precision."""
    hour_key = datetime.now(UTC).strftime('%Y-%m-%d_%H')
    
    # Clear cache if we're in a new hour, but preserve recent data for volatility calculations
    current_hour = datetime.now(UTC).strftime('%Y-%m-%d_%H')
    if hasattr(state, 'last_hour_key') and state.last_hour_key != current_hour:
        # Instead of clearing everything, only keep the last 2 hours for volatility calculations
        current_time = datetime.now(UTC)
        cutoff_time = current_time - timedelta(hours=2)
        
        # Clean old entries but keep recent ones
        keys_to_remove = []
        for key in state.hour_start_cache.keys():
            try:
                key_time = datetime.strptime(key, '%Y-%m-%d_%H').replace(tzinfo=UTC)
                if key_time < cutoff_time:
                    keys_to_remove.append(key)
            except:
                keys_to_remove.append(key)  # Remove malformed keys
        
        for key in keys_to_remove:
            del state.hour_start_cache[key]
            
        print(f"🔄 New hour detected, cleaned old p_start cache entries (kept recent 2 hours)")
    state.last_hour_key = current_hour
    
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
            # Use opening price of the first minute of the hour with full precision
            p_start = float(klines[0][1])  # open price - keep full precision
            state.hour_start_cache[hour_key][currency] = p_start
            print(f"🔒 {currency.upper()}: Fetched exact hour start price ${p_start:.8f} from Binance for {hour_key}")
            return p_start
            
    except Exception as e:
        print(f"⚠️ Failed to get hour start price from Binance for {currency}: {e}")
    
    # Fallback to current price if Binance call fails
    state.hour_start_cache[hour_key][currency] = current_crypto_price
    print(f"🔒 {currency.upper()}: Using current price ${current_crypto_price:.8f} as fallback for {hour_key}")
    return current_crypto_price

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

def is_currency_hourly_market(market, currency):
    """Checks if a market is a currency hourly market based on its question."""
    q = market["question"].lower()
    currency_pattern = CURRENCY_CONFIG[currency]['market_question_pattern']
    return currency_pattern in q and ("pm et" in q or "am et" in q)

def parse_market_datetime(market_slug, market_question, currency):
    """
    Extracts the market's date and time from its slug or question.
    Handles different slug formats and timezones, and corrects for year-end rollovers.
    """
    # Pattern 1: {currency}-up-or-down-{month}-{day}-{hour}{am|pm}-et
    pattern1 = rf'{currency}-up-or-down-(\w+)-(\d+)-(\d+)(am|pm)-et'
    match = re.match(pattern1, market_slug)
    # Pattern 2: {currency}-up-or-down-{month}-{day}-{hour}-{am|pm}-et
    if not match:
        pattern2 = rf'{currency}-up-or-down-(\w+)-(\d+)-(\d+)-(am|pm)-et'
        match = re.match(pattern2, market_slug)

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

    # Fallback: Parse date from question text, e.g., "August 4, 3AM ET"
    # Updated pattern to handle the actual format from the logs
    q_match = re.search(r'(\w+)\s+(\d+),?\s+(\d+)(am|pm)\s+et', market_question, re.IGNORECASE)
    if q_match:
        month_name, day, hour_ampm, ampm = q_match.groups()
        try:
            month = list(map(lambda x: x.lower(), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])).index(month_name.lower()) + 1
            hour = int(hour_ampm)
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

def update_markets_csv(currency):
    """
    Fetches all currency hourly markets and saves them to a CSV file.
    """
    market_data = []
    print(f"🔄 Updating {currency.upper()} market list...")
    print(f"Searching for {CURRENCY_CONFIG[currency]['name']} hourly markets...")
    market_count = 0

    for m in get_all_markets():
        if not is_currency_hourly_market(m, currency):
            continue
        
        market_count += 1
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
        parsed_date = parse_market_datetime(slug, question, currency)
        
        print(f"Found market {market_count}: {slug}")
        print(f"  Question: {question}")
        print(f"  Token YES ID: {token_id_yes}")
        print(f"  Token NO ID: {token_id_no}")
        print(f"  Parsed Date: {parsed_date}")
        
        # Debug: Show what's missing
        missing_data = []
        if not slug:
            missing_data.append("slug")
        if not question:
            missing_data.append("question")
        if not token_id_yes:
            missing_data.append("token_id_yes")
        if not token_id_no:
            missing_data.append("token_id_no")
        if parsed_date == "Could not parse date":
            missing_data.append("parsed_date")
        
        # Reject markets with missing essential data
        if not all([slug, question, token_id_yes, token_id_no, parsed_date != "Could not parse date"]):
            print(f"  -> REJECTED: Missing essential data ({', '.join(missing_data)})")
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
        print(f"No {CURRENCY_CONFIG[currency]['name']} hourly markets found.")
        return

    # Sort by date_time - only valid dates will be here now
    market_data.sort(key=lambda x: x['date_time'])

    try:
        markets_file = f"{currency}_polymarkets.csv"
        with open(markets_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["market_name", "token_id_yes", "token_id_no", "date_time", "market_slug", "condition_id"])
            writer.writeheader()
            writer.writerows(market_data)
        print(f"✅ Found {market_count} {CURRENCY_CONFIG[currency]['name']} hourly markets")
        print(f"✅ Successfully wrote {len(market_data)} valid markets to {markets_file}")
    except IOError as e:
        print(f"❌ Error writing to file {markets_file}: {e}")

def get_market_data(currency):
    """Get current market token IDs for the current hour."""
    try:
        markets_file = f"{currency}_polymarkets.csv"
        
        # Check if we need to update markets
        should_update = False
        
        if not os.path.exists(markets_file):
            print(f"⚠️ No market file found for {currency}. Updating markets...")
            should_update = True
        else:
            df = pd.read_csv(markets_file)
            if df.empty:
                print(f"⚠️ Empty market file for {currency}. Updating markets...")
                should_update = True
            else:
                # Get current hour and convert to ET timezone for market matching
                current_hour_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
                et_tz = ZoneInfo("America/New_York")
                target_hour_dt_et = current_hour_utc.astimezone(et_tz)
                target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
                
                # Find market matching current hour
                matching_rows = df[df['date_time'] == target_datetime_str]
                
                if matching_rows.empty:
                    print(f"⚠️ No market found for current hour: {target_datetime_str}")
                    should_update = True
        
        # Update markets if needed
        if should_update:
            print(f"🔄 Updating markets for {currency}...")
            update_markets_csv(currency)
            
            # Check if update was successful
            if not os.path.exists(markets_file):
                print(f"❌ Failed to create market file for {currency}")
            return None, None, None
            
        df = pd.read_csv(markets_file)
        if df.empty:
            print(f"❌ No markets found for {currency} after update")
            return None, None, None
        
        # Now try to find the current market
        df = pd.read_csv(markets_file)
        current_hour_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        et_tz = ZoneInfo("America/New_York")
        target_hour_dt_et = current_hour_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        # Find market matching current hour
        matching_rows = df[df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            print(f"⚠️ Still no market found for current hour: {target_datetime_str}")
            print(f"⚠️ Using fallback market...")
            # Fall back to most recent market as backup
            latest_market = df.iloc[-1]
            print(f"   Using fallback market: {latest_market['market_name']}")
        else:
            latest_market = matching_rows.iloc[0]
        
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
        
        # Use per-currency cache of aggTrades to maintain a rolling 60s window
        now_ms = int(time.time() * 1000)
        window_start = now_ms - 60_000

        cache = state.agg_trades_cache[currency]
        cache_before = len(cache)

        # Prune old trades outside the 60s window
        while cache and cache[0].get("T", 0) < window_start:
            cache.popleft()
        pruned = cache_before - len(cache)

        # Determine fetch start: from last cached trade time or window start
        if cache:
            fetch_start = cache[-1]["T"] + 1
        else:
            fetch_start = window_start

        # Fetch new trades since last cached time up to now
        fetched_count = 0
        while fetch_start <= now_ms:
            wait_for_rate_limit()
            resp = requests.get(
                "https://api.binance.com/api/v3/aggTrades",
                params={
                    "symbol": config['asset_symbol'],
                    "startTime": fetch_start,
                    "endTime": now_ms,
                    "limit": 1000,
                },
                timeout=5,
            )
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            # Append to cache
            cache.extend(batch)
            fetched_count += len(batch)

            # If fewer than limit returned, we've reached the end
            if len(batch) < 1000:
                break

            last_T = batch[-1]["T"]
            if last_T >= now_ms:
                break
            fetch_start = last_T + 1

        if not cache:
            return None, None

        # Compute OFI on the cached 60s window
        # Note: cache elements are raw dicts from aggTrades with keys: p, q, m, T
        df = pd.DataFrame(list(cache))
        df = df[df["T"] >= window_start]
        if df.empty:
            return None, None

        latest_price = float(df.iloc[-1]["p"])  # last trade price in window
        df["q"] = df["q"].astype(float)

        buy_vol = df[~df["m"]]["q"].sum()
        sell_vol = df[df["m"]]["q"].sum()
        total_vol = buy_vol + sell_vol
        ofi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

        # Debug logging removed after verification

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
        
        current_time = datetime.now(UTC)
        current_minute = current_time.minute
        current_second = current_time.second
        # Calculate precise time-to-expiry including seconds
        time_elapsed_in_hour = (current_minute * 60 + current_second) / 3600  # Convert to fraction of hour
        tau = max(1 - time_elapsed_in_hour, 0.01)
        
        # Enhanced volatility calculation - use last 20 data points like polyscraper.py
        vol = 0.02  # Default fallback
        vol_source = "default"
        
        if currency in state.data_cache and len(state.data_cache[currency]) >= 3:
            # Get the last 30 data points to ensure we have enough for rolling window
            recent_data = list(state.data_cache[currency])[-30:] if len(state.data_cache[currency]) >= 30 else list(state.data_cache[currency])
            
            if len(recent_data) >= 3:
                # Extract prices and create DataFrame like polyscraper.py
                prices = [data_point[5] for data_point in recent_data]  # spot_price is index 5
                price_series = pd.Series(prices)
                
                # Calculate log returns
                log_returns = np.log(price_series).diff().dropna()
                
                if len(log_returns) >= 2:
                    # Use rolling window of min(20, available_data) for std deviation like polyscraper.py
                    window_size = min(20, len(log_returns))
                    rolling_vol_series = log_returns.rolling(window=window_size, min_periods=2).std()
                    
                    if not rolling_vol_series.empty:
                        # Get the most recent volatility value and scale to hourly like polyscraper.py
                        latest_vol = rolling_vol_series.iloc[-1]
                        if not pd.isna(latest_vol) and latest_vol > 0:
                            vol = latest_vol * np.sqrt(60)  # Scale to hourly like polyscraper.py
                            vol_source = f"rolling_window_{window_size}_from_{len(recent_data)}_points"
                        
        # Ensure vol is reasonable
        if pd.isna(vol) or vol <= 0 or vol > 1.0:  # Cap at 100% hourly vol
            vol = 0.02
            vol_source = "capped_fallback"
        
        # Model features
        r_scaled = r / np.sqrt(tau)
        ofi = ofi if ofi is not None else 0.0
        
        X_live = [[r_scaled, tau, vol, ofi]]
        raw_prediction = float(state.models[currency].predict(X_live)[0])
        
        # Ensure prediction is properly bounded between 0 and 1
        prediction = max(0.0, min(1.0, raw_prediction))
        
        # Enhanced debugging - show more details at hour start (first 5 minutes)
        if current_minute <= 5:  # Debug all currencies in first 5 minutes of each hour
            print(f"    🔍 {currency.upper()} Features: r={r:.6f} r_scaled={r_scaled:.6f} τ={tau:.3f} vol={vol:.4f}({vol_source}) ofi={ofi:.4f}")
            print(f"    🔍 {currency.upper()} p_start=${p_start:.8f} current=${current_price:.8f} minute={current_minute}")
            print(f"    🔍 {currency.upper()} Prediction: raw={raw_prediction:.4f} bounded={prediction:.4f} market={market_price:.4f}")
            
            # Show cache info for debugging
            cache_size = len(state.data_cache[currency]) if currency in state.data_cache else 0
            print(f"    🔍 {currency.upper()} Cache: {cache_size} total points")
        
        # Warn if prediction was bounded
        if abs(raw_prediction - prediction) > 0.01:
            print(f"    ⚠️ {currency.upper()} prediction bounded: {raw_prediction:.4f} → {prediction:.4f}")
        
        return prediction
        
    except Exception as e:
        print(f"❌ Prediction error for {currency}: {e}")
        import traceback
        traceback.print_exc()
        return None

def place_order(side, token_id, price, size_shares, current_bid=None, current_ask=None):
    """Place a BUY or SELL order on Polymarket with optimal pricing."""
    if not state.polymarket_client:
        print(f"  ❌ Cannot place order: Polymarket client not available")
        return False
    
    try:
        wait_for_rate_limit()
        
        # Use provided order book data or fetch fresh data
        if current_bid is not None and current_ask is not None:
            best_bid, best_ask = current_bid, current_ask
        else:
            # Get current best bid and ask to calculate optimal price
            best_bid, best_ask = get_order_book_prices(token_id)
            if not best_bid or not best_ask:
                print(f"  ❌ Cannot get order book for optimal pricing")
                return False
        
        # Calculate spread and optimal price
        spread = best_ask - best_bid
        midpoint = (best_bid + best_ask) / 2
        
        # Don't trade if spread is too wide (>20 cents)
        if spread > 30:
            print(f"  ❌ Spread too wide: ${spread:.2f} (>30¢) - skipping trade")
            return False
        
        # Pricing logic: BUY at midpoint as requested; keep SELL logic unchanged (slightly above midpoint)
        if side == "BUY":
            # Place BUY at exact midpoint (bounded within the spread just in case)
            optimal_price = midpoint
            optimal_price = min(max(optimal_price, best_bid + 0.01), best_ask - 0.01)
        else:  # SELL
            # Retain previous behavior for SELL: slightly above midpoint toward ask
            half_spread_30_percent = (spread / 2) * 0.30
            optimal_price = midpoint + half_spread_30_percent
            optimal_price = min(optimal_price, best_ask - 0.01)
        
        # Round to 2 decimal places (nearest cent)
        optimal_price = round(optimal_price, 2)
        
        # Create order with 2-minute expiration
        expiration = int((datetime.now(UTC) + timedelta(minutes=2)).timestamp())
        
        order_args = OrderArgs(
            price=optimal_price,
            size=size_shares,
            side=side,
            token_id=token_id,
            expiration=expiration
        )
        
        signed_order = state.polymarket_client.create_order(order_args)
        response = state.polymarket_client.post_order(signed_order, OrderType.GTD)
        
        if response.get('success', False):
            print(f"  ✅ {side} order placed: {size_shares:.2f} shares @ ${optimal_price:.2f} (spread: ${spread:.2f})")
            return True
        else:
            error_msg = response.get('errorMsg', 'Unknown error')
            print(f"  ❌ TRADE FAILED: {side} {size_shares:.2f} shares @ ${optimal_price:.2f}")
            print(f"     └─ Error: {error_msg}")
            print(f"     └─ Token: {token_id}, Spread: ${spread:.4f}")
            return False
            
    except Exception as e:
        print(f"  ❌ TRADE FAILED: {side} {size_shares:.2f} shares - Exception: {e}")
        print(f"     └─ Token: {token_id}")
        return False

def execute_dynamic_position_management(currency, prediction, market_price, token_yes, token_no, best_bid, best_ask, current_yes, current_no):
    """Execute position adjustment based on model prediction with position tracking."""
    if not prediction or not TRADING_ENABLED:
        return {"executed": False, "reason": "no_prediction_or_disabled"}
    
    # Get current bankroll (check every cycle for now)
    current_time = time.time()
    if current_time - state.last_bankroll_check > 60:  # Check every minute
        state.current_bankroll = get_current_bankroll()
        state.last_bankroll_check = current_time
              
    if state.current_bankroll is None:
        return {"executed": False, "reason": "no_bankroll_data"}
    
    # Calculate delta and check threshold
    delta = (prediction - market_price) * 100
    
    # Note: We still need to call position management even when delta is below threshold
    # to clear existing positions, so we don't return early here anymore
    
    # Prevent trading on extreme deltas (over 20pp)
    if abs(delta) > 20.0:
        return {"executed": False, "reason": "delta_too_extreme", "delta": delta}
    

    # Check if spread is 0 cents (no trading allowed when spread is 0)
    spread = best_ask - best_bid
    if spread <= 0.0:
        return {"executed": False, "reason": "zero_spread", "bid": best_bid, "ask": best_ask}
    
    # Use passed positions (no API call needed)
    current_net = current_yes - current_no  # Net position (positive = bullish)
    
    # Calculate target position using real bankroll
    target_exposure = BANKROLL_FRACTION * state.current_bankroll * abs(delta) / 100
    
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
    
    # If delta is below threshold, clear positions (target = 0)
    if abs(delta) < PROBABILITY_DELTA_THRESHOLD:
        target_yes = 0
        target_no = 0
        target_net = 0
    
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
    
    # Check if we're clearing positions due to delta below threshold
    clearing_due_to_threshold = abs(delta) < PROBABILITY_DELTA_THRESHOLD and (current_yes > 0 or current_no > 0)
    
    # Enforce minimum order size for all trades (5 shares minimum)
    # Exception: allow clearing positions when delta is below threshold
    if abs(position_adjustment) < MINIMUM_ORDER_SIZE_SHARES and not clearing_due_to_threshold:
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
    
    # Check if we have enough balance for the trade (only for buying, not selling)
    if position_adjustment > 0:  # Only check balance when buying (positive adjustment)
        trade_cost = abs(position_adjustment) * market_price if delta > 0 else abs(position_adjustment) * (1 - market_price)
        if trade_cost > state.current_bankroll * 0.8:  # Use 80% of bankroll as safety
            return {
                "executed": False,
                "reason": "insufficient_funds",
                "delta": delta,
                "current_yes": current_yes,
                "current_no": current_no,
                "target_yes": target_yes,
                "target_no": target_no,
                "adjustment": f"{position_adjustment:+.2f}",
                "needed": trade_cost,
                "available": state.current_bankroll
            }
    
        # Execute real trades
    # Determine what we need to do based on target vs current positions
    need_more_yes = target_yes > current_yes
    need_more_no = target_no > current_no
    need_less_yes = target_yes < current_yes
    need_less_no = target_no < current_no
    

    
    # Prioritize selling positions when we need less (including when edge disappears)
    if need_less_yes:
        # Sell YES tokens (either partial or complete clearing)
        shares_to_sell = current_yes - target_yes
        action = "CLEAR" if target_yes == 0 else "SELL"
        print(f"  🔄 Attempting {action} YES: {shares_to_sell:.2f} shares @ ${best_bid/100:.4f}")
        success = place_order("SELL", token_yes, best_bid/100, shares_to_sell, best_bid/100, best_ask/100)
    elif need_less_no:
        # Sell NO tokens (either partial or complete clearing)
        shares_to_sell = current_no - target_no
        action = "CLEAR" if target_no == 0 else "SELL"
        print(f"  🔄 Attempting {action} NO: {shares_to_sell:.2f} shares @ ${1 - best_ask/100:.4f}")
        # For NO tokens: NO_bid = 1 - YES_ask, NO_ask = 1 - YES_bid
        no_bid = 1 - best_ask/100
        no_ask = 1 - best_bid/100
        success = place_order("SELL", token_no, 1 - best_ask/100, shares_to_sell, no_bid, no_ask)
    elif need_more_yes:
        # Prevent BUYs during last 2 minutes and first 1 minute of the hour (UTC)
        now_minute = datetime.utcnow().minute
        if now_minute in (58, 59, 0):
            return {"executed": False, "reason": "buy_blackout_window", "window": "minute in [58,59,0] UTC"}
        # Buy YES tokens
        shares_to_buy = target_yes - current_yes
        print(f"  🔄 Attempting BUY YES: {shares_to_buy:.2f} shares @ ${best_ask/100:.4f}")
        success = place_order("BUY", token_yes, best_ask/100, shares_to_buy, best_bid/100, best_ask/100)
    elif need_more_no:
        # Prevent BUYs during last 2 minutes and first 1 minute of the hour (UTC)
        now_minute = datetime.utcnow().minute
        if now_minute in (58, 59, 0):
            return {"executed": False, "reason": "buy_blackout_window", "window": "minute in [58,59,0] UTC"}
        # Buy NO tokens
        shares_to_buy = target_no - current_no
        print(f"  🔄 Attempting BUY NO: {shares_to_buy:.2f} shares @ ${1 - best_bid/100:.4f}")
        # For NO tokens: NO_bid = 1 - YES_ask, NO_ask = 1 - YES_bid
        no_bid = 1 - best_ask/100
        no_ask = 1 - best_bid/100
        success = place_order("BUY", token_no, 1 - best_bid/100, shares_to_buy, no_bid, no_ask)

    else:
        # No trade needed - positions are aligned
        success = True
    
    return {
        "executed": success,
        "reason": "real_trade" if success else "trade_failed",
        "delta": delta,
        "direction": "UP" if delta > 0 else "DOWN", 
        "target_exposure": target_exposure,
        "current_yes": current_yes,
        "current_no": current_no,
        "target_yes": target_yes,
        "target_no": target_no,
        "adjustment": f"{position_adjustment:+.2f}"
    }

def get_db_connection(currency):
    """Get database connection for a currency."""
    return sqlite3.connect(f"{currency}_polyscraper.db")

def log_error(operation, currency, error):
    """Consistent error logging format."""
    print(f"  ❌ Error {operation} for {currency}: {error}")

def display_trade_result(trade_result):
    """Display trade execution result with appropriate formatting."""
    if trade_result.get("executed", False):
        print(f"  🟢 TRADE EXECUTED: {trade_result['direction']} exposure ${trade_result['target_exposure']:.2f}")
        print(f"  🟥 TARGET POSITIONS: YES {trade_result['target_yes']:.2f} | NO {trade_result['target_no']:.2f}")
    elif trade_result["reason"] == "delta_too_small":
        print(f"  ⬜ NO TRADE: Delta {trade_result['delta']:.1f}pp below threshold ({PROBABILITY_DELTA_THRESHOLD}pp)")
        if 'target_yes' in trade_result and 'target_no' in trade_result:
            print(f"  🟨 CLEARING: Target YES {trade_result['target_yes']:.2f} | NO {trade_result['target_no']:.2f}")
    elif trade_result["reason"] == "insufficient_funds":
        print(f"  🟥 NO TRADE: Need ${trade_result['needed']:.2f}, have ${trade_result['available']:.2f}")
    elif trade_result["reason"] == "adjustment_too_small":
        print(f"  🟨 NO TRADE: Adjustment {trade_result['adjustment']} below minimum")
    elif trade_result["reason"] == "position_aligned":
        print(f"  🟢 NO TRADE: Position already optimal")
        print(f"  🟥 TARGET POSITIONS: YES {trade_result['target_yes']:.2f} | NO {trade_result['target_no']:.2f}")
    elif trade_result["reason"] == "no_bankroll_data":
        print(f"  🟥 NO TRADE: Cannot fetch bankroll")
    elif trade_result["reason"] == "delta_too_extreme":
        print(f"  ⬜ NO TRADE: Delta {trade_result['delta']:.1f}pp is too extreme (>20pp)")
    elif trade_result["reason"] == "zero_spread":
        print(f"  📊 NO TRADE: Zero spread (bid=${trade_result['bid']:.4f}, ask=${trade_result['ask']:.4f})")
    else:
        print(f"  ⬜ NO TRADE: {trade_result['reason']}")
        if 'target_yes' in trade_result and 'target_no' in trade_result:
            print(f"  🟥 TARGET POSITIONS: YES {trade_result['target_yes']:.2f} | NO {trade_result['target_no']:.2f}")

def save_data_point_if_new_minute(currency, timestamp, market_name, token_id_yes, best_bid, best_ask, spot_price, ofi, prediction):
    """Save data point to cache only if we're in a new minute (maintain 1-minute intervals like polyscraper.py)."""
    current_time = datetime.now(UTC)
    current_minute_key = current_time.strftime('%Y-%m-%d %H:%M')  # YYYY-MM-DD HH:MM format
    
    # Check if we've already added data for this minute
    if currency in state.last_cache_minute and state.last_cache_minute[currency] == current_minute_key:
        return  # Skip - already have data for this minute
    
    # This is a new minute, add the data point
    data_point = (timestamp, market_name, token_id_yes, best_bid, best_ask, spot_price, ofi, prediction)
    state.data_cache[currency].append(data_point)
    state.last_cache_minute[currency] = current_minute_key
    
    # Optional: Log when we add new volatility data (only for first 5 minutes of each hour)
    if current_time.minute <= 5:
        print(f"    ⬜ {currency.upper()}: Added 1-minute data point (cache size: {len(state.data_cache[currency])})")

def ensure_outcome_column_exists(currency):
    """Ensure the outcome column exists in the database table."""
    try:
        with get_db_connection(currency) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(polydata)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'outcome' not in columns:
                cursor.execute("ALTER TABLE polydata ADD COLUMN outcome TEXT")
                print(f"  📊 Added outcome column to {currency} database")
                
    except Exception as e:
        log_error("ensuring outcome column", currency, e)

def get_p_start_price_for_hour(currency, hour_utc):
    """Get p_start price for a specific hour from Binance API."""
    try:
        config = CURRENCY_CONFIG[currency]
        hour_start_ms = int(hour_utc.timestamp() * 1000)
        
        wait_for_rate_limit()
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
            return float(klines[0][1])  # open price
        return None
        
    except Exception as e:
        print(f"  ⚠️ Failed to get p_start for {currency} at {hour_utc}: {e}")
        return None

def update_outcome_for_previous_hour(currency, current_hour_utc):
    """Update outcome for the previous hour based on price comparison."""
    try:
        previous_hour_utc = current_hour_utc - timedelta(hours=1)
        
        # Check if we've already updated outcome for this hour transition
        transition_key = f"{currency}_{previous_hour_utc.strftime('%Y-%m-%d_%H')}"
        if transition_key in state.last_outcome_update:
            return  # Already updated for this transition
            
        print(f"  🔍 Checking outcome for {currency.upper()} previous hour: {previous_hour_utc.strftime('%Y-%m-%d %H:%M')}")
        
        # Try to get p_start from cache first, then fetch from API if needed
        previous_hour_key = previous_hour_utc.strftime('%Y-%m-%d_%H')
        current_hour_key = current_hour_utc.strftime('%Y-%m-%d_%H')
        
        # Get previous hour price
        p_start_previous = None
        if previous_hour_key in state.hour_start_cache and currency in state.hour_start_cache[previous_hour_key]:
            p_start_previous = state.hour_start_cache[previous_hour_key][currency]
            print(f"  📋 Found previous hour p_start in cache: ${p_start_previous:.8f}")
        else:
            print(f"  🔍 Previous hour p_start not in cache, fetching from Binance...")
            p_start_previous = get_p_start_price_for_hour(currency, previous_hour_utc)
            
        # Get current hour price
        p_start_current = None
        if current_hour_key in state.hour_start_cache and currency in state.hour_start_cache[current_hour_key]:
            p_start_current = state.hour_start_cache[current_hour_key][currency]
            print(f"  📋 Found current hour p_start in cache: ${p_start_current:.8f}")
        else:
            print(f"  🔍 Current hour p_start not in cache, fetching from Binance...")
            p_start_current = get_p_start_price_for_hour(currency, current_hour_utc)
        
        if p_start_previous is None or p_start_current is None:
            print(f"  ⚠️ Cannot determine outcome: previous=${p_start_previous}, current=${p_start_current}")
            return  # Can't determine outcome without both prices
            
        # Determine outcome
        if p_start_current > p_start_previous:
            outcome = "UP"
        elif p_start_current < p_start_previous:
            outcome = "DOWN"
        else:
            outcome = "FLAT"
            
        print(f"  📊 Outcome determined: {outcome} (${p_start_previous:.8f} → ${p_start_current:.8f})")
            
        # Update database for previous hour
        hour_start_str = previous_hour_utc.strftime('%Y-%m-%d %H:%M:%S')
        hour_end_str = current_hour_utc.strftime('%Y-%m-%d %H:%M:%S')
        
        with get_db_connection(currency) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE polydata 
                SET outcome = ? 
                WHERE timestamp >= ? AND timestamp < ?
            """, (outcome, hour_start_str, hour_end_str))
            
            if cursor.rowcount > 0:
                print(f"  ✅ Updated {cursor.rowcount} rows for {currency} previous hour: {outcome} (${p_start_previous:.2f} → ${p_start_current:.2f})")
                # Mark this transition as completed
                state.last_outcome_update[transition_key] = datetime.now(UTC)
            else:
                print(f"  ⚠️ No rows found to update for {currency} in time range {hour_start_str} to {hour_end_str}")
                
    except Exception as e:
        log_error("updating outcome", currency, e)

def write_to_database(currency, timestamp, market_name, token_id, best_bid, best_ask, spot_price, ofi, prediction):
    """Write current cycle data point to database with fresh timestamp."""
    try:
        config = CURRENCY_CONFIG[currency]
        data_tuple = (timestamp, market_name, token_id, best_bid, best_ask, spot_price, ofi, prediction)
        
        # Ensure outcome column exists
        ensure_outcome_column_exists(currency)
        
        with get_db_connection(currency) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO polydata (timestamp, market_name, token_id, best_bid, best_ask, {config['db_column']}, ofi, p_up_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_tuple)
        
    except Exception as e:
        print(f"\033[31m❌ DB write error for {currency}: {e}\033[0m")

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
            print(f"  ⚠️ {currency.upper()}: No current market found (update markets CSV?)")
            return False
        
        # Step 2: Get live price and OFI
        start = time.time()
        spot_price, ofi = get_live_price_and_ofi(currency)
        timings['binance'] = time.time() - start
        
        if not spot_price:
            print(f"  ⚠️ {currency.upper()}: No live spot price/OFI available from Binance")
            return False

        # Note: Order cancellations are now throttled in continuous_trading_loop
        
        # Step 3: Get order book prices
        start = time.time()
        best_bid, best_ask = get_order_book_prices(token_yes)
        timings['orderbook'] = time.time() - start
        
        if not best_bid or not best_ask:
            print(f"  ⚠️ {currency.upper()}: No order book data (bid/ask) for token {token_yes}")
            return False
        
        # Calculate market price
        market_price = (best_bid + best_ask) / 2
        
        # Create timestamp for this cycle
        timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        
        # Store original decimal values for cache
        original_bid = best_bid
        original_ask = best_ask
        
        # Scale prices for database storage (decimal to integer)
        best_bid = best_bid * 100 if best_bid is not None else None
        best_ask = best_ask * 100 if best_ask is not None else None
        
        # Step 4: Ensure p_start is cached for this hour (needed for outcomes)
        p_start = get_hour_start_price(currency, spot_price)
        
        # Step 4.5: Update outcomes for previous hour if we have the data
        current_hour_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        update_outcome_for_previous_hour(currency, current_hour_utc)
        
        # Step 5: Calculate prediction
        start = time.time()
        prediction = calculate_model_prediction(currency, spot_price, ofi, market_price)
        timings['prediction'] = time.time() - start
        
        # Store data in cache only if we're in a new minute (maintain 1-minute intervals)
        save_data_point_if_new_minute(currency, timestamp, market_name, token_yes, original_bid, original_ask, spot_price, ofi, prediction)
        
        # Step 6: Get current positions (do this once and reuse)
        start = time.time()
        current_yes, current_no = get_current_position(token_yes, token_no)
        timings['positions'] = time.time() - start
        
        # Step 7: Execute trading logic (pass positions to avoid redundant API call)
        start = time.time()
        trade_result = execute_dynamic_position_management(
            currency, prediction, market_price, token_yes, token_no, best_bid, best_ask, current_yes, current_no
        )
        timings['trading'] = time.time() - start
        
        
        
        # === ORGANIZED OUTPUT DISPLAY ===
        
        # 1. Market Data & Prediction
        if prediction:
            delta = (prediction - market_price) * 100
            action = "BUY UP" if delta > PROBABILITY_DELTA_THRESHOLD else "BUY DOWN" if delta < -PROBABILITY_DELTA_THRESHOLD else "HOLD"
            print(f"  🟩 PREDICTION: {prediction*100:.1f}% | MARKET: {market_price*100:.1f}% | DELTA: {delta:+.1f}pp → {action}")
        else:
            print(f"  🟩 PREDICTION: N/A | MARKET: {market_price*100:.1f}%")
        
        # 2. Model Features (include volatility!)
        r = (spot_price / p_start - 1) if p_start > 0 else 0
        current_time = datetime.now(UTC)
        current_minute = current_time.minute
        current_second = current_time.second
        # Calculate precise time-to-expiry including seconds
        time_elapsed_in_hour = (current_minute * 60 + current_second) / 3600  # Convert to fraction of hour
        tau = max(1 - time_elapsed_in_hour, 0.01)
        
        # Get volatility info from the prediction calculation
        vol_info = "N/A"
        if currency in state.data_cache and len(state.data_cache[currency]) >= 3:
            recent_data = list(state.data_cache[currency])[-30:] if len(state.data_cache[currency]) >= 30 else list(state.data_cache[currency])
            if len(recent_data) >= 3:
                prices = [data_point[5] for data_point in recent_data]
                price_series = pd.Series(prices)
                log_returns = np.log(price_series).diff().dropna()
                if len(log_returns) >= 2:
                    window_size = min(20, len(log_returns))
                    rolling_vol_series = log_returns.rolling(window=window_size, min_periods=2).std()
                    if not rolling_vol_series.empty:
                        latest_vol = rolling_vol_series.iloc[-1]
                        if not pd.isna(latest_vol) and latest_vol > 0:
                            vol_hourly = latest_vol * np.sqrt(60)
                            vol_info = f"{vol_hourly:.4f} (w={window_size})"
        
        print(f"  🟨 FEATURES: r={r:.6f} | τ={tau:.3f} | vol={vol_info} | ofi={ofi:.4f}")
        print(f"  🟦 PRICES: spot=${spot_price:.8f} | p_start=${p_start:.8f} | bid=${original_bid:.4f} | ask=${original_ask:.4f}")
        
        # 3. Positions & Bankroll (use cached positions)
        current_pos_str = f"YES {current_yes:.2f} | NO {current_no:.2f}" if (current_yes > 0 or current_no > 0) else "None"
        bankroll_str = f"${state.current_bankroll:.2f}" if state.current_bankroll is not None else "N/A"
        print(f"  🟧 POSITIONS: {current_pos_str} | BANKROLL: {bankroll_str}")
        
        # Display trade result with target positions
        display_trade_result(trade_result)
        
        # Write to database every cycle with fresh data (scale bid/ask for display)
        write_to_database(currency, timestamp, market_name, token_yes, best_bid / 100, best_ask / 100, spot_price, ofi, prediction)
        
        return True
        
    except Exception as e:
        print(f"  ❌ {currency.upper()} error: {e}")
        return False

def initialize_volatility_cache():
    """Initialize volatility cache with last 30 data points for proper rolling window calculation."""
    print("📊 Initializing volatility cache with historical data...")
    
    current_time = datetime.now(UTC)
    thirty_minutes_ago = current_time - timedelta(minutes=30)
    
    for currency in CURRENCY_CONFIG:
        try:
            config = CURRENCY_CONFIG[currency]
            
            # Get 30 minutes of 1-minute klines from Binance to match prediction calculation
            start_time_ms = int(thirty_minutes_ago.timestamp() * 1000)
            end_time_ms = int(current_time.timestamp() * 1000)
            
            wait_for_rate_limit()
            response = requests.get("https://api.binance.com/api/v3/klines", 
                                  params={
                                      "symbol": config['asset_symbol'], 
                                      "interval": "1m",
                                      "startTime": start_time_ms,
                                      "endTime": end_time_ms,
                                      "limit": 30
                                  }, 
                                  timeout=10)
            response.raise_for_status()
            
            klines = response.json()
            if klines:
                # Convert klines to data points and add to cache
                for kline in klines:
                    # kline format: [open_time, open, high, low, close, volume, close_time, ...]
                    timestamp_ms = kline[0]
                    close_price = float(kline[4])  # Use close price
                    
                    # Convert to our data format
                    timestamp_str = datetime.fromtimestamp(timestamp_ms / 1000, UTC).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Create a minimal data point for volatility calculation
                    # Format: (timestamp, market_name, token_id, best_bid, best_ask, spot_price, ofi, prediction)
                    data_point = (timestamp_str, "historical", "N/A", 0, 0, close_price, 0, None)
                    state.data_cache[currency].append(data_point)
                
                print(f"  ✅ {currency.upper()}: Loaded {len(klines)} historical price points")
            else:
                print(f"  ⚠️ {currency.upper()}: No historical data available")
                
        except Exception as e:
            print(f"  ❌ {currency.upper()}: Error loading historical data: {e}")
    
    print("📊 Volatility cache initialization complete")

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
    
    # Initialize Polymarket client and get initial bankroll
    if TRADING_ENABLED and POLYMARKET_AVAILABLE:
        try:
            if POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS:
                state.polymarket_client = ClobClient(
                    host=CLOB_API_URL,
                    key=POLYMARKET_PRIVATE_KEY,
                    chain_id=137,  # Polygon
                    signature_type=1,  # For email/magic accounts, use 2 for browser wallets
                    funder=POLYMARKET_PROXY_ADDRESS
                )
                state.polymarket_client.set_api_creds(state.polymarket_client.create_or_derive_api_creds())
                print(f"✅ Polymarket client initialized")
                
                # Get initial bankroll
                state.current_bankroll = get_current_bankroll()
                if state.current_bankroll is not None:
                    print(f"💰 Initial bankroll: ${state.current_bankroll:.2f}")
                    state.last_bankroll_check = time.time()
                else:
                    print(f"⚠️ Could not fetch initial bankroll")
            else:
                print(f"⚠️ Polymarket credentials not configured")
        except Exception as e:
            print(f"❌ Error initializing Polymarket client: {e}")
    
    # Initialize volatility cache with historical data
    initialize_volatility_cache()
    
    print("🎯 System ready for continuous trading!")

def continuous_trading_loop():
    """Main continuous trading loop."""
    currencies = list(CURRENCY_CONFIG.keys())
    cycle_count = 0
    currency_index = 0  # Track which currency to trade this cycle
    
    print(f"\n🔄 Starting continuous loop: {' → '.join(c.upper() for c in currencies)}")
    print(f"⏱️ Cycle delay: {CYCLE_DELAY_SECONDS}s | 💾 DB writes: Every cycle")
    print("-" * 60)
    
    try:
        while state.running:
            cycle_start = time.time()
            cycle_count += 1
            
            timestamp = datetime.now(UTC).strftime('%H:%M:%S')
            
            # Throttle cancel-all: run every 3 cycles to reduce API load
            if TRADING_ENABLED and state.polymarket_client and (cycle_count % 3 == 0):
                print("  🟥 CANCEL-ALL: Triggering bulk cancellation (every 3 cycles)")
                cancel_all_open_orders()

            # Determine which currency to trade this cycle
            current_currency = currencies[currency_index]
            print(f"\n\n⚡ {timestamp} [Cycle {cycle_count}] - {current_currency.upper()}")
            
            # Report API usage once per minute
            report_api_usage_if_new_minute()
            
            
            # Trade current currency
            trade_currency_cycle(current_currency)
            
            # Move to next currency for next cycle
            currency_index = (currency_index + 1) % len(currencies)
            

            
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
    """Cancel all open orders via the official client method."""
    if not state.polymarket_client:
        return 0

    try:
        wait_for_rate_limit()

        # Prefer the built-in bulk cancel endpoint
        resp = state.polymarket_client.cancel_all()

        # The client may return a dict with success/numCancelled or a bare True
        if isinstance(resp, dict):
            cancelled = int(resp.get("numCancelled", resp.get("cancelled", 0)))
            if cancelled > 0:
                print(f"  🗑️ Cancelled {cancelled} open orders")
            else:
                print("  🗑️ No open orders to cancel")
            return cancelled
        else:
            # If truthy without details, fetch current orders to infer
            wait_for_rate_limit()
            open_orders = state.polymarket_client.get_orders(OpenOrderParams())
            count = len(open_orders) if isinstance(open_orders, list) else 0
            # If still have orders, report 0; otherwise assume some were cancelled
            if count == 0:
                print("  🗑️ Cancelled all open orders")
                return -1  # unknown count
            else:
                print("  🗑️ Some orders may remain after cancel_all()")
                return 0

    except Exception as e:
        # Silently handle API errors - don't let order cancellation stop trading
        print(f"  ⚠️ Cancel-all error: {e}")
        return 0

def get_current_bankroll():
    """Get current USDC balance from Polymarket."""
    if not state.polymarket_client:
        print(f"  ❌ Cannot get bankroll: Polymarket client not available")
        return None
    
    try:
        wait_for_rate_limit()
        
        # Get USDC balance using the correct API method
        usdc_balance_params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL, 
            signature_type=-1
        )
        usdc_account_info = state.polymarket_client.get_balance_allowance(usdc_balance_params)
        usdc_balance = float(usdc_account_info["balance"]) / 1_000_000.0
        
        # Also check allowance
        allowance = float(usdc_account_info.get("allowance", 0)) / 1_000_000.0
        
        return usdc_balance
        
    except Exception as e:
        print(f"  ❌ Error getting bankroll: {e}")
        return None

def get_current_position(token_id_yes, token_id_no):
    """Get current position in YES and NO tokens for a market."""
    if not state.polymarket_client:
        return 0, 0

    try:
        wait_for_rate_limit()
        
        # Get YES token balance
        yes_balance_params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL, 
            token_id=token_id_yes, 
            signature_type=-1
        )
        yes_account_info = state.polymarket_client.get_balance_allowance(yes_balance_params)
        position_yes = float(yes_account_info["balance"]) / 1_000_000.0

        # Get NO token balance
        no_balance_params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL, 
            token_id=token_id_no, 
            signature_type=-1
        )
        no_account_info = state.polymarket_client.get_balance_allowance(no_balance_params)
        position_no = float(no_account_info["balance"]) / 1_000_000.0
        
        return position_yes, position_no
        
    except Exception:
        # Don't let position lookup errors stop trading
        return 0, 0


# === MAIN EXECUTION ===
if __name__ == "__main__":
        initialize_system()
        continuous_trading_loop() 