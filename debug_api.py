#!/usr/bin/env python3
"""
Quick debug script to test Polymarket API calls
"""

import requests
import pandas as pd
import os
from datetime import datetime, UTC
from zoneinfo import ZoneInfo

CLOB_API_URL = "https://clob.polymarket.com"

def get_market_data(currency):
    """Get current market token IDs for the current hour."""
    try:
        markets_file = f"{currency}_polymarkets.csv"
        
        if not os.path.exists(markets_file):
            print(f"❌ No markets file: {markets_file}")
            return None, None, None
            
        df = pd.read_csv(markets_file)
        if df.empty:
            return None, None, None
        
        # Get current hour and convert to ET timezone for market matching
        current_hour_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        et_tz = ZoneInfo("America/New_York")
        target_hour_dt_et = current_hour_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        print(f"🕐 Looking for market at: {target_datetime_str}")
        
        # Find market matching current hour
        matching_rows = df[df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            print(f"⚠️ No market found for current hour: {target_datetime_str}")
            print(f"📊 Available times in CSV:")
            print(df['date_time'].tail(10).tolist())
            # Fall back to most recent market as backup
            latest_market = df.iloc[-1]
            print(f"   Using fallback market: {latest_market['market_name']}")
        else:
            latest_market = matching_rows.iloc[0]
            print(f"✅ Found current hour market: {latest_market['market_name']}")
        
        token_id_yes = str(int(latest_market['token_id_yes']))
        token_id_no = str(int(latest_market['token_id_no']))
        market_name = latest_market['market_name']
        
        print(f"🔍 Market: {market_name}")
        print(f"🔍 YES token: {token_id_yes}")
        print(f"🔍 NO token: {token_id_no}")
        
        return token_id_yes, token_id_no, market_name
        
    except Exception as e:
        print(f"❌ Market data error for {currency}: {e}")
        return None, None, None

def test_order_book(token_id, name):
    """Test order book API call"""
    try:
        url = f"{CLOB_API_URL}/book"
        params = {"token_id": token_id}
        
        print(f"\n🔍 Testing order book for {name}")
        print(f"🔍 URL: {url}?token_id={token_id}")
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        print(f"📊 Response keys: {list(data.keys())}")
        print(f"📊 Got {len(bids)} bids, {len(asks)} asks")
        
        if bids:
            print(f"💰 Top 3 bids:")
            for i, bid in enumerate(bids[:3]):
                print(f"   {i+1}. {bid}")
            best_bid = float(bids[0]['price'])
        else:
            print(f"❌ No bids!")
            best_bid = None
            
        if asks:
            print(f"💸 Top 3 asks:")
            for i, ask in enumerate(asks[:3]):
                print(f"   {i+1}. {ask}")
            best_ask = float(asks[0]['price'])
        else:
            print(f"❌ No asks!")
            best_ask = None
        
        if best_bid and best_ask:
            market_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            print(f"📈 Best bid: {best_bid:.4f} ({best_bid*100:.1f}%)")
            print(f"📈 Best ask: {best_ask:.4f} ({best_ask*100:.1f}%)")
            print(f"📈 Market price: {market_price:.4f} ({market_price*100:.1f}%)")
            print(f"📈 Spread: {spread:.4f} ({spread*100:.1f}pp)")
            
        return best_bid, best_ask
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None, None

def main():
    """Test all currencies"""
    currencies = ['btc', 'eth', 'sol', 'xrp']
    
    for currency in currencies:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING {currency.upper()}")
        print(f"{'='*60}")
        
        token_yes, token_no, market_name = get_market_data(currency)
        
        if token_yes:
            test_order_book(token_yes, f"{currency.upper()}-YES")
        else:
            print(f"❌ No token IDs for {currency}")

if __name__ == "__main__":
    main() 