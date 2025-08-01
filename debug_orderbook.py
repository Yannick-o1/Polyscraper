#!/usr/bin/env python3
"""
Debug script to check order book fetching and market data flow.
The predictions look good now but order book data is still showing 1-99 spreads.
"""

import pandas as pd
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import time

def check_current_market_selection():
    """Check what market should be selected right now."""
    print("🎯 CURRENT MARKET SELECTION CHECK")
    print("=" * 50)
    
    # Get current time
    current_hour_utc = datetime.now(ZoneInfo('UTC')).replace(minute=0, second=0, microsecond=0)
    et_tz = ZoneInfo("America/New_York")
    target_hour_dt_et = current_hour_utc.astimezone(et_tz)
    target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
    
    print(f"Current UTC time: {datetime.now(ZoneInfo('UTC'))}")
    print(f"Current ET time: {datetime.now(ZoneInfo('UTC')).astimezone(et_tz)}")
    print(f"Target market string: {target_datetime_str}")
    
    # Check each currency
    for currency in ['btc', 'eth', 'sol', 'xrp']:
        markets_file = f"{currency}_polymarkets.csv"
        if not os.path.exists(markets_file):
            print(f"\n❌ {currency.upper()}: No CSV file")
            continue
            
        df = pd.read_csv(markets_file)
        matching_rows = df[df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            print(f"\n⚠️ {currency.upper()}: No match for {target_datetime_str}")
            print(f"    Using fallback: {df.iloc[-1]['market_name']}")
            selected_market = df.iloc[-1]
        else:
            print(f"\n✅ {currency.upper()}: Found match")
            selected_market = matching_rows.iloc[0]
        
        print(f"    Selected: {selected_market['market_name']}")
        print(f"    YES Token: {selected_market['token_id_yes']}")
        print(f"    NO Token: {selected_market['token_id_no']}")

def test_order_book_api(token_id, currency):
    """Test the order book API directly for a given token."""
    print(f"\n🔍 TESTING ORDER BOOK API for {currency.upper()}")
    print("-" * 30)
    
    # Polymarket order book API endpoint
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    
    try:
        print(f"Fetching: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return None, None
            
        data = response.json()
        
        if 'bids' not in data or 'asks' not in data:
            print(f"❌ Missing bids/asks in response: {data}")
            return None, None
            
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        print(f"📊 Raw order book data:")
        print(f"    Bids count: {len(bids)}")
        print(f"    Asks count: {len(asks)}")
        
        if bids:
            print(f"    Top 3 bids: {bids[:3]}")
            best_bid = float(bids[0][0]) if bids[0] else None
        else:
            print("    ❌ No bids!")
            best_bid = None
            
        if asks:
            print(f"    Top 3 asks: {asks[:3]}")
            best_ask = float(asks[0][0]) if asks[0] else None
        else:
            print("    ❌ No asks!")
            best_ask = None
            
        if best_bid and best_ask:
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            market_price = (best_bid + best_ask) / 2
            
            print(f"\n📈 CALCULATED PRICES:")
            print(f"    Best bid: {best_bid:.4f}")
            print(f"    Best ask: {best_ask:.4f}")
            print(f"    Spread: {spread:.4f} ({spread_pct:.1f}%)")
            print(f"    Market price: {market_price:.4f}")
            
            if spread_pct > 50:
                print(f"    🚨 WARNING: Huge spread {spread_pct:.1f}% suggests inactive market!")
            elif spread_pct > 5:
                print(f"    ⚠️ Warning: Large spread {spread_pct:.1f}%")
            else:
                print(f"    ✅ Normal spread {spread_pct:.1f}%")
                
        return best_bid, best_ask
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None, None

def main():
    """Main diagnostic function."""
    check_current_market_selection()
    
    print(f"\n\n🔍 ORDER BOOK API TESTING")
    print("=" * 50)
    
    # Test order book for each currency's current market
    for currency in ['btc', 'eth', 'sol', 'xrp']:
        markets_file = f"{currency}_polymarkets.csv"
        if not os.path.exists(markets_file):
            continue
            
        df = pd.read_csv(markets_file)
        
        # Get current hour market
        current_hour_utc = datetime.now(ZoneInfo('UTC')).replace(minute=0, second=0, microsecond=0)
        et_tz = ZoneInfo("America/New_York")
        target_hour_dt_et = current_hour_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        matching_rows = df[df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            selected_market = df.iloc[-1]
        else:
            selected_market = matching_rows.iloc[0]
            
        token_yes = str(int(selected_market['token_id_yes']))
        
        print(f"\n{'='*20} {currency.upper()} {'='*20}")
        print(f"Market: {selected_market['market_name']}")
        print(f"Testing YES token: {token_yes}")
        
        best_bid, best_ask = test_order_book_api(token_yes, currency)
        
        # Add small delay between API calls
        time.sleep(1)
    
    print(f"\n\n🎯 DIAGNOSIS SUMMARY:")
    print("=" * 50)
    print("1. Check if current hour market selection is working correctly")
    print("2. Check if the API is returning broken order book data")
    print("3. Look for markets with huge spreads (>50%) indicating they're inactive")
    print("4. Compare with working markets to see the difference")

if __name__ == "__main__":
    main() 