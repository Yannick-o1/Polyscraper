#!/usr/bin/env python3
"""
Fixed debug script to properly check order book data.
The previous version had a bug processing asks.
"""

import pandas as pd
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import time
import json

def test_order_book_api_detailed(token_id, currency):
    """Test the order book API with detailed error handling."""
    print(f"\n🔍 TESTING ORDER BOOK API for {currency.upper()}")
    print("-" * 50)
    
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    
    try:
        print(f"Fetching: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return None, None
            
        data = response.json()
        print(f"✅ Got response: {response.status_code}")
        
        # Show raw response structure
        print(f"📋 Response keys: {list(data.keys())}")
        
        if 'bids' not in data or 'asks' not in data:
            print(f"❌ Missing bids/asks in response")
            print(f"📄 Full response: {json.dumps(data, indent=2)}")
            return None, None
            
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        print(f"\n📊 Order book summary:")
        print(f"    Bids count: {len(bids)}")
        print(f"    Asks count: {len(asks)}")
        
        # Show raw bid data
        print(f"\n💰 BIDS (people wanting to BUY YES tokens):")
        if bids:
            for i, bid in enumerate(bids[:5]):
                print(f"    {i+1}. {bid}")
            
            try:
                best_bid = float(bids[0]['price']) if bids[0] and 'price' in bids[0] else None
                print(f"    → Best bid: {best_bid}")
            except Exception as e:
                print(f"    ❌ Error parsing best bid: {e}")
                best_bid = None
        else:
            print("    ❌ No bids!")
            best_bid = None
            
        # Show raw ask data
        print(f"\n💸 ASKS (people wanting to SELL YES tokens):")
        if asks:
            for i, ask in enumerate(asks[:5]):
                print(f"    {i+1}. {ask}")
                
            try:
                best_ask = float(asks[0]['price']) if asks[0] and 'price' in asks[0] else None
                print(f"    → Best ask: {best_ask}")
            except Exception as e:
                print(f"    ❌ Error parsing best ask: {e}")
                best_ask = None
        else:
            print("    ❌ No asks!")
            best_ask = None
            
        # Calculate spread if we have both
        if best_bid and best_ask and best_bid > 0:
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            market_price = (best_bid + best_ask) / 2
            
            print(f"\n📈 CALCULATED PRICES:")
            print(f"    Best bid: {best_bid:.4f} ({best_bid*100:.1f}%)")
            print(f"    Best ask: {best_ask:.4f} ({best_ask*100:.1f}%)")
            print(f"    Spread: {spread:.4f} ({spread_pct:.1f}%)")
            print(f"    Market price: {market_price:.4f} ({market_price*100:.1f}%)")
            
            if spread_pct > 50:
                print(f"    🚨 WARNING: Huge spread {spread_pct:.1f}%!")
            elif spread_pct > 5:
                print(f"    ⚠️ Warning: Large spread {spread_pct:.1f}%")
            else:
                print(f"    ✅ Normal spread {spread_pct:.1f}%")
        else:
            print(f"\n❌ Cannot calculate spread:")
            print(f"    best_bid: {best_bid}")
            print(f"    best_ask: {best_ask}")
                
        return best_bid, best_ask
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Test one currency in detail."""
    print("🔍 DETAILED ORDER BOOK DIAGNOSTIC")
    print("=" * 60)
    
    # Just test BTC for now to see the full details
    currency = 'btc'
    markets_file = f"{currency}_polymarkets.csv"
    
    if not os.path.exists(markets_file):
        print(f"❌ No {markets_file} found")
        return
        
    df = pd.read_csv(markets_file)
    
    # Get current hour market  
    current_hour_utc = datetime.now(ZoneInfo('UTC')).replace(minute=0, second=0, microsecond=0)
    et_tz = ZoneInfo("America/New_York")
    target_hour_dt_et = current_hour_utc.astimezone(et_tz)
    target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
    
    print(f"Looking for market: {target_datetime_str}")
    
    matching_rows = df[df['date_time'] == target_datetime_str]
    
    if matching_rows.empty:
        print(f"⚠️ No match, using fallback")
        selected_market = df.iloc[-1]
    else:
        print(f"✅ Found match")
        selected_market = matching_rows.iloc[0]
        
    print(f"\n🎯 Testing market: {selected_market['market_name']}")
    print(f"📍 Date/time: {selected_market['date_time']}")
    
    token_yes = str(int(selected_market['token_id_yes']))
    token_no = str(int(selected_market['token_id_no']))
    
    print(f"🟢 YES Token: {token_yes}")
    print(f"🔴 NO Token: {token_no}")
    
    # Test both YES and NO tokens
    print(f"\n" + "="*60)
    print(f"TESTING YES TOKEN")
    print(f"="*60)
    yes_bid, yes_ask = test_order_book_api_detailed(token_yes, "BTC-YES")
    
    print(f"\n" + "="*60)
    print(f"TESTING NO TOKEN")  
    print(f"="*60)
    no_bid, no_ask = test_order_book_api_detailed(token_no, "BTC-NO")

if __name__ == "__main__":
    main() 