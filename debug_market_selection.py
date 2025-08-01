#!/usr/bin/env python3
"""
Debug script to check market selection and data flow issues.
Run this on EC2 to diagnose the wrong predictions problem.
"""

import pandas as pd
import os
from datetime import datetime
from zoneinfo import ZoneInfo

def check_market_selection():
    """Check which markets are being selected for each currency."""
    print("🔍 MARKET SELECTION DIAGNOSTIC")
    print("=" * 50)
    
    currencies = ['btc', 'eth', 'sol', 'xrp']
    
    for currency in currencies:
        print(f"\n📊 {currency.upper()} MARKETS:")
        
        markets_file = f"{currency}_polymarkets.csv"
        if not os.path.exists(markets_file):
            print(f"   ❌ No CSV file: {markets_file}")
            continue
            
        df = pd.read_csv(markets_file)
        if df.empty:
            print(f"   ❌ Empty CSV file")
            continue
        
        # Show all available markets
        print(f"   📋 Available markets ({len(df)} total):")
        for i, row in df.iterrows():
            print(f"      {i}: {row['market_name']} | {row['date_time']}")
        
        # Test current hour selection logic
        print(f"\n   🎯 CURRENT HOUR SELECTION TEST:")
        
        # Get current hour in ET
        from zoneinfo import ZoneInfo
        current_hour_utc = datetime.now(ZoneInfo('UTC')).replace(minute=0, second=0, microsecond=0)
        et_tz = ZoneInfo("America/New_York")
        target_hour_dt_et = current_hour_utc.astimezone(et_tz)
        target_datetime_str = target_hour_dt_et.strftime('%Y-%m-%d %H:%M EDT')
        
        print(f"      Current UTC hour: {current_hour_utc}")
        print(f"      Target ET string: {target_datetime_str}")
        
        # Find matching market
        matching_rows = df[df['date_time'] == target_datetime_str]
        
        if matching_rows.empty:
            print(f"      ❌ NO MATCH for current hour!")
            print(f"      📝 Available date_time values:")
            for dt in df['date_time'].unique():
                print(f"         '{dt}'")
            print(f"      🔄 Falling back to last row:")
            latest_market = df.iloc[-1]
            print(f"         → {latest_market['market_name']}")
        else:
            print(f"      ✅ MATCH FOUND!")
            latest_market = matching_rows.iloc[0]
            print(f"         → {latest_market['market_name']}")
        
        print(f"      📋 Selected market details:")
        print(f"         Name: {latest_market['market_name']}")
        print(f"         Date: {latest_market['date_time']}")
        print(f"         YES Token: {latest_market['token_id_yes']}")
        print(f"         NO Token: {latest_market['token_id_no']}")

def check_database_state():
    """Check recent database entries to see if wrong markets are being written."""
    print(f"\n\n🗄️ DATABASE STATE DIAGNOSTIC")
    print("=" * 50)
    
    currencies = ['btc', 'eth', 'sol', 'xrp']
    
    for currency in currencies:
        db_file = f"{currency}_polyscraper.db"
        if not os.path.exists(db_file):
            print(f"\n❌ {currency.upper()}: No database file")
            continue
            
        print(f"\n📊 {currency.upper()} DATABASE:")
        
        import sqlite3
        try:
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()
                
                # Get recent entries
                cursor.execute("""
                    SELECT timestamp, market_name, token_id, best_bid, best_ask 
                    FROM polydata 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                
                rows = cursor.fetchall()
                
                if not rows:
                    print("   ❌ No data in database")
                    continue
                
                print("   📋 Recent entries:")
                for row in rows:
                    timestamp, market_name, token_id, bid, ask = row
                    market_price = (bid + ask) / 2 if bid and ask else "N/A"
                    print(f"      {timestamp} | {market_name[:30]}... | Bid:{bid} Ask:{ask} → Price:{market_price}")
                
                # Check for multiple markets in recent data
                cursor.execute("""
                    SELECT DISTINCT market_name, COUNT(*) as count
                    FROM polydata 
                    WHERE timestamp > datetime('now', '-1 hour')
                    GROUP BY market_name
                    ORDER BY count DESC
                """)
                
                recent_markets = cursor.fetchall()
                print("   🔍 Markets used in last hour:")
                for market_name, count in recent_markets:
                    print(f"      {count} entries: {market_name}")
                
                if len(recent_markets) > 1:
                    print("   🚨 WARNING: Multiple markets detected in recent data!")
                
        except Exception as e:
            print(f"   ❌ Database error: {e}")

if __name__ == "__main__":
    check_market_selection()
    check_database_state()
    
    print(f"\n\n🎯 SUMMARY & RECOMMENDATIONS:")
    print("=" * 50)
    print("1. Check if current hour market selection is working")
    print("2. Look for multiple markets in recent database entries") 
    print("3. Verify no old processes are still running:")
    print("   ps aux | grep python")
    print("   sudo systemctl status polytrader.service")
    print("4. Check cron jobs: crontab -l") 