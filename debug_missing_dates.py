"""
Check exactly which monthly investment dates are missing from trading data
This will show us if holidays/weekends are causing missed investments
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def check_missing_monthly_dates():
    """Check which monthly investment dates are missing"""
    
    print("🔍 CHECKING FOR MISSING MONTHLY INVESTMENT DATES")
    print("=" * 60)
    
    # Download VTI data
    vti = yf.Ticker("VTI")
    data = vti.history(period="max", actions=True)
    data = data.dropna()
    
    print(f"📊 Total trading days available: {len(data)}")
    print(f"📅 Date range: {data.index.min().date()} to {data.index.max().date()}")
    
    # Get theoretical monthly investment dates
    monthly_dates_first = data.resample('M').first().index
    monthly_dates_last = data.resample('M').last().index
    
    print(f"\n📈 FIRST DAY OF MONTH METHOD:")
    print("-" * 40)
    
    # Check which first-day dates actually exist in trading data
    first_day_missing = []
    first_day_found = 0
    
    for date in monthly_dates_first:
        if date in data.index:
            first_day_found += 1
        else:
            first_day_missing.append(date)
    
    print(f"✅ Investment dates found: {first_day_found}")
    print(f"❌ Investment dates missing: {len(first_day_missing)}")
    
    if first_day_missing:
        print(f"\n📋 MISSING DATES (first {min(10, len(first_day_missing))}):")
        for date in first_day_missing[:10]:
            day_of_week = date.strftime("%A")
            print(f"  {date.date()} ({day_of_week})")
    
    print(f"\n📈 LAST DAY OF MONTH METHOD:")
    print("-" * 40)
    
    # Check which last-day dates actually exist in trading data
    last_day_missing = []
    last_day_found = 0
    
    for date in monthly_dates_last:
        if date in data.index:
            last_day_found += 1
        else:
            last_day_missing.append(date)
    
    print(f"✅ Investment dates found: {last_day_found}")
    print(f"❌ Investment dates missing: {len(last_day_missing)}")
    
    if last_day_missing:
        print(f"\n📋 MISSING DATES (first {min(10, len(last_day_missing))}):")
        for date in last_day_missing[:10]:
            day_of_week = date.strftime("%A")
            print(f"  {date.date()} ({day_of_week})")
    
    # BETTER METHOD: Find actual first/last trading days
    print(f"\n🎯 BETTER METHOD - ACTUAL TRADING DAYS:")
    print("-" * 50)
    
    # Group by year-month and get actual first/last trading days
    data_with_month = data.copy()
    data_with_month['YearMonth'] = data_with_month.index.to_period('M')
    
    actual_first_days = data_with_month.groupby('YearMonth').first().index
    actual_last_days = data_with_month.groupby('YearMonth').last().index
    
    print(f"✅ Actual first trading days: {len(actual_first_days)}")
    print(f"✅ Actual last trading days: {len(actual_last_days)}")
    
    # Compare the methods
    print(f"\n📊 COMPARISON:")
    print("-" * 30)
    print(f"Pandas .first() method: {first_day_found} investments")
    print(f"Pandas .last() method: {last_day_found} investments") 
    print(f"Actual first trading days: {len(actual_first_days)} investments")
    print(f"Actual last trading days: {len(actual_last_days)} investments")
    
    # Show total investment amounts
    monthly_budget = 1000
    print(f"\n💰 TOTAL INVESTMENT COMPARISON:")
    print("-" * 40)
    print(f"Pandas first method: ${first_day_found * monthly_budget:,}")
    print(f"Pandas last method: ${last_day_found * monthly_budget:,}")
    print(f"Actual first trading: ${len(actual_first_days) * monthly_budget:,}")
    print(f"Actual last trading: ${len(actual_last_days) * monthly_budget:,}")
    
    # Daily comparison
    daily_amount = monthly_budget / 21
    daily_total = len(data) * daily_amount
    print(f"Daily strategy total: ${daily_total:,}")
    
    print(f"\n🎯 RECOMMENDATION:")
    print("-" * 30)
    
    if len(first_day_missing) > 0:
        print("❌ Current pandas .first() method is missing investment dates!")
        print("✅ Should use: data.groupby(data.index.to_period('M')).first()")
        print("   This finds actual first trading day of each month")
    
    if len(actual_first_days) == len(actual_last_days):
        print("✅ Both first and last trading day methods capture all months")
    
    # Return the corrected monthly dates for testing
    return actual_first_days, actual_last_days, first_day_missing, last_day_missing

if __name__ == "__main__":
    check_missing_monthly_dates()