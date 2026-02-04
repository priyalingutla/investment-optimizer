"""
Batch Analysis: Screen Multiple Tickers for Statistical Significance

This script runs the investment frequency analysis across many tickers
to find which ones (if any) show statistically significant differences
between daily, weekly, and monthly investing.

Usage:
    python batch_analysis.py [--top N] [--output results.csv]
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import analysis functions from main app
# We'll redefine key functions here to avoid Streamlit dependencies

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'trading_days_per_month': 21,
    'weeks_per_month': 4.33,
    'min_data_points': 1000,
    'min_window_data_points': 500,
    'min_start_buffer_days': 365,
    'step_size_divisor': 30,
    'min_step_size': 126,
    'bootstrap_iterations': 1000,
    'confidence_level': 0.95,
    'effect_size_small': 0.2,
    'effect_size_medium': 0.5,
    'effect_size_large': 0.8,
    'max_retries': 3,
    'initial_retry_delay': 2,
    'retry_backoff_factor': 2,
    'request_delay': 0.5,  # Delay between API calls to avoid rate limits
}

# Top tickers by various criteria (S&P 500 components, popular ETFs, high volume)
TOP_TICKERS = [
    # Major ETFs
    'SPY', 'QQQ', 'VTI', 'VOO', 'IWM', 'VEA', 'VWO', 'BND', 'VNQ', 'GLD',
    'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE',

    # Mega caps
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'UNH', 'HD', 'PG', 'MA', 'DIS', 'PYPL', 'ADBE', 'NFLX', 'CRM',

    # Large caps
    'INTC', 'CSCO', 'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'AVGO', 'COST', 'WMT',
    'CVX', 'XOM', 'PFE', 'NKE', 'LLY', 'ORCL', 'ACN', 'MCD', 'DHR', 'TXN',

    # Mid caps with high volume
    'AMD', 'SQ', 'ROKU', 'SNAP', 'UBER', 'LYFT', 'ZM', 'DOCU', 'CRWD', 'NET',
    'DKNG', 'PLTR', 'COIN', 'HOOD', 'RIVN', 'LCID', 'NIO', 'BABA', 'JD', 'PDD',

    # More diversified
    'BA', 'CAT', 'GE', 'MMM', 'IBM', 'GS', 'MS', 'C', 'BAC', 'WFC',
    'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'ATVI', 'EA', 'TTWO', 'MTCH', 'ABNB',
]


def download_stock_data(symbol, max_retries=3):
    """Download stock data with retry logic."""
    delay = CONFIG['initial_retry_delay']

    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="max", actions=True)

            if data is None or len(data) == 0:
                return None, f"No data found for {symbol}"

            data = data.dropna()
            if len(data) == 0:
                return None, f"Empty data for {symbol}"

            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            data['Weekday'] = data.index.day_name()
            return data, None

        except Exception as e:
            error_msg = str(e).lower()
            if 'rate' in error_msg or '429' in error_msg or 'too many' in error_msg:
                if attempt < max_retries - 1:
                    print(f"  Rate limited on {symbol}, waiting {delay}s...")
                    time.sleep(delay)
                    delay *= CONFIG['retry_backoff_factor']
                    continue
            return None, str(e)

    return None, "Max retries exceeded"


def calculate_strategy_performance(data, frequency='daily', specific_day=None, monthly_budget=1000):
    """Calculate investment returns for a strategy with NORMALIZED total investment."""
    if len(data) == 0:
        return None

    # Calculate target total investment based on data period
    data_months = (data.index.max() - data.index.min()).days / 30.44
    target_total_investment = monthly_budget * data_months

    # Get investment dates based on frequency
    if frequency == 'daily':
        investment_dates = data.index
    elif frequency == 'weekly':
        if specific_day:
            weekday_data = data[data['Weekday'] == specific_day]
            investment_dates = weekday_data.index
        else:
            weekly_groups = data.groupby(data.index.to_period('W'))
            investment_dates = weekly_groups.first().index
    elif frequency == 'monthly':
        monthly_groups = data.groupby([data.index.year, data.index.month])
        investment_dates = monthly_groups.first().index
    else:
        return None

    if len(investment_dates) == 0:
        return None

    # CRITICAL: Normalize so all strategies invest the same total
    investment_amount = target_total_investment / len(investment_dates)

    # Calculate returns
    total_invested = 0
    total_shares = 0

    for date in investment_dates:
        if date in data.index:
            price = data.loc[date, 'Close']
            if price > 0:
                shares = investment_amount / price
                total_shares += shares
                total_invested += investment_amount

    if total_invested == 0 or total_shares == 0:
        return None

    final_price = data['Close'].iloc[-1]
    final_value = total_shares * final_price
    total_return = (final_value - total_invested) / total_invested * 100

    years = (data.index.max() - data.index.min()).days / 365.25
    if years > 0:
        annualized = ((final_value / total_invested) ** (1 / years) - 1) * 100
    else:
        annualized = 0

    strategy_name = frequency if not specific_day else f"weekly_{specific_day}"

    return {
        'strategy': strategy_name,
        'annualized_return': annualized,
        'total_return': total_return,
        'total_invested': total_invested,
        'final_value': final_value,
    }


def rolling_window_analysis(data, strategies, monthly_budget=1000):
    """Test strategies across rolling windows."""
    total_years = (data.index.max() - data.index.min()).days / 365.25

    if total_years >= 20:
        window_years = [3, 5, 7, 10]
    elif total_years >= 15:
        window_years = [3, 5, 7]
    elif total_years >= 10:
        window_years = [3, 5]
    else:
        window_years = [3]

    rolling_results = []

    for window in window_years:
        if window > total_years - 1:
            continue

        window_days = window * 365
        max_start = len(data) - window_days

        if max_start < CONFIG['min_start_buffer_days']:
            continue

        step_size = max(CONFIG['min_step_size'], len(data) // CONFIG['step_size_divisor'])

        for start_idx in range(0, max_start, step_size):
            end_idx = start_idx + window_days
            window_data = data.iloc[start_idx:end_idx]

            if len(window_data) < CONFIG['min_window_data_points']:
                continue

            for strategy_name, (freq, day) in strategies.items():
                result = calculate_strategy_performance(window_data, freq, day, monthly_budget)
                if result:
                    result['window_years'] = window
                    rolling_results.append(result)

    return rolling_results


def calculate_bootstrap_ci(rolling_results):
    """Calculate bootstrap confidence intervals."""
    rolling_df = pd.DataFrame(rolling_results)
    strategies = rolling_df['strategy'].unique()

    bootstrap_results = {}

    for strategy in strategies:
        returns = rolling_df[rolling_df['strategy'] == strategy]['annualized_return'].values

        if len(returns) < 5:
            continue

        np.random.seed(42)
        bootstrap_means = []

        for _ in range(CONFIG['bootstrap_iterations']):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - CONFIG['confidence_level']

        bootstrap_results[strategy] = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'ci_lower': np.percentile(bootstrap_means, alpha / 2 * 100),
            'ci_upper': np.percentile(bootstrap_means, (1 - alpha / 2) * 100),
            'n_samples': len(returns)
        }

    return bootstrap_results


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(d):
    """Interpret Cohen's d."""
    d_abs = abs(d)
    if d_abs < CONFIG['effect_size_small']:
        return 'negligible'
    elif d_abs < CONFIG['effect_size_medium']:
        return 'small'
    elif d_abs < CONFIG['effect_size_large']:
        return 'medium'
    else:
        return 'large'


def analyze_ticker(symbol):
    """Run full analysis on a single ticker."""
    result = {
        'ticker': symbol,
        'status': 'unknown',
        'verdict': None,
        'best_strategy': None,
        'best_return': None,
        'worst_return': None,
        'spread': None,
        'data_years': None,
        'significant_pairs': None,
        'max_effect_size': None,
        'error': None
    }

    # Download data
    data, error = download_stock_data(symbol)
    if error:
        result['status'] = 'error'
        result['error'] = error
        return result

    # Check data quality
    if len(data) < CONFIG['min_data_points']:
        result['status'] = 'insufficient_data'
        result['error'] = f"Only {len(data)} days"
        return result

    data_years = (data.index.max() - data.index.min()).days / 365.25
    result['data_years'] = round(data_years, 1)

    # Define strategies (simplified - just main frequencies)
    strategies = {
        'daily': ('daily', None),
        'weekly_Monday': ('weekly', 'Monday'),
        'weekly_Wednesday': ('weekly', 'Wednesday'),
        'weekly_Friday': ('weekly', 'Friday'),
        'monthly': ('monthly', None),
    }

    # Run rolling window analysis
    rolling_results = rolling_window_analysis(data, strategies)

    if not rolling_results:
        result['status'] = 'analysis_failed'
        result['error'] = "No rolling window results"
        return result

    # Calculate bootstrap CIs
    bootstrap_cis = calculate_bootstrap_ci(rolling_results)

    if len(bootstrap_cis) < 2:
        result['status'] = 'insufficient_strategies'
        return result

    # Find best and worst
    best_strategy = max(bootstrap_cis.keys(), key=lambda s: bootstrap_cis[s]['mean'])
    worst_strategy = min(bootstrap_cis.keys(), key=lambda s: bootstrap_cis[s]['mean'])

    best_mean = bootstrap_cis[best_strategy]['mean']
    worst_mean = bootstrap_cis[worst_strategy]['mean']
    best_ci = (bootstrap_cis[best_strategy]['ci_lower'], bootstrap_cis[best_strategy]['ci_upper'])

    result['best_strategy'] = best_strategy
    result['best_return'] = round(best_mean, 2)
    result['worst_return'] = round(worst_mean, 2)
    result['spread'] = round(best_mean - worst_mean, 2)

    # Check for statistical significance
    rolling_df = pd.DataFrame(rolling_results)
    significant_pairs = []
    max_effect = 0

    for strategy in bootstrap_cis.keys():
        if strategy == best_strategy:
            continue

        other_ci = (bootstrap_cis[strategy]['ci_lower'], bootstrap_cis[strategy]['ci_upper'])
        ci_overlap = not (best_ci[0] > other_ci[1] or other_ci[0] > best_ci[1])

        best_returns = rolling_df[rolling_df['strategy'] == best_strategy]['annualized_return'].values
        other_returns = rolling_df[rolling_df['strategy'] == strategy]['annualized_return'].values

        cohens_d = calculate_cohens_d(best_returns, other_returns)
        effect_size = interpret_effect_size(cohens_d)

        max_effect = max(max_effect, abs(cohens_d))

        is_significant = not ci_overlap and effect_size in ['medium', 'large']
        if is_significant:
            significant_pairs.append(f"{best_strategy} vs {strategy}")

    result['max_effect_size'] = round(max_effect, 3)
    result['significant_pairs'] = len(significant_pairs)

    # Determine verdict
    if len(significant_pairs) == 0:
        result['verdict'] = 'no_difference'
        result['status'] = 'success'
    elif len(significant_pairs) == len(bootstrap_cis) - 1:
        result['verdict'] = 'SIGNIFICANT'
        result['status'] = 'success'
    else:
        result['verdict'] = 'partial'
        result['status'] = 'success'

    return result


def run_batch_analysis(tickers, max_workers=3):
    """Run analysis on multiple tickers."""
    results = []
    total = len(tickers)

    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS: Screening {total} tickers")
    print(f"{'='*60}\n")

    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{total}] Analyzing {ticker}...", end=" ", flush=True)

        try:
            result = analyze_ticker(ticker)
            results.append(result)

            if result['status'] == 'success':
                if result['verdict'] == 'SIGNIFICANT':
                    print(f"✅ SIGNIFICANT! Best: {result['best_strategy']} ({result['best_return']}%)")
                elif result['verdict'] == 'partial':
                    print(f"⚠️  Partial significance. Best: {result['best_strategy']} ({result['best_return']}%)")
                else:
                    print(f"➖ No significant difference (spread: {result['spread']}%)")
            else:
                print(f"❌ {result['status']}: {result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            results.append({
                'ticker': ticker,
                'status': 'exception',
                'error': str(e)
            })

        # Rate limit protection
        time.sleep(CONFIG['request_delay'])

    return results


def print_summary(results):
    """Print summary of batch analysis."""
    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    # Success rate
    success = df[df['status'] == 'success']
    print(f"Successfully analyzed: {len(success)}/{len(df)} tickers")

    if len(success) == 0:
        print("No successful analyses to summarize.")
        return df

    # Significance breakdown
    significant = success[success['verdict'] == 'SIGNIFICANT']
    partial = success[success['verdict'] == 'partial']
    no_diff = success[success['verdict'] == 'no_difference']

    print(f"\nStatistical Significance Results:")
    print(f"  - SIGNIFICANT difference: {len(significant)} ({len(significant)/len(success)*100:.1f}%)")
    print(f"  - Partial significance:   {len(partial)} ({len(partial)/len(success)*100:.1f}%)")
    print(f"  - No difference:          {len(no_diff)} ({len(no_diff)/len(success)*100:.1f}%)")

    # Show significant tickers
    if len(significant) > 0:
        print(f"\n{'='*60}")
        print("TICKERS WITH SIGNIFICANT FREQUENCY DIFFERENCES:")
        print(f"{'='*60}")
        for _, row in significant.iterrows():
            print(f"  {row['ticker']}: Best={row['best_strategy']} ({row['best_return']}%), "
                  f"Spread={row['spread']}%, Effect={row['max_effect_size']}")

    # Show partial significance
    if len(partial) > 0:
        print(f"\n{'='*60}")
        print("TICKERS WITH PARTIAL SIGNIFICANCE:")
        print(f"{'='*60}")
        for _, row in partial.iterrows():
            print(f"  {row['ticker']}: Best={row['best_strategy']} ({row['best_return']}%), "
                  f"Spread={row['spread']}%")

    # Top spreads (regardless of significance)
    print(f"\n{'='*60}")
    print("TOP 10 LARGEST RETURN SPREADS:")
    print(f"{'='*60}")
    top_spreads = success.nlargest(10, 'spread')
    for _, row in top_spreads.iterrows():
        sig_marker = "✅" if row['verdict'] == 'SIGNIFICANT' else "⚠️" if row['verdict'] == 'partial' else "➖"
        print(f"  {sig_marker} {row['ticker']}: {row['spread']}% spread "
              f"({row['best_strategy']} @ {row['best_return']}%)")

    return df


def main():
    parser = argparse.ArgumentParser(description='Batch analysis of investment frequency across tickers')
    parser.add_argument('--top', type=int, default=50, help='Number of tickers to analyze (default: 50)')
    parser.add_argument('--output', type=str, default='batch_results.csv', help='Output CSV file')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of specific tickers')

    args = parser.parse_args()

    # Determine ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = TOP_TICKERS[:args.top]

    print(f"Starting batch analysis of {len(tickers)} tickers...")
    print(f"Results will be saved to: {args.output}")

    # Run analysis
    results = run_batch_analysis(tickers)

    # Print summary
    df = print_summary(results)

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    return df


if __name__ == '__main__':
    main()
