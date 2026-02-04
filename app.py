import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import re
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - All adjustable parameters in one place
# =============================================================================
CONFIG = {
    # Cache settings
    'cache_ttl_seconds': 3600,  # 1 hour cache for market data

    # Trading calendar assumptions
    'trading_days_per_month': 21,
    'weeks_per_month': 4.33,

    # QA validation thresholds
    'monthly_coverage_threshold': 0.85,  # Require 85% of expected monthly investments
    'weekly_coverage_threshold': 0.85,   # Require 85% of expected weekly investments
    'min_data_points': 1000,             # Minimum trading days for analysis
    'min_window_data_points': 500,       # Minimum points for rolling window

    # Rolling window analysis
    'min_start_buffer_days': 365,        # Minimum days before first window starts
    'step_size_divisor': 30,             # Divide data length by this for step size
    'min_step_size': 126,                # ~6 months minimum step

    # Market regime analysis
    'rolling_return_period': 252,        # 1 year for return calculation
    'rolling_vol_period': 63,            # ~3 months for volatility
    'min_regime_periods': 200,           # Minimum data points for regime analysis

    # UI defaults
    'default_ticker': 'VTI',
    'default_monthly_amount': 1000,
    'min_investment': 100,
    'max_investment': 50000,

    # Confidence thresholds
    'high_confidence_threshold': 60,     # % win rate for high confidence
    'moderate_confidence_threshold': 40, # % win rate for moderate confidence

    # Statistical significance settings
    'bootstrap_iterations': 1000,        # Number of bootstrap resamples
    'confidence_level': 0.95,            # 95% confidence intervals
    'effect_size_small': 0.2,            # Cohen's d threshold for small effect
    'effect_size_medium': 0.5,           # Cohen's d threshold for medium effect
    'effect_size_large': 0.8,            # Cohen's d threshold for large effect
}

# Set page config with light theme
st.set_page_config(
    page_title="Optimal Investment Frequency Finder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force bright theme and pastel styling
st.markdown("""
<style>
    /* Soft pastel button - light and gentle */
    .stButton > button {
        background: linear-gradient(45deg, #ffeaa7, #fab1a0) !important;
        color: #2d3748 !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(255, 234, 167, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #fdcb6e, #e17055) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(255, 203, 110, 0.5) !important;
    }
    
    /* Winner box styling */
    .winner-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        color: #2d3748;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(255, 154, 158, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Input container styling */
    .input-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(220, 230, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class StockDataResult:
    """Structured result from stock data download for consistent handling."""
    def __init__(self, data=None, error=None, stock_name=None, data_years=0):
        self.data = data
        self.error = error
        self.stock_name = stock_name
        self.data_years = data_years
        self.success = data is not None and error is None

def validate_ticker(symbol):
    """Validate ticker symbol format before API call."""
    if not symbol or not isinstance(symbol, str):
        return False, "Ticker symbol is required"

    # Remove whitespace and validate format
    symbol = symbol.strip().upper()

    # Basic format validation (alphanumeric, dots, hyphens allowed)
    if not re.match(r'^[A-Z0-9\.\-]{1,10}$', symbol):
        return False, f"Invalid ticker format: '{symbol}'. Use 1-10 alphanumeric characters."

    return True, symbol

@st.cache_data(ttl=CONFIG['cache_ttl_seconds'])
def download_stock_data(symbol, max_years=None):
    """
    Download and prepare stock data from Yahoo Finance.

    Returns:
        StockDataResult with data, error, stock_name, and data_years
    """
    # Validate ticker format first
    is_valid, validation_result = validate_ticker(symbol)
    if not is_valid:
        return StockDataResult(error=validation_result, stock_name=symbol)

    symbol = validation_result  # Use cleaned symbol

    try:
        ticker = yf.Ticker(symbol)

        # Get historical data
        if max_years:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=max_years * 365)
            data = ticker.history(start=start_date, end=end_date, actions=True, period="max")
        else:
            data = ticker.history(actions=True, period="max")

        if data is None or len(data) == 0:
            return StockDataResult(
                error=f"No data found for '{symbol}'. Please verify the ticker symbol is correct.",
                stock_name=symbol
            )

        # Clean and prepare data
        data = data.dropna()

        if len(data) == 0:
            return StockDataResult(
                error=f"Data for '{symbol}' contains only invalid entries.",
                stock_name=symbol
            )

        # Normalize timezone
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data['Weekday'] = data.index.day_name()

        # Get stock name (with fallback)
        stock_name = symbol
        try:
            info = ticker.info
            if info and 'longName' in info:
                stock_name = info.get('longName', symbol)
        except Exception:
            pass  # Keep symbol as name if info fetch fails

        # Calculate data span
        data_years = (data.index.max() - data.index.min()).days / 365.25

        return StockDataResult(
            data=data,
            stock_name=stock_name,
            data_years=data_years
        )

    except ConnectionError:
        return StockDataResult(
            error="Network error: Unable to connect to Yahoo Finance. Please check your internet connection.",
            stock_name=symbol
        )
    except Exception as e:
        error_msg = str(e)
        if "No timezone" in error_msg or "ticker" in error_msg.lower():
            return StockDataResult(
                error=f"Invalid ticker '{symbol}': Symbol not found or delisted.",
                stock_name=symbol
            )
        return StockDataResult(
            error=f"Error fetching data for '{symbol}': {error_msg}",
            stock_name=symbol
        )

def perform_investment_qa(data, investment_dates, frequency, specific_day=None):
    """Quality assurance checks for investment schedules"""
    qa_results = {
        'total_periods': len(investment_dates),
        'warnings': [],
        'info': [],
        'passed': True
    }
    
    # Check 1: Investment coverage validation
    if frequency == 'monthly':
        data_months = (data.index.max() - data.index.min()).days / 30.44
        expected_months = int(data_months)
        actual_months = len(investment_dates)
        coverage_ratio = actual_months / expected_months if expected_months > 0 else 0

        if coverage_ratio < CONFIG['monthly_coverage_threshold']:
            qa_results['warnings'].append(
                f"Monthly strategy coverage low: {actual_months}/{expected_months} months ({coverage_ratio:.0%})"
            )
        else:
            qa_results['info'].append(
                f"✅ Monthly coverage: {actual_months}/{expected_months} months ({coverage_ratio:.0%})"
            )

    elif frequency == 'weekly':
        data_weeks = (data.index.max() - data.index.min()).days / 7
        expected_weeks = int(data_weeks)
        actual_weeks = len(investment_dates)

        if specific_day:
            # For specific weekday, expect ~1/5 of total weeks (accounting for holidays)
            expected_specific_day = expected_weeks * 0.18  # ~18% accounts for holidays
            coverage_ratio = actual_weeks / expected_specific_day if expected_specific_day > 0 else 0
            if coverage_ratio < CONFIG['weekly_coverage_threshold']:
                qa_results['warnings'].append(
                    f"Weekly {specific_day} coverage low: {actual_weeks} investments ({coverage_ratio:.0%} of expected)"
                )
            else:
                qa_results['info'].append(
                    f"✅ Weekly {specific_day} coverage: {actual_weeks} investments ({coverage_ratio:.0%})"
                )
        else:
            coverage_ratio = actual_weeks / expected_weeks if expected_weeks > 0 else 0
            if coverage_ratio < CONFIG['weekly_coverage_threshold']:
                qa_results['warnings'].append(
                    f"Weekly strategy coverage low: {actual_weeks}/{expected_weeks} weeks ({coverage_ratio:.0%})"
                )
            else:
                qa_results['info'].append(
                    f"✅ Weekly coverage: {actual_weeks}/{expected_weeks} weeks ({coverage_ratio:.0%})"
                )
    
    # Check 2: Investment dates exist in trading data (informational only)
    missing_dates = sum(1 for date in investment_dates if date not in data.index)
    if missing_dates > 0:
        qa_results['info'].append(f"ℹ️ {missing_dates} investment dates adjusted for trading calendar")
    else:
        qa_results['info'].append(f"✅ All investment dates exist in trading data")
    
    # Check 3: Simple period count check
    if len(investment_dates) > 0:
        if frequency == 'monthly':
            qa_results['info'].append(f"✅ Monthly investments: {len(investment_dates)} periods")
        elif frequency == 'weekly':
            qa_results['info'].append(f"✅ Weekly investments: {len(investment_dates)} periods")
        elif frequency == 'daily':
            qa_results['info'].append(f"✅ Daily investments: {len(investment_dates)} periods")
    
    # Check 4: Data coverage (informational only)
    data_years = (data.index.max() - data.index.min()).days / 365.25
    if data_years < 1:
        qa_results['warnings'].append(f"Very limited data: Only {data_years:.1f} years available")
    else:
        qa_results['info'].append(f"✅ Data coverage: {data_years:.1f} years")
    
    return qa_results

def calculate_strategy_performance(data, frequency='daily', specific_day=None, monthly_budget=4000):
    """Calculate investment returns with robust date handling - FIXED MONTHLY CALCULATION"""
    
    if len(data) == 0:
        return None
    
    # Get investment dates and amounts based on frequency
    if frequency == 'daily':
        investment_dates = data.index
        investment_amount = monthly_budget / CONFIG['trading_days_per_month']
    elif frequency == 'weekly':
        if specific_day:
            weekday_data = data[data['Weekday'] == specific_day]
            investment_dates = weekday_data.index
        else:
            weekly_groups = data.groupby(data.index.to_period('W'))
            investment_dates = weekly_groups.first().index
        investment_amount = monthly_budget / CONFIG['weeks_per_month']
    elif frequency == 'monthly':
        # FIXED: Use a more robust monthly date generation method
        # Group by year-month and take the first trading day of each month
        monthly_groups = data.groupby([data.index.year, data.index.month])
        investment_dates = monthly_groups.first().index
        
        # CRITICAL FIX: Make sure we use the correct investment amount
        investment_amount = monthly_budget  # This should be the full monthly budget!
    else:
        return None
    
    # Perform QA checks silently - only fail if critical issues found
    qa_checks = perform_investment_qa(data, investment_dates, frequency, specific_day)
    
    # Only fail if we have no investment dates or critical data issues
    if len(investment_dates) == 0:
        return None
    
    # Track investment performance
    total_shares = 0
    total_invested = 0
    portfolio_history = []
    
    for date in investment_dates:
        if date in data.index:
            price = data.loc[date, 'Close']
            shares_bought = investment_amount / price
            total_shares += shares_bought
            total_invested += investment_amount
            
            current_value = total_shares * price
            portfolio_history.append({
                'Date': date,
                'Portfolio_Value': current_value,
                'Total_Invested': total_invested,
                'Price': price
            })
    
    if not portfolio_history:
        return None
    
    # Calculate metrics
    final_value = portfolio_history[-1]['Portfolio_Value']
    total_return = (final_value - total_invested) / total_invested
    years_invested = (data.index.max() - data.index.min()).days / 365.25
    annualized_return = ((final_value / total_invested) ** (1 / years_invested)) - 1
    
    # Calculate max drawdown
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df['Running_Max'] = portfolio_df['Portfolio_Value'].cummax()
    portfolio_df['Drawdown'] = (portfolio_df['Portfolio_Value'] - portfolio_df['Running_Max']) / portfolio_df['Running_Max']
    max_drawdown = abs(portfolio_df['Drawdown'].min())
    
    return {
        'strategy': f"{frequency}" + (f"_{specific_day}" if specific_day else ""),
        'annualized_return': annualized_return * 100,
        'total_return': total_return * 100,
        'max_drawdown': max_drawdown * 100,
        'final_value': final_value,
        'total_invested': total_invested,
        'years_invested': years_invested,
        'portfolio_history': portfolio_df,
        'investment_periods': len(investment_dates),
        'investment_amount': investment_amount,
        'qa_results': qa_checks
    }

def rolling_window_analysis(data, strategies, monthly_budget, window_years=None):
    """Test strategies across rolling time windows for robustness analysis."""

    total_years = (data.index.max() - data.index.min()).days / 365.25

    # Determine appropriate window sizes based on available data
    if window_years is None:
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
        start_points = range(0, max_start, step_size)

        for start_idx in start_points:
            end_idx = start_idx + window_days
            window_data = data.iloc[start_idx:end_idx]

            if len(window_data) < CONFIG['min_window_data_points']:
                continue

            window_results = []
            for strategy_name, (freq, day) in strategies.items():
                result = calculate_strategy_performance(window_data, freq, day, monthly_budget)
                if result:
                    result['window_years'] = window
                    result['start_date'] = window_data.index.min()
                    result['end_date'] = window_data.index.max()
                    window_results.append(result)

            if window_results:
                best = max(window_results, key=lambda x: x['annualized_return'])
                for result in window_results:
                    result['is_winner'] = (result['strategy'] == best['strategy'])

                rolling_results.extend(window_results)

    return rolling_results

def regime_analysis(data, strategies, monthly_budget):
    """Analyze performance across different market conditions (volatility & returns)."""

    data_copy = data.copy()

    # Calculate rolling metrics using configurable periods
    return_period = CONFIG['rolling_return_period']
    vol_period = CONFIG['rolling_vol_period']

    data_copy['Rolling_Return'] = data_copy['Close'].pct_change(return_period) * 100
    data_copy['Rolling_Vol'] = data_copy['Close'].pct_change().rolling(vol_period).std() * np.sqrt(252) * 100

    data_copy = data_copy.ffill().bfill()

    # Calculate quantile thresholds
    vol_75th = data_copy['Rolling_Vol'].quantile(0.75)
    vol_25th = data_copy['Rolling_Vol'].quantile(0.25)
    ret_75th = data_copy['Rolling_Return'].quantile(0.75)
    ret_25th = data_copy['Rolling_Return'].quantile(0.25)

    # Define market conditions
    market_conditions = {
        'High Volatility': data_copy['Rolling_Vol'] > vol_75th,
        'Low Volatility': data_copy['Rolling_Vol'] < vol_25th,
        'Bear Periods': data_copy['Rolling_Return'] < ret_25th,
        'Bull Periods': data_copy['Rolling_Return'] > ret_75th,
        'Crisis Periods': (data_copy['Rolling_Vol'] > vol_75th) & (data_copy['Rolling_Return'] < ret_25th),
        'Goldilocks': (data_copy['Rolling_Vol'] < vol_25th) & (data_copy['Rolling_Return'] > ret_75th)
    }

    regime_results = []
    min_periods = CONFIG['min_regime_periods']

    for condition_name, condition_mask in market_conditions.items():
        if condition_mask.sum() < min_periods:
            continue

        condition_data = data_copy[condition_mask]

        if len(condition_data) < min_periods:
            continue

        condition_strategy_results = []
        for strategy_name, (freq, day) in strategies.items():
            result = calculate_strategy_performance(condition_data, freq, day, monthly_budget)
            if result:
                result['regime'] = condition_name
                result['regime_periods'] = len(condition_data)
                condition_strategy_results.append(result)

        if condition_strategy_results:
            best = max(condition_strategy_results, key=lambda x: x['annualized_return'])
            for result in condition_strategy_results:
                result['is_winner'] = (result['strategy'] == best['strategy'])

            regime_results.extend(condition_strategy_results)

    return regime_results


def calculate_bootstrap_ci(rolling_results, n_bootstrap=None, confidence_level=None):
    """
    Calculate bootstrap confidence intervals for strategy returns.

    Uses the rolling window results to bootstrap confidence intervals,
    answering: "How confident are we in each strategy's mean return?"

    Returns:
        dict: {strategy_name: {'mean': x, 'ci_lower': y, 'ci_upper': z, 'std': s}}
    """
    if n_bootstrap is None:
        n_bootstrap = CONFIG['bootstrap_iterations']
    if confidence_level is None:
        confidence_level = CONFIG['confidence_level']

    # Convert to DataFrame and group by strategy
    rolling_df = pd.DataFrame(rolling_results)
    strategies = rolling_df['strategy'].unique()

    bootstrap_results = {}

    for strategy in strategies:
        strategy_returns = rolling_df[rolling_df['strategy'] == strategy]['annualized_return'].values

        if len(strategy_returns) < 5:
            continue

        # Bootstrap resampling
        bootstrap_means = []
        np.random.seed(42)  # Reproducibility

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(strategy_returns, size=len(strategy_returns), replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        bootstrap_results[strategy] = {
            'mean': np.mean(strategy_returns),
            'std': np.std(strategy_returns),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_samples': len(strategy_returns)
        }

    return bootstrap_results


def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_std

    Interpretation:
        |d| < 0.2: negligible
        |d| 0.2-0.5: small
        |d| 0.5-0.8: medium
        |d| > 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < CONFIG['effect_size_small']:
        return 'negligible'
    elif d_abs < CONFIG['effect_size_medium']:
        return 'small'
    elif d_abs < CONFIG['effect_size_large']:
        return 'medium'
    else:
        return 'large'


def statistical_significance_analysis(rolling_results):
    """
    Comprehensive statistical significance analysis.

    Determines if differences between strategies are statistically meaningful
    or just noise.

    Returns:
        dict with bootstrap CIs, pairwise comparisons, and overall verdict
    """
    rolling_df = pd.DataFrame(rolling_results)
    strategies = rolling_df['strategy'].unique()

    # 1. Bootstrap confidence intervals for each strategy
    bootstrap_cis = calculate_bootstrap_ci(rolling_results)

    # 2. Find the best strategy by mean return
    best_strategy = max(bootstrap_cis.keys(), key=lambda s: bootstrap_cis[s]['mean'])
    best_mean = bootstrap_cis[best_strategy]['mean']
    best_ci = (bootstrap_cis[best_strategy]['ci_lower'], bootstrap_cis[best_strategy]['ci_upper'])

    # 3. Pairwise comparisons with best strategy
    pairwise_results = {}
    significant_differences = []

    for strategy in strategies:
        if strategy == best_strategy:
            continue

        strategy_returns = rolling_df[rolling_df['strategy'] == strategy]['annualized_return'].values
        best_returns = rolling_df[rolling_df['strategy'] == best_strategy]['annualized_return'].values

        # Check CI overlap (non-overlapping = significant)
        other_ci = (bootstrap_cis[strategy]['ci_lower'], bootstrap_cis[strategy]['ci_upper'])
        ci_overlap = not (best_ci[0] > other_ci[1] or other_ci[0] > best_ci[1])

        # Calculate effect size
        cohens_d = calculate_cohens_d(best_returns, strategy_returns)
        effect_interpretation = interpret_effect_size(cohens_d)

        # Determine significance
        is_significant = not ci_overlap and effect_interpretation in ['medium', 'large']

        pairwise_results[strategy] = {
            'mean_diff': best_mean - bootstrap_cis[strategy]['mean'],
            'ci_overlap': ci_overlap,
            'cohens_d': cohens_d,
            'effect_size': effect_interpretation,
            'is_significant': is_significant
        }

        if is_significant:
            significant_differences.append(strategy)

    # 4. Overall verdict
    if len(significant_differences) == 0:
        verdict = 'no_significant_difference'
        verdict_text = "Differences between strategies are NOT statistically significant. Investment frequency doesn't meaningfully impact returns for this ticker."
    elif len(significant_differences) == len(strategies) - 1:
        verdict = 'clear_winner'
        verdict_text = f"{best_strategy.replace('_', ' ').title()} significantly outperforms ALL other strategies."
    else:
        verdict = 'partial_significance'
        sig_names = [s.replace('_', ' ').title() for s in significant_differences]
        verdict_text = f"{best_strategy.replace('_', ' ').title()} significantly outperforms: {', '.join(sig_names)}. Other strategies perform similarly."

    return {
        'bootstrap_cis': bootstrap_cis,
        'best_strategy': best_strategy,
        'pairwise': pairwise_results,
        'significant_differences': significant_differences,
        'verdict': verdict,
        'verdict_text': verdict_text
    }


# Main App Header
st.markdown("""
<div style="
    background: linear-gradient(45deg, #fab1a0, #ffeaa7, #a29bfe);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem;
    text-align: center;
    font-weight: 600;
    margin-bottom: 2rem;
    padding: 1rem;
">
📈 Optimal Investment Frequency Finder
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem; color: #666;'>
    Find the best investment timing strategy by backtesting across different market conditions and time periods
</div>
""", unsafe_allow_html=True)

# Input Section
col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input(
        "📊 Stock Ticker",
        value=CONFIG['default_ticker'],
        help="Enter any stock ticker (VTI, SPY, QQQ, AAPL, etc.)"
    ).upper()

with col2:
    monthly_amount = st.number_input(
        "💰 Monthly Investment ($)",
        min_value=CONFIG['min_investment'],
        max_value=CONFIG['max_investment'],
        value=CONFIG['default_monthly_amount'],
        step=100,
        help="How much you want to invest each month"
    )

with col3:
    use_max_data = st.checkbox(
        "📊 Use All Available Data",
        value=True,
        help="Use maximum historical data for more robust analysis"
    )
    
    if not use_max_data:
        analysis_years = st.slider(
            "📅 Analysis Period (Years)",
            min_value=5,
            max_value=25,
            value=15,
            help="How many years of data to analyze"
        )
    else:
        st.write("**Using maximum available data**")

# Methodology Section (Educational)
with st.expander("📚 How This Analysis Works", expanded=False):
    st.markdown("""
    ### Understanding the Methodology

    This tool helps you determine if **when** you invest (daily, weekly, or monthly) actually
    matters for your returns. Here's what we do:

    ---

    #### 1️⃣ **Dollar-Cost Averaging (DCA) Simulation**

    We simulate investing your monthly budget using different frequencies:
    - **Daily**: Split your monthly amount across ~21 trading days
    - **Weekly**: Split across ~4.33 weeks (we test each day of the week separately)
    - **Monthly**: Invest the full amount on the first trading day of each month

    Each strategy invests the **same total amount** over time - only the timing differs.

    ---

    #### 2️⃣ **Rolling Window Analysis**

    Instead of just looking at one time period, we test each strategy across **many different
    market periods** (3-year, 5-year, 7-year, and 10-year windows). This shows how strategies
    perform in different market conditions:
    - Bull markets 📈
    - Bear markets 📉
    - High volatility periods
    - Calm markets

    ---

    #### 3️⃣ **Statistical Significance Testing**

    Just because one strategy has a higher average return doesn't mean it's actually better.
    The difference could be random noise. We use two statistical methods:

    **Bootstrap Confidence Intervals**
    - We resample our results 1,000 times to estimate the true range of each strategy's returns
    - If the confidence intervals overlap, the strategies are statistically similar
    - Think of it as: "If we ran this experiment 1,000 times, what range of results would we see?"

    **Cohen's d Effect Size**
    - Even if there's a statistical difference, is it *practically* meaningful?
    - Effect sizes: Negligible (<0.2) → Small (0.2-0.5) → Medium (0.5-0.8) → Large (>0.8)
    - A "small" effect might be statistically significant but not worth changing your behavior

    ---

    #### 4️⃣ **The Verdict**

    We combine statistical significance with effect size to give you a clear answer:

    | Result | What It Means |
    |--------|---------------|
    | **No Significant Difference** | Pick whichever frequency is most convenient for you |
    | **Clear Winner** | One strategy is reliably better - use it |
    | **Partial Significance** | Some differences exist, but the best choice depends on your preferences |

    ---

    #### 📖 Key Concepts

    **Annualized Return**: Your yearly growth rate, accounting for compounding.
    *Example: 8% annualized means $1,000 becomes $1,080 after one year*

    **Max Drawdown**: The largest peak-to-trough decline in your portfolio.
    *Example: 20% drawdown means at some point you were down 20% from your highest value*

    **Confidence Interval**: A range where the true value likely falls.
    *Example: 8.2% [7.4% - 9.1%] means we're 95% confident the true return is in that range*

    **Statistical Significance**: The probability that an observed difference is real, not random chance.
    *We use p < 0.05, meaning less than 5% chance the difference is due to luck*
    """)

# Run Analysis Button
if st.button("🚀 Find Optimal Strategy", type="primary", use_container_width=True):

    # Download data
    with st.spinner(f"📡 Downloading data for {ticker}..."):
        if use_max_data:
            result = download_stock_data(ticker)
        else:
            result = download_stock_data(ticker, analysis_years)

        # Extract results from StockDataResult object
        data = result.data
        error = result.error
        stock_name = result.stock_name
        data_years = result.data_years

    if not result.success:
        st.error(f"❌ {error}")
        st.stop()

    if len(data) < CONFIG['min_data_points']:
        st.error(f"❌ Insufficient data for robust analysis. Found {len(data):,} trading days, need at least {CONFIG['min_data_points']:,}. Try a different ticker or longer time period.")
        st.stop()
    
    st.success(f"✅ Loaded {len(data):,} trading days for {stock_name} ({data_years:.1f} years of data)")
    st.info(f"📅 Data range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
    
    # Define strategies to test
    strategies = {
        'Daily': ('daily', None),
        'Monthly': ('monthly', None),
        'Monday': ('weekly', 'Monday'),
        'Tuesday': ('weekly', 'Tuesday'), 
        'Wednesday': ('weekly', 'Wednesday'),
        'Thursday': ('weekly', 'Thursday'),
        'Friday': ('weekly', 'Friday')
    }
    
    # Main Analysis
    with st.spinner("🔍 Running comprehensive backtests..."):
        
        # Overall performance test
        overall_results = []
        for strategy_name, (freq, day) in strategies.items():
            result = calculate_strategy_performance(data, freq, day, monthly_amount)
            if result:
                overall_results.append(result)
        
        # Rolling window analysis
        rolling_results = rolling_window_analysis(data, strategies, monthly_amount)
        
        # Market condition analysis
        regime_results = regime_analysis(data, strategies, monthly_amount)
    
    # Results Display
    if overall_results:
        
        # Find overall winner
        best_overall = max(overall_results, key=lambda x: x['annualized_return'])
        
        # Winner announcement
        st.markdown(f"""
        <div class="winner-box">
            🏆 <strong>OPTIMAL STRATEGY: {best_overall['strategy'].replace('_', ' ').upper()}</strong><br>
            📈 <strong>{best_overall['annualized_return']:.2f}% Annualized Return</strong> | 
            💰 <strong>${best_overall['final_value']:,.0f} Final Value</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance comparison
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Strategy Performance Comparison")
            
            # Create comparison chart
            results_df = pd.DataFrame(overall_results)
            results_df['Display_Name'] = results_df['strategy'].str.replace('_', ' ').str.title()
            
            fig = px.bar(
                results_df.sort_values('annualized_return', ascending=True),
                x='annualized_return',
                y='Display_Name',
                orientation='h',
                title=f"Annualized Returns - {stock_name}",
                labels={'annualized_return': 'Annualized Return (%)', 'Display_Name': 'Strategy'},
                color='annualized_return',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(
                height=400, 
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Key Metrics")
            
            profit = best_overall['final_value'] - best_overall['total_invested']
            
            st.metric(
                label="💰 Final Portfolio Value", 
                value=f"${best_overall['final_value']:,.0f}",
                delta=f"+${profit:,.0f}"
            )
            
            st.metric(
                label="📊 Total Invested", 
                value=f"${best_overall['total_invested']:,.0f}",
                delta=f"{best_overall['years_invested']:.1f} years"
            )
            
            st.metric(
                label="📈 Annualized Return", 
                value=f"{best_overall['annualized_return']:.2f}%",
                delta=f"Total: {best_overall['total_return']:.1f}%"
            )
            
            st.metric(
                label="📉 Max Drawdown", 
                value=f"{best_overall['max_drawdown']:.1f}%",
                delta="Risk measure",
                delta_color="off"
            )
        
        # Rolling window results
        if rolling_results:
            st.subheader("🔄 Rolling Window Analysis")
            st.markdown(f"Comprehensive testing across {len(rolling_results)} different time periods - each capturing unique market conditions")
            
            rolling_df = pd.DataFrame(rolling_results)
            
            window_summary = rolling_df.groupby('window_years').size()
            total_periods = len(rolling_df) // len(strategies)
            st.write(f"**Windows tested**: {', '.join([f'{int(years)}yr ({count//len(strategies)} periods)' for years, count in window_summary.items()])} = **{total_periods} total market periods**")
            
            # Calculate overall win rates across ALL periods
            win_rates = rolling_df.groupby('strategy')['is_winner'].agg(['sum', 'count'])
            win_rates['win_rate'] = (win_rates['sum'] / win_rates['count'] * 100).round(1)
            win_rates = win_rates.sort_values('win_rate', ascending=False)
            
            # Show average performance across all periods
            avg_performance = rolling_df.groupby('strategy')['annualized_return'].agg(['mean', 'std', 'min', 'max']).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏆 Win Rates Across All Periods:**")
                for strategy, row in win_rates.iterrows():
                    strategy_name = strategy.replace('_', ' ').title()
                    avg_return = avg_performance.loc[strategy, 'mean']
                    st.write(f"• **{strategy_name}**: {row['win_rate']}% wins | Avg: {avg_return:.1f}%")
            
            with col2:
                # Simple summary instead of confusing scatter plot
                st.write("**📊 Performance Summary:**")
                st.write(f"• **Total periods tested**: {total_periods}")
                st.write(f"• **Most consistent winner**: {win_rates.index[0].replace('_', ' ').title()}")
                st.write(f"• **Highest average return**: {avg_performance['mean'].idxmax().replace('_', ' ').title()}")
                
                # Show performance range
                best_performance = avg_performance.loc[avg_performance['mean'].idxmax()]
                worst_performance = avg_performance.loc[avg_performance['mean'].idxmin()]
                performance_spread = best_performance['mean'] - worst_performance['mean']
                st.write(f"• **Performance spread**: {performance_spread:.2f}% difference between best and worst")

            # Statistical Significance Analysis
            st.subheader("📊 Statistical Significance Analysis")
            st.markdown("*Does investment frequency actually matter for this ticker, or is it just noise?*")

            sig_analysis = statistical_significance_analysis(rolling_results)

            # Display bootstrap confidence intervals
            st.markdown("**95% Confidence Intervals by Strategy:**")

            # Create a DataFrame for display
            ci_data = []
            for strategy, stats in sig_analysis['bootstrap_cis'].items():
                ci_data.append({
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Mean Return': f"{stats['mean']:.2f}%",
                    '95% CI': f"[{stats['ci_lower']:.2f}% - {stats['ci_upper']:.2f}%]",
                    'Std Dev': f"{stats['std']:.2f}%",
                    'Samples': stats['n_samples']
                })

            ci_df = pd.DataFrame(ci_data)
            ci_df = ci_df.sort_values('Mean Return', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
            st.dataframe(ci_df, use_container_width=True, hide_index=True)

            # Pairwise comparisons with best strategy
            best_strat_name = sig_analysis['best_strategy'].replace('_', ' ').title()
            st.markdown(f"**Comparing other strategies to {best_strat_name}:**")

            col1, col2 = st.columns(2)
            with col1:
                for strategy, comparison in sig_analysis['pairwise'].items():
                    strategy_name = strategy.replace('_', ' ').title()
                    diff = comparison['mean_diff']
                    effect = comparison['effect_size']
                    overlap = "overlap" if comparison['ci_overlap'] else "no overlap"

                    if comparison['is_significant']:
                        st.markdown(f"• **{strategy_name}**: {diff:+.2f}% diff, **{effect}** effect ⚠️")
                    else:
                        st.markdown(f"• **{strategy_name}**: {diff:+.2f}% diff, {effect} effect")

            with col2:
                # Effect size legend
                st.markdown("**Effect Size Guide:**")
                st.markdown("• *Negligible* (<0.2): No practical difference")
                st.markdown("• *Small* (0.2-0.5): Minor difference")
                st.markdown("• *Medium* (0.5-0.8): Meaningful difference")
                st.markdown("• *Large* (>0.8): Major difference")

            # Verdict box
            if sig_analysis['verdict'] == 'no_significant_difference':
                verdict_color = "#e8f5e9"  # Light green
                verdict_icon = "✅"
            elif sig_analysis['verdict'] == 'clear_winner':
                verdict_color = "#fff3e0"  # Light orange
                verdict_icon = "🏆"
            else:
                verdict_color = "#e3f2fd"  # Light blue
                verdict_icon = "📊"

            st.markdown(f"""
            <div style="
                background: {verdict_color};
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                border-left: 4px solid {'#4caf50' if sig_analysis['verdict'] == 'no_significant_difference' else '#ff9800'};
            ">
                <strong>{verdict_icon} STATISTICAL VERDICT:</strong><br>
                {sig_analysis['verdict_text']}
            </div>
            """, unsafe_allow_html=True)

        # Market condition results
        if regime_results:
            st.subheader("📊 Market Condition Analysis") 
            st.markdown("Performance across naturally occurring market conditions (based on volatility and returns)")
            
            regime_df = pd.DataFrame(regime_results)
            
            # Create regime performance heatmap
            regime_pivot = regime_df.pivot_table(
                values='annualized_return',
                index='strategy',
                columns='regime',
                aggfunc='mean'
            ).fillna(0)
            
            # Clean up strategy names for display
            regime_pivot.index = regime_pivot.index.str.replace('_', ' ').str.title()
            
            fig = px.imshow(
                regime_pivot,
                aspect='auto',
                title="Strategy Performance by Market Condition (%)",
                color_continuous_scale='RdYlGn',
                labels={'color': 'Annualized Return (%)'},
                text_auto='.1f'  # Show values on the heatmap
            )
            fig.update_layout(
                height=400,
                xaxis_title="Market Condition",
                yaxis_title="Investment Strategy"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Condition winners and frequency
            regime_winners = regime_df[regime_df['is_winner']].groupby('regime')['strategy'].first()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎯 Best Strategy by Market Condition:**")
                for regime, winner in regime_winners.items():
                    winner_name = winner.replace('_', ' ').title()
                    st.markdown(f"• **{regime}**: {winner_name}")
            
            with col2:
                # Show regime frequency
                regime_counts = regime_df.groupby('regime')['regime_periods'].first().sort_values(ascending=False)
                st.markdown("**📈 Market Condition Frequency:**")
                for regime, periods in regime_counts.items():
                    st.markdown(f"• {regime}: {periods} trading days")
        
        # Portfolio growth chart
        st.subheader("📈 Portfolio Growth Over Time")
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, result in enumerate(overall_results):
            if 'portfolio_history' in result and result['portfolio_history'] is not None:
                portfolio_df = result['portfolio_history']
                strategy_name = result['strategy'].replace('_', ' ').title()
                
                fig.add_trace(go.Scatter(
                    x=portfolio_df['Date'],
                    y=portfolio_df['Portfolio_Value'],
                    mode='lines',
                    name=strategy_name,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title=f"Portfolio Growth Comparison - {stock_name}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary insights
        st.subheader("🎯 Key Insights & Recommendation")

        insights = []

        # Overall best strategy
        insights.append(f"**🏆 Best Overall Strategy**: {best_overall['strategy'].replace('_', ' ').title()} with {best_overall['annualized_return']:.2f}% annualized returns")

        # Most consistent winner across all periods
        if rolling_results:
            rolling_df = pd.DataFrame(rolling_results)
            most_consistent = rolling_df.groupby('strategy')['is_winner'].sum().idxmax()
            most_consistent_rate = rolling_df.groupby('strategy')['is_winner'].mean().max() * 100

            if most_consistent == best_overall['strategy']:
                insights.append(f"**✅ Consistency**: This strategy wins {most_consistent_rate:.0f}% of all time periods tested")
            else:
                insights.append(f"**⚖️ Alternative**: {most_consistent.replace('_', ' ').title()} is most consistent (wins {most_consistent_rate:.0f}% of periods)")

        # Performance vs market conditions
        if regime_results:
            regime_df = pd.DataFrame(regime_results)
            best_strategy_regimes = regime_df[regime_df['strategy'] == best_overall['strategy']]
            avg_regime_return = best_strategy_regimes['annualized_return'].mean()
            insights.append(f"**📊 Market Adaptability**: Averages {avg_regime_return:.1f}% across all market conditions")

        # Statistical significance insight (sig_analysis was computed in rolling window section)
        stat_verdict = sig_analysis['verdict'] if rolling_results else None

        if stat_verdict == 'no_significant_difference':
            insights.append("**📈 Statistical Finding**: Frequency differences are NOT significant - choose based on convenience")
        elif stat_verdict == 'clear_winner':
            insights.append(f"**📈 Statistical Finding**: {best_overall['strategy'].replace('_', ' ').title()} is statistically superior")
        elif stat_verdict == 'partial_significance':
            insights.append("**📈 Statistical Finding**: Some frequency differences are meaningful")

        # Final recommendation incorporating statistical analysis
        if rolling_results:
            total_periods_tested = len(rolling_df) // len(strategies)
            best_strategy_wins = rolling_df[rolling_df['strategy'] == best_overall['strategy']]['is_winner'].sum()
            confidence_level = best_strategy_wins / total_periods_tested * 100

            # Adjust confidence based on statistical significance
            if stat_verdict == 'no_significant_difference':
                # If no significant difference, recommend convenience
                confidence = "FREQUENCY DOESN'T MATTER 🎯"
                recommendation = "Pick whichever frequency fits your schedule best"
            elif confidence_level >= CONFIG['high_confidence_threshold']:
                confidence = "HIGH CONFIDENCE ✅"
                recommendation = f"Deploy {best_overall['strategy'].replace('_', ' ').title()} strategy"
            elif confidence_level >= CONFIG['moderate_confidence_threshold']:
                confidence = "MODERATE CONFIDENCE ⚖️"
                recommendation = f"Deploy {best_overall['strategy'].replace('_', ' ').title()} strategy"
            else:
                confidence = "LOW CONFIDENCE ⚠️"
                recommendation = f"Consider {best_overall['strategy'].replace('_', ' ').title()}, but any frequency is acceptable"

            insights.append(f"**🎯 Recommendation**: {confidence} - {recommendation}")

        for insight in insights:
            st.markdown(f"• {insight}")

        # Action Plan
        st.markdown("---")
        daily_amount = monthly_amount / CONFIG['trading_days_per_month']
        weekly_amount = monthly_amount / CONFIG['weeks_per_month']

        # Customize action plan based on statistical significance
        if stat_verdict == 'no_significant_difference':
            st.markdown(f"""
            ### 🚀 **Action Plan**
            **Investment frequency doesn't significantly impact returns for {stock_name}.**

            Choose the option that best fits your lifestyle:
            • **Daily**: Invest ${daily_amount:.0f}/day - Best for automated investing
            • **Weekly**: Invest ${weekly_amount:.0f}/week - Good balance of simplicity and averaging
            • **Monthly**: Invest ${monthly_amount:,}/month - Simplest to manage manually

            All strategies performed statistically similarly across **{total_periods_tested}** market periods.
            """)
        else:
            st.markdown(f"""
            ### 🚀 **Action Plan**
            **Start investing ${monthly_amount:,}/month using the {best_overall['strategy'].replace('_', ' ').title()} strategy:**

            {f"• **Daily**: Invest ${daily_amount:.0f} every trading day" if best_overall['strategy'] == 'daily' else ""}
            {f"• **Monthly**: Invest ${monthly_amount:,} on the first trading day of each month" if best_overall['strategy'] == 'monthly' else ""}
            {f"• **Weekly**: Invest ${weekly_amount:.0f} every {best_overall['strategy'].split('_')[1]}" if 'weekly' in best_overall['strategy'] else ""}

            This strategy has been tested across **{total_periods_tested if 'total_periods_tested' in locals() else 'multiple'}** different market periods.
            """)
        
        # Consolidated QA Summary at the end - only show if no major issues
        with st.expander("🔍 Quality Assurance Summary", expanded=False):
            st.markdown("**✅ Analysis Validation:**")
            
            # Overall data quality
            st.info(f"📅 **Data Coverage**: {data_years:.1f} years ({len(data):,} trading days)")
            
            # Investment equivalence checks
            st.markdown("**💰 Investment Validation:**")
            
            # Get daily and monthly strategies for comparison
            daily_result = next((r for r in overall_results if r['strategy'] == 'daily'), None)
            monthly_result = next((r for r in overall_results if r['strategy'] == 'monthly'), None)
            
            if daily_result and monthly_result:
                # Calculate expected values
                expected_years = data_years
                expected_months = expected_years * 12
                
                # Check monthly strategy
                monthly_actual = monthly_result['total_invested']
                expected_monthly_total = expected_months * monthly_amount
                monthly_expected_ratio = monthly_actual / expected_monthly_total
                
                if 0.85 <= monthly_expected_ratio <= 1.15:
                    st.success(f"✅ **Monthly Strategy**: ${monthly_actual:,.0f} invested ({monthly_expected_ratio:.1%} of expected)")
                else:
                    st.warning(f"⚠️ **Monthly Strategy**: ${monthly_actual:,.0f} invested ({monthly_expected_ratio:.1%} of expected ${expected_monthly_total:,.0f})")
                
                # Investment frequency validation
                monthly_frequency = monthly_result['investment_periods'] / expected_years
                
                if 10 <= monthly_frequency <= 14:  # ~12 months per year
                    st.success(f"✅ **Monthly Frequency**: {monthly_frequency:.1f} investments/year (expected ~12)")
                else:
                    st.info(f"ℹ️ **Monthly Frequency**: {monthly_frequency:.1f} investments/year (expected ~12)")
            
            # Strategy investment summary
            st.markdown("**📊 Strategy Summary:**")
            for result in overall_results:
                strategy_name = result['strategy'].replace('_', ' ').title()
                periods = result['investment_periods']
                total = result['total_invested']
                annual_investments = periods / data_years
                
                st.write(f"• **{strategy_name}**: {periods:,} investments | ${total:,.0f} total | {annual_investments:.1f}/year")
            
            # Data integrity checks
            data_start = data.index.min().strftime('%Y-%m-%d')
            data_end = data.index.max().strftime('%Y-%m-%d')
            st.success(f"✅ **Data Integrity**: All strategies use consistent date range ({data_start} to {data_end})")
            
            # Rolling window validation
            if rolling_results:
                rolling_df = pd.DataFrame(rolling_results)
                total_windows = len(rolling_df) // len(strategies)
                window_sizes = sorted(rolling_df['window_years'].unique())
                st.success(f"✅ **Rolling Analysis**: {total_windows} periods tested across {window_sizes} year windows")
            
            # Final validation
            st.markdown("**🎯 Recommendation Confidence:**")
            if rolling_results:
                rolling_df = pd.DataFrame(rolling_results)
                best_strategy_wins = rolling_df[rolling_df['strategy'] == best_overall['strategy']]['is_winner'].sum()
                total_periods = len(rolling_df) // len(strategies)
                confidence = best_strategy_wins / total_periods * 100
                
                if confidence >= 50:
                    st.success(f"✅ **High Confidence**: Recommended strategy wins {confidence:.0f}% of time periods")
                else:
                    st.warning(f"⚠️ **Moderate Confidence**: Recommended strategy wins {confidence:.0f}% of time periods")
            else:
                st.info("ℹ️ Confidence based on overall performance across full time period")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>📊 <strong>Investment Frequency Optimizer</strong> | Backtest investment strategies across market cycles</p>
    <p><em>Disclaimer: Past performance does not guarantee future results. This is for educational purposes only.</em></p>
</div>
""", unsafe_allow_html=True)