import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config with light theme
st.set_page_config(
    page_title="Optimal Investment Frequency Finder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .winner-box {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .regime-section {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# No need for hardcoded market regimes - rolling windows capture all market conditions naturally!

@st.cache_data(ttl=300)  # Cache for 5 minutes
def download_stock_data(symbol, max_years=None):
    """Download and prepare stock data from the earliest available date"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get the maximum available history
        # For most stocks, this goes back 20+ years
        if max_years:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=max_years*365)
            data = ticker.history(start=start_date, end=end_date, actions=True, period="max")
        else:
            # Get ALL available data
            data = ticker.history(actions=True, period="max")
        
        if len(data) == 0:
            return None, f"No data found for {symbol}"
        
        # Clean and prepare data
        data = data.dropna()
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        data['Weekday'] = data.index.day_name()
        
        # Get basic info about the stock
        info = ticker.info
        stock_name = info.get('longName', symbol)
        
        # Calculate how many years of data we have
        data_years = (data.index.max() - data.index.min()).days / 365.25
        
        return data, None, stock_name, data_years
    except Exception as e:
        return None, f"Error downloading {symbol}: {str(e)}", symbol, 0

def calculate_strategy_performance(data, frequency, specific_day=None, monthly_budget=1000):
    """Calculate performance for a specific investment strategy"""
    
    if frequency == 'daily':
        investment_dates = data.index
        investment_amount = monthly_budget / 21
    elif frequency == 'weekly':
        if specific_day:
            weekday_data = data[data['Weekday'] == specific_day]
            investment_dates = weekday_data.index
        else:
            investment_dates = data.resample('W').first().index
        investment_amount = monthly_budget / 4.33
    elif frequency == 'monthly':
        investment_dates = data.resample('M').first().index
        investment_amount = monthly_budget
    
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
        'portfolio_history': portfolio_df
    }

def rolling_window_analysis(data, strategies, monthly_budget, window_years=None):
    """Test strategies across rolling time windows - each window captures different market conditions"""
    
    # Auto-determine appropriate window sizes based on data length
    total_years = (data.index.max() - data.index.min()).days / 365.25
    
    if window_years is None:
        if total_years >= 20:
            window_years = [3, 5, 7, 10]
        elif total_years >= 15:
            window_years = [3, 5, 7]
        elif total_years >= 10:
            window_years = [3, 5]
        else:
            window_years = [3]  # Minimum viable window
    
    rolling_results = []
    
    for window in window_years:
        if window > total_years - 1:  # Skip if window is too large
            continue
            
        window_days = window * 365
        max_start = len(data) - window_days
        
        if max_start < 365:  # Need at least 1 year of data
            continue
        
        # Test every 6 months for comprehensive coverage
        step_size = max(126, len(data) // 30)  # ~6 months, but not too many windows
        start_points = range(0, max_start, step_size)
        
        for start_idx in start_points:
            end_idx = start_idx + window_days
            window_data = data.iloc[start_idx:end_idx]
            
            if len(window_data) < 500:  # Need sufficient data
                continue
            
            window_results = []
            for strategy_name, (freq, day) in strategies.items():
                result = calculate_strategy_performance(window_data, freq, day, monthly_budget)
                if result:
                    result['window_years'] = window
                    result['start_date'] = window_data.index.min()
                    result['end_date'] = window_data.index.max()
                    
                    # Add market condition context
                    avg_return = window_data['Close'].pct_change().mean() * 252 * 100  # Annualized market return
                    volatility = window_data['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
                    
                    result['market_return'] = avg_return
                    result['market_volatility'] = volatility
                    result['period_label'] = f"{window_data.index.min().strftime('%Y-%m')} to {window_data.index.max().strftime('%Y-%m')}"
                    
                    window_results.append(result)
            
            if window_results:
                # Find best strategy for this window
                best = max(window_results, key=lambda x: x['annualized_return'])
                for result in window_results:
                    result['is_winner'] = (result['strategy'] == best['strategy'])
                
                rolling_results.extend(window_results)
    
    return rolling_results

def regime_analysis(data, strategies, monthly_budget):
    """Analyze performance across different market conditions based on volatility and returns"""
    
    # Calculate rolling market metrics to identify natural regimes
    data_copy = data.copy()
    data_copy['Rolling_Return_252'] = data_copy['Close'].pct_change(252) * 100  # 1-year return
    data_copy['Rolling_Vol_63'] = data_copy['Close'].pct_change().rolling(63).std() * np.sqrt(252) * 100  # 3-month vol
    
    # Fill NaN values
    data_copy = data_copy.fillna(method='ffill').fillna(method='bfill')
    
    # Define market conditions based on actual data percentiles
    vol_75th = data_copy['Rolling_Vol_63'].quantile(0.75)
    vol_25th = data_copy['Rolling_Vol_63'].quantile(0.25)
    ret_75th = data_copy['Rolling_Return_252'].quantile(0.75)
    ret_25th = data_copy['Rolling_Return_252'].quantile(0.25)
    
    # Create natural market regimes
    market_conditions = {
        'High Volatility': data_copy['Rolling_Vol_63'] > vol_75th,
        'Low Volatility': data_copy['Rolling_Vol_63'] < vol_25th,
        'Bear Periods': data_copy['Rolling_Return_252'] < ret_25th,
        'Bull Periods': data_copy['Rolling_Return_252'] > ret_75th,
        'Crisis Periods': (data_copy['Rolling_Vol_63'] > vol_75th) & (data_copy['Rolling_Return_252'] < ret_25th),
        'Goldilocks': (data_copy['Rolling_Vol_63'] < vol_25th) & (data_copy['Rolling_Return_252'] > ret_75th)
    }
    
    regime_results = []
    
    for condition_name, condition_mask in market_conditions.items():
        if condition_mask.sum() < 200:  # Need sufficient data points
            continue
        
        condition_data = data_copy[condition_mask]
        
        if len(condition_data) < 200:
            continue
        
        condition_strategy_results = []
        for strategy_name, (freq, day) in strategies.items():
            result = calculate_strategy_performance(condition_data, freq, day, monthly_budget)
            if result:
                result['regime'] = condition_name
                result['regime_periods'] = len(condition_data)
                condition_strategy_results.append(result)
        
        if condition_strategy_results:
            # Find best strategy for this regime
            best = max(condition_strategy_results, key=lambda x: x['annualized_return'])
            for result in condition_strategy_results:
                result['is_winner'] = (result['strategy'] == best['strategy'])
            
            regime_results.extend(condition_strategy_results)
    
    return regime_results

# Main App
# Main App Header with soft pastel styling
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
        value="VTI", 
        help="Enter any stock ticker (VTI, SPY, QQQ, AAPL, etc.)"
    ).upper()

with col2:
    monthly_amount = st.number_input(
        "💰 Monthly Investment ($)",
        min_value=100,
        max_value=50000,
        value=1000,
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

# Run Analysis Button - simple and functional
if st.button("🚀 Find Optimal Strategy", type="primary", use_container_width=True):
    
    # Download data
    with st.spinner(f"📡 Downloading maximum available data for {ticker}..."):
        if use_max_data:
            result = download_stock_data(ticker)
        else:
            result = download_stock_data(ticker, analysis_years)
            
        if len(result) == 4:
            data, error, stock_name, data_years = result
        else:
            data, error = result[:2]
            stock_name = ticker
            data_years = 0
    
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    if len(data) < 1000:
        st.error("❌ Insufficient data for robust analysis. Try a different ticker or longer time period.")
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
        
        # Rolling window analysis - captures all market conditions naturally
        rolling_results = rolling_window_analysis(data, strategies, monthly_amount)
        
        # Market condition analysis - data-driven regimes
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
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Key Metrics")
            
            # Use Streamlit's built-in metric widgets for better display
            profit = best_overall['final_value'] - best_overall['total_invested']
            profit_pct = (profit / best_overall['total_invested']) * 100
            
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
            st.markdown(f"Testing strategy robustness across {len(set(r['window_years'] for r in rolling_results))} different window sizes")
            
            rolling_df = pd.DataFrame(rolling_results)
            
            # Show window details
            window_summary = rolling_df.groupby('window_years').size()
            st.write(f"**Windows tested**: {', '.join([f'{int(years)}yr ({count} periods)' for years, count in window_summary.items()])}")
            
            # Win rate calculation
            win_rates = rolling_df.groupby('strategy')['is_winner'].agg(['sum', 'count'])
            win_rates['win_rate'] = (win_rates['sum'] / win_rates['count'] * 100).round(1)
            win_rates = win_rates.sort_values('win_rate', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Win Rates Across Time Windows:**")
                for strategy, row in win_rates.iterrows():
                    strategy_name = strategy.replace('_', ' ').title()
                    st.write(f"• {strategy_name}: {row['win_rate']}% ({int(row['sum'])}/{int(row['count'])} wins)")
            
            with col2:
                # Average performance by window size
                avg_by_window = rolling_df.groupby(['window_years', 'strategy'])['annualized_return'].mean().reset_index()
                
                fig = px.line(
                    avg_by_window,
                    x='window_years',
                    y='annualized_return',
                    color='strategy',
                    title="Average Returns by Time Window",
                    labels={'window_years': 'Window Size (Years)', 'annualized_return': 'Avg Return (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Market condition results - data-driven regimes
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
            
            fig = px.imshow(
                regime_pivot,
                aspect='auto',
                title="Strategy Performance by Market Condition (%)",
                color_continuous_scale='RdYlGn',
                labels={'color': 'Annualized Return (%)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Condition winners
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
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, result in enumerate(overall_results[:5]):  # Show top 5 strategies
            if 'portfolio_history' in result:
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
        
        # Summary insights - focus on the BEST performer
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
                insights.append(f"**✅ High Confidence**: This strategy wins {most_consistent_rate:.0f}% of all time periods tested")
            else:
                insights.append(f"**⚖️ Alternative**: {most_consistent.replace('_', ' ').title()} is most consistent (wins {most_consistent_rate:.0f}% of periods)")
        
        # Performance vs market conditions
        if regime_results:
            regime_df = pd.DataFrame(regime_results)
            best_strategy_regimes = regime_df[regime_df['strategy'] == best_overall['strategy']]
            avg_regime_return = best_strategy_regimes['annualized_return'].mean()
            insights.append(f"**📊 Market Adaptability**: Averages {avg_regime_return:.1f}% across all market conditions")
        
        # Final recommendation
        if rolling_results:
            total_periods_tested = len(rolling_df) // len(strategies)
            best_strategy_wins = rolling_df[rolling_df['strategy'] == best_overall['strategy']]['is_winner'].sum()
            confidence_level = best_strategy_wins / total_periods_tested * 100
            
            if confidence_level >= 60:
                confidence = "HIGH CONFIDENCE ✅"
                color = "green"
            elif confidence_level >= 40:
                confidence = "MODERATE CONFIDENCE ⚖️"
                color = "orange"
            else:
                confidence = "LOW CONFIDENCE ⚠️"
                color = "red"
            
            insights.append(f"**🎯 Recommendation**: {confidence} - Deploy {best_overall['strategy'].replace('_', ' ').title()} strategy")
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Simple action item
        st.markdown("---")
        st.markdown(f"""
        ### 🚀 **Action Plan**
        **Start investing ${monthly_amount:,}/month using the {best_overall['strategy'].replace('_', ' ').title()} strategy:**
        
        {f"• **Daily**: Invest ${monthly_amount/21:.0f} every trading day" if best_overall['strategy'] == 'daily' else ""}
        {f"• **Monthly**: Invest ${monthly_amount:,} on the first trading day of each month" if best_overall['strategy'] == 'monthly' else ""}
        {f"• **Weekly**: Invest ${monthly_amount/4.33:.0f} every {best_overall['strategy'].split('_')[1]}" if 'weekly' in best_overall['strategy'] else ""}
        
        This strategy has been tested across **{total_periods_tested if 'total_periods_tested' in locals() else 'multiple'}** different market periods for maximum robustness.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>📊 <strong>Investment Frequency Optimizer</strong> | Backtest investment strategies across market cycles</p>
    <p><em>Disclaimer: Past performance does not guarantee future results. This is for educational purposes only.</em></p>
</div>
""", unsafe_allow_html=True)