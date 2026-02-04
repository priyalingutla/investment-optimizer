# Investment Strategy Optimizer

Find the optimal investment frequency by backtesting across different market conditions and time periods.

## Live App
https://optimal-investing-strategy.streamlit.app/

## Features
- Comprehensive backtesting across rolling time windows
- Data-driven market condition analysis (bull/bear, high/low volatility, crisis periods)
- Support for any stock ticker
- Professional visualization and clear recommendations
- Tests 7 strategies: Daily, Monthly, and Weekly (Mon-Fri)

## Usage
1. Enter your stock ticker (VTI, SPY, AAPL, etc.)
2. Set your monthly investment amount
3. Click "Find Optimal Strategy"
4. Get your personalized recommendation!

## Build Your Own Streamlit App

This repo includes a Claude Code skill to help you build your own Streamlit apps!

### Using the Skill

If you have [Claude Code](https://claude.ai/claude-code) installed:

```bash
# Clone this repo
git clone https://github.com/your-username/investment-optimizer.git
cd investment-optimizer

# Run Claude Code and invoke the skill
claude
> /build-streamlit-app
```

The skill will guide you through:
1. Defining your app concept
2. Choosing data sources and features
3. Generating production-ready code
4. Deploying to Streamlit Cloud

### What You'll Learn
- Streamlit app architecture patterns
- Caching strategies for performance
- Custom CSS styling
- Plotly visualizations
- Error handling best practices
- Deployment workflow

## Tech Stack
- **Streamlit** - Web app framework
- **yfinance** - Stock data API
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computing

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Project Structure

```
investment-optimizer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .claude/
    └── commands/
        └── build-streamlit-app.md  # Claude Code skill
```

## License
MIT
