# Build a Streamlit App

Guide users through building a Streamlit data application following production-ready patterns.

## Description

This skill helps you build Streamlit apps from scratch using best practices learned from building the investment-optimizer app. It will ask you questions about your app concept, then generate appropriate code with proper structure, caching, styling, and deployment configuration.

**Live Example:** https://optimal-investing-strategy.streamlit.app/

---

## Instructions for Claude

When the user invokes this skill, follow this workflow:

### Phase 1: Discovery

Ask the user these questions using AskUserQuestion (ask 2-3 at a time max):

1. **App Concept**: "What problem will your app solve? Describe it in 1-2 sentences."

2. **Data Source**: Ask which data source they'll use:
   - External API (like yfinance, weather API, etc.)
   - File upload (CSV, Excel, JSON)
   - Database connection
   - Static/hardcoded data
   - User input only

3. **Key Features**: Ask which features they need (multi-select):
   - Interactive charts/visualizations
   - Data tables
   - File export (CSV, PDF)
   - Calculations/analysis
   - Forms/user input
   - Multiple pages

4. **Styling**: Ask their preference:
   - Default Streamlit theme
   - Custom pastel/modern styling (like investment-optimizer)
   - Dark theme
   - Minimal/clean

### Phase 2: Architecture Recommendation

Based on answers, recommend:

**For simple, focused apps (1-2 features):**
```
project/
├── app.py              # Everything in one file
├── requirements.txt
└── README.md
```

**For medium complexity (3+ features):**
```
project/
├── app.py              # Main entry point
├── utils.py            # Helper functions
├── requirements.txt
└── README.md
```

**For complex apps (multiple pages):**
```
project/
├── app.py              # Home page
├── pages/
│   ├── 1_Feature_One.py
│   └── 2_Feature_Two.py
├── utils/
│   ├── data.py
│   └── charts.py
├── requirements.txt
└── README.md
```

### Phase 3: Generate Code

Generate the app using these patterns:

---

## Code Templates

### Base App Template

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Your App Title",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM STYLING (Optional - for modern look)
# ============================================
st.markdown("""
<style>
    /* Pastel gradient button */
    .stButton > button {
        background: linear-gradient(45deg, #ffeaa7, #fab1a0) !important;
        color: #2d3748 !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 15px rgba(255, 234, 167, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, #fdcb6e, #e17055) !important;
        transform: translateY(-1px) !important;
    }

    /* Result highlight box */
    .result-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3748;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(168, 237, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA FUNCTIONS
# ============================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(source):
    """
    Load data with caching.
    TTL (time-to-live) prevents stale data for external APIs.
    """
    try:
        # Replace with your data loading logic
        # Examples:
        # - API: response = requests.get(url); return response.json()
        # - File: return pd.read_csv(source)
        # - yfinance: return yf.Ticker(symbol).history(period="max")

        data = pd.DataFrame()  # Your data here
        return data, None  # Return (data, error)
    except Exception as e:
        return None, str(e)

def process_data(data):
    """
    Process/analyze the loaded data.
    Keep calculation logic separate from UI.
    """
    results = {
        'metric_1': 0,
        'metric_2': 0,
        'processed_df': data
    }
    return results

# ============================================
# HEADER
# ============================================
st.markdown("""
<div style="
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    text-align: center;
    font-weight: 600;
    margin-bottom: 1rem;
">
📊 Your App Title
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem; color: #666;'>
    Brief description of what your app does
</div>
""", unsafe_allow_html=True)

# ============================================
# INPUT SECTION
# ============================================
col1, col2, col3 = st.columns(3)

with col1:
    input_1 = st.text_input(
        "📝 Input Label",
        value="default",
        help="Help text for this input"
    )

with col2:
    input_2 = st.number_input(
        "🔢 Number Input",
        min_value=1,
        max_value=100,
        value=10,
        help="Help text"
    )

with col3:
    input_3 = st.selectbox(
        "📋 Select Option",
        options=["Option A", "Option B", "Option C"],
        help="Help text"
    )

# ============================================
# ACTION BUTTON
# ============================================
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):

    # Load data with spinner
    with st.spinner("Loading data..."):
        data, error = load_data(input_1)

    # Error handling
    if error:
        st.error(f"❌ {error}")
        st.stop()

    if data is None or len(data) == 0:
        st.error("❌ No data found. Please check your input.")
        st.stop()

    st.success(f"✅ Loaded {len(data):,} records")

    # Process data
    with st.spinner("Analyzing..."):
        results = process_data(data)

    # ========================================
    # RESULTS DISPLAY
    # ========================================

    # Highlight box for main result
    st.markdown(f"""
    <div class="result-box">
        🎯 <strong>Main Result:</strong> {results['metric_1']}
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Metric 1",
            value=f"{results['metric_1']:,.0f}",
            delta="+10%"
        )

    with col2:
        st.metric(
            label="📈 Metric 2",
            value=f"{results['metric_2']:,.2f}",
            delta="-5%"
        )

    # Charts
    st.subheader("📊 Visualization")

    # Bar chart example
    fig = px.bar(
        results['processed_df'],
        x='category',  # Replace with your column
        y='value',     # Replace with your column
        title="Bar Chart Title",
        color='value',
        color_continuous_scale='RdYlGn'  # Red-Yellow-Green scale
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Line chart example
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=results['processed_df']['date'],  # Replace
        y=results['processed_df']['value'],  # Replace
        mode='lines',
        name='Trend',
        line=dict(color='#667eea', width=2)
    ))
    fig2.update_layout(
        title="Line Chart Title",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Data table in expander
    with st.expander("📋 View Raw Data", expanded=False):
        st.dataframe(results['processed_df'], use_container_width=True)

    # Download button
    csv = results['processed_df'].to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"results_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | <a href="https://github.com/your-repo">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
```

---

### Requirements.txt Template

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
# Add based on data source:
# yfinance>=0.2.18      # For stock data
# requests>=2.28.0      # For API calls
# openpyxl>=3.0.0       # For Excel files
# sqlalchemy>=2.0.0     # For database
```

---

### Caching Patterns

**For external API data (changes frequently):**
```python
@st.cache_data(ttl=300)  # Refresh every 5 minutes
def fetch_api_data(endpoint):
    ...
```

**For file uploads (user-specific):**
```python
@st.cache_data
def process_uploaded_file(file_content, filename):
    # Use filename as part of cache key
    ...
```

**For expensive calculations (doesn't change):**
```python
@st.cache_data
def heavy_computation(data_hash):
    # Pass a hash of data instead of data itself
    ...
```

**For database connections:**
```python
@st.cache_resource  # Use cache_resource for connections
def get_db_connection():
    return create_engine(connection_string)
```

---

### UI Layout Patterns

**Two-column layout:**
```python
col1, col2 = st.columns([2, 1])  # 2:1 ratio
with col1:
    # Main content (charts)
with col2:
    # Sidebar content (metrics)
```

**Tabs for different views:**
```python
tab1, tab2, tab3 = st.tabs(["Overview", "Details", "Settings"])
with tab1:
    st.write("Overview content")
```

**Sidebar for inputs:**
```python
with st.sidebar:
    st.header("Settings")
    option = st.selectbox("Choose", ["A", "B"])
```

**Progressive disclosure:**
```python
with st.expander("Advanced Options", expanded=False):
    advanced_setting = st.slider("Threshold", 0, 100, 50)
```

---

### Visualization Color Scales

**Financial data (good/bad):**
```python
color_continuous_scale='RdYlGn'  # Red=bad, Yellow=neutral, Green=good
```

**Sequential data:**
```python
color_continuous_scale='Blues'   # Light to dark blue
color_continuous_scale='Viridis' # Perceptually uniform
```

**Categorical data:**
```python
color_discrete_sequence=px.colors.qualitative.Set2
```

---

## Deployment Instructions

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial Streamlit app"
   git remote add origin https://github.com/username/repo.git
   git push -u origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app" and connect your GitHub repo**

4. **Configure:**
   - Repository: your-username/your-repo
   - Branch: main
   - Main file path: app.py

5. **Click Deploy**

Your app will be live at: `https://your-app-name.streamlit.app`

---

## Example Reference

This skill is based on the **Investment Optimizer** app:
- **Live app:** https://optimal-investing-strategy.streamlit.app/
- **GitHub:** Check the `app.py` in this repository for a complete working example

The investment-optimizer demonstrates:
- External API data fetching (yfinance)
- Complex calculations with proper caching
- Multiple visualization types (bar, line, heatmap)
- Custom pastel styling
- Progressive disclosure with expanders
- Quality assurance validation
- Professional metrics display

---

## Quick Start

If you want to start immediately, tell me:
1. What your app will do (one sentence)
2. Where your data comes from

I'll generate a complete, working `app.py` tailored to your needs!
