# AntiQuant Analyzer Streamlit App
# This is a complete Streamlit app for deploying the AntiQuant strategy.
# It integrates data fetching, heatmap, risk scoring, and backtesting.
# Run with: streamlit run this_file.py
# Requirements: pip install streamlit pandas numpy influxdb-client tushare scikit-learn matplotlib

import streamlit as st
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from sklearn.ensemble import GradientBoostingClassifier  # Simplified ML model
import tushare as ts
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# Configuration
INFLUXDB_URL = "http://localhost:8086"  # Replace with your InfluxDB URL
INFLUXDB_TOKEN = "your_influxdb_token"
INFLUXDB_ORG = "your_org"
INFLUXDB_BUCKET = "a_share_data"
TS_TOKEN = "your_tushare_token"  # Tushare Pro token
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# InfluxDB Client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

# Helper Functions

def fetch_and_store_data():
    """Fetch and store small cap data (demo: top 10 small caps)"""
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,market,list_date,market_cap')
    small_caps = df[(df['market'].isin(['中小板', '创业板'])) & (df['market_cap'] < 50e9)]  # Small caps <50B
    small_caps = small_caps.head(10)  # Limit for demo
    for _, row in small_caps.iterrows():
        code = row['ts_code']
        # Simulate Level-2: Get daily data as proxy
        quote = pro.daily(ts_code=code, start_date=(datetime.now() - timedelta(days=1)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
        if not quote.empty:
            point = Point("level2_data") \
                .tag("stock", code) \
                .field("price", quote['close'].iloc[-1]) \
                .field("volume", quote['vol'].iloc[-1]) \
                .field("bid_ask_ratio", np.random.uniform(0.5, 2.0))  # Simulate
            write_api.write(bucket=INFLUXDB_BUCKET, record=point, write_precision=WritePrecision.NS)
    return small_caps['ts_code'].tolist()

# Crowding Degree (Simplified HHI)
def calculate_crowding(holdings: pd.DataFrame) -> float:
    # Placeholder: In real, fetch fund holdings
    return np.random.uniform(0.1, 0.5)  # Simulate

# Detect Disintegration
def detect_disintegration(stock_code: str, time_window: timedelta = timedelta(minutes=5)) -> bool:
    start_time = (datetime.utcnow() - time_window).strftime('%Y-%m-%dT%H:%M:%SZ')
    flux_query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: {start_time})
      |> filter(fn: (r) => r["_measurement"] == "level2_data" and r["stock"] == "{stock_code}")
      |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
    '''
    tables = query_api.query(flux_query)
    if not tables:
        return False
    records = [record.values for table in tables for record in table.records]
    if not records:
        return False
    df = pd.DataFrame(records)
    if 'volume' not in df.columns or 'price' not in df.columns:
        return False
    volume_spike = df['volume'].max() > 3 * df['volume'].mean()
    price_drop = (df['price'].max() - df['price'].min()) / df['price'].max() > 0.05
    return volume_spike and price_drop

# Abnormal Orders (Placeholder ML)
model = GradientBoostingClassifier()  # Assume pre-trained; in real, train on data
def detect_abnormal_orders(stock_code: str) -> bool:
    # Simulate fetch
    return np.random.choice([True, False])

# Risk Score
def calculate_risk_score(stock_code: str) -> float:
    crowding = calculate_crowding(pd.DataFrame())
    disintegration_prob = 1 if detect_disintegration(stock_code) else 0
    abnormal = 1 if detect_abnormal_orders(stock_code) else 0
    score = 0.4 * crowding + 0.3 * disintegration_prob + 0.3 * abnormal
    return score * 100

# Backtesting (Simplified)
def backtest(stock_codes, start_date, end_date, crowding_threshold=0.25):
    results = {}
    for code in stock_codes:
        df = pro.daily(ts_code=code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
        if df.empty:
            continue
        # Simulate signals
        df['crowding'] = np.random.uniform(0.1, 0.5, len(df))
        df['signal'] = np.where(df['crowding'] > crowding_threshold, 1, 0)  # Buy on high crowding
        df['returns'] = df['close'].pct_change() * df['signal'].shift()
        sharpe = np.mean(df['returns']) / np.std(df['returns']) * np.sqrt(252) if np.std(df['returns']) != 0 else 0
        max_drawdown = (df['close'].cummax() - df['close']).max() / df['close'].cummax().max() if not df.empty else 0
        results[code] = {'sharpe': sharpe, 'max_drawdown': max_drawdown}
    return results

# Streamlit App
st.set_page_config(page_title="AntiQuant Analyzer", layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("Controls")
refresh_button = st.sidebar.button("Refresh Data")
if refresh_button:
    with st.spinner("Fetching and storing data..."):
        stock_codes = fetch_and_store_data()
        st.session_state['stock_codes'] = stock_codes
        st.success("Data refreshed!")

# Load stock codes
stock_codes = st.session_state.get('stock_codes', fetch_and_store_data())

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Heatmap", "Risk Scoring", "Strategy Backtest"])

with tab1:
    st.header("Heatmap of Small Cap Crowding")
    # Fetch data for heatmap
    heatmap_data = []
    for code in stock_codes:
        risk = calculate_risk_score(code)
        crowding = calculate_crowding(pd.DataFrame())
        heatmap_data.append({'stock': code, 'crowding': crowding, 'risk': risk})
    df_heatmap = pd.DataFrame(heatmap_data)
    
    # Simple Heatmap with Matplotlib
    fig, ax = plt.subplots()
    cmap = cm.get_cmap('RdYlGn_r')  # Red high risk
    norm = plt.Normalize(df_heatmap['risk'].min(), df_heatmap['risk'].max())
    colors = cmap(norm(df_heatmap['risk']))
    ax.barh(df_heatmap['stock'], df_heatmap['risk'], color=colors)
    ax.set_xlabel('Risk Score')
    st.pyplot(fig)

with tab2:
    st.header("Risk Scoring")
    selected_stock = st.selectbox("Select Stock", stock_codes)
    if st.button("Calculate Risk"):
        with st.spinner("Calculating..."):
            score = calculate_risk_score(selected_stock)
            signal = "BUY (Reverse)" if score > 80 else "HOLD"
            st.metric("Risk Score", f"{score:.2f}")
            st.write(f"Signal: {signal}")
            if detect_disintegration(selected_stock):
                st.warning("Disintegration detected!")
            if detect_abnormal_orders(selected_stock):
                st.warning("Abnormal orders detected!")

with tab3:
    st.header("Strategy Backtest")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", value=datetime.now())
    with col2:
        crowding_threshold = st.number_input("Crowding Threshold", min_value=0.1, max_value=0.5, value=0.25)
    selected_stocks = st.multiselect("Select Stocks", stock_codes, default=stock_codes[:3])
    
    if st.button("Run Backtest"):
        with st.spinner("Backtesting..."):
            results = backtest(selected_stocks, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), crowding_threshold)
            st.table(pd.DataFrame(results).T)
            # Plot cumulative returns (simplified)
            fig, ax = plt.subplots()
            for code in selected_stocks:
                df = pro.daily(ts_code=code, start_date=start_date.strftime('%Y%m%d'), end_date=end_date.strftime('%Y%m%d'))
                ax.plot(df['trade_date'], df['close'].cumsum(), label=code)  # Placeholder cumsum
            ax.legend()
            st.pyplot(fig)

# Real-time simulation (refresh every 60s)
if 'last_refresh' not in st.session_state or (datetime.now() - st.session_state['last_refresh']).seconds > 60:
    st.session_state['last_refresh'] = datetime.now()
    st.experimental_rerun()

# Note: For production, secure tokens, handle errors, and train ML model properly.
