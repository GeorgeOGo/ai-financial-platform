import os
import streamlit as st
import plotly.express as px
import pandas as pd
from financial_platform import FinancialPlatform

st.set_page_config(page_title="AI Financial Platform", layout="wide")
st.title("📊 AI Financial Analysis & Trading Platform")

# Sidebar
st.sidebar.header("⚙️ Settings")

# --- Data Source Option ---
data_source = st.sidebar.radio("Select Data Source", ["Stock Symbol", "Upload CSV"])

csv_path, symbol = None, "AAPL"
if data_source == "Stock Symbol":
    # Dropdown with famous tickers
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "Other"]
    choice = st.sidebar.selectbox("Choose Stock Symbol", tickers)

    if choice == "Other":
        symbol = st.sidebar.text_input("Enter Stock Symbol", "GOOG").upper()
    else:
        symbol = choice
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        csv_path = f"data/{uploaded_file.name}"
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

# --- Date Range Control ---
st.sidebar.subheader("📅 Date Range")
start_date = st.sidebar.date_input("From:", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("To:", pd.to_datetime("2024-01-01"))

# --- Model Choice ---
model_choice = st.sidebar.selectbox("Choose Model", ["RandomForest", "LSTM"])
future_days = st.sidebar.slider("Days to Forecast", 5, 60, 30)

# Load Data
platform = FinancialPlatform(
    symbol=symbol,
    start=str(start_date),
    end=str(end_date),
    csv_path=csv_path
)

st.subheader("📈 Historical Prices")
st.line_chart(platform.data["Close"])

# Train & Forecast
if st.button("🚀 Train Model & Forecast"):
    with st.spinner("Training model..."):
        if model_choice == "RandomForest":
            rmse = platform.train_randomforest()
        else:
            rmse = platform.train_lstm()
    st.success(f"✅ Model trained. RMSE = {rmse:.4f}")

    # Forecast
    forecast = platform.forecast_future(future_days)
    st.subheader("🔮 Forecast")
    fig = px.line(title=f"{symbol} {model_choice} Forecast")
    fig.add_scatter(x=platform.data.index, y=platform.data["Close"], name="Historical")
    fig.add_scatter(x=forecast.index, y=forecast, name="Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # Backtesting
    st.subheader("📊 Backtesting Results")
    result, metrics, returns = platform.backtest_strategy()

    fig2 = px.line(result, x=result.index, y=["Portfolio", "Buy_Hold"], title="Backtest: Portfolio vs Buy & Hold")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📌 Performance Metrics")
    st.json(metrics)

    # Histogram Risk
    st.subheader("📉 Risk Distribution")
    var = metrics["VaR 95%"] / 100
    cvar = metrics["CVaR 95%"] / 100

    returns_df = returns.to_frame(name="Returns")
    returns_df["Category"] = ["Below VaR" if r <= var else "Above VaR" for r in returns_df["Returns"]]

    hist_fig = px.histogram(
        returns_df,
        x="Returns",
        color="Category",
        nbins=50,
        title="Distribution of Portfolio Returns",
        color_discrete_map={"Below VaR": "red", "Above VaR": "blue"},
        opacity=0.7
    )
    hist_fig.add_vline(x=var, line_dash="dash", line_color="red",
                       annotation_text="VaR 95%", annotation_position="top right")
    hist_fig.add_vline(x=cvar, line_dash="dot", line_color="orange",
                       annotation_text="CVaR 95%", annotation_position="top right")

    st.plotly_chart(hist_fig, use_container_width=True)
