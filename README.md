# AI-Powered Financial Analysis & Trading Platform  

This repository contains an **offline, AI-powered financial platform** for **stock market prediction, risk assessment, and trading strategy backtesting**.  
It was developed as part of a technical interview assessment for the AI Development Team.  

The platform integrates **machine learning (RandomForest)** and **deep learning (LSTM)** models, works fully **on-device without cloud APIs**, and provides **interactive dashboards** for visualization.

---

##  Features  

 **Data Input**  
- Fetch stock data from **Yahoo Finance** (e.g., AAPL, NVDA, TSLA).  
- Upload your own **CSV file** for offline analysis.  

 **Prediction Models**  
- **RandomForest Regressor** → Fast, interpretable, robust.  
- **LSTM Neural Network** → Captures time dependencies and trends.  

 **Forecasting**  
- Predict stock prices up to **60 days ahead**.  
- Visual comparison: *Historical vs Predicted vs Forecast*.  

 **Risk Assessment**  
- **Annualized Volatility**  
- **Sharpe Ratio** (risk-adjusted return)  
- **Value at Risk (VaR)**  
- **Conditional VaR (CVaR)**  
- **Max Drawdown**  

 **Backtesting**  
- Simulates a **trading strategy** (buy/sell based on predictions).  
- Compares against a **Buy & Hold benchmark**.  
- Supports **transaction costs, slippage, and liquidity constraints**.  

 **Visualization**  
- Interactive **Plotly charts**.  
- Risk **distribution histograms** with red (losses) vs blue (gains).  
- Portfolio performance curves.  

---------------------------------------------------------------------------------------
  
  ***Install dependencies*** :
  pip install -r requirements.txt
  
  ***Usage*** :
  streamlit run app.py

-----------------------------------------------------------------------------------

# Example Workflow

**Choose Data Source** :

- Select from famous tickers (AAPL, NVDA, MSFT, TSLA, AMZN).

-OR upload your own CSV.


**Set Date Range** :

-Example: From 2015-01-01 to 2024-01-01.


**Select Model** :

-RandomForest (fast, interpretable).

-LSTM (deep learning).

**Train & Forecast** :

-Model RMSE displayed.

-Forecast shown in chart with future predictions.


**Backtesting & Risk**

-Portfolio vs Buy & Hold chart.

-Risk metrics JSON.

-Risk histogram with VaR and CVaR highlighted.


---------------------------------------------------------------------------------------------------
** FAQs & Troubleshooting**

 Q: I uploaded a CSV but got KeyError: 'Returns'. ?
 
 A: Make sure your CSV has at least these columns: Date, Open, High, Low, Close, Volume. The platform will calculate Returns automatically.


 Q: The LSTM throws shape errors. ?
 
 A: Ensure enough data points (> 200) and correct columns. LSTM needs sequential data to train properly.


 Q: Forecast chart is empty. ?
 
 A: Train the model first, then run forecast. Check date range includes enough historical data.

 ------------------------------------------------------------------------------------------------------------
 <img width="1934" height="878" alt="image" src="https://github.com/user-attachments/assets/21f8a30d-14ba-4d53-8065-55a26853a6b8" />

<img width="1026" height="784" alt="image" src="https://github.com/user-attachments/assets/ca1a7369-f128-462a-bb26-0613a9bc9562" />

<img width="1460" height="702" alt="image" src="https://github.com/user-attachments/assets/fb1414ba-89d7-4795-a553-4d4d77c8248e" />

<img width="1462" height="554" alt="image" src="https://github.com/user-attachments/assets/ae6b0fe4-7599-426e-9057-4e96c0533035" />

