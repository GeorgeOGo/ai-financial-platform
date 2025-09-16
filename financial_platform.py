import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class FinancialPlatform:
    def __init__(self, symbol="AAPL", start="2015-01-01", end="2024-01-01", csv_path=None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.csv_path = csv_path
        self.model = None
        self.X_test, self.y_test, self.y_pred = None, None, None
        self.data = self.load_data()

    def load_data(self):
        if self.csv_path:
            df = pd.read_csv(self.csv_path, parse_dates=True, index_col=0)
        else:
            df = yf.download(self.symbol, start=self.start, end=self.end, progress=False)

        df.dropna(inplace=True)
        df["Returns"] = df["Close"].pct_change()
        df.dropna(inplace=True)
        return df

    def prepare_features(self, lag_days=5):
        df = self.data.copy()
        for i in range(1, lag_days + 1):
            df[f"Close_lag_{i}"] = df["Close"].shift(i)
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["MA_30"] = df["Close"].rolling(30).mean()
        df["Volatility"] = df["Close"].rolling(30).std()
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)
        features = [c for c in df.columns if "lag" in c or "MA" in c or "Volatility" in c]
        return df[features], df["Target"]

    # ----------------- Random Forest -----------------
    def train_randomforest(self, n_estimators=100, test_size=0.2):
        X, y = self.prepare_features()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        return rmse

    # ----------------- LSTM -----------------
    def train_lstm(self, epochs=10, batch_size=16):
        df = self.data.copy()
        values = df["Close"].values.reshape(-1, 1)
        X, y = [], []
        window = 10
        for i in range(len(values) - window):
            X.append(values[i:i+window])
            y.append(values[i+window])
        X, y = np.array(X), np.array(y)

        split = int(len(X) * 0.8)
        X_train, self.X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential()
        model.add(LSTM(50, activation="relu", input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        self.model = model
        self.y_test = y_test.reshape(-1)                   # ✅ Fix
        self.y_pred = model.predict(self.X_test).flatten() # ✅ Fix
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        return rmse

    # ----------------- Forecast -----------------
    def forecast_future(self, days=30):
        if self.model is None:
            raise ValueError("Model not trained")
        if isinstance(self.model, RandomForestRegressor):
            X, _ = self.prepare_features()
            last_features = X.iloc[-1:].values
            preds, features = [], last_features
            for _ in range(days):
                pred = self.model.predict(features)[0]
                preds.append(pred)
                features = np.roll(features, -1)
                features[-1] = pred
            future_dates = pd.date_range(self.data.index[-1], periods=days+1, freq="B")[1:]
            return pd.Series(preds, index=future_dates, name="Forecast")
        else:
            last_seq = self.data["Close"].values[-10:].reshape(1, 10, 1)
            preds = []
            for _ in range(days):
                pred = self.model.predict(last_seq)[0][0]
                preds.append(pred)
                last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
            future_dates = pd.date_range(self.data.index[-1], periods=days+1, freq="B")[1:]
            return pd.Series(preds, index=future_dates, name="Forecast")

    # ----------------- Backtesting -----------------
    def backtest_strategy(self, initial_capital=10000, commission=0.001, slippage=0.001, confidence_level=0.95):
        if self.y_pred is None or self.y_test is None:
            raise ValueError("Train model first")

        df = pd.DataFrame({"Actual": self.y_test, "Predicted": self.y_pred})
        cash, holdings = initial_capital, 0
        portfolio = []

        for date, row in df.iterrows():
            price, pred = row["Actual"], row["Predicted"]
            buy_price = price * (1 + slippage)
            sell_price = price * (1 - slippage)

            if pred > price and cash > 0:
                holdings = (cash * (1 - commission)) / buy_price
                cash = 0
            elif pred < price and holdings > 0:
                cash = holdings * sell_price * (1 - commission)
                holdings = 0

            portfolio.append({"Date": date, "Portfolio": cash + holdings * price})

        result = pd.DataFrame(portfolio).set_index("Date")
        result["Buy_Hold"] = initial_capital * (df["Actual"] / df["Actual"].iloc[0])

        # ---- 📊 Metrics ----
        result["Returns"] = result["Portfolio"].pct_change().fillna(0)
        total_return = (result["Portfolio"].iloc[-1] / initial_capital - 1) * 100
        sharpe_ratio = (result["Returns"].mean() / result["Returns"].std()) * np.sqrt(252) if result["Returns"].std() != 0 else 0
        cumulative_max = result["Portfolio"].cummax()
        drawdown = (result["Portfolio"] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100

        # 📌 Value at Risk (VaR) & Conditional Value at Risk (CVaR)
        var = result["Returns"].quantile(1 - confidence_level) * 100
        cvar = result["Returns"][result["Returns"] <= result["Returns"].quantile(1 - confidence_level)].mean() * 100

        metrics = {
            "Total Return (%)": round(total_return, 2),
            "Sharpe Ratio": round(sharpe_ratio, 2),
            "Max Drawdown (%)": round(max_drawdown, 2),
            f"VaR {int(confidence_level*100)}%": round(var, 2),
            f"CVaR {int(confidence_level*100)}%": round(cvar, 2),
        }

        return result, metrics, result["Returns"]
