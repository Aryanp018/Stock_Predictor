import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime
import ta
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# StockPredictor class (from your notebook)
class StockPredictor:
    def __init__(self, stock_symbol, api_key, look_back=60, prediction_days=1, max_retries=3):
        self.stock_symbol = stock_symbol.upper()
        self.api_key = api_key
        self.look_back = look_back
        self.prediction_days = prediction_days
        self.max_retries = max_retries
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.models = {}
        
    def get_stock_data(self):
        retries = 0
        while retries < self.max_retries:
            try:
                ts = TimeSeries(key=self.api_key, output_format='pandas')
                data, meta_data = ts.get_daily(symbol=self.stock_symbol, outputsize='full')
                data = data.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
                data = data.sort_index()
                data = data.loc['2015-01-01':datetime.today().strftime('%Y-%m-%d')]
                if data.empty:
                    raise ValueError(f"No data found for {self.stock_symbol}")
                self.data = data
                logging.info(f"Successfully downloaded data for {self.stock_symbol}")
                return True
            except Exception as e:
                retries += 1
                logging.warning(f"Attempt {retries} failed: {str(e)}")
                if retries == self.max_retries:
                    logging.error(f"Error fetching data for {self.stock_symbol} after {self.max_retries} attempts")
                    return False
                time.sleep(2 ** retries)
                
    def add_technical_indicators(self):
        if self.data is None:
            return None
        df = self.data.copy()
        try:
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MA20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['MA50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_lower'] = bb.bollinger_lband()
            df = df.dropna()
            self.data = df
            return df
        except Exception as e:
            logging.error(f"Error adding technical indicators: {str(e)}")
            return None

    def prepare_lstm_data(self):
        prices = self.data['Close'].values
        scaled_data = self.scaler.fit_transform(prices.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.look_back, len(scaled_data) - self.prediction_days):
            X.append(scaled_data[i-self.look_back:i])
            y.append(scaled_data[i:i+self.prediction_days]) 
        
        X, y = np.array(X), np.array(y)
        train_size = int(len(X) * 0.8)
        return (X[:train_size], y[:train_size], 
                X[train_size:], y[train_size:], scaled_data)

    def create_lstm_model(self):
        from tensorflow.keras.optimizers import Adam
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.look_back, 1)),
            Dropout(0.3),
            LSTM(100),
            Dropout(0.3),
            Dense(50),
            Dense(self.prediction_days)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_models(self):
        if self.data is None:
            return None, None, None, None
            
        import tensorflow as tf
        import random
        np.random.seed(42)
        tf.random.set_seed(42)
        random.seed(42)

        features = ['Close', 'RSI', 'MA20', 'MA50', 'BB_upper', 'BB_lower']
        X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test, scaled_data = self.prepare_lstm_data()
        
        self.models['lstm'] = self.create_lstm_model()
        self.models['lstm'].fit(X_lstm_train, y_lstm_train, epochs=50, 
                              batch_size=32, validation_split=0.1, verbose=0)
        
        X = self.data[features].values
        y = self.data['Close'].shift(-self.prediction_days).values
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        
        total_samples = len(scaled_data) - self.look_back - self.prediction_days + 1
        train_size = int(total_samples * 0.8)
        
        rf_gb_scaler = MinMaxScaler()
        X_scaled = rf_gb_scaler.fit_transform(X)
        
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:train_size + len(X_lstm_test)]
        y_train, y_test = y[:train_size], y[train_size:train_size + len(X_lstm_test)]
        
        self.models['rf'] = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.models['rf'].fit(X_train, y_train)
        
        self.models['gb'] = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        self.models['gb'].fit(X_train, y_train)
        
        return X_lstm_test, y_lstm_test, X_test, y_test

    def predict(self, X_lstm_test, X_test):
        lstm_pred = self.models['lstm'].predict(X_lstm_test, verbose=0)
        rf_pred = self.models['rf'].predict(X_test)
        gb_pred = self.models['gb'].predict(X_test)
        
        lstm_pred_inv = self.scaler.inverse_transform(lstm_pred[:, -1].reshape(-1, 1))
        lstm_pred_inv = lstm_pred_inv.flatten()
        
        lstm_rmse, rf_rmse, gb_rmse = 11.73, 72.60, 74.10
        total_inverse_rmse = (1/lstm_rmse) + (1/rf_rmse) + (1/gb_rmse)
        lstm_weight = (1/lstm_rmse) / total_inverse_rmse
        rf_weight = (1/rf_rmse) / total_inverse_rmse
        gb_weight = (1/gb_rmse) / total_inverse_rmse
        
        ensemble_pred = (lstm_weight * lstm_pred_inv + rf_weight * rf_pred + gb_weight * gb_pred)
        return ensemble_pred, lstm_pred_inv, rf_pred, gb_pred

# Streamlit UI
st.title("📈 Stock Price Predictor")
st.write("Enter a stock symbol and Alpha Vantage API key to get a 1-day-ahead price prediction.")

# Sidebar instructions
st.sidebar.markdown("""
### How to Use
1. Get a free API key from [Alpha Vantage](https://www.alphavantage.co).
2. Enter a stock symbol (e.g., NVDA, AAPL) and your API key.
3. Click "Predict" to see results in ~2–5 minutes.
""")

# User inputs
symbol = st.text_input("Stock Symbol (e.g., NVDA, AAPL):", "NVDA")
api_key = st.text_input("Alpha Vantage API Key:", type="password")

if st.button("Predict"):
    if not symbol or not api_key:
        st.error("Please provide both a stock symbol and API key.")
    else:
        with st.spinner("Fetching data and running models..."):
            try:
                # Progress bar
                progress = st.progress(0)
                predictor = StockPredictor(symbol, api_key, prediction_days=1)
                
                # Step 1: Fetch data
                progress.progress(0.25)
                if not predictor.get_stock_data():
                    st.error(f"Failed to fetch data for {symbol}. Check symbol or API key.")
                    st.stop()
                
                # Step 2: Add indicators
                progress.progress(0.5)
                df = predictor.add_technical_indicators()
                if df is None:
                    st.error("Failed to add technical indicators.")
                    st.stop()
                
                # Step 3: Train models
                progress.progress(0.75)
                X_lstm_test, y_lstm_test, X_test, y_test = predictor.train_models()
                if X_lstm_test is None:
                    st.error("Failed to train models.")
                    st.stop()
                
                # Step 4: Predict
                progress.progress(1.0)
                ensemble_pred, lstm_pred, rf_pred, gb_pred = predictor.predict(X_lstm_test, X_test)
                y_test_inv = predictor.scaler.inverse_transform(y_lstm_test[:, -1].reshape(-1, 1)).flatten()
                
                # Calculate RMSE
                rmse_results = {}
                for name, pred in [('Ensemble', ensemble_pred), ('LSTM', lstm_pred), 
                                 ('RF', rf_pred), ('GB', gb_pred)]:
                    rmse = np.sqrt(mean_squared_error(y_test_inv, pred))
                    rmse_results[name] = rmse
                
                # Display results
                st.success("Prediction completed!")
                st.subheader("1-Day-Ahead Closing Price Predictions")
                st.write(f"**Ensemble**: ${ensemble_pred[-1]:.2f}")
                st.write(f"**LSTM**: ${lstm_pred[-1]:.2f}")
                st.write(f"**Random Forest**: ${rf_pred[-1]:.2f}")
                st.write(f"**Gradient Boosting**: ${gb_pred[-1]:.2f}")
                
                st.subheader("Model Performance (RMSE)")
                for name, rmse in rmse_results.items():
                    st.write(f"**{name}**: {rmse:.2f}")
                
                # Plot
                st.subheader("Prediction vs. Actual Prices")
                fig = plt.figure(figsize=(10, 5))
                plt.plot(df.index[-len(y_test):], y_test_inv, label='Actual')
                plt.plot(df.index[-len(y_test):], ensemble_pred, label='Ensemble')
                plt.plot(df.index[-len(y_test):], lstm_pred, label='LSTM')
                plt.plot(df.index[-len(y_test):], rf_pred, label='Random Forest')
                plt.plot(df.index[-len(y_test):], gb_pred, label='Gradient Boosting')
                plt.title(f'{symbol} Stock Price Prediction (1-Day Ahead)')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error: {str(e)}. Check your API key, symbol, or try again later.")

st.markdown("""
**Note**: This app uses the Alpha Vantage API (free tier: 5 requests/min). Predictions may take a few minutes due to model training.
""")
