import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
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

# StockPredictor class (LSTM and RF only)
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
                if data.empty or len(data) < self.look_back + 50:  # Ensure enough data
                    raise ValueError(f"Insufficient data for {self.stock_symbol}: {len(data)} rows")
                self.data = data
                logging.info(f"Fetched {len(data)} rows for {self.stock_symbol}")
                return True
            except Exception as e:
                retries += 1
                logging.warning(f"Attempt {retries} failed: {str(e)}")
                if retries == self.max_retries:
                    logging.error(f"Failed to fetch data for {self.stock_symbol}")
                    return False
                time.sleep(2 ** retries)
                
    def add_technical_indicators(self):
        if self.data is None:
            return False
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
            if df.empty or len(df) < self.look_back + 10:
                logging.error(f"After indicators, insufficient data: {len(df)} rows")
                return False
            self.data = df
            logging.info(f"Indicators added, {len(df)} rows remain")
            return True
        except Exception as e:
            logging.error(f"Error adding indicators: {str(e)}")
            return False

    def prepare_lstm_data(self):
        if self.data is None or len(self.data) < self.look_back + 10:
            logging.error("Not enough data for LSTM")
            return None, None, None, None, None
        prices = self.data['Close'].values
        if np.any(np.isnan(prices)):
            logging.error("NaN values in Close prices")
            return None, None, None, None, None
        scaled_data = self.scaler.fit_transform(prices.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.look_back, len(scaled_data) - self.prediction_days):
            X.append(scaled_data[i-self.look_back:i])
            y.append(scaled_data[i:i+self.prediction_days]) 
        
        X, y = np.array(X), np.array(y)
        if X.size == 0 or y.size == 0:
            logging.error("Empty LSTM data arrays")
            return None, None, None, None, None
        train_size = int(len(X) * 0.8)
        if train_size < 10:
            logging.error("Training data too small")
            return None, None, None, None, None
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
            logging.error("No data to train models")
            return None, None, None, None
            
        import tensorflow as tf
        import random
        np.random.seed(42)
        tf.random.set_seed(42)
        random.seed(42)

        features = ['Close', 'RSI', 'MA20', 'MA50', 'BB_upper', 'BB_lower']
        X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test, scaled_data = self.prepare_lstm_data()
        if X_lstm_train is None:
            logging.error("Failed to prepare LSTM data")
            return None, None, None, None
        
        try:
            self.models['lstm'] = self.create_lstm_model()
            self.models['lstm'].fit(X_lstm_train, y_lstm_train, epochs=20, 
                                  batch_size=32, validation_split=0.1, verbose=0)
        except Exception as e:
            logging.error(f"LSTM training failed: {str(e)}")
            return None, None, None, None
        
        X = self.data[features].values
        y = self.data['Close'].shift(-self.prediction_days).values
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        if X.size == 0 or y.size == 0:
            logging.error("Empty RF data arrays")
            return None, None, None, None
        
        total_samples = len(scaled_data) - self.look_back - self.prediction_days + 1
        train_size = int(total_samples * 0.8)
        if train_size < 10:
            logging.error("RF training data too small")
            return None, None, None, None
        
        rf_scaler = MinMaxScaler()
        try:
            X_scaled = rf_scaler.fit_transform(X)
        except Exception as e:
            logging.error(f"RF scaling failed: {str(e)}")
            return None, None, None, None
        
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:train_size + len(X_lstm_test)]
        y_train, y_test = y[:train_size], y[train_size:train_size + len(X_lstm_test)]
        if X_train.size == 0 or y_train.size == 0:
            logging.error("Empty RF training data")
            return None, None, None, None
        
        try:
            self.models['rf'] = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
            self.models['rf'].fit(X_train, y_train)
        except Exception as e:
            logging.error(f"RF training failed: {str(e)}")
            return None, None, None, None
        
        return X_lstm_test, y_lstm_test, X_test, y_test

    def predict(self, X_lstm_test, X_test):
        if X_lstm_test is None or X_test is None:
            logging.error("Invalid prediction inputs")
            return None, None
        try:
            lstm_pred = self.models['lstm'].predict(X_lstm_test, verbose=0)
            lstm_pred_inv = self.scaler.inverse_transform(lstm_pred[:, -1].reshape(-1, 1)).flatten()
        except Exception as e:
            logging.error(f"LSTM prediction failed: {str(e)}")
            return None, None
        try:
            rf_pred = self.models['rf'].predict(X_test)
        except Exception as e:
            logging.error(f"RF prediction failed: {str(e)}")
            return None, None
        return lstm_pred_inv, rf_pred

# Streamlit UI
st.title("📈 Stock Price Predictor")
st.write("Enter a stock symbol and Alpha Vantage API key to get a 1-day-ahead price prediction.")

# Sidebar instructions
st.sidebar.markdown("""
### How to Use
1. Get a free API key from [Alpha Vantage](https://www.alphavantage.co).
2. Enter a stock symbol (e.g., MSFT, NVDA) and your API key.
3. Click "Predict" to see results in ~2–5 minutes.
""")

# User inputs
symbol = st.text_input("Stock Symbol (e.g., MSFT, NVDA):", "MSFT")
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
                if not predictor.add_technical_indicators():
                    st.error("Failed to compute technical indicators. Try another symbol.")
                    st.stop()
                
                # Step 3: Train models
                progress.progress(0.75)
                X_lstm_test, y_lstm_test, X_test, y_test = predictor.train_models()
                if X_lstm_test is None:
                    st.error("Failed to train models. Data may be insufficient.")
                    st.stop()
                
                # Step 4: Predict
                progress.progress(1.0)
                lstm_pred, rf_pred = predictor.predict(X_lstm_test, X_test)
                if lstm_pred is None or rf_pred is None:
                    st.error("Prediction failed. Check data or try again.")
                    st.stop()
                
                y_test_inv = predictor.scaler.inverse_transform(y_lstm_test[:, -1].reshape(-1, 1)).flatten()
                if len(y_test_inv) == 0:
                    st.error("No valid test data for evaluation.")
                    st.stop()
                
                # Calculate RMSE
                rmse_results = {}
                for name, pred in [('LSTM', lstm_pred), ('RF', rf_pred)]:
                    try:
                        rmse = np.sqrt(mean_squared_error(y_test_inv, pred))
                        rmse_results[name] = rmse
                    except Exception as e:
                        logging.error(f"RMSE calculation failed for {name}: {str(e)}")
                        rmse_results[name] = float('nan')
                
                # Display results
                st.success("Prediction completed!")
                st.subheader("1-Day-Ahead Closing Price Predictions")
                st.write(f"**LSTM**: ${lstm_pred[-1]:.2f}")
                st.write(f"**Random Forest**: ${rf_pred[-1]:.2f}")
                
                st.subheader("Model Performance (RMSE)")
                for name, rmse in rmse_results.items():
                    if np.isnan(rmse):
                        st.write(f"**{name}**: Unable to compute")
                    else:
                        st.write(f"**{name}**: {rmse:.2f}")
                
                # Plot
                st.subheader("Prediction vs. Actual Prices")
                fig = plt.figure(figsize=(10, 5))
                plt.plot(predictor.data.index[-len(y_test):], y_test_inv, label='Actual', linewidth=2)
                plt.plot(predictor.data.index[-len(y_test):], lstm_pred, label='LSTM', linestyle='--')
                plt.plot(predictor.data.index[-len(y_test):], rf_pred, label='Random Forest', linestyle='--')
                plt.title(f'{symbol} Stock Price Prediction (1-Day Ahead)')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error: {str(e)}. Check your API key, symbol, or try again later.")
                logging.error(f"General error: {str(e)}")

st.markdown("""
**Note**: This app uses the Alpha Vantage API (free tier: 5 requests/min). Predictions may take a few minutes due to model training.
""")
