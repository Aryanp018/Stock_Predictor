import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
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

# Custom CSS for UI styling
st.markdown("""
    <style>
    /* General styling */
    body {
        background-color: #FFFFFF;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #FFFFFF;
    }
    h1 {
        color: #003087 !important;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 0.5em;
    }
    h2 {
        color: #003087 !important;
        font-size: 1.8em;
        border-bottom: 2px solid #28A745;
        padding-bottom: 0.2em;
        margin-top: 1em;
    }
    /* Main page text */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: #333 !important;
        font-size: 1.1em;
    }
    .stTextInput > div > div > div > label {
        color: #003087 !important;
        font-size: 1.1em;
        font-weight: bold;
    }
    /* Input form */
    .stTextInput > div > div > input {
        border: 2px solid #003087;
        border-radius: 8px;
        padding: 10px;
        font-size: 1.1em;
    }
    .stButton > button {
        background-color: #28A745;
        color: white !important;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.2em;
        border: none;
        transition: background-color 0.3s;
        display: block;
        margin: 1em auto;
    }
    .stButton > button:hover {
        background-color: #218838;
    }
    /* Output cards */
    .price-card {
        background-color: #F5F6F5;
        border: 1px solid #003087;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .price-card h3 {
        color: #003087 !important;
        margin: 0;
        font-size: 1.4em;
    }
    .price-card p {
        color: #333 !important;
        font-size: 1.6em;
        margin: 5px 0 0;
        font-weight: bold;
    }
    .price-card p:last-child {
        font-size: 0.9em;
        color: #666 !important;
    }
    /* Plot */
    .plot-container {
        background-color: #FFFFFF;
        border: 1px solid #003087;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Sidebar */
    .stSidebar {
        background-color: #E8ECEF;
        border-right: 1px solid #003087;
    }
    .stSidebar h3 {
        color: #003087 !important;
        font-size: 1.5em;
    }
    .stSidebar .stMarkdown {
        color: #333 !important;
        font-size: 1.1em;
    }
    .stSidebar .stExpander {
        background-color: #FFFFFF;
        border: 1px solid #003087;
        border-radius: 8px;
    }
    .stSidebar .stExpander > div > div > div {
        color: #333 !important;
    }
    /* Alerts */
    .stAlert[role="alert"] {
        border-radius: 8px;
        font-size: 1.1em;
    }
    /* Success banner */
    .stAlert[role="alert"] {
        background-color: #C3E6CB !important;
    }
    .stAlert[role="alert"] div {
        color: #333 !important;
        font-size: 1.2em;
        font-weight: bold;
    }
    /* Warning and error banners */
    .stAlert[role="alert"][style*="background-color: rgb(255, 243, 205)"] div {
        color: #333 !important;
    }
    .stAlert[role="alert"][style*="background-color: rgb(248, 215, 218)"] div {
        color: #333 !important;
    }
    /* Status messages */
    .status-message {
        text-align: center;
        font-size: 1.2em;
        color: #003087 !important;
        margin: 1em 0;
    }
    </style>
""", unsafe_allow_html=True)

# StockPredictor class (LSTM only)
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
                if data.empty or len(data) < self.look_back + 50:
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
        return X[:train_size], y[:train_size], X[train_size:], y[train_size:], scaled_data

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
            return None, None
        
        import tensorflow as tf
        import random
        np.random.seed(42)
        tf.random.set_seed(42)
        random.seed(42)

        X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test, scaled_data = self.prepare_lstm_data()
        if X_lstm_train is None:
            logging.error("Failed to prepare LSTM data")
            return None, None
        
        try:
            self.models['lstm'] = self.create_lstm_model()
            self.models['lstm'].fit(X_lstm_train, y_lstm_train, epochs=20, 
                                  batch_size=32, validation_split=0.1, verbose=0)
        except Exception as e:
            logging.error(f"LSTM training failed: {str(e)}")
            return None, None
        
        return X_lstm_test, y_lstm_test

    def predict(self, X_lstm_test):
        if X_lstm_test is None:
            logging.error("Invalid prediction inputs")
            return None
        try:
            lstm_pred = self.models['lstm'].predict(X_lstm_test, verbose=0)
            lstm_pred_inv = self.scaler.inverse_transform(lstm_pred[:, -1].reshape(-1, 1)).flatten()
        except Exception as e:
            logging.error(f"LSTM prediction failed: {str(e)}")
            return None
        return lstm_pred_inv

# Streamlit UI
st.title("📈 Stock Price Predictor")
st.markdown("Predict tomorrow's stock price with advanced AI. Enter a symbol and API key below.", unsafe_allow_html=True)

# Input form
st.markdown("### Enter Stock Details")
col1, col2 = st.columns([1, 1])
with col1:
    symbol = st.text_input("Stock Symbol", value="MSFT", placeholder="e.g., MSFT, NVDA", help="Enter a valid stock ticker (e.g., MSFT for Microsoft)")
with col2:
    api_key = st.text_input("Alpha Vantage API Key", type="password", placeholder="Enter your API key", help="Get a free key at alphavantage.co")

# Predict button
if st.button("Predict", key="predict_button"):
    if not symbol or not api_key:
        st.error("Please provide both a stock symbol and API key.")
    else:
        with st.spinner("Processing..."):
            try:
                # Progress messages
                status = st.empty()
                status.markdown("<div class='status-message'>🔄 Fetching stock data...</div>", unsafe_allow_html=True)
                predictor = StockPredictor(symbol, api_key, prediction_days=1)
                
                if not predictor.get_stock_data():
                    status.empty()
                    st.error(f"Failed to fetch data for {symbol}. Check if the symbol is valid (e.g., MSFT).")
                    st.stop()
                
                status.markdown("<div class='status-message'>🔄 Computing technical indicators...</div>", unsafe_allow_html=True)
                if not predictor.add_technical_indicators():
                    status.empty()
                    st.error("Failed to compute technical indicators. Try another symbol.")
                    st.stop()
                
                status.markdown("<div class='status-message'>🔄 Training LSTM model...</div>", unsafe_allow_html=True)
                X_lstm_test, y_lstm_test = predictor.train_models()
                if X_lstm_test is None:
                    status.empty()
                    st.error("Failed to train LSTM model. Data may be insufficient.")
                    st.stop()
                
                status.markdown("<div class='status-message'>🔄 Generating prediction...</div>", unsafe_allow_html=True)
                lstm_pred = predictor.predict(X_lstm_test)
                if lstm_pred is None:
                    status.empty()
                    st.error("Prediction failed. Check data or try again.")
                    st.stop()
                
                y_test_inv = predictor.scaler.inverse_transform(y_lstm_test[:, -1].reshape(-1, 1)).flatten()
                if len(y_test_inv) == 0:
                    status.empty()
                    st.error("No valid test data for evaluation.")
                    st.stop()
                
                # Get last actual price
                last_actual_price = predictor.data['Close'][-1]
                last_actual_date = predictor.data.index[-1].strftime('%Y-%m-%d')
                
                # Calculate RMSE
                try:
                    lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_pred))
                except Exception as e:
                    logging.error(f"RMSE calculation failed: {str(e)}")
                    lstm_rmse = float('nan')
                
                # Display results
                status.empty()
                st.success("Prediction completed! 🎉")
                
                st.markdown("### Stock Price Information")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(
                        f"""
                        <div class="price-card">
                            <h3>Last Actual Closing Price</h3>
                            <p>${last_actual_price:.2f}</p>
                            <p style="font-size: 0.9em; color: #666;">({last_actual_date})</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="price-card">
                            <h3>1-Day-Ahead Prediction</h3>
                            <p>${lstm_pred[-1]:.2f}</p>
                            <p style="font-size: 0.9em; color: #666;">(LSTM Model)</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.markdown("### Model Performance")
                st.markdown(
                    f"""
                    <div class="price-card">
                        <h3>LSTM RMSE</h3>
                        <p>{'Unable to compute' if np.isnan(lstm_rmse) else f'{lstm_rmse:.2f}'}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Plot
                st.markdown("### Actual vs. Predicted Prices")
                fig = plt.figure(figsize=(10, 5))
                plt.plot(predictor.data.index[-len(y_test_inv):], y_test_inv, label='Actual', linewidth=2, color='blue')
                plt.plot(predictor.data.index[-len(y_test_inv):], lstm_pred, label='LSTM Prediction', linewidth=2, linestyle='--', color='orange')
                plt.title(f'{symbol} Stock Price Prediction (1-Day Ahead)', fontsize=14, color='#003087')
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Price ($)', fontsize=12)
                plt.legend(fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                status.empty()
                if "rate limit" in str(e).lower():
                    st.warning("Alpha Vantage API rate limit reached (5 requests/min). Please wait a minute and try again.")
                else:
                    st.error(f"Error: {str(e)}. Check your API key, symbol, or try again later.")
                logging.error(f"General error: {str(e)}")

# Sidebar help
with st.sidebar:
    st.markdown("### About")
    st.markdown("This app uses AI to predict stock prices one day ahead. Powered by LSTM models and Alpha Vantage data.")
    
    with st.expander("Help & FAQ"):
        st.markdown("""
        **How to get an API key?**
        - Visit [alphavantage.co](https://www.alphavantage.co) and sign up for a free key.
        
        **What symbols are valid?**
        - Use stock tickers like MSFT (Microsoft), NVDA (NVIDIA), or AAPL (Apple).
        
        **Why does it take a few minutes?**
        - The app fetches fresh data and trains an LSTM model, which takes ~1–3 minutes.
        
        **Error messages?**
        - **Invalid symbol**: Check the ticker (e.g., MSFT, not Microsoft).
        - **Rate limit**: Wait a minute due to Alpha Vantage’s 5 requests/min limit.
        - **Other errors**: Ensure your API key is correct or try again later.
        """)

st.markdown(
    """
    <div style='text-align: center; color: #666; margin-top: 20px;'>
        Powered by Alpha Vantage API. Free tier limited to 5 requests/min.
    </div>
    """,
    unsafe_allow_html=True
)
