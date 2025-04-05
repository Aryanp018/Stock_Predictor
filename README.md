# 📈 Stock Predictor

**Stock Price Prediction with LSTM, Random Forest, and Gradient Boosting**

This project implements a stock price forecasting system for 1-day-ahead predictions using a blend of deep learning and machine learning models. It uses an LSTM model for temporal pattern recognition, along with Random Forest (RF) and Gradient Boosting (GB) models using technical indicators. Predictions are combined using a weighted ensemble approach based on inverse RMSE to improve accuracy.

---

## 🚀 Overview

- **Data Source**: [Alpha Vantage API](https://www.alphavantage.co/support/#api-key)
- **Forecast Target**: 1-day-ahead closing price
- **Models Used**:
  - Long Short-Term Memory (LSTM)
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Weighted Ensemble (based on 1/RMSE)

### 📊 Performance (on MSFT stock)

| Model         | RMSE   |
|---------------|--------|
| LSTM          | 7.83   |
| Random Forest | 73.75  |
| Gradient Boosting | 78.03 |
| **Ensemble**     | **22.10** |

> LSTM closely tracks the actual stock price. RF and GB provide supplementary signals, contributing to the ensemble.

---

## 📦 Features

- Historical stock data fetching via Alpha Vantage
- Technical indicators: RSI, MA20, MA50, Bollinger Bands
- LSTM model with 2 layers and dropout regularization
- Tree-based models for feature-driven learning
- Weighted ensemble combining all model outputs
- Visualizations for predictions vs. actual stock prices

---

## 🛠 Installation

### ✅ Prerequisites

- Python 3.7+
- Alpha Vantage API key (free at [Alpha Vantage](https://www.alphavantage.co/support/#api-key))

### 📥 Install Dependencies

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn alpha-vantage pandas-ta
```

### 📂 Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

## 💡 Usage

### 1. 🔐 Set Up Your API Key

In your script or notebook:

```python
api_key = 'YOUR_API_KEY'  # Replace with your Alpha Vantage key
```

### 2. ▶️ Run the Script

**Using Python Script:**

```bash
python stock_predictor.py
```

**Using Jupyter Notebook:**

```bash
jupyter notebook stock_predictor.ipynb
```

Run all cells to fetch data, train models, and visualize predictions.

### 3. 📈 Sample Output

```
Ensemble RMSE: 22.10
LSTM RMSE: 7.83
RF RMSE: 73.75
GB RMSE: 78.03
```

A plot of actual vs. predicted prices will be displayed, with the LSTM tracking the trend and RF/GB providing flatter outputs.

---

## 🔁 Predict a Different Stock

To use this for your stock of choice change the MSFT to your symbol of choice, modify the function call:

```python
main('AAPL', api_key)
```

**Note:** Alpha Vantage's free tier has a rate limit of 5 requests per minute.

---

## 📂 Code Structure

```
├── STOCK.ipynb                 # Main script
├── README.md                   # Project README
```

- **StockPredictor class**: Handles data fetching, preprocessing, training, and prediction
- **main() function**: Orchestrates the end-to-end workflow

---

## 📈 Model Details

### 🧠 LSTM

- 2 LSTM layers (100 units each)
- Dropout: 0.3
- Dense output layers
- Trained for 50 epochs, batch size 32, with 10% validation split

### 🌲 Random Forest

- 200 trees
- Max depth: 10

### 🔁 Gradient Boosting

- 200 estimators
- Max depth: 5
- Learning rate: 0.05

### ⚖️ Ensemble

- Weighted average of all models using inverse RMSE:
  
  ```python
  weight = 1 / RMSE
  ```

---

## 🔧 Future Improvements

- Add more indicators (e.g., MACD, ADX, lagged features)
- Grid search for RF/GB hyperparameter tuning
- Train a generalized model across multiple stocks
- Implement rolling window prediction for RF/GB
- Fine-tune LSTM learning rate and architecture

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a pull request


---

## 🙏 Acknowledgments

- **Alpha Vantage** – Free stock market data API
- **TensorFlow** – LSTM modeling
- **Scikit-learn** – Machine learning models
- **pandas-ta** – Technical indicators
