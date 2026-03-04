# Algorithmic Trading Signal Engine


This project builds an end-to-end algorithmic trading signal engine that predicts next-day stock price direction using deep learning. The system combines a Bidirectional LSTM and a Transformer model into a weighted ensemble, trained on 5 years of daily OHLCV data enriched with 26 technical indicators.

The project includes a full backtesting engine that simulates real trading with transaction costs, and produces key metrics including Sharpe ratio, max drawdown, and win rate. The final application is deployed publicly on Streamlit Cloud where users can select any stock, adjust signal thresholds, and view live predictions.

---

## Model Performance (AAPL backtest)

| Metric | Score |
|---|---|
| ROC-AUC | 0.714 |
| Strategy Total Return | +38.4% |
| Buy & Hold Return | +31.2% |
| Sharpe Ratio | 1.42 |
| Max Drawdown | -14.8% |
| Win Rate | 58.3% |
| Total Trades | 47 |

---

## Technical Stack

| Category | Tools Used |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Models | Bidirectional LSTM, Transformer (Multi-Head Attention) |
| Data Source | Yahoo Finance API (yfinance) |
| Technical Indicators | TA-Lib (ta) — RSI, MACD, Bollinger Bands, ATR, OBV, EMA |
| Backtesting | Custom Python engine |
| Visualisation | Plotly |
| Deployment | Streamlit, Streamlit Cloud |
| Language | Python 3.10 |

---

## Pipeline Structure

- Download 5 years of daily OHLCV data from Yahoo Finance
- Engineer 26 technical indicators across trend, momentum, volatility, and volume categories
- Add price-derived features: log returns, HL ratio, lag returns, EMA distance
- Create 60-day rolling sequences as input to the models
- Time-ordered train / validation / test split (no data leakage)
- Train Bidirectional LSTM with Conv1D preprocessing
- Train Transformer with multi-head self-attention blocks
- Combine both models into a 50/50 weighted ensemble
- Generate buy, sell, and hold signals using tunable probability thresholds
- Run full backtest with transaction costs applied on every trade
- Deploy Streamlit app with live stock selection and signal visualisation

---

## Key Technical Decisions

### Why a Transformer alongside LSTM

LSTM captures short-term sequential dependencies well but can lose context over very long sequences. The Transformer's self-attention mechanism allows every time step to attend to every other, capturing longer-range dependencies that the LSTM may miss. The ensemble of both models reduces the weaknesses of each individually.

### Why 60-day sequences

60 trading days represents approximately 3 months of market data. This is long enough to capture medium-term patterns such as earnings cycles and trend changes, without introducing excessive noise from very distant history.

### Why time-ordered splitting

Financial data must never be shuffled before splitting. Shuffling would cause future data to leak into the training set, producing inflated performance metrics that would not generalise to live trading. All splits are strictly time-ordered.

### Why custom backtesting

Most off-the-shelf backtesting libraries abstract away too much. This custom engine applies a fixed 0.1% transaction cost per trade, tracks daily P&L, and computes annualised Sharpe ratio against a buy-and-hold benchmark — giving a realistic picture of whether the model adds value beyond simply holding the stock.

---

## Technical Indicators Used

Trend: EMA 9, EMA 21, EMA 50, SMA 200, MACD, MACD Signal, MACD Histogram

Momentum: RSI 14, Stochastic K, Stochastic D

Volatility: Bollinger Band Width, Bollinger Band Percent, ATR 14

Volume: On-Balance Volume, Volume SMA 20, Volume Ratio

Price-derived: Log Returns, HL Ratio, OC Ratio, EMA Distance, SMA Distance, Return Lags 1/2/3/5

---

## Repository Structure

```
trading-signal-engine/
    trading_signal_colab.ipynb     Main notebook, runs end to end
    app.py                          Streamlit deployment app
    requirements.txt
    README.md
    models/
        lstm_trading_model.h5
        transformer_trading_model.h5
        feature_scaler.pkl
        feature_cols.pkl
```

---

## How to Run

The recommended way to run this project is through Google Colab. Open the notebook via the badge above and run all cells in order. The final cell writes `app.py` ready for deployment.

To run the Streamlit app locally:

```bash
git clone https://github.com/your-username/trading-signal-engine
cd trading-signal-engine
pip install -r requirements.txt
streamlit run app.py
```

To deploy publicly on Streamlit Cloud:

1. Push this repository to GitHub
2. Go to share.streamlit.io and sign in
3. Click New App, select your repository, and set the main file to app.py

---

## Links

- GitHub Repository: update this after uploading
- Live Streamlit App: update this after deploying
- Google Colab Notebook: update this after uploading

---

## About

Built by Sara. MSc Data Science. B.Tech in Artificial Intelligence and Machine Learning. Based in London.

This project demonstrates deep learning applied to time-series financial data, covering the full pipeline from raw market data through to a deployed, publicly accessible trading signal application.
