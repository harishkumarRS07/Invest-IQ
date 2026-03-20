# 🧠 InvestIQ — Complete Architecture & Workflow Guide

> A comprehensive breakdown of how InvestIQ works end-to-end: from data ingestion through AI prediction to mobile app delivery.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Data Pipeline (Training)](#data-pipeline-training)
4. [Inference Pipeline (Prediction)](#inference-pipeline-prediction)
5. [Backend API Structure](#backend-api-structure)
6. [Mobile App Flow](#mobile-app-flow)
7. [Feature Engineering Details](#feature-engineering-details)
8. [Models & AI Components](#models--ai-components)
9. [Key Workflows & Use Cases](#key-workflows--use-cases)
10. [Database & Authentication](#database--authentication)
11. [Deployment & Scheduling](#deployment--scheduling)

---

## Project Overview

**InvestIQ** is a full-stack AI-powered stock prediction platform:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI (Python) | REST API, ML inference, data pipeline |
| **Models** | PyTorch (Transformer), XGBoost | Stock price & signal prediction |
| **Mobile Frontend** | React Native (Expo) | iOS/Android user interface |
| **Data Source** | yfinance, FinBERT, News APIs | Real-time stock data & sentiment |
| **Authentication** | JWT (homebrew HS256) | Stateless user auth |
| **Scheduling** | APScheduler | Background training & data updates |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       INVESTIQ ECOSYSTEM                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MOBILE APP (React Native/Expo)               │
│  [Login] → [Dashboard] → [Stock Detail] → [Portfolio] → [News] │
└──────────────┬────────────────────────────────────────────┬─────┘
               │                                            │
               ├──── HTTP/REST (Bearer JWT) ──────────────┬│
               │                                          ││
┌──────────────▼──────────────────────────────────────────▼▼─────┐
│                    BACKEND (FastAPI + PyTorch)                 │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐        │
│  │  API Routes  │  │  Auth Layer │  │ Inference Engine │        │
│  │ /predict     │  │  (JWT)      │  │   (Predictor)    │        │
│  │ /signals     │  │             │  │                  │        │
│  │ /portfolio   │  │  users.json │  │  ▼ Load Model    │        │
│  │ /auth/login  │  └─────────────┘  │  ▼ Fetch Data    │        │
│  └──────────────┘                    │  ▼ Features      │        │
│                                       │  ▼ Predict       │        │
│                                       └──────────────────┘        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           DATA & FEATURE ENGINEERING PIPELINE            │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  1. Load historical CSV (25-year window)                │    │
│  │  2. Add Technical Indicators (RSI, MACD, BB, etc)      │    │
│  │  3. Add Sentiment (FinBERT on live news)               │    │
│  │  4. Add Risk Metrics (Sharpe, Drawdown, etc)           │    │
│  │  5. Add External Data (Macro, Market Correlation)      │    │
│  │  6. Scale Features (StandardScaler)                     │    │
│  │  7. Create Sequences for Transformer                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           TRAINED MODELS (in models/saved_models/)      │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  • transformer_TICKER.pth (PyTorch weights)             │    │
│  │  • scaler_TICKER.pkl (StandardScaler fitted on data)   │    │
│  │  • (XGBoost, LSTM-Attention models also available)     │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
               │                                          │
        ┌──────┴──────────┐                      ┌────────▼────────┐
        │                 │                      │                 │
   ┌────▼─────┐     ┌────▼─────┐         ┌──────▼──────┐    ┌─────▼──┐
   │ yfinance  │     │ FinBERT  │         │ APScheduler │    │ Models │
   │ Stock     │     │ News     │         │ (Training   │    │ Cache  │
   │ Data CSV  │     │ Sentiment│         │  Updates)   │    │ (10min)│
   └───────────┘     └──────────┘         └─────────────┘    └────────┘
```

---

## Data Pipeline (Training)

### 1⃣ **Data Acquisition Phase**

```
Step 1: Collect Historical Data
├── Input: Ticker symbol (e.g., "HDFCBANK")
├── Source: yfinance.download(ticker, period='25y')
├── Output: CSV with OHLCV (Open, High, Low, Close, Volume)
└── Location: backend/data/stock_data/HDFCBANK.csv

Step 2: Data Cleaning
├── Convert Date to datetime, sort by date
├── Remove duplicates, fill missing values (ffill, bfill)
└── Validate required columns exist
```

**File:** `backend/preprocessing/cleaning.py:clean_data()`

---

### 2⃣ **Feature Engineering Phase**

This is where raw price data transforms into ML-ready features:

```
Input: Raw OHLCV DataFrame (25 years of daily data)
       │
       ├─► Technical Indicators
       │   ├── SMA_20, SMA_50 (Moving Averages)
       │   ├── RSI (Relative Strength Index)
       │   ├── MACD + Signal + Histogram
       │   ├── Bollinger Bands (High, Low)
       │   ├── ATR (Average True Range)
       │   ├── VWAP (Volume Weighted Average Price)
       │   ├── Log_Return = ln(Close_t / Close_t-1)
       │   ├── Volume_Change = (Volume % change)
       │   └── Rolling_Volatility = σ(Log_Return, window=20)
       │
       ├─► Market Correlation
       │   ├── Fetch NIFTY 50 prices (market index)
       │   ├── Compute rolling correlation with stock returns
       │   └── Add Market_Correlation column
       │
       ├─► Multi-Timeframe Features (OPTIONAL)
       │   ├── Resample to Weekly (W), Monthly (ME)
       │   ├── Merge back with suffix _W, _M
       │   └── Forward-fill to propagate to daily rows
       │
       ├─► External Features (SIMULATED for training, REAL for inference)
       │   ├── Sentiment: News sentiment score (-1 to +1)
       │   ├── Macro_Score: Economic health indicator (0-100)
       │   └── (Can be randomized for robustness or fetched real-time)
       │
       └─► Risk Metrics Computed Per Window
           ├── Sharpe Ratio
           ├── Max Drawdown
           ├── Sortino Ratio
           └── Value at Risk (95%)

Output: DataFrame with ~25+ features, named:
        ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Log_Return', ...]
```

**Files:**
- `backend/features/indicators.py:add_technical_indicators()`
- `backend/features/sentiment.py:SentimentAnalyzer.analyze()`
- `backend/features/external_data.py:ExternalDataSimulator.add_external_features()`
- `backend/features/timeframes.py:TimeFrameProcessor.create_multiframe_features()`
- `backend/features/risk.py:RiskEngine.get_risk_profile()`

---

### 3️⃣ **Scaling & Sequence Creation Phase**

```
Input: Feature-engineered DataFrame
│
├─► Scaling (StandardScaler)
│   ├── Fit on training data (normalize all columns)
│   ├── Mean=0, Std=1 for each feature
│   └── Save scaler_TICKER.pkl for inference reuse
│
├─► Drop NaN rows (from indicator calculation windows)
│
└─► Create Sequences for Transformer
    ├── SEQ_LENGTH = 90 days (input window)
    ├── FORECAST_HORIZON = 7 days (output window)
    │
    ├── For each position i in data:
    │   ├── Input X[i]: data[i : i+90, all_features]     → shape (90, 25)
    │   ├── Target y[i]: data[i+90 : i+90+7, Log_Return] → shape (7,)
    │   └── Repeat for all valid i
    │
    └── Output:
        ├── X shape: (num_samples, 90, 25)
        └── y shape: (num_samples, 7)
```

**File:** `backend/training/train.py:create_sequences()`

---

### 4️⃣ **Model Training Phase**

```
Model: TimeSeriesTransformer
├── Architecture:
│   ├── Input: (batch_size, seq_len=90, features=25)
│   ├── Embedding + Positional Encoding
│   ├── Transformer Encoder (2 layers, 4 heads, 64 d_model)
│   ├── Decoder → Linear projection to forecast_horizon=7
│   └── Output: (batch_size, 7, 1)
│
├── Loss: MSE (Mean Squared Error)
├── Optimizer: Adam (lr=0.001)
├── Scheduler: ReduceLROnPlateau (factor=0.5 if val_loss plateaus)
│
├── Training Loop (100 epochs):
│   ├── Split data: 80% train, 20% validation (time-series aware)
│   ├── For each epoch:
│   │   ├── Shuffle training batches (batch_size=32)
│   │   ├── Forward pass, compute loss, backprop
│   │   ├── Validate on hold-out set
│   │   └── Log train/val loss
│   └── Save best checkpoint
│
└── Output:
    ├── transformer_TICKER.pth (model weights)
    └── scaler_TICKER.pkl (for scaling inference data)
```

**File:** `backend/training/train.py:train_pipeline()`

---

## Inference Pipeline (Prediction)

### When a user requests a prediction (e.g., `/api/v1/predict`):

```
1️⃣ LOAD MODEL & SCALER
   ├── Model file: models/saved_models/transformer_TICKER.pth
   ├── Scaler file: scaler_TICKER.pkl
   └── Load into memory (CPU or GPU)

2️⃣ FETCH LATEST DATA
   ├── Historical CSV: backend/data/stock_data/TICKER.csv
   ├── Latest price point: yfinance.Ticker(TICKER).history(period="1d")
   └── Merge live data if newer than CSV

3️⃣ APPLY FEATURE ENGINEERING (SAME AS TRAINING)
   ├── Clean data
   ├── Add Technical Indicators
   ├── Add Market Correlation
   ├── Add Real-time Sentiment (FinBERT on live news)
   ├── Add External Features
   └── Drop NaN rows

4️⃣ SCALE FEATURES
   ├── Get last 90 days of data
   ├── Transform using loaded scaler
   └── Create input tensor: shape (1, 90, 25)

5️⃣ RUN INFERENCE
   ├── Input to model: (1, 90, 25)
   ├── Model predicts: (1, 7, 1) → 7-day log returns
   └── Raw output: array of log returns

6️⃣ POST-PROCESS PREDICTIONS
   ├── Inverse-transform log returns to prices
   ├── Convert to cumulative returns
   ├── Compute confidence score
   ├── Determine signal (BUY/SELL/HOLD)
   ├── Extract technical indicators from last rows
   └── Build response JSON

7️⃣ RETURN TO CLIENT
   └── {
         "symbol": "HDFCBANK",
         "current_price": 1850.50,
         "predicted_price": 1875.25,
         "7_day_forecast": [1855, 1860, 1865, ...],
         "signal": "BUY",
         "signal_confidence": 0.75,
         "risk_level": "Medium",
         "indicators": {...},
         "explanation": "..."
       }
```

**File:** `backend/inference/predict.py:Predictor.predict()`

---

## Backend API Structure

### Routes Overview

```python
# backend/app/routes.py

GET  /health
     └─ Returns: {"status": "ok", "version": "2.0.0"}

POST /auth/register
     ├─ Input: {email, password, name}
     └─ Output: {token, email, name}

POST /auth/login
     ├─ Input: {email, password}
     └─ Output: {token, email, name}

GET  /auth/me
     ├─ Header: Authorization: Bearer <token>
     └─ Output: {email, name}

POST /predict (Protected)
     ├─ Header: Authorization: Bearer <token>
     ├─ Input: {symbol, file_path?}
     └─ Output: PredictionResponse (complex object)

POST /signals/batch (Protected)
     ├─ Header: Authorization: Bearer <token>
     ├─ Input: {symbols: [TICKER1, TICKER2, ...]}
     └─ Output: {signals: [SignalSummary, ...]}

GET  /tickers (Protected)
     ├─ Header: Authorization: Bearer <token>
     └─ Output: {tickers: ["HDFCBANK", "INFY", ...]}

POST /retrain/trigger (Protected)
     ├─ Endpoint: Manual retraining trigger
     └─ Starts background thread to retrain all models

GET  /retrain/status (Protected)
     └─ Returns: {is_running, last_run, last_status, log}
```

### Request/Response Schemas

```python
# backend/app/schemas.py

class PredictionRequest:
    symbol: str              # "HDFCBANK"
    file_path: Optional[str] # path to CSV

class PredictionResponse:
    symbol: str
    current_price: float
    predicted_price: float
    seven_day_forecast: List[float]
    confidence_interval: Tuple[float, float]
    signal: str              # "BUY" | "SELL" | "HOLD"
    signal_confidence: float # 0.0 to 1.0
    risk_level: str          # "Low" | "Medium" | "High"
    indicators: Indicators   # RSI, MACD, SMA_20, etc.
    explanation: str

class Indicators:
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    sma_20: Optional[float]
    sma_50: Optional[float]
    bb_high: Optional[float]
    bb_low: Optional[float]
    vwap: Optional[float]
    atr: Optional[float]

class BatchSignalRequest:
    symbols: List[str]  # ["HDFCBANK", "INFY", "TCS"]

class SignalSummary:
    symbol: str
    current_price: float
    predicted_price: float
    signal: str
    signal_confidence: float
    risk_level: str
    pct_change: float
    indicators: Indicators
    explanation: str
```

### Caching Strategy

```python
_signals_cache: dict = {}        # symbol → (timestamp, result_dict)
_CACHE_TTL = 600                 # 10 minutes

When /signals/batch is called:
├── For each symbol:
│   ├── Check if in cache AND cache age < 10 min
│   ├── If yes: use cached prediction
│   └── If no: run fresh prediction, cache it
└── Return all signals
```

---

## Mobile App Flow

### Architecture (React Native + Expo)

```
App Entry Point: InvestIQ-App/App.js
│
├─► Root Layout: app/_layout.js
│   ├── Setup Providers (ThemeProvider, AuthProvider, GestureHandlerRootView)
│   ├── Show AnimatedSplash (1-2 seconds)
│   └── Render Stack navigation
│
├─► Auth Guard (AuthGuard component)
│   ├── If authenticated → show (tabs) group
│   └── If not authenticated → redirect to (auth)/login
│
├─► (auth) Group: app/(auth)/_layout.js
│   ├── login.js (Login Screen)
│   │   ├── Email + Password TextInputs
│   │   ├── Call POST /auth/login
│   │   ├── Store JWT in SecureStore
│   │   └── Redirect to Dashboard
│   │
│   └── register.js (Register Screen)
│       ├── Name, Email, Password inputs
│       ├── Call POST /auth/register
│       ├── Store JWT in SecureStore
│       └── Redirect to Dashboard
│
└─► (tabs) Group: app/(tabs)/_layout.js
    ├── Bottom Tab Bar with 4 screens
    │
    ├─► Dashboard Tab: dashboard.js
    │   ├── Fetch GET /api/v1/tickers → list available stocks
    │   ├── Call POST /api/v1/signals/batch (with Bearer token)
    │   ├── Display SignalCard grid or list
    │   │   └── Each card shows: symbol, signal, price, % change
    │   ├── Pull-to-refresh → refetch signals
    │   └── Tap card → navigate to Stock Detail
    │
    ├─► Stock Detail: stock/[symbol].js
    │   ├── Dynamic route param: symbol = route.params.symbol
    │   ├── Call GET /api/v1/predict?symbol=TICKER
    │   ├── Render:
    │   │   ├── Current price chart (7-day forecast)
    │   │   ├── Technical indicators (RSI, MACD, Bollinger Bands)
    │   │   ├── AI explanation text
    │   │   ├── Signal badge (BUY/SELL/HOLD)
    │   │   └── Confidence bar
    │   └── Refresh button to reload prediction
    │
    ├─► Portfolio Tab: portfolio.js
    │   ├── Call GET /api/v1/portfolio
    │   ├── Display Pie chart of allocation
    │   ├── Show allocation percentages
    │   └── Rebalance button
    │
    ├─► Settings Tab: settings.js
    │   ├── User profile
    │   ├── Dark/Light mode toggle
    │   ├── Notification settings
    │   ├── Logout button
    │   └── About / Help
    │
    └─► News Tab: news.js
        ├── Fetch live news (optional, can integrate news API)
        └── Display news cards with sentiment
```

### Key React Components

```javascript
// src/components/ui.js
├── GradientButton      // CTA button with purple gradient
├── SecondaryButton     // Outline button
├── Card                // Rounded surface with shadow
├── SignalBadge         // BUY/SELL/HOLD colored pill
├── ErrorBanner         // Error message display
├── LoadingSpinner      // Animated loading indicator
└── EmptyState          // Placeholder for empty lists

// src/components/StockSignalCard.js
└── Card component showing:
    ├── Ticker & company name
    ├── Current price
    ├── Signal (BUY/SELL/HOLD)
    ├── Confidence %
    ├── % change
    └── TapHandler → navigate to stock detail

// src/context/AuthContext.js
└── Global auth state:
    ├── user (email, name)
    ├── token (JWT)
    ├── loading (session restore)
    ├── login() function
    ├── register() function
    └── logout() function

// src/services/api.js
└── Axios instance with interceptors:
    ├── authApi.login(email, password)
    ├── authApi.register(email, password, name)
    ├── stockApi.predict(symbol)
    ├── stockApi.signals(symbols)
    ├── stockApi.tickers()
    └── All requests auto-attach Bearer token
```

### Data Flow: User Sees Dashboard

```
1. User taps Dashboard tab
   └─ dashboard.js component mounts

2. useEffect hook fires:
   ├─ Call GET /api/v1/tickers
   ├─ Get list: ["HDFCBANK", "INFY", "TCS", ...]
   └─ Store in state

3. Call POST /api/v1/signals/batch with token
   ├─ Request: {symbols: ["HDFCBANK", "INFY", ...]}
   ├─ Backend runs inference for each ticker (uses 10-min cache)
   └─ Response: {signals: [{symbol, signal, price, ...}, ...]}

4. Process response:
   ├─ Parse JSON
   ├─ Create array of SignalCards
   └─ Set state (triggers re-render)

5. User sees:
   ├─ Grid of cards
   │   ├── HDFCBANK: BUY 1857.50 +1.20%
   │   ├── INFY: HOLD 1650.00 -0.50%
   │   └── ...
   └─ Pull-to-refresh available

6. User taps a card:
   ├─ Navigate to stock/[symbol]
   ├─ Pass symbol as route param
   └─ Detail screen loads
```

---

## Feature Engineering Details

### Complete Feature List (After Engineering)

```
TECHNICAL INDICATORS (10 features):
├── SMA_20               Moving Average (20-day)
├── SMA_50               Moving Average (50-day)
├── RSI                  Relative Strength Index (14-day)
├── MACD                 MACD line
├── MACD_Signal          MACD signal line
├── MACD_Hist            MACD histogram
├── BB_High              Bollinger Band upper line
├── BB_Low               Bollinger Band lower line
├── VWAP                 Volume Weighted Average Price
└── ATR                  Average True Range

RETURNS & VOLATILITY (3 features):
├── Log_Return           Natural log return (target variable)
├── Volume_Change        Daily volume percentage change
└── Rolling_Volatility   20-day rolling stdev of returns

MARKET CORRELATION (1 feature):
└── Market_Correlation   Correlation with NIFTY 50 returns

MULTI-TIMEFRAME (Optional, 6 features):
├── SMA_20_W, SMA_20_M   Weekly/Monthly moving averages
├── RSI_W, RSI_M         Weekly/Monthly RSI
└── (and other OHLCV aggregates)

EXTERNAL FEATURES (2 features):
├── Sentiment            News sentiment score (-1.0 to 1.0)
└── Macro_Score          Economic health (0 to 100)

TOTAL: ~28 features → all scaled to mean=0, std=1
```

### Sentiment Analysis Workflow

```
Live Sentiment Detection (during inference):

1. Fetch live news for ticker:
   └─ yfinance.Ticker(symbol).news → list of news objects

2. Extract titles:
   ├─ For each article:
   │   ├── Get article['content']['title']
   │   └── Collect into list: ["HDFCBANK gains...", "HDFC stock..."]
   └─ Output: list of ~5-10 recent titles

3. Analyze with FinBERT:
   ├── Load ProsusAI/finbert model
   ├── For each title:
   │   ├── Tokenize (max 512 tokens, chunk if longer)
   │   ├── Forward pass through FinBERT
   │   └── Get scores: {positive, negative, neutral}
   └── Compute: compound = positive_score - negative_score

4. Return average sentiment:
   └─ (-1.0) Negative ← → Neutral (0.0) ← → Positive (+1.0)

Example:
   "HDFCBANK beats profit estimates" → +0.85
   "INFY lays off 2000 employees" → -0.72
   "TCS shares trade flat" → 0.05
```

**File:** `backend/features/sentiment.py:SentimentAnalyzer`

---

## Models & AI Components

### 1. Main Production Model: **TimeSeriesTransformer**

```python
# backend/models/transformer.py

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim=25,              # Number of features
        d_model=64,                # Embedding dimension
        nhead=4,                   # Number of attention heads
        num_layers=2,              # Transformer encoder layers
        dropout=0.1,
        output_dim=1,              # Predict 1 value (Log_Return)
        forecast_horizon=7         # Predict 7 days ahead
    ):
        ├── Input Linear Layer: projects (batch, seq, 25) → (batch, seq, 64)
        ├── Positional Encoding: adds position info to embeddings
        ├── Transformer Encoder: 2 layers of multi-head attention
        ├── Decoder: projects back to output dimension
        └── Output Projection: per-timestep + fully connected layers

Architecture:
    Input (batch, 90, 25)
        ↓ [Linear Embedding]
    (batch, 90, 64)
        ↓ [Positional Encoding]
    (batch, 90, 64)
        ↓ [Transformer Encoder: 4-head attention × 2 layers]
    (batch, 90, 64)
        ↓ [Take last timestep]
    (batch, 64)
        ↓ [Reshape to forecasting head]
    (batch, 7)
        ↓ [Linear layer to 1 output per step]
    (batch, 7, 1)  ← Predictions for next 7 days
```

### 2. Alternative Models (Available but not default)

```
A) LSTM with Attention (backend/models/lstm_attention.py)
   ├── Input: (batch, 90, 25)
   ├── LSTM encoder (bi-directional)
   ├── Attention mechanism on encoder output
   ├── LSTM decoder, teacher forcing during training
   └── Output: (batch, 7, 1)

B) XGBoost Fusion (backend/models/xgboost_fusion.py)
   ├── Ensemble combination:
   │   ├── Transformer output + confidence scores
   │   ├── XGBoost predictions from tabular features
   │   └── LSTM predictions
   ├── Weighted voting → final prediction
   └── Used for robustness (experimental)

C) Ensemble (backend/models/ensemble.py)
   └── Combines outputs from multiple models
```

### 3. Model Evaluation Metrics

```python
# backend/evaluation/metrics.py

Computed after training/testing:
├── MAE (Mean Absolute Error)
├── RMSE (Root Mean Squared Error)
├── MAPE (Mean Absolute Percentage Error) ← best for forecasting
├── Directional Accuracy → % of correctly predicted up/down moves
├── Sharpe Ratio of trading strategy based on predictions
└── Max Drawdown if trading on signals
```

---

## Key Workflows & Use Cases

### Use Case 1: First-Time User Registration & Login

```
┌─ User opens app
├─ Sees login screen
└─ Taps "Create Account"

FLOW:
1. Register Screen → enters: email, password, confirm, name
2. Clicks "Create Account"
3. POST /auth/register
   ├─ Backend:
   │   ├── Check if email already exists
   │   ├── Hash password (SHA256)
   │   ├── Store in users.json
   │   ├── Generate JWT token (24h expiry)
   │   └── Return {token, email, name}
4. App receives response
5. SecureStore.store('jwt_token', token)
6. Set auth state: {user, token, isAuthenticated=true}
7. AuthGuard redirects to Dashboard
8. User sees stock signals
```

---

### Use Case 2: Get AI Prediction for a Stock

```
┌─ User in dashboard, taps HDFCBANK card
└─ Navigate to stock/HDFCBANK

DETAIL SCREEN:
1. Route param: symbol = "HDFCBANK"
2. useEffect: Call GET /api/v1/predict?symbol=HDFCBANK
3. Add header: Authorization: Bearer <jwt_token>

BACKEND /predict endpoint:
├─ 1. Validate JWT token
├─ 2. Load model: transformer_HDFCBANK.pth
├─ 3. Load scaler: scaler_HDFCBANK.pkl
├─ 4. Load data: HDFCBANK.csv
├─ 5. Fetch live price from yfinance
├─ 6. Merge live data with historical
├─ 7. Apply feature engineering:
│   ├── Add technical indicators
│   ├── Add sentiment (FinBERT on live news)
│   ├── Add market correlation
│   └── Drop NaNs
├─ 8. Get last 90 days
├─ 9. Scale features
├─ 10. Run transformer: (1, 90, 25) → (1, 7, 1)
├─ 11. Post-process:
│   ├── Inverse-transform log returns
│   ├── Convert to prices
│   ├── Compute signal: if 7-day return > 0.5% → BUY, else...
│   ├── Compute confidence (dynamic based on magnitude)
│   ├── Extract technical indicators
│   └── Build explanation text
└─ 12. Return PredictionResponse JSON

FRONTEND:
├─ Receive response
├─ Format for display:
│   ├── Chart: 7-day forecast line
│   ├── Metrics: current price, predicted, confidence
│   ├── Indicators: RSI, MACD, Bollinger Bands
│   ├── Signal badge: BUY/SELL/HOLD
│   └── AI explanation: human-readable rationale
└─ Render on screen

RESULT:
User sees: "HDFCBANK: Current 1857.50 → Predicted 1895 (BUY, 78% confidence)"
```

---

### Use Case 3: Weekly Auto-Retraining

```
SCHEDULED TASK (APScheduler in main.py lifespan):
├─ Trigger: Every Sunday 00:00 (disabled as of recent fix)
└─ Manual trigger: POST /retrain/trigger (if re-enabled)

RETRAINING WORKFLOW:
1. Spawn background thread: _run_full_retrain()
2. For each CSV in data/stock_data/:
   ├─ Load historical data (25-year window)
   ├─ Apply feature engineering
   ├─ Train Transformer model
   ├─ Evaluate on validation set
   ├─ Save: transformer_TICKER.pth
   ├─ Save: scaler_TICKER.pkl
   └─ Log results

3. Clear prediction cache → new models immediately active
4. Return retrain status via /retrain/status endpoint

MONITORING:
├─ User can poll /retrain/status
│   └─ Response: {is_running, last_run, last_status, log}
├─ Log: ["Step 1/2: Updating stock data...", "  ✓ Stock data updated", ...]
└─ Optional: Send notification when retraining completes (future)
```

**File:** `backend/training/auto_retrain.py`

---

### Use Case 4: Portfolio Optimization

```
SCENARIO:
User has portfolio of stocks: HDFCBANK, INFY, TCS, LTCON
Wants to optimize allocation for max Sharpe Ratio

FLOW:
1. GET /api/v1/portfolio (with symbols list)
2. Backend:
   ├─ Fetch 3-year historical prices for each stock
   ├─ Compute log returns
   ├─ Build covariance matrix
   ├─ Run Mean-Variance Optimization
   │   ├── Objective: maximize Sharpe Ratio
   │   ├── Constraint: sum(weights) = 1
   │   ├── Bounds: 0 ≤ weight ≤ 1
   │   └── Solver: SLSQP
   ├─ If degenerate (one stock >60%):
   │   └── Fall back to Inverse-Volatility weighting
   ├─ Apply floor: each allocation ≥ 5%
   ├─ Normalize to sum=1
   └─ Return: {HDFCBANK: 0.25, INFY: 0.30, TCS: 0.25, LTCON: 0.20}

3. Frontend:
   ├─ Render Pie Chart with allocations
   ├── Show recommended weights
   └─ User can accept or ignore

METRICS RETURNED:
├─ Expected Annual Return
├─ Annual Volatility
├─ Sharpe Ratio
├─ Max Drawdown (historical)
└─ Value at Risk (95%)
```

**File:** `backend/features/portfolio.py:PortfolioOptimizer`

---

## Database & Authentication

### User Storage (Current: File-Based)

```
Location: backend/app/users.json

Format:
{
  "user@example.com": {
    "name": "John Doe",
    "password_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "created_at": 1710525600.0
  }
}

NOTE: This is for development. In production, migrate to:
├─ PostgreSQL with SQLAlchemy ORM
├─ Hash passwords with bcrypt
└─ Use proper session/token management (e.g., redis)
```

### JWT Token Structure (Homebrew HS256)

```
Token Format: header.payload.signature

Example decoded:
{
  "alg": "HS256",
  "typ": "JWT"
}
.
{
  "sub": "user@example.com",
  "name": "John Doe",
  "exp": 1710612000       ← 24 hours from now
}
.
<HMAC-SHA256 signature>

Validation:
├── Recompute signature with SECRET_KEY
├── Compare with received signature
├── Check if exp > current time
└── Return payload if valid, None otherwise
```

### Token Lifecycle

```
1. User registers/logs in
   └─ Backend: _create_token({"sub": email, "name": name}, expires_in_seconds=86400)

2. Client receives token
   └─ SecureStore.setItem('jwt_token', token)

3. Client makes API requests
   ├─ All routes append header: Authorization: Bearer <token>
   └─ Backend middleware validates token

4. Token expires after 24 hours
   └─ Client catches 401 error, redirects to login

5. User logs out
   └─ Client: SecureStore.removeItem('jwt_token')
```

---

## Deployment & Scheduling

### Backend Startup (main.py Lifespan)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ─── STARTUP ─────────────────────────────────────────
    scheduler = AsyncIOScheduler()
    
    # ① Daily Stock Data Update (18:00 / 6 PM)
    scheduler.add_job(
        refined_update,
        'cron',
        hour=18,
        minute=0,
        id='daily_data_update',
        name='Daily Stock Data Update'
    )
    
    # ② Weekly Model Retraining (DISABLED as of recent fix)
    # Was: Sunday 00:00
    # Now: Commented out / trigger-only
    
    scheduler.start()
    print("Scheduler started.")
    
    yield  ← App runs while this yields
    
    # ─── SHUTDOWN ────────────────────────────────────────
    scheduler.shutdown()
    print("Scheduler stopped.")
```

### Running the Backend

```bash
# Development (with reload):
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Production (no reload):
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Via Docker:
docker build -t investiq .
docker run -p 8000:8000 investiq
```

### Environment Variables

```
JWT_SECRET = "investiq-super-secret-key-change-in-production"
DATABASE_URL = "postgresql://user:pass@localhost/investiq" (future)
OPEN_AI_KEY = "" (for future explanations)
NEWS_API_KEY = "" (for real news fetching)
```

---

## Summary Diagram: Data → Model → Prediction

```
┌─────────────────────────────────────────────────────────────┐
│                  END-TO-END PREDICTION FLOW                 │
└─────────────────────────────────────────────────────────────┘

                    [TRAINING PHASE]
                           ↓
        Raw OHLCV (yfinance) + News (FinBERT)
                           ↓
        Feature Engineering (20+ technical + sentiment)
                           ↓
        Scaling (StandardScaler) + Sequencing (90/7 window)
                           ↓
        Transformer Encoder-Decoder (100 epochs)
                           ↓
        Save: transformer_TICKER.pth + scaler_TICKER.pkl
                           ↓
    ┌─────────────────────────────────────────────────────┐
    │         [INFERENCE PHASE - Per Request]              │
    │                                                       │
    │  1. Load model + scaler                             │
    │  2. Fetch latest data (24h old CSV + live price)   │
    │  3. Apply same feature engineering                  │
    │  4. Take last 90 days (scaled)                     │
    │  5. Forward pass through transformer                │
    │  6. Output: 7-day log return predictions            │
    │  7. Convert to prices, generate signal              │
    │  8. Cache result (10 min)                           │
    │  9. Return to mobile app                            │
    │                                                       │
    │         [User sees on phone]                        │
    │      Ticker | Current | Predicted | Signal         │
    │      HDFC   | 1850    | 1895      | BUY 78%        │
    └─────────────────────────────────────────────────────┘
```

---

## Conclusion

**InvestIQ** architecture is a well-designed full-stack ML system:

- **Data Flow:** yfinance → Feature Engineering → Scaling → Model
- **Inference:** Real-time predictions with 10-min caching for efficiency
- **Security:** JWT-based auth with SecureStore on mobile
- **Scalability:** Modular design allows swapping models, features, data sources
- **UX:** Intuitive React Native app with dark mode, real-time signals, portfolio tracking

**Next Steps for Production:**
1. Migrate user store to PostgreSQL
2. Use Redis for token blacklist + prediction cache
3. Add monitoring (Prometheus, Grafana)
4. Implement proper error handling & rollback strategies
5. Add A/B testing framework for signal quality
6. Scale inference (GPU cluster, model partitioning)

---

**Generated**: March 15, 2026  
**For**: InvestIQ Architecture Documentation
