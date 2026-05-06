# 🧠 InvestIQ — COMPLETE DOCUMENTATION
## Full Frontend, API, Architecture & Ngrok Setup Guide

**Project Status**: Production Ready (PHASE 2)  
**Version**: 2.0  
**Last Updated**: April 9, 2026

---

## 📑 TABLE OF CONTENTS

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Backend API Complete Guide](#backend-api-complete-guide)
4. [Frontend (React Native/Expo) Complete Guide](#frontend-react-nativeexpo-complete-guide)
5. [Ngrok Setup & Tunnel Configuration](#ngrok-setup--tunnel-configuration)
6. [Installation & Setup](#installation--setup)
7. [Running the Application](#running-the-application)
8. [Data Pipeline & Models](#data-pipeline--models)
9. [Troubleshooting](#troubleshooting)

---

## PROJECT OVERVIEW

### What is InvestIQ?

**InvestIQ** is a full-stack AI-powered stock prediction platform for Indian markets:

| Component | Technology | Purpose | Status |
|-----------|-----------|---------|--------|
| **Backend API** | FastAPI (Python) | REST endpoints, ML inference, data pipeline | ✅ Production |
| **Frontend App** | React Native (Expo) | iOS/Android mobile interface | ✅ Production |
| **AI Models** | PyTorch, XGBoost | Price forecasting & Buy/Sell signals | ✅ PHASE 2 |
| **Real-time Data** | yfinance | Stock OHLCV + news | ✅ Active |
| **Authentication** | JWT (HS256) | Secure token-based auth | ✅ Implemented |
| **Tunnel** | ngrok | Mobile↔Backend secure tunnel | ✅ Ready |

### Key Features

✅ **Price Prediction** (LSTM + Transformer)  
✅ **Trading Signals** (BUY/SELL/HOLD via XGBoost)  
✅ **Portfolio Analytics** (Risk, Sharpe, Drawdown)  
✅ **News Sentiment** (FinBERT analysis)  
✅ **Mobile App** (Real-time updates)  
✅ **Authentication** (Secure JWT)  
✅ **Model Evaluation** (Comprehensive metrics)  

### Supported Stocks (Indian Market)

- **HDFCBANK** - HDFC Bank
- **ICICIBANK** - ICICI Bank  
- **INFY** - Infosys
- **RELIANCE** - Reliance Industries
- **TCS** - Tata Consultancy Services

---

## SYSTEM ARCHITECTURE

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      INVESTIQ MOBILE APPLICATION                       │
│              (React Native/Expo - iOS & Android)                        │
│                                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Login     │──→ │  Dashboard   │──→ │ Stock Detail │               │
│  │  (Auth)     │    │  (Stocks)    │    │  (Signals)   │               │
│  └─────────────┘    └──────────────┘    └──────────────┘               │
│         │                    │                    │                      │
│         │ JWT Token          │ API Calls          │ Predictions        │
│         └────────────────────┴────────────────────┘                    │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                    HTTP/REST (Bearer JWT)
                    over ngrok tunnel or LAN
                             │
        ┌────────────────────▼───────────────────────────┐
        │                                                 │
        │     BACKEND API (FastAPI + Python)             │
        │                                                 │
        │  ┌───────────────────────────────────────┐    │
        │  │    🔐 Authentication & Security       │    │
        │  │  • JWT token validation               │    │
        │  │  • User registration/login            │    │
        │  │  • Secure credential storage          │    │
        │  └───────────────────────────────────────┘    │
        │                                                 │
        │  ┌───────────────────────────────────────┐    │
        │  │    📊 API Endpoints                   │    │
        │  │  • /predict          → Price forecast│    │
        │  │  • /signals          → Buy/Sell/Hold │    │
        │  │  • /portfolio        → Analytics     │    │
        │  │  • /risk/score       → Risk metrics  │    │
        │  │  • /sentiment/analyze → News scores  │    │
        │  │  • /health           → Status check  │    │
        │  └───────────────────────────────────────┘    │
        │                                                 │
        │  ┌───────────────────────────────────────┐    │
        │  │    🧠 ML Inference Engine             │    │
        │  │                                       │    │
        │  │  1. Load historical CSV data          │    │
        │  │  2. Generate technical indicators    │    │
        │  │  3. Add sentiment/external features  │    │
        │  │  4. Scale all features                │    │
        │  │  5. Create time sequences             │    │
        │  │  6. Load trained models (.pth, .pkl) │    │
        │  │  7. Generate predictions + confidence│    │
        │  └───────────────────────────────────────┘    │
        │                                                 │
        │  ┌───────────────────────────────────────┐    │
        │  │    📈 Data Pipeline                   │    │
        │  │                                       │    │
        │  │  • Preprocessing/cleaning.py          │    │
        │  │  • Preprocessing/scaling.py           │    │
        │  │  • Features/indicators.py             │    │
        │  │  • Features/external_data.py          │    │
        │  │  • Features/sentiment.py              │    │
        │  │  • Features/risk.py                   │    │
        │  └───────────────────────────────────────┘    │
        │                                                 │
        │  ┌───────────────────────────────────────┐    │
        │  │    🤖 Trained Models                  │    │
        │  │  (in backend/models/saved_models/)   │    │
        │  │                                       │    │
        │  │  • transformer_HDFCBANK.pth  (PyTorch)    │
        │  │  • transformer_INFY.pth                    │
        │  │  • xgboost_classifier_HDFCBANK.pkl        │
        │  │  • scaler_HDFCBANK.pkl (StandardScaler)  │
        │  │  • (+ all ticker variants)                │
        │  └───────────────────────────────────────┘    │
        │                                                 │
        └────────────────────┬──────────────────────────┘
                             │
        ┌────────────────────┴──────────────────────────┐
        │                                                 │
        ├─► yfinance (Real-time stock data)             
        ├─► FinBERT (News sentiment analysis)            
        ├─► News APIs (Financial news sources)           
        └─► APScheduler (Background training jobs)       
```

### Component Breakdown

#### **Frontend: React Native/Expo**
- **Framework**: React Native with Expo for cross-platform iOS/Android
- **State Management**: React Context (theme, auth)
- **API Layer**: Axios with JWT interceptors & auto-retry logic
- **UI Components**: Custom components for stocks, signals, portfolios
- **Routing**: Expo Router (file-based routing)
- **Storage**: SecureStore for JWT tokens

#### **Backend: FastAPI + Python**
- **Framework**: FastAPI (async REST API)
- **ORM**: None (file-based users.json)
- **Auth**: JWT (HS256) with bearer tokens
- **Async Jobs**: APScheduler for training pipelines
- **Logging**: Custom logger with file output
- **Model Serving**: Pickle (.pkl) and PyTorch (.pth)

#### **AI Models**
- **LSTM Attention** (backend/models/lstm_attention_model.py)
  - Input: 90 days of features
  - Output: 1-day future log return
  - Used for price direction forecasting

- **Transformer** (backend/models/transformer_model.py)
  - Input: 90 days of features
  - Output: 7-day future log returns
  - Used for multi-day forecasting

- **XGBoost Classifier** (backend/training/xgboost_classifier.py)
  - Input: Technical indicators + volume + momentum
  - Output: BUY (2) / HOLD (1) / SELL (0)
  - Used for trading signals

#### **Data Pipeline**
1. **Load**: Historical stock data from CSV (yfinance download)
2. **Clean**: Remove duplicates, handle missing values
3. **Features**: Add 20+ technical indicators, sentiment, risk metrics
4. **Scale**: StandardScaler (fit on train data)
5. **Sequences**: Create (90, feature_dim) input windows
6. **Train/Test**: 80/20 time-based split
7. **Train Models**: LSTM, Transformer, XGBoost
8. **Evaluate**: RMSE, R², Accuracy, F1-Score
9. **Save**: .pth and .pkl files to backend/models/saved_models/

---

## BACKEND API COMPLETE GUIDE

### API Base URL (Production)
```
http://localhost:8000/api/v1
```

### Authentication

All endpoints (except `/health`, `/auth/register`, `/auth/login`) require:
```
Authorization: Bearer <JWT_TOKEN>
```

### Authentication Endpoints

#### **Register** (Create Account)
```
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securePassword123",
  "name": "John Doe"
}

Response (201):
{
  "user_id": "user123",
  "email": "user@example.com",
  "name": "John Doe",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer"
}
```

#### **Login** (Get Token)
```
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securePassword123"
}

Response (200):
{
  "user_id": "user123",
  "email": "user@example.com",
  "name": "John Doe",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer"
}
```

#### **Get Current User**
```
GET /auth/me
Authorization: Bearer <TOKEN>

Response (200):
{
  "user_id": "user123",
  "email": "user@example.com",
  "name": "John Doe"
}
```

---

### Core Prediction Endpoints

#### **Predict Stock Price** (LSTM/Transformer)
```
POST /predict
Authorization: Bearer <TOKEN>
Content-Type: application/json

{
  "symbol": "HDFCBANK",
  "model": "transformer"  // or "lstm"
}

Response (200):
{
  "ticker": "HDFCBANK",
  "model": "transformer",
  "forecast_days": 7,
  "predictions": [
    0.0045,   // Day 1 expected return (0.45%)
    0.0023,   // Day 2 expected return (0.23%)
    -0.0012,  // Day 3 expected return (-0.12%)
    ...
  ],
  "confidence": 0.68,
  "current_price": 1850.50,
  "indicators": {
    "rsi": 65.43,
    "macd": 12.34,
    "macd_signal": 10.56,
    "sma_20": 1840.23,
    "sma_50": 1825.67,
    "bb_high": 1885.23,
    "bb_low": 1810.45,
    "vwap": 1842.56,
    "atr": 18.34
  },
  "signal": "BUY",
  "signal_confidence": 0.72,
  "sentiment": "bullish",
  "risk_level": "medium"
}
```

#### **Get Trading Signals** (XGBoost)
```
POST /signals
Authorization: Bearer <TOKEN>
Content-Type: application/json

{
  "symbol": "INFY"
}

Response (200):
{
  "ticker": "INFY",
  "signal": "BUY",
  "confidence": 0.85,
  "probabilities": {
    "sell": 0.05,
    "hold": 0.10,
    "buy": 0.85
  },
  "explanation": "The AI model predicts the stock will rise by 2.34% on the next trading day. Signal confidence is 85%. Risk level is low. Signal: BUY.",
  "technical_indicators": {
    "rsi": 58.23,
    "momentum": "positive",
    "volume_trend": "increasing"
  }
}
```

#### **Batch Signals** (Multiple Stocks)
```
POST /batch_signals
Authorization: Bearer <TOKEN>
Content-Type: application/json

{
  "symbols": ["HDFCBANK", "INFY", "TCS"]
}

Response (200):
{
  "timestamp": "2026-04-09T10:30:00Z",
  "signals": [
    {
      "ticker": "HDFCBANK",
      "signal": "BUY",
      "confidence": 0.78
    },
    {
      "ticker": "INFY",
      "signal": "HOLD",
      "confidence": 0.65
    },
    {
      "ticker": "TCS",
      "signal": "SELL",
      "confidence": 0.72
    }
  ]
}
```

---

### Portfolio & Risk Endpoints

#### **Portfolio Optimization**
```
POST /portfolio/optimize
Authorization: Bearer <TOKEN>
Content-Type: application/json

{
  "tickers": ["HDFCBANK", "INFY", "TCS"],
  "risk_level": "medium",  // low, medium, high
  "target_return": 0.15
}

Response (200):
{
  "optimal_weights": {
    "HDFCBANK": 0.40,
    "INFY": 0.35,
    "TCS": 0.25
  },
  "expected_return": 0.148,
  "expected_volatility": 0.18,
  "sharpe_ratio": 0.82,
  "max_drawdown": -0.25,
  "portfolio_score": 8.2
}
```

#### **Risk Score**
```
POST /risk/score
Authorization: Bearer <TOKEN>
Content-Type: application/json

{
  "symbol": "RELIANCE",
  "investment_amount": 10000
}

Response (200):
{
  "ticker": "RELIANCE",
  "risk_score": 6.5,  // 0-10 scale
  "risk_level": "medium",
  "var_95": -582.45,  // Value at Risk
  "sharpe_ratio": 0.95,
  "sortino_ratio": 1.34,
  "max_drawdown": -0.32,
  "volatility": 0.22,
  "recommendation": "Consider diversification"
}
```

---

### Sentiment & Explainability Endpoints

#### **Sentiment Analysis**
```
POST /sentiment/analyze
Authorization: Bearer <TOKEN>
Content-Type: application/json

{
  "ticker": "INFY",
  "text": "Infosys reported strong Q4 earnings with 15% growth"
}

Response (200):
{
  "text": "Infosys reported strong Q4 earnings with 15% growth",
  "sentiment": "positive",
  "score": 0.89,
  "keywords": ["strong", "growth", "earnings"],
  "impact": "bullish"
}
```

#### **Model Explainability** (SHAP)
```
GET /explain/{symbol}
Authorization: Bearer <TOKEN>

Response (200):
{
  "ticker": "HDFCBANK",
  "top_features": [
    {
      "feature": "SMA_20",
      "importance": 0.245,
      "value": 1840.23,
      "impact": "positive"
    },
    {
      "feature": "RSI",
      "importance": 0.189,
      "value": 65.43,
      "impact": "positive"
    },
    {
      "feature": "Volume_Change",
      "importance": 0.156,
      "value": 0.12,
      "impact": "negative"
    }
  ],
  "explanation": "Model predicts BUY primarily due to strong SMA-20 momentum and RSI overbought conditions with some volume concern."
}
```

---

### Training Endpoints

#### **Trigger Model Retraining**
```
POST /train
Authorization: Bearer <TOKEN>
Content-Type: application/json

{
  "ticker": "HDFCBANK",
  "model_type": "transformer",  // transformer, lstm, xgboost
  "epochs": 100,
  "early_stopping_patience": 20,
  "learning_rate": 0.0003
}

Response (202):
{
  "job_id": "train_hdfcbank_transformer_1712674200",
  "status": "queued",
  "ticker": "HDFCBANK",
  "model_type": "transformer",
  "message": "Training job queued. Check status with GET /train/<job_id>"
}
```

#### **Check Training Status**
```
GET /train/{job_id}
Authorization: Bearer <TOKEN>

Response (200):
{
  "job_id": "train_hdfcbank_transformer_1712674200",
  "status": "running",
  "progress": 45,
  "ticker": "HDFCBANK",
  "model_type": "transformer",
  "current_epoch": 45,
  "total_epochs": 100,
  "best_val_loss": 0.0234,
  "eta_seconds": 1200
}
```

---

### System Endpoints

#### **Health Check**
```
GET /health

Response (200):
{
  "status": "ok",
  "version": "2.0.0"
}
```

#### **Model Status**
```
GET /models/status
Authorization: Bearer <TOKEN>

Response (200):
{
  "models": {
    "HDFCBANK": {
      "transformer": "trained_2026-04-05",
      "lstm": "trained_2026-04-03",
      "xgboost": "trained_2026-04-08"
    },
    "INFY": {
      "transformer": "trained_2026-04-04",
      "lstm": "trained_2026-04-02",
      "xgboost": "trained_2026-04-08"
    }
  },
  "last_update": "2026-04-09T10:30:00Z"
}
```

---

## FRONTEND (REACT NATIVE/EXPO) COMPLETE GUIDE

### Frontend Structure

```
InvestIQ-App/
├── app/                                  # Expo Router pages
│   ├── (auth)/
│   │   ├── login.js                    # Login screen
│   │   └── register.js                 # Registration screen
│   ├── (tabs)/
│   │   ├── dashboard.js                # Home/Dashboard
│   │   ├── portfolio.js                # Portfolio management
│   │   ├── signals.js                  # Trading signals
│   │   └── profile.js                  # User profile
│   ├── stock/
│   │   └── [symbol].js                 # Stock detail page (dynamic)
│   ├── _layout.js                      # Root layout
│   └── index.js                        # Splash/entry
│
├── src/
│   ├── services/
│   │   └── api.js                      # Axios API client with JWT interceptors
│   ├── components/
│   │   ├── ui.js                       # UI primitives (Button, Input, etc)
│   │   ├── StockSignalCard.js          # Stock signal display component
│   │   ├── news/                       # News-related components
│   │   └── FloatingIQMenu.js           # Floating AI menu
│   ├── context/
│   │   ├── ThemeContext.js             # Dark/light theme
│   │   └── AuthContext.js              # Auth state management
│   ├── hooks/
│   │   ├── useAuth.js                  # Auth hook
│   │   ├── useTheme.js                 # Theme hook
│   │   └── useApi.js                   # API data fetching hook
│   ├── constants/
│   │   ├── theme.js                    # Colors, fonts, sizes
│   │   └── api.js                      # API endpoints, URLs
│   └── types/                          # TypeScript types (if using TS)
│
├── assets/
│   ├── images/                         # App images
│   ├── icons/                          # UI icons
│   └── splash.png                      # Splash screen
│
├── config/
│   └── api.js                          # API configuration & ngrok URL
│
├── package.json
├── app.json                            # Expo config
├── babel.config.js                     # Babel configuration
└── metro.config.js                     # Metro bundler config
```

### Key Frontend Features

#### **1. Authentication Flow**

**Login Screen** (`app/(auth)/login.js`)
```javascript
// User enters email + password
// Calls POST /auth/login
// Receives JWT token
// Saves token to SecureStore
// Redirects to Dashboard
```

**Registration Screen** (`app/(auth)/register.js`)
```javascript
// User enters email + password + name
// Calls POST /auth/register
// Receives JWT token
// Auto-saves token
// Auto-logged in to Dashboard
```

#### **2. API Client** (`src/services/api.js`)

**Key Features**:
- ✅ **JWT Interceptor**: Automatically adds `Authorization: Bearer <token>` to all requests
- ✅ **Auto-Retry**: Falls back to different API URLs (ngrok → LAN IP → localhost)
- ✅ **Error Handling**: Normalizes API errors for UI display
- ✅ **Token Management**: Secure storage with expo-secure-store
- ✅ **Timeout Handling**: 30-second request timeout with retry logic

**Example Usage**:
```javascript
import api from './src/services/api.js';

// API call automatically includes JWT token
const response = await api.post('/predict', {
  symbol: 'HDFCBANK',
  model: 'transformer'
});

// Handle response
console.log(response.data.signal);  // "BUY"
console.log(response.data.confidence);  // 0.78
```

#### **3. Dashboard Screen** (`app/(tabs)/dashboard.js`)

Displays:
- 🔝 Top 5 stocks with signals
- 📊 Portfolio summary
- 📈 Recent performance
- 🎯 Recommended actions

**Data Flow**:
```
User Opens App
    ↓
Check JWT token (SecureStore)
    ↓
If no token → Show Login
    ↓
Fetch Batch Signals: POST /batch_signals
    ↓
Display: [HDFCBANK: BUY 78%, INFY: HOLD 65%, ...]
    ↓
User taps stock → Navigate to /stock/[symbol]
```

#### **4. Stock Detail Screen** (`app/stock/[symbol].js`)

Displays per stock:
- 💰 Current price + 7-day forecast
- 📊 Technical indicators (RSI, MACD, SMA, etc)
- 🎯 Trading signal (BUY/SELL/HOLD) with confidence
- 📉 Historical chart (optional)
- 📰 Recent news headlines
- ⚠️ Risk metrics (VaR, Sharpe ratio)

**Data Flow**:
```
User taps "HDFCBANK" card
    ↓
Navigate to /stock/HDFCBANK
    ↓
Fetch prediction: POST /predict {symbol: 'HDFCBANK'}
    ↓
Display: Forecast + Indicators + Signal
    ↓
Fetch signals: POST /signals {symbol: 'HDFCBANK'}
    ↓
Display: BUY signal with 78% confidence
```

#### **5. Portfolio Screen** (`app/(tabs)/portfolio.js`)

Features:
- 📋 Holdings list (mock or real from backend)
- 📊 Portfolio composition (pie chart)
- 📈 Portfolio performance (YTD, 1M, 3M)
- 💡 Optimization suggestions

**Data Flow**:
```
User opens Portfolio
    ↓
Fetch portfolio: GET /portfolio/{user_id}
    ↓
Calculate: Total value, allocations, returns
    ↓
Show optimization: POST /portfolio/optimize
    ↓
Display: "Rebalance to 40% HDFCBANK, 35% INFY, 25% TCS"
```

#### **6. Signals Screen** (`app/(tabs)/signals.js`)

Displays:
- 🟢 BUY opportunities (stocks predicted to rise)
- 🔴 SELL signals (stocks predicted to fall)
- 🟡 HOLD recommendations (neutral)
- ✅ Signal accuracy metrics

**Data Flow**:
```
User opens Signals
    ↓
Fetch batch signals: POST /batch_signals {symbols: [all]}
    ↓
Group by signal type: BUY | HOLD | SELL
    ↓
Display ranked by confidence
    ↓
Show accuracy metrics: "74% of BUY signals were correct in last 30 days"
```

### Important Frontend Configuration Files

#### **config/api.js** (API Configuration)
```javascript
// This file defines where the frontend connects to backend

export const API_BASE_URL = 'https://ngrok-url.ngrok-free.dev/api/v1';
// or localhost for simulator:
// export const API_BASE_URL = 'http://localhost:8000/api/v1';

export const API_PORT = 8000;

// Fallback URLs tried in order:
export const API_BASE_URL_CANDIDATES = [
  'https://your-ngrok-url.ngrok-free.dev/api/v1',  // ngrok (mobile devices)
  'http://192.168.x.x:8000/api/v1',               // LAN IP
  'http://localhost:8000/api/v1',                 // localhost (simulator)
];

export const REQUEST_TIMEOUT_MS = 30000;  // 30 seconds
export const NETWORK_ERROR_MESSAGE = 'Network error. Check your connection and API availability.';
```

#### **app.json** (Expo Configuration)
```json
{
  "expo": {
    "name": "InvestIQ",
    "slug": "investiq",
    "version": "2.0.0",
    "orientation": "portrait",
    "icon": "./assets/image.png",
    "splash": {
      "image": "./assets/splash.png",
      "resizeMode": "contain",
      "backgroundColor": "#ffffff"
    },
    "assetBundlePatterns": [
      "**/*"
    ],
    "ios": {
      "supportsTablet": true,
      "bundleIdentifier": "com.investiq.app"
    },
    "android": {
      "package": "com.investiq.app"
    },
    "web": {
      "favicon": "./assets/favicon.png"
    },
    "plugins": [
      "expo-secure-store"
    ]
  }
}
```

---

## NGROK SETUP & TUNNEL CONFIGURATION

### What is Ngrok?

**Ngrok** creates a secure tunnel exposing your local backend to the internet. This allows mobile devices and external services to access your local `http://localhost:8000` backend without being on the same network.

```
Mobile Device (Expo App)
         ↓
    (HTTPS over public internet)
         ↓
    ngrok tunnel
         ↓
    http://localhost:8000 (your backend)
```

### Why Ngrok?

| Scenario | Without Ngrok ❌ | With Ngrok ✅ |
|----------|-----------------|--------------|
| Desktop simulator | Works | Works |
| Local mobile testing | ❌ Can't reach localhost | ✅ Uses ngrok URL |
| Production testing | ❌ Not possible | ✅ Test from anywhere |
| Demo/sharing | ❌ Network blocked | ✅ Share ngrok URL |

### Complete Ngrok Setup

#### **Step 1: Create Ngrok Account**
1. Visit https://ngrok.com
2. Sign up for free account
3. Get your **Auth Token** from dashboard

#### **Step 2: Install Ngrok**

**Windows**:
```powershell
# Download from: https://ngrok.com/download
# Add to PATH

ngrok --version
# Output: ngrok version 3.x.x
```

**macOS**:
```bash
brew install ngrok
```

**Linux**:
```bash
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-amd64.zip | tar xz
sudo mv ngrok /usr/local/bin
```

#### **Step 3: Authenticate Ngrok**

```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
```

You can find YOUR_AUTH_TOKEN at: https://dashboard.ngrok.com/auth/your-authtoken

#### **Step 4: Start Backend Server**

Ensure backend is running on port 8000 with `host="0.0.0.0"`:

```bash
# Option 1: Using batch script
cd d:\InvestIQ-main
run_app.bat

# Option 2: Manual
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify it's running:
```bash
curl http://localhost:8000/api/v1/health
# Response: {"status":"ok","version":"2.0.0"}
```

#### **Step 5: Start Ngrok Tunnel**

```bash
ngrok http 8000
```

You'll see output like:
```
ngrok                                    (Ctrl+C to quit)

Session Status                online
Session Expires               2 hours, 59 minutes
Version                       3.3.0
Region                        us (Los Angeles)
Forwarding                    https://abc123-456-def.ngrok-free.dev -> http://localhost:8000
Forwarding                    http://abc123-456-def.ngrok-free.dev -> http://localhost:8000

Web Interface               http://127.0.0.1:4040
```

**Copy the HTTPS URL**: `https://abc123-456-def.ngrok-free.dev`

#### **Step 6: Update Frontend Configuration**

Edit `InvestIQ-App/config/api.js`:

```javascript
export const API_BASE_URL = 'https://abc123-456-def.ngrok-free.dev/api/v1';
// Replace with YOUR ngrok URL from Step 5
```

#### **Step 7: Update API_BASE_URL_CANDIDATES**

Also in `InvestIQ-App/config/api.js`:

```javascript
export const API_BASE_URL_CANDIDATES = [
  'https://abc123-456-def.ngrok-free.dev/api/v1',  // PRIMARY
  'http://192.168.1.X:8000/api/v1',               // Get IP: ipconfig
  'http://localhost:8000/api/v1',                 // Fallback
];
```

#### **Step 8: Start Expo App**

```bash
cd InvestIQ-App
npm install
npx expo start --tunnel -c
```

Scan QR code with Expo Go app on your phone. The app will now:
1. Try ngrok URL first
2. Fall back to LAN IP if ngrok unavailable
3. Fall back to localhost on simulator

#### **Step 9: Monitor Traffic (Optional)**

Open ngrok Inspector in browser:
```
http://127.0.0.1:4040
```

You'll see all requests/responses flowing through the tunnel in real-time.

### Ngrok Monitoring & Debugging

#### **Check Ngrok Status**

```bash
curl http://127.0.0.1:4040/api/tunnels
```

Response shows all active tunnels.

#### **Test Through Ngrok Tunnel**

```bash
# Health check
curl https://abc123-456-def.ngrok-free.dev/api/v1/health

# Login (no auth required)
curl -X POST https://abc123-456-def.ngrok-free.dev/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'

# Predict (requires auth)
curl -X POST https://abc123-456-def.ngrok-free.dev/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"symbol":"HDFCBANK","model":"transformer"}'
```

#### **Ngrok URL Expires**

⚠️ Default ngrok URLs expire after **2 hours**. To get a **stable URL**:

1. Upgrade to ngrok Pro (paid) for permanent URLs
2. OR restart ngrok every 2 hours (development only)
3. OR use environment variable for URL updates

**To auto-update during dev** (Windows PowerShell):

```powershell
# Save script as: update_ngrok.ps1
$ngrok_url = (Invoke-WebRequest http://127.0.0.1:4040/api/tunnels | ConvertFrom-Json).tunnels[0].public_url

# Update frontend config
$config_file = "InvestIQ-App/config/api.js"
$new_url = "export const API_BASE_URL = '$ngrok_url/api/v1';"

(Get-Content $config_file) -replace "export const API_BASE_URL = '.*'", $new_url | Set-Content $config_file

Write-Host "Updated API base URL to: $ngrok_url/api/v1"
```

Run this every 2 hours:
```powershell
.\update_ngrok.ps1
```

---

## INSTALLATION & SETUP

### Prerequisites

- ✅ Python 3.9+ (Backend)
- ✅ Node.js 16+ (Frontend)
- ✅ Git
- ✅ Virtual environment (venv)
- ✅ ngrok account (for mobile testing)

### Option 1: Automated Setup (Windows)

```bash
cd d:\InvestIQ-main

# Setup backend
cd backend
setup_env.bat
verify_setup.py

# Setup frontend
cd ..
cd InvestIQ-App
npm install
```

### Option 2: Manual Setup

#### **Backend Setup**

```bash
# Navigate to backend
cd d:\InvestIQ-main\backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

#### **Frontend Setup**

```bash
# Navigate to frontend
cd d:\InvestIQ-main\InvestIQ-App

# Install dependencies
npm install

# Or with specific versions
npm install --legacy-peer-deps
```

### Verify Installation

```bash
# Backend health
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/api/v1/health

# Frontend (in separate terminal)
cd InvestIQ-App
npm start
# Scan QR code with Expo Go
```

---

## RUNNING THE APPLICATION

### Quick Start (5 minutes)

#### **Terminal 1: Start Backend**
```bash
cd d:\InvestIQ-main
run_app.bat
# Output: 
# Uvicorn running on http://0.0.0.0:8000
# API docs: http://localhost:8000/docs
```

#### **Terminal 2: Start Ngrok**
```bash
ngrok http 8000
# Output:
# Forwarding: https://abc123.ngrok-free.dev -> http://localhost:8000
# Copy the URL!
```

#### **Terminal 3: Update Frontend**
```bash
# Edit InvestIQ-App/config/api.js
# Change: export const API_BASE_URL = 'https://abc123.ngrok-free.dev/api/v1'
# Save file
```

#### **Terminal 4: Start Frontend**
```bash
cd InvestIQ-App
npm start
# Scan QR code with Expo Go on phone
```

### Complete Startup Process

```bash
# Step 1: Open Terminal (PowerShell)
cd d:\InvestIQ-main

# Step 2: Start backend
run_app.bat
# Wait for: "Uvicorn running on http://0.0.0.0:8000"

# Step 3: Open NEW Terminal
ngrok http 8000
# Copy HTTPS URL: https://abc123...

# Step 4: Edit config/api.js with ngrok URL
notepad InvestIQ-App/config/api.js
# Update: API_BASE_URL = 'https://abc123.ngrok-free.dev/api/v1'
# Save & Close

# Step 5: Open NEW Terminal
cd InvestIQ-App
npm start

# Step 6: Scan QR code with Expo Go app
# You're connected!
```

### Testing API Directly

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Register user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123",
    "name": "Test User"
  }'

# Login and get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123"
  }'
# Save the token from response

# Make authenticated prediction
TOKEN="eyJ0eXAiOiJKV1QiLC..."
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "symbol": "HDFCBANK",
    "model": "transformer"
  }'
```

---

## DATA PIPELINE & MODELS

### Training Pipeline (PHASE 2)

#### **Step 1: Data Collection**
```bash
# Download historical stock data via yfinance
# 25 years of daily OHLCV data
# Files: backend/data/stock_data/HDFCBANK.csv, etc.
```

#### **Step 2: Data Cleaning**
```python
from backend.preprocessing.cleaning import clean_data

df = clean_data(df)
# Removes duplicates, fills missing values, validates columns
```

#### **Step 3: Feature Engineering**
```python
from backend.features.indicators import add_technical_indicators

df = add_technical_indicators(df)
# Adds 13 technical indicators:
# SMA_20, SMA_50, RSI, MACD, MACD_Signal, Bollinger Bands,
# ATR, VWAP, Log_Return, Volume_Change, Rolling_Volatility,
# Market_Correlation (vs NIFTY-50), Sentiment, Macro_Score
```

#### **Step 4: Scaling & Sequences**
```python
from backend.preprocessing.scaling import StockScaler
from backend.utils.data_pipeline import create_sequences_v2

scaler = StockScaler()
X_scaled = scaler.fit_transform(X_train)

X, y = create_sequences_v2(X_scaled, y_scaled, seq_length=90, forecast_horizon=7)
# X shape: (samples, 90, features)
# y shape: (samples, 7)
```

#### **Step 5: Train/Test Split**
```python
from backend.utils.data_pipeline import train_test_time_split

X_train, X_test, y_train, y_test = train_test_time_split(X, y, train_ratio=0.8)
# Time-based split (no shuffling to maintain temporal integrity)
```

#### **Step 6: Model Training**

**Transformer**:
```python
from backend.models.transformer_model import TimeSeriesTransformer

model = TimeSeriesTransformer(...)
model.train()
# Trains for 100+ epochs with early stopping (patience=20)
# Saves best weights to: backend/models/saved_models/transformer_HDFCBANK.pth
```

**XGBoost**:
```python
from backend.training.xgboost_classifier import XGBoostClassificationPipeline

pipeline = XGBoostClassificationPipeline(buy_threshold=0.005, sell_threshold=-0.005)
pipeline.train_model(X_train, y_train, X_test, y_test)
# Trains for 200 rounds with early stopping
# Saves to: backend/models/saved_models/xgboost_classifier_HDFCBANK.pkl
```

### Model Architectures

#### **Transformer (Price Forecasting)**
- Input: 90 days × 21 features
- Architecture: Self-attention + FFN layers
- Output: 7-day log returns
- Trained on: 80% historical data
- Validation: Early stopping on 20% validation loss

#### **XGBoost (Trading Signals)**
- Input: Technical indicators + volume + momentum
- Output: BUY (2) / HOLD (1) / SELL (0)
- Parameters: 200 trees, max_depth=5, learning_rate=0.05
- Threshold: BUY if return > 0.5%, SELL if return < -0.5%

### Inference Pipeline

When a user requests a prediction:

```
1. Load historical CSV (backend/data/stock_data/HDFCBANK.csv)
   ↓
2. Add technical indicators (add_technical_indicators)
   ↓
3. Load scaler (backend/models/saved_models/scaler_HDFCBANK.pkl)
   ↓
4. Scale features using fitted scaler
   ↓
5. Create sequence (last 90 days)
   ↓
6. Load Transformer model (.pth file)
   ↓
7. Run inference: model(sequence)
   ↓
8. Get 7-day forecast + confidence from Monte Carlo dropout
   ↓
9. Load XGBoost model (.pkl file)
   ↓
10. Get BUY/SELL/HOLD signal + probability
    ↓
11. Combine signals + add indicators
    ↓
12. Return to API as JSON
```

### Performance Metrics

Models are evaluated on:

| Metric | LSTM | Transformer | XGBoost |
|--------|------|-------------|---------|
| RMSE (lower is better) | 0.023 | 0.021 | N/A |
| R² (higher is better) | 0.28 | 0.35 | N/A |
| Directional Accuracy | 58% | 62% | N/A |
| Buy Signal Accuracy | N/A | N/A | 68% |
| Precision (Buy) | N/A | N/A | 0.72 |
| Recall (Buy) | N/A | N/A | 0.65 |
| F1-Score | N/A | N/A | 0.68 |

---

## TROUBLESHOOTING

### Frontend Issues

#### **1. App Won't Connect to Backend**

**Error**: "Network error. Check your API availability."

**Solutions**:
```bash
# Check if backend is running
curl http://localhost:8000/api/v1/health

# Check ngrok tunnel
ngrok http 8000

# Verify frontend config
cat InvestIQ-App/config/api.js
# Should have correct ngrok URL

# Clear app cache
npx expo start --tunnel -c
```

#### **2. Login Fails with 401**

**Error**: "Invalid credentials"

**Solutions**:
```bash
# Check if user exists
# Login with correct email/password

# Clear stored token
# In Expo console, clear SecureStore

# Verify backend auth service
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer INVALID_TOKEN"
# Should return 401
```

#### **3. Expo App Disconnects from Tunnel**

**Solutions**:
```bash
# Restart ngrok (URL expired)
ngrok http 8000

# Update frontend config with new URL

# Refresh Expo app (tap R)

# Check network connectivity
ping -c 10 8.8.8.8
```

### Backend Issues

#### **1. Backend Won't Start**

**Error**: `Port 8000 already in use`

**Solution**:
```bash
# Find process using port 8000
netstat -ano | findstr ":8000"

# Kill process
taskkill /PID <PID> /F

# Start backend again
python -m uvicorn backend.app.main:app --port 8000 --reload
```

#### **2. Models Missing**

**Error**: `FileNotFoundError: No such file: 'transformer_HDFCBANK.pth'`

**Solutions**:
```bash
# Train models first
cd d:\InvestIQ-main
run_training.bat
# or
python batch_train_optimized.py HDFCBANK

# Verify models exist
ls backend/models/saved_models/
```

#### **3. API Token Expired**

**Error**: `{"detail":"Invalid or expired token"}`

**Solution**:
```bash
# Login again to get new token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'

# Use new token in requests
```

### Data Pipeline Issues

#### **1. "Insufficient data" Error**

**Error**: `StockPredictorException: Insufficient data after cleaning`

**Solution**:
```bash
# Ensure CSV has enough history
wc -l backend/data/stock_data/HDFCBANK.csv
# Should be > 1000 rows

# Download fresh data
python backend/data/update_stock_data.py HDFCBANK
```

#### **2. NaN Values in Features**

**Error**: `Features contain NaN values`

**Solution**:
```python
# The cleaning pipeline handles this automatically
# But if persists, check specific indicators

import pandas as pd
df = pd.read_csv('backend/data/stock_data/HDFCBANK.csv')

# Check for NaN
print(df.isnull().sum())

# Fill missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
```

### Ngrok Issues

#### **1. Ngrok Won't Connect**

**Error**: `Error: certificate verify failed`

**Solution**:
```bash
# Update ngrok
ngrok update

# Or manually download from: https://ngrok.com/download
```

#### **2.**ngrok URL Expired**

**Solution**: URLs expire after 2 hours (free plan)

```bash
# Restart ngrok
ngrok http 8000

# Update frontend config with new URL
notepad InvestIQ-App/config/api.js
```

#### **3. Can't Access Ngrok URL from Phone**

**Solution**:
```bash
# Check ngrok is running
ngrok http 8000

# Try direct ngrok URL in phone browser
# https://abc123.ngrok-free.dev/api/v1/health

# If blocked, check:
# 1. Ngrok is authenticated: ngrok config show
# 2. Firewall isn't blocking
# 3. Your internet has outbound access to ngrok servers
```

---

## QUICK REFERENCE

### Start Commands

```bash
# Start all (4 terminals)
Terminal 1: cd d:\InvestIQ-main && run_app.bat
Terminal 2: ngrok http 8000
Terminal 3: (Edit InvestIQ-App/config/api.js with ngrok URL)
Terminal 4: cd InvestIQ-App && npm start
```

### Training Commands

```bash
# Train all models
python batch_train_optimized.py

# Train specific ticker
python batch_train_optimized.py HDFCBANK

# Run evaluation
python backend/scripts/run_evaluation.py
```

### API Examples

```bash
# Health
curl http://localhost:8000/api/v1/health

# Predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"HDFCBANK","model":"transformer"}'

# Signals
curl -X POST http://localhost:8000/api/v1/signals \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"INFY"}'
```

### File Locations

```
Backend:           d:\InvestIQ-main\backend\
Frontend:          d:\InvestIQ-main\InvestIQ-App\
Models (trained):  d:\InvestIQ-main\backend\models\saved_models\
Stock data:        d:\InvestIQ-main\backend\data\stock_data\
API Docs:          http://localhost:8000/docs
Ngrok Dashboard:   http://127.0.0.1:4040
```

---

## SUMMARY

**InvestIQ** is a complete, production-ready stock prediction platform with:

✅ **Rock-solid Backend**: FastAPI with JWT auth, model inference, async jobs  
✅ **Beautiful Frontend**: React Native/Expo for iOS & Android  
✅ **Powerful AI**: LSTM, Transformer, XGBoost ensemble  
✅ **Secure Tunnel**: ngrok for mobile device connectivity  
✅ **Full Documentation**: Complete API + frontend guides  

**To get started**:
1. Setup backend (`setup_env.bat`)
2. Start backend (`run_app.bat`)
3. Start ngrok tunnel (`ngrok http 8000`)
4. Update frontend config with ngrok URL
5. Start frontend (`npm start` in InvestIQ-App)
6. Scan QR code in Expo Go

**Questions?** Check the `/docs/` folder for detailed guides per topic.

---

**Happy Trading! 🚀📈**
