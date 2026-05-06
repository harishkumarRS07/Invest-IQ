# ⚡ InvestIQ API Quick Reference Guide

**All API endpoints with curl examples - Copy & Paste Ready!**

---

## 📋 Table of Contents

1. [Health & Status](#health--status)
2. [Authentication](#authentication)
3. [Predictions](#predictions)
4. [Trading Signals](#trading-signals)
5. [Portfolio Management](#portfolio-management)
6. [Risk & Analytics](#risk--analytics)
7. [Sentiment Analysis](#sentiment-analysis)
8. [Model Management](#model-management)
9. [Testing Checklist](#testing-checklist)

---

## HEALTH & STATUS

### Health Check
```bash
curl -X GET http://localhost:8000/api/v1/health

# Response (200)
{
  "status": "ok",
  "version": "2.0.0"
}
```

### Model Status
```bash
TOKEN="your_jwt_token_here"

curl -X GET http://localhost:8000/api/v1/models/status \
  -H "Authorization: Bearer $TOKEN"

# Response (200)
{
  "models": {
    "HDFCBANK": {
      "transformer": "trained_2026-04-05",
      "lstm": "trained_2026-04-03",
      "xgboost": "trained_2026-04-08"
    },
    "INFY": {...}
  },
  "last_update": "2026-04-09T10:30:00Z"
}
```

---

## AUTHENTICATION

### Step 1: Register User
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "trader@example.com",
    "password": "SecurePass123!",
    "name": "John Trader"
  }'

# Response (201)
{
  "user_id": "user_abc123",
  "email": "trader@example.com",
  "name": "John Trader",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

### Step 2: Login User
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "trader@example.com",
    "password": "SecurePass123!"
  }'

# Response (200)
{
  "user_id": "user_abc123",
  "email": "trader@example.com",
  "name": "John Trader",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}

# ✅ Save this token! Use it in all other requests!
```

### Step 3: Get Current User
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."

curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"

# Response (200)
{
  "user_id": "user_abc123",
  "email": "trader@example.com",
  "name": "John Trader"
}
```

---

## PREDICTIONS

### Single Stock Prediction (Transformer)
```bash
TOKEN="your_token_here"

curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "HDFCBANK",
    "model": "transformer"
  }'

# Response (200)
{
  "ticker": "HDFCBANK",
  "model": "transformer",
  "forecast_days": 7,
  "predictions": [
    0.0045,    # Day 1: +0.45%
    0.0023,    # Day 2: +0.23%
    -0.0012,   # Day 3: -0.12%
    0.0089,
    0.0012,
    -0.0034,
    0.0056
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

### Single Stock Prediction (LSTM)
```bash
TOKEN="your_token_here"

curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "INFY",
    "model": "lstm"
  }'

# Response format same as transformer, but using LSTM model
```

---

## TRADING SIGNALS

### Get BUY/SELL/HOLD Signal (Single Stock)
```bash
TOKEN="your_token_here"

curl -X POST http://localhost:8000/api/v1/signals \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "INFY"
  }'

# Response (200)
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

### Get Signals for Multiple Stocks (Batch)
```bash
TOKEN="your_token_here"

curl -X POST http://localhost:8000/api/v1/batch_signals \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["HDFCBANK", "INFY", "TCS", "ICICIBANK", "RELIANCE"]
  }'

# Response (200)
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
    },
    {
      "ticker": "ICICIBANK",
      "signal": "BUY",
      "confidence": 0.81
    },
    {
      "ticker": "RELIANCE",
      "signal": "HOLD",
      "confidence": 0.68
    }
  ]
}
```

### Signal Interpretation
- 🟢 **BUY** (confidence > 0.70): Stock likely to rise ↗️
- 🟡 **HOLD** (0.40 < confidence < 0.70): Wait for clarity ➡️
- 🔴 **SELL** (confidence > 0.70): Stock likely to fall ↘️

---

## PORTFOLIO MANAGEMENT

### Get Portfolio Summary
```bash
TOKEN="your_token_here"

curl -X GET http://localhost:8000/api/v1/portfolio \
  -H "Authorization: Bearer $TOKEN"

# Response (200)
{
  "holdings": [
    {
      "symbol": "HDFCBANK",
      "quantity": 10,
      "avg_price": 1800,
      "current_price": 1850,
      "value": 18500,
      "gain_loss": 500,
      "gain_loss_pct": 2.78
    },
    {
      "symbol": "INFY",
      "quantity": 20,
      "avg_price": 1600,
      "current_price": 1620,
      "value": 32400,
      "gain_loss": 400,
      "gain_loss_pct": 1.25
    }
  ],
  "total_value": 50900,
  "total_invested": 50000,
  "total_gain_loss": 900,
  "total_gain_loss_pct": 1.80
}
```

### Optimize Portfolio
```bash
TOKEN="your_token_here"

curl -X POST http://localhost:8000/api/v1/portfolio/optimize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["HDFCBANK", "INFY", "TCS"],
    "risk_level": "medium",
    "target_return": 0.15
  }'

# Response (200)
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
  "portfolio_score": 8.2,
  "recommendation": "Balanced portfolio with good risk-return profile"
}
```

**Rebalance instruction**: 
- If portfolio worth ₹100,000:
  - HDFCBANK: ₹40,000 (40%)
  - INFY: ₹35,000 (35%)
  - TCS: ₹25,000 (25%)

---

## RISK & ANALYTICS

### Get Risk Score for Stock
```bash
TOKEN="your_token_here"

curl -X POST http://localhost:8000/api/v1/risk/score \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "investment_amount": 10000
  }'

# Response (200)
{
  "ticker": "RELIANCE",
  "risk_score": 6.5,      # 0-10 scale
  "risk_level": "medium",
  "var_95": -582.45,      # Max loss at 95% confidence
  "sharpe_ratio": 0.95,   # Return per unit risk
  "sortino_ratio": 1.34,  # Return per downside risk
  "max_drawdown": -0.32,  # Worst decline from peak
  "volatility": 0.22,     # Daily price fluctuation
  "recommendation": "Consider diversification"
}
```

**Risk Levels**:
- 0-3: Low risk (conservative stocks)
- 4-6: Medium risk (balanced)
- 7-10: High risk (volatile, high growth)

### Risk Score Interpretation
- **VaR (Value at Risk)**: On ₹10,000 investment, 5% chance of loss > ₹582
- **Sharpe Ratio**: 0.95 means 0.95 units of return per unit of risk
- **Max Drawdown**: Stock can drop up to 32% from peak
- **Volatility**: Stock moves ~22% annually on average

---

## SENTIMENT ANALYSIS

### Analyze Sentiment of News Text
```bash
TOKEN="your_token_here"

curl -X POST http://localhost:8000/api/v1/sentiment/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "INFY",
    "text": "Infosys reported strong Q4 earnings with 15% revenue growth and raised FY25 guidance"
  }'

# Response (200)
{
  "text": "Infosys reported strong Q4 earnings with 15% revenue growth and raised FY25 guidance",
  "sentiment": "positive",
  "score": 0.89,          # 0 to 1 scale
  "keywords": ["strong", "growth", "raised", "earnings"],
  "impact": "bullish"     # How it affects stock price
}
```

**Sentiment Scores**:
- 0.0 - 0.3: Negative 🔴
- 0.3 - 0.7: Neutral 🟡
- 0.7 - 1.0: Positive 🟢

---

## MODEL MANAGEMENT

### Get Model Explainability (SHAP)
```bash
TOKEN="your_token_here"

curl -X GET http://localhost:8000/api/v1/explain/HDFCBANK \
  -H "Authorization: Bearer $TOKEN"

# Response (200)
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
    },
    {
      "feature": "MACD",
      "importance": 0.134,
      "value": 12.34,
      "impact": "positive"
    }
  ],
  "explanation": "Model predicts BUY primarily due to strong SMA-20 momentum (24.5% importance) and RSI overbought conditions (18.9% importance) with some volume concern."
}
```

### Trigger Model Retraining
```bash
TOKEN="your_token_here"

curl -X POST http://localhost:8000/api/v1/train \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "HDFCBANK",
    "model_type": "transformer",
    "epochs": 100,
    "early_stopping_patience": 20,
    "learning_rate": 0.0003
  }'

# Response (202 Accepted)
{
  "job_id": "train_hdfcbank_transformer_1712674200",
  "status": "queued",
  "ticker": "HDFCBANK",
  "model_type": "transformer",
  "message": "Training job queued. Check status with GET /train/<job_id>"
}
```

### Check Training Progress
```bash
TOKEN="your_token_here"
JOB_ID="train_hdfcbank_transformer_1712674200"

curl -X GET http://localhost:8000/api/v1/train/$JOB_ID \
  -H "Authorization: Bearer $TOKEN"

# Response (200)
{
  "job_id": "train_hdfcbank_transformer_1712674200",
  "status": "running",     # queued | running | completed | failed
  "progress": 45,          # 0-100%
  "ticker": "HDFCBANK",
  "model_type": "transformer",
  "current_epoch": 45,
  "total_epochs": 100,
  "best_val_loss": 0.0234,
  "eta_seconds": 1200      # ~20 minutes remaining
}
```

---

## ⚠️ ERROR RESPONSES

### 401 Unauthorized
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"HDFCBANK","model":"transformer"}'

# Response (401)
{
  "detail": "Missing or invalid Authorization header"
}
```
**Solution**: Add valid JWT token: `-H "Authorization: Bearer $TOKEN"`

### 404 Not Found
```bash
# Response (404)
{
  "detail": "No data found for UNKNOWN_TICKER"
}
```
**Solution**: Use valid ticker: HDFCBANK, INFY, TCS, ICICIBANK, RELIANCE

### 422 Unprocessable Entity
```bash
# Response (422)
{
  "detail": [
    {
      "loc": ["body", "symbol"],
      "msg": "Field required",
      "type": "value_error.missing"
    }
  ]
}
```
**Solution**: Ensure all required fields are present in request body

### 500 Internal Server Error
```bash
# Response (500)
{
  "detail": "Internal server error"
}
```
**Solution**: Check backend logs, ensure models are trained, data files exist

---

## TESTING CHECKLIST

### ✅ Quick Test Flow (5 minutes)

```bash
# 1. Health check
curl http://localhost:8000/api/v1/health

# 2. Register
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test123","name":"Test"}' \
  | jq -r '.token')

# 3. Get current user
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"

# 4. Get predictions
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"HDFCBANK","model":"transformer"}'

# 5. Get signals
curl -X POST http://localhost:8000/api/v1/signals \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"INFY"}'

# 6. Get batch signals
curl -X POST http://localhost:8000/api/v1/batch_signals \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbols":["HDFCBANK","INFY","TCS"]}'

# All passed? ✅ API is working!
```

### ✅ Endpoint Checklist

- [ ] GET `/health` returns 200
- [ ] POST `/auth/register` creates user
- [ ] POST `/auth/login` returns token
- [ ] GET `/auth/me` returns user info
- [ ] POST `/predict` returns forecast
- [ ] POST `/signals` returns signal
- [ ] POST `/batch_signals` returns all signals
- [ ] POST `/portfolio/optimize` returns weights
- [ ] POST `/risk/score` returns risk metrics
- [ ] POST `/sentiment/analyze` returns sentiment
- [ ] GET `/explain/{symbol}` returns features
- [ ] GET `/models/status` returns model info

---

## 🎯 COMMON WORKFLOWS

### Workflow 1: Daily Trading Signals
```bash
# Get signals for all stocks
TOKEN="your_token"

curl -X POST http://localhost:8000/api/v1/batch_signals \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbols":["HDFCBANK","INFY","TCS","ICICIBANK","RELIANCE"]}' \
  | jq '.signals | sort_by(.confidence) | reverse'

# Shows signals ranked by confidence (highest first)
```

### Workflow 2: Analyze Single Stock
```bash
TOKEN="your_token"
SYMBOL="HDFCBANK"

# Get prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"symbol\":\"$SYMBOL\",\"model\":\"transformer\"}"

# Get signal
curl -X POST http://localhost:8000/api/v1/signals \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"symbol\":\"$SYMBOL\"}"

# Get risk
curl -X POST http://localhost:8000/api/v1/risk/score \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"symbol\":\"$SYMBOL\",\"investment_amount\":10000}"
```

### Workflow 3: Portfolio Rebalancing
```bash
TOKEN="your_token"

# Get optimization
curl -X POST http://localhost:8000/api/v1/portfolio/optimize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["HDFCBANK","INFY","TCS"],
    "risk_level": "medium",
    "target_return": 0.15
  }' \
  | jq '.optimal_weights'

# Apply new weights to your portfolio
```

---

## 📱 FRONTEND INTEGRATION EXAMPLE

```javascript
// React Native / Expo Example
import api from './src/services/api.js';

// Get signals
const fetchSignals = async () => {
  try {
    const response = await api.post('/batch_signals', {
      symbols: ['HDFCBANK', 'INFY', 'TCS']
    });
    
    console.log('Signals:', response.data.signals);
    // Display to user
    
  } catch (error) {
    console.error('Error:', error.message);
  }
};

// Get prediction
const fetchPrediction = async (symbol) => {
  try {
    const response = await api.post('/predict', {
      symbol: symbol,
      model: 'transformer'
    });
    
    console.log('Prediction:', response.data);
    // Display forecast + indicators
    
  } catch (error) {
    console.error('Error:', error.message);
  }
};
```

---

## 🤝 SUPPORT

**API Swagger UI**: http://localhost:8000/docs  
**Full Documentation**: See `FULL_DOCUMENTATION.md`  
**Backend Repo**: `d:\InvestIQ-main\backend\`  
**Frontend Repo**: `d:\InvestIQ-main\InvestIQ-App\`

Happy Trading! 📈✨
