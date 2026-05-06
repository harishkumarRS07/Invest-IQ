# 🧠 InvestIQ — AI-Powered Financial Intelligence Platform

InvestIQ is an AI-driven stock prediction and portfolio management platform built with a Python/FastAPI backend and a React Native (Expo) mobile frontend.

---

## 📁 Project Structure

```
InvestIQ-main/
├── backend/                  ← Python AI/API backend
│   ├── app/                  ← FastAPI server (main, routes, auth, schemas)
│   ├── core/                 ← Config, logging, exceptions
│   ├── data/                 ← Data fetching & stock CSVs
│   ├── evaluation/           ← Model evaluation utilities
│   ├── explainability/       ← SHAP / AI explainability
│   ├── features/             ← Indicators, sentiment, risk, portfolio
│   ├── inference/            ← Prediction engine
│   ├── models/               ← Trained model files (.pth, .pkl)
│   ├── preprocessing/        ← Data cleaning & scaling
│   ├── scripts/              ← Utility scripts (demo, train, predict, etc.)
│   ├── tests/                ← Unit tests
│   ├── training/             ← Model training pipelines
│   ├── utils/                ← Shared utilities
│   ├── requirements.txt      ← Python dependencies
│   ├── run_server.bat        ← Starts backend API server
│   └── setup_env.bat         ← Creates venv & installs dependencies
│
├── InvestIQ-App/             ← React Native / Expo mobile frontend
│   ├── app/                  ← Expo Router pages & layouts
│   │   ├── (auth)/           ← Login, register screens
│   │   ├── (tabs)/           ← Main tab navigation
│   │   └── stock/            ← Stock detail screen
│   ├── src/                  ← Source code
│   │   ├── components/       ← Reusable UI components
│   │   ├── constants/        ← App constants & theme
│   │   ├── context/          ← React context (theme, auth)
│   │   ├── hooks/            ← Custom React hooks
│   │   └── services/         ← API service layer
│   ├── assets/               ← Images, icons, splash
│   └── package.json
│
├── Dockerfile                ← Docker container config
├── docker-compose.yml        ← Docker Compose config
├── run_app.bat               ← Start backend API server (dev)
├── run_demo.bat              ← Run full AI platform demo
├── run_training.bat          ← Train all AI models
├── pyrightconfig.json        ← Pyright/Pylance config
└── venv/                     ← Python virtual environment
```

---

## 🚀 Quick Start

### 1. Setup Backend Environment
```bash
cd backend
setup_env.bat
```
Or manually:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```

### 2. Start Backend API
```bash
run_app.bat
```
API runs at: `http://localhost:8000`  
Docs (Swagger): `http://localhost:8000/docs`

### 3. Train AI Models
```bash
run_training.bat
```

### 4. Run Demo
```bash
run_demo.bat
```

### 5. Start Mobile App
```bash
cd InvestIQ-App
npm install
npm start
```
Scan QR code with Expo Go app (iOS/Android).

---

## 🧠 AI Features

| Feature | Description |
|--------|-------------|
| **LSTM + Attention** | Deep learning for price sequence prediction |
| **XGBoost Fusion** | Gradient boosting with technical + sentiment fusion |
| **Sentiment Analysis** | FinBERT-based news sentiment scoring |
| **Portfolio Optimizer** | Mean-variance optimization (Max Sharpe) |
| **Risk Engine** | VaR, Sharpe Ratio, Max Drawdown |
| **Technical Indicators** | RSI, MACD, Bollinger Bands, ATR, VWAP |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Get AI prediction for a ticker |
| `GET` | `/api/v1/signals` | Get all trading signals |
| `POST` | `/auth/login` | User authentication |
| `GET` | `/api/v1/portfolio` | Portfolio optimization |

---

## 📦 Tech Stack

**Backend:** Python · FastAPI · PyTorch · XGBoost · FinBERT · yfinance  
**Frontend:** React Native · Expo · Expo Router · React Context  
**Infrastructure:** Docker · Uvicorn · SHAP

---

## 📄 License
MIT
