# 📚 InvestIQ - Complete Documentation Index

**Status**: ✅ Production Ready | **Version**: 2.0 | **Updated**: April 15, 2026

---

## 🎯 NEW: Comprehensive Documentation Suite

### ⭐ **COMPLETE_SYSTEM_DOCUMENTATION.md** (Main Reference)
**10,000+ lines covering everything from backend to frontend**

```
✓ System Architecture & Diagrams
✓ Data Pipeline (Load → Clean → Features → Models)
✓ Feature Engineering (50+ features with formulas)
✓ ML Pipeline (LSTM + XGBoost Hybrid Ensemble)
✓ Training Process (Walk-Forward Validation)
✓ Inference Pipeline (Real-time Predictions)
✓ Performance Metrics (Accuracy breakdown: 59% avg)
✓ Trading Metrics (Sharpe 1.72, Win Rate 58%, Max DD -9%)
✓ API Endpoints (Request/Response examples)
✓ Frontend Integration (React Native)
✓ Deployment Guide (Docker & AWS)
✓ Performance Benchmarks (350ms per prediction)
```

**When to read this**: First time understanding the system  
**Time**: ~4-6 hours for complete understanding  
**Best for**: Developers, Data Scientists, Product Managers

---

### 🧮 **CALCULATIONS_AND_METRICS_QUICK_REFERENCE.md** (Formulas)
**2,000+ lines of calculations and formulas**

```
✓ Feature Calculation Formulas
  ├─ Momentum: RSI, MACD, ROC
  ├─ Volatility: Bollinger Bands, ATR
  ├─ Volume: OBV, Volume MA, VROC
  ├─ Trend: SMA, EMA, ADX
  ├─ Lag: Historical returns & prices
  └─ Market: Correlation, NIFTY context

✓ Accuracy Metrics (with Python code)
  ├─ Accuracy, Precision, Recall, F1-Score
  ├─ ROC-AUC, Confusion Matrix
  └─ Performance Tiers (Excellent, Good, Poor)

✓ Trading Metrics
  ├─ Win Rate, Total Return, Sharpe Ratio, Max Drawdown, Profit Factor
  └─ Detailed calculation examples with numbers

✓ API Response Formulas
  ├─ Ensemble Score: (LSTM × 0.5) + (XGBoost × 0.5)
  ├─ Signal Generation: if score > 0.60 → BUY
  ├─ Price Prediction: predicted_return formula
  └─ Risk Level: volatility-based classification

✓ 3 Complete Calculation Examples
  ├─ Example 1: Full prediction workflow
  ├─ Example 2: Trading performance calculation
  └─ Example 3: Sharpe ratio computation
```

**When to read this**: Need to verify calculations or understand formulas  
**Time**: ~20 minutes for specific formula lookup  
**Best for**: Engineers, Analysts, QA Testers  

---

### 🚀 **DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md** (Setup & Ops)
**3,000+ lines of deployment & infrastructure guide**

```
✓ High-Level System Architecture
  ├─ System Components Diagram
  ├─ Data Flow Diagram
  └─ Real-Time Prediction Flow Visualization

✓ Local Development Setup (Windows/Mac/Linux)
  ├─ Backend setup (Python venv)
  ├─ Data download (5 stocks)
  ├─ Model training
  ├─ API server startup
  ├─ Frontend setup (React Native)
  └─ Integration testing

✓ Docker Deployment
  ├─ Build Docker image
  ├─ Run single container
  └─ Docker Compose (PostgreSQL + Redis + Backend + Nginx)

✓ Production Deployment (AWS)
  ├─ AWS infrastructure setup
  ├─ RDS PostgreSQL configuration
  ├─ ElastiCache Redis setup
  ├─ ECS Fargate deployment
  ├─ Load balancer configuration
  └─ Environment variables

✓ Monitoring & Maintenance
  ├─ Health check endpoints
  ├─ Logging configuration
  ├─ Model retraining procedures
  └─ Performance monitoring

✓ Troubleshooting Guide
  ├─ Models not found
  ├─ Out of memory errors
  ├─ Database connection issues
  ├─ API timeouts
  └─ Prediction accuracy problems
```

**When to read this**: Setting up locally, deploying to production, troubleshooting  
**Time**: 1 hour for local setup, 2-3 hours for AWS production  
**Best for**: DevOps, Backend Engineers, System Administrators

---

## 📊 Key Performance Summary

### Accuracy Improvements (Baseline → Improved)

| Metric | Baseline | Improved | Gain |
|--------|:--------:|:--------:|:----:|
| **Accuracy** | 33% | 59% | **+26pp** |
| **Precision** | 50% | 69% | **+19pp** |
| **Recall** | 30% | 62% | **+32pp** |
| **F1-Score** | 0.35 | 0.65 | **+86%** |
| **ROC-AUC** | 0.60 | 0.80 | **+33%** |
| **Win Rate** | 45% | 58% | **+13pp** |
| **Sharpe Ratio** | 0.80 | 1.72 | **+115%** |

### Performance Benchmarks

- **Single Stock Prediction**: 350-450ms
- **Batch (5 stocks)**: ~600ms (parallel)
- **Training Duration**: ~5-8 minutes per stock
- **API Server Memory**: ~110MB idle
- **Model Size per Stock**: ~4.6MB (LSTM + XGBoost)

---

## 🎯 Quick Start Based on Your Role

### I'm a Frontend Developer
**Read**: DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md → Local Development Setup  
**Then**: COMPLETE_SYSTEM_DOCUMENTATION.md → Frontend Integration  
**Time**: ~2 hours | **Result**: Local app connected to backend

### I'm a Data Scientist
**Read**: COMPLETE_SYSTEM_DOCUMENTATION.md → Feature Engineering + ML Pipeline  
**Then**: CALCULATIONS_AND_METRICS_QUICK_REFERENCE.md → All Metrics  
**Time**: ~3 hours | **Result**: Understanding model architecture & accuracy

### I'm a DevOps Engineer
**Read**: DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md → Architecture + Production Setup  
**Then**: COMPLETE_SYSTEM_DOCUMENTATION.md → Monitoring & Maintenance  
**Time**: ~3 hours | **Result**: Production deployment & monitoring setup

### I'm a Backend Engineer
**Read**: COMPLETE_SYSTEM_DOCUMENTATION.md → API Endpoints  
**Then**: DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md → Local Setup  
**Time**: ~2 hours | **Result**: Backend running locally with familiarity

### I'm a Product Manager
**Read**: COMPLETE_SYSTEM_DOCUMENTATION.md → System Overview + Performance  
**Then**: Quick reference: Performance Summary (above)  
**Time**: ~30 minutes | **Result**: System capabilities & improvements understanding

### I Need to Verify Calculations
**Read**: CALCULATIONS_AND_METRICS_QUICK_REFERENCE.md  
**Use**: Specific formula section + calculation examples  
**Time**: ~15 minutes per formula | **Result**: Verified calculations with code

---

## 📁 Documentation Files Location

```
d:\InvestIQ-main\
├─ COMPLETE_SYSTEM_DOCUMENTATION.md              ← Main (10K lines)
├─ CALCULATIONS_AND_METRICS_QUICK_REFERENCE.md   ← Formulas (2K lines)
├─ DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md          ← Setup (3K lines)
├─ DOCUMENTATION_INDEX.md                        ← This file
│
├─ FULL_DOCUMENTATION.md                         ← Old reference docs
├─ BASELINE_VS_IMPROVED_DETAILED_COMPARISON.md   ← Comparison
├─ IMPROVED_HYBRID_MODELS_IMPLEMENTATION_SUMMARY.md ← Implementation notes
│
└─ backend/
   ├─ training/
   │  ├─ improved_hybrid_model.py                ← ML models
   │  ├─ train_improved_hybrid_models.py         ← Training script
   │  └─ evaluation_module.py                    ← Metrics
   ├─ inference/
   │  └─ predict.py                              ← Inference
   └─ app/
      └─ routes.py                               ← API endpoints
```

---

## 🚀 Getting Started in 3 Steps

### Step 1: Understand the System (30 minutes)
```bash
# Read the system overview
# Open: COMPLETE_SYSTEM_DOCUMENTATION.md
# Read sections:
#   1. System Overview
#   2. Architecture Diagram
#   3. Key Performance Improvements
```

### Step 2: Set Up Locally (1 hour)
```bash
# Follow setup guide
# Open: DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md
# Follow: Local Development Setup section
# Instructions are step-by-step with all commands
```

### Step 3: Run & Test (30 minutes)
```bash
# Backend
cd backend
python training/train_improved_hybrid_models.py --ticker HDFCBANK
uvicorn app.main:app --reload --port 8000

# Frontend (in new terminal)
cd InvestIQ-App
npm start
```

---

## 🔍 Find Information By Topic

### **How do features work?**
→ COMPLETE_SYSTEM_DOCUMENTATION.md → Section: Feature Engineering

### **How accurate is the model?**
→ COMPLETE_SYSTEM_DOCUMENTATION.md → Section: Model Performance & Accuracy Metrics  
→ And: CALCULATIONS_AND_METRICS_QUICK_REFERENCE.md → Performance Summary

### **How do I deploy?**
→ DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md → Local/Docker/AWS sections

### **What are the API endpoints?**
→ COMPLETE_SYSTEM_DOCUMENTATION.md → Section: API Endpoints & Calculations

### **What calculations are used?**
→ CALCULATIONS_AND_METRICS_QUICK_REFERENCE.md → All calculation sections

### **How does the training work?**
→ COMPLETE_SYSTEM_DOCUMENTATION.md → Section: Training Process

### **How does inference work?**
→ COMPLETE_SYSTEM_DOCUMENTATION.md → Section: Inference Pipeline

### **What's the system architecture?**
→ DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md → Section: High-Level Architecture

---

## 📈 System at a Glance

```
Frontend (React Native)
    ↓ REST API
Backend (FastAPI)
    ├─ Data Loading (CSV files)
    ├─ Feature Engineering (50+ features)
    ├─ ML Models
    │  ├─ LSTM: Temporal patterns
    │  └─ XGBoost: Feature patterns
    ├─ Ensemble: 50-50 weighted average
    ├─ API Endpoints: /predict, /train, /portfolio, etc.
    └─ Metrics: 59% accuracy, 1.72 Sharpe, 58% win rate
```

---

## ✅ Complete Documentation Checklist

- [x] **System Architecture** - Fully documented with diagrams
- [x] **Data Pipeline** - Load → Clean → Feature → Model flow
- [x] **Feature Engineering** - All 50+ features explained with formulas
- [x] **ML Pipeline** - LSTM + XGBoost hybrid ensemble
- [x] **Training** - Walk-forward validation procedure
- [x] **Inference** - Real-time prediction pipeline
- [x] **Accuracy Metrics** - All metrics with formulas & examples
- [x] **Trading Metrics** - Sharpe, drawdown, win rate calculations
- [x] **API Endpoints** - All endpoints with request/response examples
- [x] **Frontend Integration** - React Native component examples
- [x] **Local Setup** - Step-by-step development setup
- [x] **Docker Deployment** - Container & compose files
- [x] **AWS Production** - ECS, RDS, ElastiCache setup
- [x] **Monitoring** - Health checks & logging
- [x] **Troubleshooting** - Common issues & solutions

---

## 📞 Support Resources

### **Code Examples**
- ML Models: `backend/training/improved_hybrid_model.py`
- Inference: `backend/inference/predict.py`
- API Routes: `backend/app/routes.py`
- Frontend API: `InvestIQ-App/src/services/api.js`
- Components: `InvestIQ-App/src/components/StockSignalCard.js`

### **Configuration Files**
- Config: `backend/core/config.py`
- Frontend Config: `InvestIQ-App/config/api.js`
- Requirements: `backend/requirements.txt`
- Package.json: `InvestIQ-App/package.json`

### **Data & Models**
- Stock Data: `backend/data/stock_data/`
- Trained Models: `backend/models/saved_models/`
- Training Scripts: `batch_train_optimized.py`

---

## 🎓 Recommended Reading Order

**For Complete Understanding** (4-6 hours):
1. COMPLETE_SYSTEM_DOCUMENTATION.md (start to finish)
2. CALCULATIONS_AND_METRICS_QUICK_REFERENCE.md (formulas)
3. DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md (setup)

**For Implementation** (2-3 hours):
1. DEPLOYMENT_AND_ARCHITECTURE_GUIDE.md (local setup)
2. COMPLETE_SYSTEM_DOCUMENTATION.md (API endpoints)
3. Follow step-by-step instructions in guides

**For Verification** (15-30 minutes):
1. CALCULATIONS_AND_METRICS_QUICK_REFERENCE.md (formulas)
2. COMPLETE_SYSTEM_DOCUMENTATION.md (architecture)

---

## 🎉 You're All Set!

**Everything you need is documented above.**  
Choose your starting point based on your role/needs.

**Next Steps**:
1. ✅ Pick a documentation file
2. ✅ Follow the instructions
3. ✅ Set up locally or deploy
4. ✅ Start making predictions!

---

## Previous Documentation (Still Useful Reference)

---

## 🚀 GETTING STARTED IMMEDIATELY

### 1️⃣ First Time Setup (5 mins)
- Read: [FULL_DOCUMENTATION.md - Installation & Setup](#installation--setup)
- Run: `cd backend && setup_env.bat`
- Run: `cd InvestIQ-App && npm install`

### 2️⃣ Start the Application (5 mins)
- **Terminal 1**: `run_app.bat` → Backend starts on port 8000
- **Terminal 2**: `ngrok http 8000` → Copy ngrok URL
- **Terminal 3**: Update `InvestIQ-App/config/api.js` with ngrok URL
- **Terminal 4**: `cd InvestIQ-App && npm start` → Scan QR code

### 3️⃣ First API Call (2 mins)
```bash
# Register user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123","name":"Test"}'

# Login & get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'

# Make prediction (use token from login response)
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"HDFCBANK","model":"transformer"}'
```

---

## 📖 COMPLETE DOCUMENTATION STRUCTURE

### In FULL_DOCUMENTATION.md:

#### **Section 1: Project Overview**
- What is InvestIQ?
- Key features checklist
- Supported stocks
- Technology stack

#### **Section 2: System Architecture**
- High-level data flow diagram
- Component breakdown
- Frontend architecture
- Backend architecture
- AI Models overview
- Data pipeline flow

#### **Section 3: Backend API FULL GUIDE**
- ✅ All authentication endpoints (register, login, get user)
- ✅ All prediction endpoints (predict, signals, batch signals)
- ✅ All portfolio endpoints (optimize, risk score)
- ✅ All sentiment endpoints (analyze)
- ✅ All training endpoints (trigger, status)
- ✅ All system endpoints (health, model status)

**Every endpoint includes**:
- Full URL with examples
- Request JSON format
- Response JSON format
- Status codes

#### **Section 4: Frontend COMPLETE GUIDE**
- 📱 Frontend directory structure
- 📱 All 5 main screens explained
- 📱 Authentication flow
- 📱 Data fetching flow
- 📱 API client details
- 📱 Configuration files
- 📱 Key features per screen

#### **Section 5: Ngrok Setup & Tunnel**
- ⚡ What is ngrok?
- ⚡ Why use ngrok?
- ⚡ Complete 9-step setup
- ⚡ Monitoring & debugging
- ⚡ URL expiration handling
- ⚡ Auto-update script

#### **Section 6: Installation & Setup**
- Automated setup (Windows)
- Manual setup (all platforms)
- Verification steps
- Dependencies list

#### **Section 7: Running the Application**
- Quick 5-minute start
- Complete startup process
- Testing API directly
- All curl examples

#### **Section 8: Data Pipeline & Models**
- Training pipeline (6 steps)
- Model architectures
- Inference pipeline
- Performance metrics table

#### **Section 9: Troubleshooting**
- Frontend issues (3 common problems)
- Backend issues (3 common problems)
- Data pipeline issues (2 common problems)
- Ngrok issues (3 common problems)
- All with solutions

---

## 📂 DOCUMENTATION FILES ORGANIZATION

### In `/docs/` folder:

```
docs/README.md                              ← Main docs index
docs/QUICK_START.md                        ← 3-step setup
docs/PROJECT_ARCHITECTURE.md               ← Full system design
docs/COMPLETE_BACKEND_INVENTORY.md         ← All backend modules
docs/NGROK_SETUP_GUIDE.md                  ← Ngrok tunneling
docs/TRAINING_PIPELINE_COMPLETE.md         ← Model training guide
docs/MODEL_EVALUATION_GUIDE.md              ← Evaluation metrics
docs/QUICK_FIX_EVALUATION.md                ← Fast evaluation
docs/PHASE_1_QUICKSTART.md                 ← Earlier version docs
docs/PHASE_2_QUICKSTART.md                 ← Current version docs
docs/XGBOOST_CLASSIFICATION_GUIDE.md       ← Trading signal model
```

### In root folder:

```
FULL_DOCUMENTATION.md                      ← NEW: Complete guide (THIS FILE)
PRODUCTION_READY.md                        ← Production checklist
FINAL_PRODUCTION_CLEANUP.md                ← Last optimizations
PHASE2_FIXES_SUMMARY.md                    ← Recent improvements
XGBOOST_QUICK_START.md                     ← Trading signals setup
```

---

## 🔗 DIRECT FILE REFERENCES

### Frontend Files
- **App entry**: `InvestIQ-App/app/(auth)/login.js` - Login screen
- **Dashboard**: `InvestIQ-App/app/(tabs)/dashboard.js` - Main screen
- **Stock detail**: `InvestIQ-App/app/stock/[symbol].js` - Per-stock view
- **API client**: `InvestIQ-App/src/services/api.js` - JWT & requests
- **Config**: `InvestIQ-App/config/api.js` - API URLs (EDIT THIS for ngrok)
- **Package**: `InvestIQ-App/package.json` - Dependencies & scripts

### Backend Files
- **Main app**: `backend/app/main.py` - FastAPI initialization
- **Routes**: `backend/app/routes.py` - All API endpoints
- **Auth**: `backend/app/auth.py` - JWT token logic
- **Schemas**: `backend/app/schemas.py` - Request/response models
- **Config**: `backend/core/config.py` - Settings & hyperparameters
- **Predictor**: `backend/inference/predict.py` - Prediction engine
- **Transformer model**: `backend/models/transformer_model.py` - Architecture
- **XGBoost pipeline**: `backend/training/xgboost_classifier.py` - Trading signals

### Data Files
- **Stock data**: `backend/data/stock_data/` - CSV files (HDFCBANK, INFY, etc)
- **Trained models**: `backend/models/saved_models/` - .pth and .pkl files
- **User accounts**: `backend/app/users.json` - Stored credentials

### Training Scripts
- **Optimized training**: `batch_train_optimized.py` - Main training script
- **Run training**: `run_training.bat` - Windows batch script
- **Run app**: `run_app.bat` - Start backend server
- **Demo**: `run_demo.bat` - Full system demo

---

## 🎯 BY USE CASE

### "I want to run the app locally with mobile device"
1. Read: [FULL_DOCUMENTATION.md - Ngrok Setup](#ngrok-setup--tunnel-configuration)
2. Run: `ngrok http 8000`
3. Update: `InvestIQ-App/config/api.js` with ngrok URL
4. Run: `npm start` in InvestIQ-App

### "I want to understand all API endpoints"
- Go to: [FULL_DOCUMENTATION.md - Backend API](#backend-api-complete-guide)
- See every endpoint with request/response formats
- Or open: `http://localhost:8000/docs` (Swagger UI)

### "I want to train custom models"
1. Read: [FULL_DOCUMENTATION.md - Data Pipeline](#data-pipeline--models)
2. Run: `python batch_train_optimized.py HDFCBANK`
3. Check: `backend/models/saved_models/transformer_HDFCBANK.pth`

### "I want to evaluate model performance"
1. Run: `run_model_evaluation.bat`
2. Results: `backend/models/saved_models/evaluation_results/`

### "I want to understand frontend screens"
- Dashboard: Shows top 5 stocks with signals
- Stock Detail: Shows 7-day forecast + indicators + news
- Signals: Lists BUY/SELL/HOLD opportunities
- Portfolio: Shows holdings + optimization suggestions
- Profile: User settings + preferences

### "I want to debug API issues"
1. Backend health: `curl http://localhost:8000/api/v1/health`
2. API docs: `http://localhost:8000/docs`
3. Ngrok inspector: `http://127.0.0.1:4040`
4. Check logs: `backend/logs/` folder

---

## 📊 ARCHITECTURE QUICK REFERENCE

### Data Flow (Inference)
```
User Request
    ↓
API Receives: POST /predict {symbol, model}
    ↓
Load CSV: backend/data/stock_data/{symbol}.csv
    ↓
Add Features: Technical indicators (13)
    ↓
Load Scaler: .pkl file
    ↓
Scale Features: Apply fitted scaler
    ↓
Create Sequence: Last 90 days
    ↓
Load Model: .pth file (Transformer)
    ↓
Inference: Model(sequence)
    ↓
Get Forecast: 7-day returns + confidence
    ↓
Load XGBoost: Signal classifier (.pkl)
    ↓
Get Signal: BUY/SELL/HOLD + probability
    ↓
Response: JSON with forecast + signal + indicators
    ↓
Return to Frontend/App
```

### API Authentication Flow
```
1. User enters email + password
    ↓
2. POST /auth/login
    ↓
3. Backend validates in users.json
    ↓
4. Generate JWT token (HS256)
    ↓
5. Return token to client
    ↓
6. Client stores in SecureStore
    ↓
7. Client includes in all requests: Authorization: Bearer {token}
    ↓
8. Backend validates JWT on every request
    ↓
9. Request succeeds or returns 401
```

### Feature Engineering Pipeline
```
Raw OHLCV
    ↓
+ Technical Indicators (13)
    ├─ SMA_20, SMA_50
    ├─ RSI, MACD, Bollinger Bands
    ├─ ATR, VWAP
    ├─ Log_Return, Volume_Change
    └─ Rolling_Volatility
    ↓
+ Momentum Features
    ├─ return_3d, return_5d, return_7d
    └─ momentum_3d, momentum_5d
    ↓
+ Volume Features
    ├─ volume_change
    ├─ volume_ma_5, volume_ma_20
    └─ volume_ratio
    ↓
+ External Features
    ├─ Market_Correlation (vs NIFTY-50)
    ├─ Sentiment_Score
    └─ Macro_Score
    ↓
= 25+ features total
    ↓
StandardScaler
    ↓
Ready for Model Input
```

---

## 🔧 CONFIGURATION FILES TO KNOW

### `InvestIQ-App/config/api.js`
**Purpose**: Set API base URL (CRITICAL for ngrok)
**Edit this when**:
- Starting with ngrok → Add ngrok URL
- Switching to LAN → Add LAN IP
- Local testing → Use localhost

### `backend/core/config.py`
**Purpose**: Model hyperparameters, data paths
**Contains**:
- Learning rate: 0.0003
- Batch size: 128 (GPU) / 64 (CPU)
- Epochs: 100
- Early stopping patience: 20
- Dropout: 0.2

### `InvestIQ-App/app.json`
**Purpose**: Expo app configuration
**Contains**:
- App name, version, slug
- iOS/Android bundle IDs
- Permissions needed

### `backend/requirements.txt`
**Purpose**: Python dependencies
**Contains**:
- FastAPI, Uvicorn
- PyTorch, XGBoost
- scikit-learn, pandas, numpy
- yfinance, APScheduler, etc.

---

## 📞 NEED HELP?

### Common Questions Answered in FULL_DOCUMENTATION.md:

**Q: How do I connect my phone to the backend?**  
A: Use ngrok (Section: Ngrok Setup & Tunnel Configuration)

**Q: What are all the API endpoints?**  
A: Complete list in Section: Backend API Complete Guide

**Q: How do I train a custom model?**  
A: Instructions in Section: Data Pipeline & Models

**Q: Why is my app disconnecting?**  
A: Troubleshooting in Section: Troubleshooting

**Q: Where do I put the ngrok URL?**  
A: File: `InvestIQ-App/config/api.js` (see config reference above)

**Q: What models are available?**  
A: Transformer, LSTM, XGBoost (see Architecture section)

---

## ✅ CHECKLIST TO GO LIVE

- [ ] Backend running (`run_app.bat`)
- [ ] ngrok tunnel active (`ngrok http 8000`)
- [ ] Frontend config updated with ngrok URL
- [ ] Frontend running (`npm start`)
- [ ] Can login with test account
- [ ] Can request predictions
- [ ] Models trained (`run_training.bat`)
- [ ] Data available (`backend/data/stock_data/`)
- [ ] No 404 errors in logs

---

## 🎓 LEARNING PATH

**New to the project?** Follow this order:

1. **Understanding** (10 mins)
   - Read: FULL_DOCUMENTATION.md - Project Overview
   - Read: FULL_DOCUMENTATION.md - System Architecture

2. **Setup** (15 mins)
   - Follow: FULL_DOCUMENTATION.md - Installation & Setup
   - Run: `setup_env.bat` and `npm install`

3. **Running** (10 mins)
   - Follow: FULL_DOCUMENTATION.md - Running the Application
   - Get backend + frontend + ngrok running

4. **Testing** (10 mins)
   - Read: FULL_DOCUMENTATION.md - Backend API Complete Guide
   - Try API calls with curl

5. **Exploring** (30 mins)
   - Open app on phone
   - Try different stocks
   - Check different signals

6. **Advanced** (Optional)
   - Train models: `batch_train_optimized.py`
   - Evaluate: `run_model_evaluation.bat`
   - Customize features in `backend/features/`

---

## 📚 EXTERNAL RESOURCES

- **Expo**: https://expo.dev/
- **React Native Docs**: https://reactnative.dev/
- **FastAPI**: https://fastapi.tiangolo.com/
- **ngrok**: https://ngrok.com/
- **PyTorch**: https://pytorch.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **Swagger API Docs**: http://localhost:8000/docs (when running)

---

## 🚀 YOU'RE ALL SET!

Everything you need is in **`FULL_DOCUMENTATION.md`**  
Start with Section 1 or jump to your use case above.

**Happy Trading! 📈**

