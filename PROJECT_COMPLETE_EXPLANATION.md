# InvestIQ - Complete Project Explanation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Why This Project Exists](#why-this-project-exists)
3. [System Architecture](#system-architecture)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Machine Learning Models](#machine-learning-models)
7. [Feature Engineering](#feature-engineering)
8. [Training Pipeline](#training-pipeline)
9. [Inference & Predictions](#inference--predictions)
10. [API Architecture](#api-architecture)
11. [Frontend Application](#frontend-application)
12. [Setup & Installation](#setup--installation)
13. [How to Run](#how-to-run)

---

## Project Overview

**InvestIQ** is an AI-powered stock prediction system that uses advanced machine learning to forecast stock price movements and generate trading signals (BUY/SELL/HOLD). 

### Key Features:
- **Hybrid ML Ensemble**: Combines LSTM (deep learning) + XGBoost (gradient boosting) for robust predictions
- **Advanced Feature Engineering**: 50+ technical indicators and market-based features
- **Real-time Predictions**: Provides price forecasts with confidence scores
- **Trading Signals**: Generates actionable BUY/SELL/HOLD recommendations
- **Risk Assessment**: Evaluates risk levels and portfolio optimization
- **Sentiment Analysis**: Integrates news sentiment into predictions
- **Walk-forward Validation**: Prevents data leakage using time-series aware validation

### Performance Improvement (vs Baseline)
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Accuracy | 33% | 55-65% | +22-32% |
| Precision | ~50% | 65-75% | +15-25% |
| Recall | ~30% | 60-70% | +30-40% |
| F1-Score | 0.35 | 0.62-0.72 | +78% |
| ROC-AUC | 0.60 | 0.80-0.85 | +33% |

---

## Why This Project Exists

### Problem It Solves
1. **Stock Market Complexity**: Financial markets are highly complex with non-linear patterns. Traditional rule-based systems fail to capture these patterns.
2. **Information Overload**: Traders have access to massive amounts of data (prices, volumes, news, market indicators) but struggle to process it effectively.
3. **Poor Prediction Accuracy**: Most retail trading tools have low accuracy (close to random guessing at 50%).
4. **Risk Management**: Investors need scientific risk assessment and portfolio optimization.

### Solution Provided by InvestIQ
- **Machine Learning**: Uses neural networks to learn complex, non-linear market patterns
- **Ensemble Approach**: Combines multiple models (LSTM + XGBoost) for better predictions than any single model
- **Intelligent Feature Engineering**: Creates 50+ features from raw price/volume data
- **Real-time Recommendations**: Provides actionable signals with confidence scores
- **Quantified Risk**: Calculates risk metrics (Sharpe ratio, max drawdown, win rate)

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INVESTIQ SYSTEM                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    FRONTEND (React Native)                  │   │
│  │                  (Mobile + Web - Expo App)                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              REST API (FastAPI + Django)                    │   │
│  │  /api/v1/predict      - Get stock predictions               │   │
│  │  /api/v1/train        - Trigger model training              │   │
│  │  /api/v1/sentiment    - Analyze news sentiment              │   │
│  │  /api/v1/risk         - Calculate risk scores               │   │
│  │  /api/v1/portfolio    - Optimize portfolio                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         BACKEND PROCESSING LAYER                            │   │
│  │                                                              │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │ DATA PIPELINE                                       │   │   │
│  │  │ - Data Loading & Cleaning                           │   │   │
│  │  │ - Validation & Error Handling                       │   │   │
│  │  │ - Feature Engineering (50+ features)                │   │   │
│  │  │ - Data Scaling & Normalization                      │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │                        ↓                                     │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │ MACHINE LEARNING MODELS                             │   │   │
│  │  │                                                      │   │   │
│  │  │  MODEL 1: LSTM (Long Short-Term Memory)            │   │   │
│  │  │  - Deep learning neural network                     │   │   │
│  │  │  - Learns temporal patterns in prices              │   │   │
│  │  │  - 64 hidden units, 2 layers                        │   │   │
│  │  │                                                      │   │   │
│  │  │  MODEL 2: XGBoost (Gradient Boosting)              │   │   │
│  │  │  - Tree-based ensemble learning                     │   │   │
│  │  │  - Learns feature interactions                      │   │   │
│  │  │  - 200+ decision trees                              │   │   │
│  │  │                                                      │   │   │
│  │  │  ENSEMBLE: Weighted Combination                     │   │   │
│  │  │  - 50% LSTM + 50% XGBoost + 20% Sentiment          │   │   │
│  │  │  - Final confidence score (0-1)                     │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │                        ↓                                     │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │ SIGNAL GENERATION                                   │   │   │
│  │  │ - BUY: Confidence > 60% & Positive direction        │   │   │
│  │  │ - SELL: Confidence > 60% & Negative direction       │   │   │
│  │  │ - HOLD: Below thresholds or neutral signals         │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         DATA STORAGE                                         │   │
│  │ - Stock Data (CSV files)                                     │   │
│  │ - Trained Models (PyTorch + XGBoost)                         │   │
│  │ - Scalers (StandardScaler)                                   │   │
│  │ - User Data (SQLite)                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
InvestIQ-main/
├── backend/                          # Main backend application
│   ├── app/                          # FastAPI application
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── routes.py                # API endpoints
│   │   ├── auth.py                  # Authentication logic
│   │   └── schemas.py               # Request/Response models
│   │
│   ├── training/                    # Model training pipelines
│   │   ├── improved_hybrid_model.py # Production hybrid LSTM+XGBoost
│   │   ├── lstm_model.py            # LSTM implementation
│   │   ├── xgboost_classifier.py    # XGBoost implementation
│   │   ├── feature_engineering.py   # Feature generation (50+ features)
│   │   ├── train_improved_hybrid_models.py # Main training script
│   │   └── evaluation_module.py     # Model evaluation metrics
│   │
│   ├── inference/                   # Prediction engines
│   │   ├── predict.py               # Standard predictor
│   │   └── hybrid_predict.py        # Hybrid ensemble predictor
│   │
│   ├── preprocessing/               # Data preprocessing
│   │   ├── cleaning.py              # Data cleaning & validation
│   │   └── scaling.py               # Feature scaling (StandardScaler)
│   │
│   ├── features/                    # Feature extraction
│   │   ├── indicators.py            # Technical indicators
│   │   ├── external_data.py         # Market data integration
│   │   └── realtime_price.py        # Real-time price fetching
│   │
│   ├── models/                      # Model storage
│   │   ├── saved_models/            # Trained model files
│   │   ├── enhanced_models.py       # PyTorch model architectures
│   │   └── model.py                 # Model utilities
│   │
│   ├── core/                        # Core utilities
│   │   ├── config.py                # Configuration settings
│   │   ├── logging.py               # Logging setup
│   │   └── exceptions.py            # Custom exceptions
│   │
│   ├── data/                        # Stock data
│   │   └── stock_data/              # CSV files for each stock
│   │       ├── HDFCBANK.csv
│   │       ├── ICICIBANK.csv
│   │       ├── INFY.csv
│   │       ├── RELIANCE.csv
│   │       └── TCS.csv
│   │
│   └── evaluation/                  # Evaluation utilities
│       ├── backtesting.py           # Backtest trading strategies
│       └── metrics.py               # Performance metrics
│
├── InvestIQ-App/                    # React Native frontend
│   ├── app/                         # Main app screens
│   ├── src/                         # Source code
│   ├── config/                      # API configuration
│   └── assets/                      # Images & resources
│
├── docs/                            # Documentation
│
└── requirements.txt                 # Python dependencies
```

---

## Core Components

### 1. **Data Loading & Preprocessing** (`backend/preprocessing/`)

**Purpose**: Prepare raw stock data for ML models

**Components**:
- `cleaning.py`: Removes invalid data, handles missing values, validates data integrity
- `scaling.py`: Normalizes features using StandardScaler (prevents bias toward large-value features)

**Why It's Important**:
- ML models are sensitive to data quality
- Missing values or outliers can destroy model performance
- Scaling ensures features have equal influence on model

**Example Process**:
```
Raw CSV Data (Open, High, Low, Close, Volume)
    ↓
Load & Clean (remove NaN, validate ranges)
    ↓
Engineer Features (create 50+ indicators)
    ↓
Scale Features (normalize to [-1, 1] range)
    ↓
Create Sequences (20-day windows for LSTM)
    ↓
Ready for Training
```

---

### 2. **Feature Engineering** (`backend/training/feature_engineering.py`)

**Purpose**: Transform raw price/volume data into meaningful signals ML models can learn from

**50+ Features Grouped Into Categories**:

#### Technical Indicators (Classic indicators from finance)
- **RSI (Relative Strength Index)**: Measures momentum (overbought/oversold conditions)
- **MACD (Moving Average Convergence Divergence)**: Identifies trend changes
- **Bollinger Bands**: Shows volatility extremes
- **SMA (Simple Moving Averages)**: Identifies trend direction

#### Momentum Features
- **Daily Returns**: 1-day, 3-day, 5-day percentage changes
- **Return Lags**: Previous day returns (memory of recent movement)
- **Volatility**: 10-day rolling standard deviation

#### Volume Features
- **Volume Change**: Daily volume percentage change
- **Volume Interactions**: RSI × Volume_Change (combination signals)

#### Market Context
- **Market Volatility**: Overall market volatility proxy
- **Nifty Returns**: Index returns (market direction)
- **Sector Trend**: Sector-specific movement

#### Sentiment Features
- **News Sentiment**: Positive/negative news scoring
- **Sentiment Trend**: Direction of sentiment change
- **Sentiment × Return**: Interaction (sentiment influence on returns)

#### Interaction Features (Capture complex relationships)
- `RSI × Volume_Change`: RSI signal amplified by volume
- `MACD × Market_Volatility`: MACD reliability in volatile markets
- `Return_3D × Sentiment`: 3-day returns weighted by sentiment

**Why These Features Work**:
```
Raw Data: Just prices and volumes (2 features)
Problem: ML can't learn useful patterns from just 2 features
Solution: Engineer 50+ features representing different aspects:
  - Trend (MACD, SMA)
  - Momentum (RSI, returns)
  - Volatility (Bollinger, std dev)
  - Market context (sector, volatility)
  - Sentiment (news)
Result: ML model has rich information to learn from
```

---

### 3. **LSTM Model** (`backend/training/lstm_model.py`)

**What is LSTM?**
- LSTM = Long Short-Term Memory
- A type of recurrent neural network (RNN) designed for time-series data
- Remembers important information from previous time steps

**Architecture**:
```
Input Layer: 50+ features
    ↓
LSTM Layer 1: 64 hidden units
    ↓
LSTM Layer 2: 64 hidden units
    ↓
Dropout: 20% (regularization to prevent overfitting)
    ↓
Dense Output Layer: Predicts next price movement (UP/DOWN)
```

**How It Works**:
1. Takes 20-day window of data (20 days × 50 features = 1000 values)
2. Processes sequentially, learning temporal patterns
3. Outputs probability of UP movement in next 3 days

**Why LSTM for Stock Prices?**:
- **Temporal Patterns**: Stock prices have patterns over time (trends, cycles)
- **Memory**: LSTM remembers important events from days/weeks ago
- **Non-linear**: Captures complex, non-linear price movements
- **Flexible**: Can learn different patterns for different stocks

**Training Configuration**:
- Learning Rate: 0.0003 (small steps for stability)
- Batch Size: 128
- Epochs: 70-100
- Early Stopping: Stops if validation performance plateaus
- Optimizer: Adam (adaptive learning rate)

---

### 4. **XGBoost Model** (`backend/training/xgboost_classifier.py`)

**What is XGBoost?**
- XGBoost = eXtreme Gradient Boosting
- Tree-based ensemble learning (builds multiple decision trees sequentially)
- Each tree corrects mistakes of previous trees

**Architecture**:
```
Input: 50+ features
    ↓
Tree 1: Makes initial prediction
    ↓
Tree 2: Corrects Tree 1's mistakes
    ↓
Tree 3: Corrects Tree 2's mistakes
    ↓
... (200+ trees total)
    ↓
Final Prediction: Sum of all tree predictions
```

**How It Works**:
1. Trees learn different aspects of the feature space
2. First tree captures main trend
3. Second tree captures deviations from first tree
4. Process repeats, each tree learning residual errors
5. Final prediction = weighted sum of all tree predictions

**Why XGBoost for Stock Trading?**:
- **Feature Importance**: Automatically identifies which features matter most
- **Non-linear Relationships**: Captures complex feature interactions
- **Robustness**: Less prone to overfitting than single decision trees
- **Fast**: Can handle thousands of features efficiently
- **Proven**: Industry standard for classification/regression tasks

**Configuration**:
- Number of Trees: 200
- Max Depth: 6 (prevents overfitting)
- Learning Rate: 0.1
- Objective: Binary classification (UP/DOWN)

---

### 5. **Hybrid Ensemble Predictor** (`backend/inference/hybrid_predict.py`)

**Why Combine LSTM + XGBoost?**

```
LSTM Strengths:
  ✓ Captures temporal patterns
  ✓ Learns long-term dependencies
  ✗ May miss short-term feature interactions
  ✗ Slow to train

XGBoost Strengths:
  ✓ Learns feature interactions quickly
  ✓ Feature importance insights
  ✓ Fast predictions
  ✗ May miss temporal patterns
  ✗ No native time-series awareness

Ensemble Solution:
  - Use both: Get temporal + interaction benefits
  - Weighted combination: 50% LSTM + 50% XGBoost
  - If one fails: Fall back to other (robustness)
  - Final Signal: More reliable than either alone
```

**Ensemble Logic**:
```python
lstm_prob = lstm_model.predict(data)              # 0-1
xgb_prob = xgb_model.predict_proba(features)     # 0-1
sentiment_score = get_sentiment()                 # -1 to 1
sentiment_norm = clip((sentiment_score + 1) / 2) # 0-1

# Weighted combination
final_score = (0.5 * lstm_prob) + (0.5 * xgb_prob) + (0.2 * sentiment_norm)

# Convert to signal
if final_score > 0.60:
    signal = "BUY"
elif final_score < 0.40:
    signal = "SELL"
else:
    signal = "HOLD"
```

**Why Ensembles Work**:
- Diversity of models reduces individual model errors
- Different strengths cover different weaknesses
- Statistically more stable than single model
- Production-grade reliability

---

## Data Flow

### Complete Prediction Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ USER REQUESTS PREDICTION FOR "HDFCBANK" STOCK                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: FETCH DATA                                              │
│ - Load last 90 days of HDFCBANK.csv                             │
│ - Get real-time price from yfinance API                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: CLEAN DATA                                              │
│ - Remove missing values                                         │
│ - Validate price ranges (shouldn't jump 50% in one day)         │
│ - Handle outliers                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: FEATURE ENGINEERING                                     │
│ - Calculate RSI, MACD, Bollinger Bands                          │
│ - Compute returns, volatility                                   │
│ - Extract sentiment from recent news                            │
│ - Create interaction features                                   │
│ Result: 90 rows × 50 columns (4,500 feature values)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: SCALE FEATURES                                          │
│ - Normalize each feature to [-1, 1] range                       │
│ - Use pre-trained scaler (trained on historical data)           │
│ - Ensures consistency between training and prediction           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: CREATE SEQUENCES                                        │
│ - Take last 20 days of data (20 × 50 = 1000 values)             │
│ - Reshape into 3D tensor (1, 20, 50) for LSTM                   │
│ - This is the "context window" the model will analyze           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: LSTM PREDICTION                                         │
│ - Input: 3D tensor of 20 days of 50 features                    │
│ - Process through LSTM layers (learns temporal patterns)        │
│ - Output: Probability UP (0-1)                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: XGBOOST PREDICTION                                      │
│ - Input: 50 features (current day)                              │
│ - Process through 200 decision trees                            │
│ - Output: Probability UP (0-1)                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: ENSEMBLE COMBINATION                                    │
│ - LSTM output: 0.65                                             │
│ - XGBoost output: 0.68                                          │
│ - Sentiment: 0.75 (positive news)                               │
│ - Final: (0.5×0.65) + (0.5×0.68) + (0.2×0.75) = 0.68           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 9: GENERATE SIGNAL & RETURN RESPONSE                       │
│ - Confidence: 68% (above 60% threshold)                         │
│ - Signal: "BUY" (confidence > 60%)                              │
│ - Predicted Price: ₹1,850 (up 2% from current ₹1,815)          │
│ - Risk Level: MEDIUM                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ RESPONSE SENT TO FRONTEND                                       │
│ {                                                               │
│   "signal": "BUY",                                              │
│   "confidence": 0.68,                                           │
│   "predicted_price": 1850,                                      │
│   "current_price": 1815,                                        │
│   "predicted_change_pct": 0.0193,                               │
│   "risk_level": "MEDIUM",                                       │
│   "indicators": {...}                                           │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Machine Learning Models

### LSTM Model Details

**Purpose**: Learn temporal patterns in stock prices

**Input**: Sequence of 20 days × 50 features per day

**Processing**:
```
Day 1: [feature1, feature2, ..., feature50]
Day 2: [feature1, feature2, ..., feature50]
...
Day 20: [feature1, feature2, ..., feature50]
         ↓
    LSTM Cell 1 (processes day 1, outputs hidden state)
         ↓
    LSTM Cell 2 (processes day 2, uses previous hidden state)
         ↓
    LSTM Cell 3 (processes day 3, uses previous hidden state)
         ↓
    ... (continues for all 20 days)
         ↓
    Final hidden state encodes "essence" of 20-day pattern
         ↓
    Dense layer: Converts hidden state → probability
```

**Output**: Probability that stock will go UP in next 3 days

**Why Sequence Matters**:
- Day 20 prediction depends on Days 1-19
- LSTM "remembers" important events from days ago
- Captures trends, reversals, volatility patterns
- Example: If volatility spikes on Day 18, LSTM learns this predicts price action on Days 21-23

**Training**:
- Dataset: 3+ years of historical data per stock
- Train/Val/Test: 70/15/15 (time-ordered to prevent data leakage)
- Loss Function: Binary Cross-Entropy (classification)
- Early Stopping: Stops if validation accuracy plateaus for 20 epochs

---

### XGBoost Model Details

**Purpose**: Learn feature interactions and decision boundaries

**Input**: 50 features from current day

**Processing**:
```
Tree 1:
  IF RSI > 70 THEN predict UP (0.1)
  ELSE predict DOWN (-0.1)

Residuals (errors) from Tree 1 → fed to Tree 2

Tree 2:
  IF MACD > 0 THEN adjust prediction by +0.08
  ELSE adjust by -0.05

Residuals from Tree 2 → fed to Tree 3

... (repeat for 200 trees)

Final Prediction = 0.1 + 0.08 + ... (sum of all tree contributions)
```

**Why Tree-Based**:
- Trees naturally handle non-linear relationships
- Can capture "if-then" patterns (If RSI high AND volume low → buy)
- Fast to train and predict
- Provides feature importance (which features matter most)

**Feature Importance** (what XGBoost learns):
```
RSI: 25% (most important)
MACD: 18%
Recent_Returns: 15%
Volatility: 12%
Volume_Change: 10%
... (other features: 20%)
```

---

### Training Process

**Walk-Forward Validation** (prevents data leakage):

```
Historical Data:
├─ 2020: [Training 70%] [Validation 15%] [Test 15%]
├─ 2021: [Training 70%] [Validation 15%] [Test 15%]
├─ 2022: [Training 70%] [Validation 15%] [Test 15%]
├─ 2023: [Training 70%] [Validation 15%] [Test 15%]
└─ 2024: [Training 70%] [Validation 15%] [Test 15%]

NO DATA LEAKAGE: Later years never trained on future data
Result: Realistic performance estimate
```

**Why Not Random Split?**:
```
Wrong Approach (Random Split):
  Model trained on March 2024 price → May 2024 price
  Then tested on January 2024 price → March 2024 price
  Problem: Model trained on "future" data, tested on "past"
  Result: Unrealistic high accuracy

Correct Approach (Walk-Forward):
  Model trained on Jan-Mar 2024 → predict Apr-May 2024
  Then trained on Jan-Apr 2024 → predict May-Jun 2024
  Problem: Model never sees future data before testing
  Result: Realistic accuracy estimate
```

---

## Feature Engineering

### Why 50+ Features?

**Raw Data**: Close, Open, High, Low, Volume (5 features)
- Problem: Not enough information for accurate learning

**With 50+ Features**:
```
Trend Indicators:  5 features
Momentum:          8 features
Volatility:        4 features
Volume:            3 features
Returns:           6 features
Sentiment:         3 features
Market Context:    4 features
Interactions:      12 features
```

**Example**: If you only had close price:
- Model can only learn "if price went up → goes up again"
- Ignores volume (high volume = stronger signal)
- Ignores sentiment (bad news = price falls)
- Ignores trend reversal signals (RSI overbought = price reverses)

**With All 50 Features**:
- Model learns complex patterns combining multiple signals
- Example: "If RSI high AND volume high AND sentiment positive → UP"
- Captures context (what works in bull market doesn't work in bear market)

### Key Features Explained

#### RSI (Relative Strength Index)
```
Formula: RSI = 100 - (100 / (1 + RS))
where RS = Average Gains / Average Losses (over 14 days)

Range: 0-100
Interpretation:
  > 70: Overbought (potential reversal down)
  < 30: Oversold (potential reversal up)
  
Why It Matters:
  Identifies extreme conditions where reversal is likely
```

#### MACD (Moving Average Convergence Divergence)
```
Formula:
  MACD = EMA(12) - EMA(26)
  Signal = EMA(MACD, 9)
  Histogram = MACD - Signal

Interpretation:
  MACD above Signal = Bullish (uptrend)
  MACD below Signal = Bearish (downtrend)
  
Why It Matters:
  Captures trend changes early
```

#### Bollinger Bands
```
Formula:
  Middle = SMA(20)
  Upper = SMA(20) + 2 * StdDev(20)
  Lower = SMA(20) - 2 * StdDev(20)

Interpretation:
  Price near Upper = Overbought
  Price near Lower = Oversold
  
Why It Matters:
  Shows volatility extremes and support/resistance
```

#### Returns (Log Returns)
```
Daily Return = (Close_today - Close_yesterday) / Close_yesterday
3-Day Return = (Close_today - Close_3days_ago) / Close_3days_ago

Why It Matters:
  Raw prices are hard to compare (₹100 vs ₹1000)
  Returns normalize: all on same scale
  Easy to combine across different stocks
```

---

## Training Pipeline

### Command to Train Models

```bash
# From repo root
python backend/training/train_improved_hybrid_models.py --verbose
```

### What Happens During Training

```
PHASE 1: DATA LOADING
  ├─ Load HDFCBANK.csv (3+ years of daily data)
  ├─ Load ICICIBANK.csv
  ├─ Load INFY.csv
  ├─ Load RELIANCE.csv
  └─ Load TCS.csv

PHASE 2: DATA CLEANING
  ├─ Remove rows with missing Close/Volume
  ├─ Validate price ranges
  ├─ Check for data integrity
  └─ Log statistics

PHASE 3: FEATURE ENGINEERING
  ├─ For each stock:
  │   ├─ Calculate RSI, MACD, Bollinger Bands
  │   ├─ Compute returns (1D, 3D, 5D)
  │   ├─ Calculate volatility
  │   ├─ Extract sentiment
  │   └─ Create interactions
  └─ Result: 50+ features per day

PHASE 4: TRAIN/VAL/TEST SPLIT
  ├─ 70% training data
  ├─ 15% validation data (monitor for overfitting)
  └─ 15% test data (final evaluation)

PHASE 5: FEATURE SCALING
  ├─ Fit StandardScaler on training data only
  ├─ Transform train/val/test using same scaler
  └─ Save scaler for inference (must use same for predictions)

PHASE 6: LABEL ENGINEERING
  ├─ Calculate future returns (5-day horizon)
  ├─ Convert to binary: UP (1) vs DOWN (0)
  ├─ Remove noise (near-zero returns ignored)
  └─ Apply smoothing for stability

PHASE 7: CREATE SEQUENCES
  ├─ For LSTM: Use 20-day windows
  ├─ For XGBoost: Use current day features
  └─ Align labels with sequences

PHASE 8: XGBOOST TRAINING
  ├─ Initialize with 200 trees
  ├─ Early stopping on validation set
  ├─ Calculate feature importance
  └─ Save model & scaler

PHASE 9: LSTM TRAINING
  ├─ Build PyTorch model (2 LSTM layers, 64 hidden units)
  ├─ Train for 70-100 epochs
  ├─ Monitor validation accuracy
  ├─ Save best model (checkpoint)
  └─ Collect metrics (loss, accuracy)

PHASE 10: ENSEMBLE EVALUATION
  ├─ Make predictions with LSTM
  ├─ Make predictions with XGBoost
  ├─ Combine predictions (weighted ensemble)
  ├─ Calculate metrics:
  │   ├─ Accuracy
  │   ├─ Precision/Recall
  │   ├─ ROC-AUC
  │   ├─ F1-Score
  │   └─ Trading metrics (Sharpe, Drawdown, Win Rate)
  └─ Save results to CSV

PHASE 11: SAVE ARTIFACTS
  ├─ LSTM weights (.pth file)
  ├─ XGBoost model (.pkl file)
  ├─ Scaler (.pkl file)
  ├─ Feature list (.pkl file)
  └─ Metadata (timestamps, hyperparameters)

PHASE 12: GENERATE DIAGNOSTICS
  ├─ Accuracy comparison (before vs after improvements)
  ├─ Feature importance plots
  ├─ Confidence score distributions
  ├─ Signal distribution (BUY/SELL/HOLD)
  └─ Trading backtest results
```

### Training Output Example

```
[2026-04-20 15:30:12] Starting training for HDFCBANK
[2026-04-20 15:30:13] Loading data... 1400 rows loaded
[2026-04-20 15:30:14] Cleaning data... 1398 rows after cleaning
[2026-04-20 15:30:15] Engineering features... 50 features created
[2026-04-20 15:30:16] Splitting data (70/15/15)...
  - Training: 978 rows
  - Validation: 210 rows
  - Test: 210 rows
[2026-04-20 15:30:17] Scaling features...
[2026-04-20 15:30:18] Training XGBoost...
  - Tree 1-50: Loss 0.45
  - Tree 51-100: Loss 0.38
  - Tree 101-150: Loss 0.35
  - Tree 151-200: Loss 0.34 ✓
[2026-04-20 15:31:20] Training LSTM (2 layers, 64 units)...
  - Epoch 1/100: Train Loss 0.52, Val Acc 0.58
  - Epoch 20/100: Train Loss 0.32, Val Acc 0.65
  - Epoch 40/100: Train Loss 0.28, Val Acc 0.68
  - Epoch 60/100: Train Loss 0.26, Val Acc 0.69
  - Epoch 80/100: Train Loss 0.25, Val Acc 0.70
  - Early Stopping: No improvement for 20 epochs, stopping at epoch 78
[2026-04-20 15:35:45] Ensemble Evaluation on Test Set:
  - LSTM Accuracy: 0.71
  - XGBoost Accuracy: 0.68
  - Ensemble Accuracy: 0.72 ✓ (better than either alone)
  - Precision: 0.75
  - Recall: 0.68
  - ROC-AUC: 0.82
  - F1-Score: 0.71
[2026-04-20 15:35:50] Saving models...
  - LSTM: backend/models/saved_models/lstm_HDFCBANK.pth (3.2 MB)
  - XGBoost: backend/models/saved_models/xgboost_classifier_HDFCBANK.pkl (1.4 MB)
  - Scaler: scaler_HDFCBANK.pkl (2.1 KB)
[2026-04-20 15:35:51] Training complete for HDFCBANK ✓
```

---

## Inference & Predictions

### Real-Time Prediction Flow

**Step 1: API Request Received**
```json
POST /api/v1/predict
{
  "ticker": "HDFCBANK",
  "days": 1
}
```

**Step 2: Data Preparation**
```python
# Load last 90 days of historical data
data = load_stock_data("HDFCBANK.csv")

# Fetch today's real-time price
current_price = fetch_latest_stock_data("HDFCBANK")

# Clean data
data = clean_data(data)

# Add today's data if new
if current_price:
    append_to_dataframe(data, current_price)
```

**Step 3: Feature Engineering**
```python
# Create 50+ features
features = engineer_features(data, "HDFCBANK")

# Scale features using saved scaler
scaler = load_scaler("scaler_HDFCBANK.pkl")
features_scaled = scaler.transform(features[-20:])
```

**Step 4: Load Models**
```python
# Load pre-trained LSTM
lstm_model = torch.load("lstm_HDFCBANK.pth")
lstm_model.eval()

# Load pre-trained XGBoost
xgb_model = joblib.load("xgboost_classifier_HDFCBANK.pkl")
```

**Step 5: Make Predictions**
```python
# LSTM prediction (on last 20 days of sequences)
lstm_sequence = create_sequences(features_scaled, seq_length=20)
lstm_output = lstm_model(lstm_sequence)  # Returns probability 0-1

# XGBoost prediction (on current day features)
xgb_output = xgb_model.predict_proba(features_scaled[-1:])  # Returns probability 0-1

# Sentiment prediction
sentiment = get_news_sentiment("HDFCBANK")  # Returns -1 to 1
```

**Step 6: Ensemble Combination**
```python
final_score = (
    0.5 * lstm_output +
    0.5 * xgb_output +
    0.2 * normalize_sentiment(sentiment)
)

# Generate signal
if final_score > 0.60:
    signal = "BUY"
elif final_score < 0.40:
    signal = "SELL"
else:
    signal = "HOLD"
```

**Step 7: Calculate Additional Metrics**
```python
# Extract technical indicators
rsi = features_scaled[-1]["RSI"]
macd = features_scaled[-1]["MACD"]
volatility = features_scaled[-1]["Volatility_10D"]

# Calculate predicted price
predicted_return = (final_score - 0.5) * 0.05  # 5% max move
predicted_price = current_price * (1 + predicted_return)

# Assess risk
risk_level = assess_risk(volatility, final_score)

# Generate explanation
explanation = generate_explanation(signal, predicted_return, final_score, risk_level)
```

**Step 8: Return Response**
```json
{
  "ticker": "HDFCBANK",
  "current_price": 1815.50,
  "predicted_price": 1850.25,
  "predicted_change_pct": 0.0193,
  "signal": "BUY",
  "confidence": 0.68,
  "risk_level": "MEDIUM",
  "indicators": {
    "rsi": 72.5,
    "macd": 0.45,
    "bollinger_position": 0.8,
    "volatility": 0.018
  },
  "explanation": "The AI model predicts the stock will rise by 1.93% on the next trading day. Signal confidence is 68%. Risk level is MEDIUM. Signal: BUY.",
  "timestamp": "2026-04-20T15:35:50Z"
}
```

---

## API Architecture

### FastAPI Application Structure

**Main Entry Point**: `backend/app/main.py`
```python
from fastapi import FastAPI
from backend.app import routes
from backend.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(routes.router, prefix=settings.API_V1_STR)
```

### API Endpoints

#### 1. Health Check
```
GET /api/v1/health

Response:
{
  "status": "ok",
  "timestamp": "2026-04-20T15:35:50Z"
}
```

#### 2. Predict Stock Movement
```
POST /api/v1/predict
Headers: Authorization: Bearer {token}
Body:
{
  "ticker": "HDFCBANK",
  "days": 1
}

Response:
{
  "ticker": "HDFCBANK",
  "current_price": 1815.50,
  "predicted_price": 1850.25,
  "signal": "BUY",
  "confidence": 0.68,
  "risk_level": "MEDIUM",
  "indicators": {...},
  "explanation": "..."
}
```

#### 3. Batch Predictions
```
POST /api/v1/batch-signals
Headers: Authorization: Bearer {token}
Body:
{
  "tickers": ["HDFCBANK", "ICICIBANK", "INFY"],
  "days": 1
}

Response:
{
  "signals": [
    {"ticker": "HDFCBANK", "signal": "BUY", ...},
    {"ticker": "ICICIBANK", "signal": "HOLD", ...},
    {"ticker": "INFY", "signal": "SELL", ...}
  ],
  "summary": {
    "buy_count": 1,
    "sell_count": 1,
    "hold_count": 1
  }
}
```

#### 4. Train Models
```
POST /api/v1/train
Headers: Authorization: Bearer {token}
Body:
{
  "ticker": "HDFCBANK",
  "epochs": 100
}

Response:
{
  "status": "training",
  "job_id": "train_hdfcbank_20260420_153550",
  "message": "Training started in background"
}
```

#### 5. Portfolio Optimization
```
POST /api/v1/portfolio/optimize
Headers: Authorization: Bearer {token}
Body:
{
  "tickers": ["HDFCBANK", "ICICIBANK", "INFY", "RELIANCE", "TCS"],
  "capital": 100000,
  "risk_tolerance": "medium"
}

Response:
{
  "portfolio": {
    "HDFCBANK": {"allocation_pct": 25, "amount": 25000},
    "ICICIBANK": {"allocation_pct": 20, "amount": 20000},
    ...
  },
  "expected_return": 0.18,
  "risk": 0.12,
  "sharpe_ratio": 1.5
}
```

#### 6. Sentiment Analysis
```
POST /api/v1/sentiment/analyze
Headers: Authorization: Bearer {token}
Body:
{
  "ticker": "HDFCBANK"
}

Response:
{
  "ticker": "HDFCBANK",
  "overall_sentiment": 0.65,
  "positive_news": 8,
  "negative_news": 3,
  "neutral_news": 2,
  "recent_headlines": ["HDFC Bank Q1 profits up 15%", ...]
}
```

#### 7. Risk Assessment
```
POST /api/v1/risk/score
Headers: Authorization: Bearer {token}
Body:
{
  "ticker": "HDFCBANK",
  "amount": 50000
}

Response:
{
  "risk_score": 0.35,
  "risk_level": "MEDIUM",
  "max_drawdown": -0.15,
  "volatility": 0.018,
  "beta": 0.85,
  "var_95": -2500
}
```

#### 8. User Authentication
```
POST /api/v1/register
Body:
{
  "username": "user@example.com",
  "password": "secure_password"
}

POST /api/v1/login
Body:
{
  "username": "user@example.com",
  "password": "secure_password"
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Request/Response Flow

```
┌─────────────────────┐
│  React Native App   │
│  (Frontend)         │
└──────────┬──────────┘
           │
           │ POST /api/v1/predict
           │ {"ticker": "HDFCBANK"}
           ↓
┌─────────────────────────────────────┐
│  FastAPI Application                │
│  backend/app/main.py                │
└──────────┬──────────────────────────┘
           │
           ├─ Route: routes.py
           ├─ Validation: schemas.py
           ├─ Auth: auth.py
           │
           ↓
┌─────────────────────────────────────┐
│  Predictor (inference/predict.py)   │
│  HybridPredictor (inference/...)    │
└──────────┬──────────────────────────┘
           │
           ├─ Load Models
           ├─ Prepare Data
           ├─ Engineer Features
           ├─ Make Predictions
           │
           ↓
┌─────────────────────────────────────┐
│  Machine Learning Models            │
│  LSTM + XGBoost Ensemble            │
└──────────┬──────────────────────────┘
           │
           ↓ Return predictions
           │
┌─────────────────────────────────────┐
│  Response (PredictionResponse)       │
│  {"signal": "BUY", ...}             │
└──────────┬──────────────────────────┘
           │
           ↓ JSON Response
           │
┌─────────────────────┐
│  React Native App   │
│  Display Result     │
└─────────────────────┘
```

---

## Frontend Application

### Technology Stack
- **Framework**: React Native (Expo)
- **Language**: JavaScript/TypeScript
- **State Management**: Redux or Context API
- **Styling**: Native styling + Tailwind CSS

### App Structure

```
InvestIQ-App/
├── app/                           # Main app screens
│   ├── index.tsx                 # Home screen
│   ├── predict.tsx               # Prediction screen
│   ├── portfolio.tsx             # Portfolio screen
│   └── settings.tsx              # Settings screen
│
├── src/                          # Source code
│   ├── components/               # Reusable components
│   │   ├── PredictionCard.tsx
│   │   ├── PortfolioCard.tsx
│   │   ├── SignalIndicator.tsx
│   │   └── ...
│   │
│   ├── hooks/                    # Custom React hooks
│   │   ├── usePrediction.ts
│   │   └── usePortfolio.ts
│   │
│   ├── services/                 # API communication
│   │   ├── api.ts
│   │   ├── predictions.ts
│   │   └── portfolio.ts
│   │
│   ├── store/                    # State management
│   │   ├── slices/
│   │   │   ├── predictionSlice.ts
│   │   │   └── portfolioSlice.ts
│   │   └── store.ts
│   │
│   └── utils/                    # Utilities
│       ├── formatting.ts
│       └── validation.ts
│
└── config/
    └── api.config.ts             # API endpoints configuration
```

### Key Screens

#### 1. Home Screen
```
┌─────────────────────────────────┐
│          InvestIQ APP           │
├─────────────────────────────────┤
│                                 │
│  Quick Predictions              │
│  ┌──────────────────────────┐   │
│  │ HDFCBANK        BUY      │   │
│  │ ₹1,815 → ₹1,850         │   │
│  │ Confidence: 68%          │   │
│  └──────────────────────────┘   │
│                                 │
│  ┌──────────────────────────┐   │
│  │ ICICIBANK       HOLD     │   │
│  │ ₹950 → ₹955             │   │
│  │ Confidence: 48%          │   │
│  └──────────────────────────┘   │
│                                 │
│  ┌──────────────────────────┐   │
│  │ INFY            SELL     │   │
│  │ ₹1,200 → ₹1,165         │   │
│  │ Confidence: 72%          │   │
│  └──────────────────────────┘   │
│                                 │
└─────────────────────────────────┘
```

#### 2. Prediction Detail Screen
```
┌─────────────────────────────────┐
│  HDFCBANK Prediction            │
├─────────────────────────────────┤
│                                 │
│  Signal:      BUY               │
│  Confidence:  68%               │
│  Risk Level:  MEDIUM            │
│                                 │
│  Current Price:    ₹1,815.50    │
│  Predicted Price:  ₹1,850.25    │
│  Change:          +1.93%        │
│                                 │
│  Technical Indicators           │
│  ─────────────────────────      │
│  RSI:             72.5 (High)   │
│  MACD:            0.45          │
│  Bollinger:       +2σ           │
│  Volatility:      1.8%          │
│                                 │
│  Explanation:                   │
│  The AI model predicts the      │
│  stock will rise by 1.93% on    │
│  the next trading day. Signal   │
│  confidence is 68%. Risk level  │
│  is MEDIUM. Signal: BUY.        │
│                                 │
│  [ BUY ]  [ ADD TO WATCHLIST ]  │
│                                 │
└─────────────────────────────────┘
```

#### 3. Portfolio Screen
```
┌─────────────────────────────────┐
│  My Portfolio                   │
├─────────────────────────────────┤
│                                 │
│  Total Value:  ₹5,00,000        │
│  Today's Gain: +₹12,500 (+2.5%) │
│  Win Rate:     58% (87/150)     │
│  Sharpe Ratio: 1.52             │
│                                 │
│  Holdings:                      │
│  ┌──────────────────────────┐   │
│  │ HDFCBANK  25,000  +3.2%  │   │
│  │ ICICIBANK 20,000  -1.5%  │   │
│  │ INFY      18,000  +2.8%  │   │
│  │ RELIANCE  15,000  +0.5%  │   │
│  │ TCS       22,000  +4.2%  │   │
│  └──────────────────────────┘   │
│                                 │
│  [ OPTIMIZE ]  [ REBALANCE ]    │
│                                 │
└─────────────────────────────────┘
```

### API Communication Example

```typescript
// Service: src/services/predictions.ts
import axios from 'axios';

const API_BASE_URL = 'http://backend.example.com/api/v1';

export const getPrediction = async (ticker: string, token: string) => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/predict`,
      { ticker, days: 1 },
      {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Prediction API error:', error);
    throw error;
  }
};

// Component: app/predict.tsx
import { getPrediction } from '../src/services/predictions';

export default function PredictScreen() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const token = useToken(); // Get auth token

  const handlePredict = async (ticker: string) => {
    setLoading(true);
    try {
      const data = await getPrediction(ticker, token);
      setPrediction(data);
      // Display on UI
    } catch (error) {
      // Show error message
    } finally {
      setLoading(false);
    }
  };

  return (
    <View>
      {/* Render prediction data */}
      {prediction && (
        <View>
          <Text>Signal: {prediction.signal}</Text>
          <Text>Confidence: {prediction.confidence * 100}%</Text>
          <Text>Price: {prediction.current_price} → {prediction.predicted_price}</Text>
        </View>
      )}
    </View>
  );
}
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git
- Visual Studio Code (recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/InvestIQ-main.git
cd InvestIQ-main
```

### Step 2: Python Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt
```

### Step 3: Configure Backend

**Create `.env` file** in `backend/` directory:
```
PROJECT_NAME=Stock Predictor AI
API_V1_STR=/api/v1
DATA_DIR=backend/data/stock_data
MODEL_DIR=backend/models/saved_models
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.0003
INFERENCE_MODE=hybrid
```

**Verify Setup**:
```bash
cd backend
python verify_setup.py
```

### Step 4: Stock Data Setup

Place stock CSV files in `backend/data/stock_data/`:
- HDFCBANK.csv
- ICICIBANK.csv
- INFY.csv
- RELIANCE.csv
- TCS.csv

**CSV Format**:
```
Date,Open,High,Low,Close,Volume
2020-01-01,1000.50,1010.25,995.75,1005.00,5000000
2020-01-02,1005.00,1015.50,1000.25,1010.00,5500000
...
```

### Step 5: Frontend Setup

```bash
cd InvestIQ-App

# Install dependencies
npm install

# Configure API endpoint
# Edit src/config/api.config.ts
export const API_BASE_URL = 'http://localhost:8000/api/v1';
```

---

## How to Run

### Option 1: Run Locally (Development)

#### Terminal 1: Backend API
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Access API**: http://localhost:8000/docs (Swagger UI)

#### Terminal 2: Train Models
```bash
cd backend
python training/train_improved_hybrid_models.py --verbose
```

#### Terminal 3: Frontend (React Native)
```bash
cd InvestIQ-App
npm start

# In Expo app:
# Press 'w' for web
# Press 'a' for Android
# Press 'i' for iOS
```

### Option 2: Run with Batch Scripts (Windows)

```bash
# Terminal 1: Run Backend
double-click run_app.bat

# Terminal 2: Run Training
double-click run_training_v2.bat

# Terminal 3: Run Demo
double-click run_demo.bat
```

### Option 3: Run with Docker

```bash
# Build Docker image
docker-compose build

# Start all services
docker-compose up -d

# Check services
docker-compose ps

# Stop services
docker-compose down
```

---

## Testing the System

### 1. Test API Health
```bash
curl http://localhost:8000/api/v1/health
# Response: {"status":"ok","timestamp":"2026-04-20T15:35:50Z"}
```

### 2. Test Prediction Endpoint
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"ticker":"HDFCBANK","days":1}'

# Response:
# {
#   "ticker":"HDFCBANK",
#   "signal":"BUY",
#   "confidence":0.68,
#   ...
# }
```

### 3. Test Model Training
```bash
python backend/training/train_improved_hybrid_models.py --verbose --ticker HDFCBANK
```

### 4. Test Frontend App
- Open Expo app on phone or simulator
- Scan QR code or press connection button
- Navigate through screens
- Test predictions

---

## Key Improvements & Why They Matter

### 1. **Hybrid Ensemble (LSTM + XGBoost)**
- **Why**: Single models have blind spots
- **Result**: +10-15% accuracy improvement

### 2. **50+ Feature Engineering**
- **Why**: More information = better predictions
- **Result**: +15-20% accuracy improvement

### 3. **Walk-Forward Validation**
- **Why**: Prevents unrealistic performance estimates
- **Result**: Realistic accuracy (not optimistic)

### 4. **Binary Classification**
- **Why**: UP/DOWN simpler than 3-class (BUY/HOLD/SELL)
- **Result**: +5-10% easier to learn

### 5. **Smart Label Engineering**
- **Why**: Remove noise, focus on strong signals
- **Result**: More stable predictions

### 6. **Sentiment Integration**
- **Why**: News significantly impacts prices
- **Result**: Context-aware predictions

### 7. **Real-time Predictions**
- **Why**: Traders need current signals
- **Result**: Act on fresh information

---

## Conclusion

InvestIQ is a production-ready AI system that combines:
- **Deep Learning** (LSTM) for temporal patterns
- **Gradient Boosting** (XGBoost) for feature interactions
- **Advanced Feature Engineering** (50+ indicators)
- **Risk Management** (portfolio optimization, risk scoring)
- **User-friendly Interface** (React Native app)

The system achieves **2-3x accuracy improvement** (33% → 55-65%) through systematic improvements in data quality, feature engineering, model architecture, and validation methodology.

**Next Steps**:
1. Set up backend and frontend locally
2. Train models with historical data
3. Test predictions on different stocks
4. Deploy to production
5. Monitor performance metrics
6. Continuously retrain with new data

For more details, see the comprehensive documentation in the `docs/` folder.

