# Complete InvestIQ Backend Inventory & Documentation

**Generated**: 2026-04-09  
**Project**: InvestIQ - Stock Prediction AI System  
**Backend Version**: 2.0 (with retraining & evaluation suite)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Module Documentation](#module-documentation)
4. [File Inventory by Size](#file-inventory-by-size)
5. [Dependencies & Requirements](#dependencies--requirements)
6. [Data Flow & Architecture](#data-flow--architecture)
7. [Key Components](#key-components)
8. [Model Architecture](#model-architecture)
9. [API Endpoints](#api-endpoints)
10. [Scripts & Utilities](#scripts--utilities)

---

## Project Overview

### Purpose
A machine learning-powered stock prediction system using:
- **LSTM Attention**: 1-day return forecasting
- **Transformer**: 7-day return forecasting  
- **XGBoost**: Buy/Hold/Sell signal classification
- **Ensemble**: Combined predictions from all models

### Indian Stock Universe
- HDFCBANK (HDFC Bank)
- ICICIBANK (ICICI Bank)
- INFY (Infosys)
- RELIANCE (Reliance Industries)
- TCS (Tata Consultancy Services)

### Features Generated: **21 per stock per day**
- **Price Data** (5): Open, High, Low, Close, Volume
- **Technical Indicators** (13): SMA-20, SMA-50, RSI, Bollinger Bands, VWAP, MACD, ATR, Log_Return, Volume_Change, Rolling_Volatility
- **Market Dynamics** (1): Market_Correlation (vs NIFTY-50)
- **External Factors** (2): Sentiment (simulated), Macro_Score (simulated)

### Data Timeline
- **Training**: 2015-2024 (9 years per ticker)
- **Test**: 2023-2024 (1 year per ticker)
- **Frequency**: Daily

---

## Directory Structure

```
backend/
├── __init__.py                          (Empty package marker)
├── README.md                            (Backend overview)
├── requirements.txt                     (Python dependencies)
├── verify_setup.py                      (Setup verification script)
├── setup_env.bat                        (Windows environment setup)
├── run_server.bat                       (Launch backend server)
│
├── app/                                 (FastAPI Application)
│   ├── __init__.py
│   ├── main.py                          (FastAPI app initialization, routes)
│   ├── routes.py                        (API endpoint handlers)
│   ├── auth.py                          (Authentication & JWT tokens)
│   ├── schemas.py                       (Pydantic data models for API)
│   └── users.json                       (User credentials storage)
│
├── core/                                (Core Configuration)
│   ├── __init__.py
│   ├── config.py                        (Settings: paths, hyperparameters, model config)
│   ├── logging.py                       (Logger setup with file & console output)
│   └── exceptions.py                    (Custom exception classes)
│
├── data/                                (Data Management)
│   ├── __init__.py
│   ├── realtime.py                      (Fetch real-time stock data via yfinance)
│   ├── update_stock_data.py             (Update historical stock data)
│   └── stock_data/                      (Historical CSV data)
│       ├── README.md
│       ├── HDFCBANK.csv
│       ├── ICICIBANK.csv
│       ├── INFY.csv
│       ├── RELIANCE.csv
│       └── TCS.csv
│
├── preprocessing/                       (Data Preprocessing)
│   ├── __init__.py
│   ├── cleaning.py                      (Load, clean, handle missing data)
│   └── scaling.py                       (StandardScaler wrapper for feature scaling)
│
├── features/                            (Feature Engineering)
│   ├── __init__.py
│   ├── indicators.py                    (Technical indicators: SMA, RSI, MACD, etc.)
│   ├── external_data.py                 (Market correlation, sentiment, macro data)
│   ├── risk.py                          (Risk metrics: VaR, Sharpe ratio)
│   ├── portfolio.py                     (Portfolio optimization)
│   ├── sentiment.py                     (News sentiment analysis)
│   ├── realtime_price.py                (Real-time price features)
│   └── timeframes.py                    (Multi-timeframe indicators)
│
├── models/                              (Neural Network Models)
│   ├── __init__.py
│   ├── lstm_attention.py                (LSTM with attention mechanism, 87 lines)
│   ├── transformer.py                   (Multi-head transformer, 150+ lines)
│   ├── xgboost_fusion.py                (XGBoost classifier wrapper, 65 lines)
│   ├── ensemble.py                      (Ensemble combining all models, 80 lines)
│   └── saved_models/                    (Trained model files)
│       ├── lstm_attention_HDFCBANK.pth
│       ├── lstm_attention_ICICIBANK.pth
│       ├── lstm_attention_INFY.pth
│       ├── lstm_attention_RELIANCE.pth
│       ├── lstm_attention_TCS.pth
│       ├── transformer_HDFCBANK.pth
│       ├── transformer_ICICIBANK.pth
│       ├── transformer_INFY.pth
│       ├── transformer_RELIANCE.pth
│       ├── transformer_TCS.pth
│       ├── xgboost_fusion_HDFCBANK.pkl
│       ├── xgboost_fusion_ICICIBANK.pkl
│       ├── xgboost_fusion_INFY.pkl
│       ├── xgboost_fusion_RELIANCE.pkl
│       ├── xgboost_fusion_TCS.pkl
│       ├── scaler_HDFCBANK.pkl         (Feature scaler object)
│       ├── scaler_ICICIBANK.pkl
│       ├── scaler_INFY.pkl
│       ├── scaler_RELIANCE.pkl
│       ├── scaler_TCS.pkl
│       └── evaluation_results/         (Evaluation outputs)
│           ├── comprehensive_evaluation_report.txt    (Detailed metrics)
│           ├── 01_rmse_comparison.png                 (Graph)
│           ├── 02_r2_comparison.png
│           ├── 03_directional_accuracy.png
│           ├── 04_xgboost_metrics.png
│           ├── 05_performance_heatmap.png
│           ├── 06_ticker_performance.png
│           ├── 07_ensemble_improvement.png
│           ├── 08_box_plots.png
│           ├── model_summary.csv
│           ├── detailed_comparison.csv
│           ├── statistical_analysis.txt
│           └── paper_tables.tex
│
├── training/                            (Model Training)
│   ├── __init__.py
│   ├── train.py                         (Main training pipeline, 300+ lines)
│   ├── train_remaining.py               (Retrain specific tickers)
│   ├── auto_retrain.py                  (Automatic retraining scheduler)
│   └── tuner.py                         (Hyperparameter tuning)
│
├── inference/                           (Prediction & Inference)
│   ├── __init__.py
│   ├── predict.py                       (Make predictions with trained models)
│   └── stock_predictor.py               (High-level prediction API)
│
├── evaluation/                          (Model Evaluation)
│   ├── __init__.py
│   ├── metrics.py                       (Evaluation metrics: RMSE, R², Accuracy)
│   └── evaluate.py                      (Comprehensive model evaluation)
│
├── explainability/                      (Model Interpretability)
│   ├── __init__.py
│   └── shap_explainer.py               (SHAP values for feature importance)
│
├── utils/                               (Utility Functions)
│   ├── __init__.py
│   └── training_utils.py               (Helper functions for training)
│
├── scripts/                             (Standalone Scripts)
│   ├── calculate_accuracy.py           (Compute accuracy metrics)
│   ├── check_setup.py                   (Verify environment setup)
│   ├── debug_yfinance.py                (Debug data fetching)
│   ├── demo.py                          (Demo predictions)
│   ├── fetch_latest_news.py            (Fetch news articles)
│   ├── predict_all.py                   (Predict for all tickers)
│   ├── test_imports.py                  (Verify imports work)
│   ├── test_real_news.py                (Test news sentiment)
│   ├── test_write.py                    (Test file writing)
│   ├── train_all.py                     (Train all models)
│   ├── verify_all.py                    (Full system verification)
│   ├── verify_prediction.py             (Verify prediction outputs)
│   ├── run_evaluation.py                (Run comprehensive evaluation)
│   ├── comprehensive_model_evaluation.py (Main evaluation suite, 700+ lines)
│   ├── generate_paper_reports.py        (Generate CSV/LaTeX reports)
│   ├── generate_prediction_plots.py     (Generate detailed prediction graphs)
│   └── retrain_for_evaluation.py        (Retrain models for evaluation)
│
├── tests/                               (Unit Tests)
│   ├── __init__.py
│   ├── test_concept.py                  (Concept validation tests)
│   ├── test_features.py                 (Feature engineering tests)
│   ├── test_sentiment_manual.py        (Manual sentiment tests)
│   └── output.txt                       (Test results log)
│
├── backtesting/                         (Backtest Engine)
│   └── backtest.py                      (Backtest trading strategies)
```

---

## Module Documentation

### 1. **core/config.py** (Configuration & Settings)

**Size**: ~1 KB  
**Purpose**: Centralized configuration management

**Key Components**:
```python
class Settings(BaseSettings):
    PROJECT_NAME: str = "Stock Predictor AI"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    DATA_DIR: str = "backend/data/stock_data"
    MODEL_DIR: str = "backend/models/saved_models"
    
    # Model Configuration
    SEQ_LENGTH: int = 90          # 90-day lookback window
    TEST_SIZE: float = 0.2         # 80/20 train/test split
    FORECAST_HORIZON: int = 7      # 7-day forecast
    
    # Training Parameters
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    
    # Advanced Model Params
    DROPOUT: float = 0.1
    NHEAD: int = 4                # Transformer heads
    NUM_LAYERS: int = 2           # Transformer/LSTM layers
```

---

### 2. **core/logging.py** (Logging System)

**Size**: ~1 KB  
**Purpose**: Centralized logging with file & console output

**Features**:
- Console handler (INFO level)
- File handler (DEBUG level)
- Format: `[%(asctime)s] - %(name)s - %(levelname)s - %(message)s`
- Logger name: `stock_predictor`

---

### 3. **preprocessing/cleaning.py** (Data Cleaning)

**Size**: ~3 KB  
**Purpose**: Load, clean, and validate stock data

**Functions**:
```python
def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data from file"""
    
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, drop duplicates, drop NaN rows"""
    
def validate_data(df: pd.DataFrame) -> bool:
    """Validate data integrity"""
```

**Handles**:
- Missing values (forward fill, drop)
- Duplicate rows
- Data type conversion
- Index management

---

### 4. **preprocessing/scaling.py** (Feature Scaling)

**Size**: ~2 KB  
**Purpose**: Normalize features for neural networks

**Class: StockScaler**
```python
class StockScaler:
    def __init__(self, scaler_type='standard')  # StandardScaler or MinMaxScaler
    
    def fit_transform(self, df, feature_cols):
        """Fit scaler and transform data"""
    
    def transform(self, df):
        """Apply fitted scaler"""
    
    def inverse_transform(self, df_scaled):
        """Reverse scaling (denormalize)"""
    
    def save(self, name):
        """Pickle scaler to disk"""
    
    def load(self, name):
        """Load scaler from disk"""
```

---

### 5. **features/indicators.py** (Technical Indicators)

**Size**: ~4 KB  
**Purpose**: Generate 13 technical indicators

**Indicators Generated**:
1. **Moving Averages**: SMA-20, SMA-50
2. **Momentum**: RSI (14-day)
3. **Bollinger Bands**: Upper, Lower bands
4. **Volume**: VWAP (Volume Weighted Average Price)
5. **Trend**: MACD, MACD_Signal, MACD_Histogram
6. **Volatility**: ATR (Average True Range)
7. **Returns**: Log_Return, Volume_Change, Rolling_Volatility

**Functions**:
```python
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add 13 technical indicators"""
    
def add_market_correlation(df: pd.DataFrame, market_df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Calculate rolling correlation with market index (NIFTY-50)"""
```

---

### 6. **features/external_data.py** (External Features)

**Size**: ~3 KB  
**Purpose**: Add sentiment and macro indicators

**Class: ExternalDataSimulator**
```python
@staticmethod
def get_sentiment(ticker, date=None) -> float:
    """Random sentiment score (-1.0 to 1.0)"""
    
@staticmethod
def get_macro_score(date=None) -> float:
    """Random macro score (0 to 100)"""
    
@staticmethod
def add_external_features(df, ticker, deterministic=False):
    """Add Sentiment + Macro_Score columns"""
    
@staticmethod
def fetch_market_index(ticker="^NSEI", start_date=None, end_date=None):
    """Fetch NIFTY-50 data via yfinance"""
    
@staticmethod
def fetch_live_news(ticker_symbol):
    """Fetch real news via yfinance"""
    
@staticmethod
def fetch_live_sentiment(ticker_symbol):
    """Fetch live sentiment via FinBERT"""
```

---

### 7. **models/lstm_attention.py** (LSTM with Attention)

**Size**: ~2 KB  
**Lines**: 87  
**Purpose**: 1-day return prediction using LSTM + Attention

**Architecture**:
```
Input (batch, 90, 21)
    ↓
Bidirectional LSTM (2 layers, 128 hidden units)
    ↓
Layer Normalization
    ↓
Attention Mechanism (soft attention over sequence)
    ↓
Context Vector (batch, 256)
    ↓
FC1 (256 → 128) + ReLU + Dropout
    ↓
FC2 (128 → 1)
    ↓
Output (batch, 1) - 1-day return prediction
```

**Classes**:
```python
class Attention(nn.Module):
    """Soft attention over LSTM outputs"""
    forward(lstm_output) -> (context, weights)

class LSTMAttentionModel(nn.Module):
    """Full LSTM + Attention model"""
    __init__(input_dim, hidden_dim=128, num_layers=2, ...)
    forward(x) -> prediction
```

**Key Features**:
- Bidirectional LSTM captures patterns in both directions
- Attention weights show which timesteps are important
- Output: 1-day return (scalar)
- Target: Log_Return column

---

### 8. **models/transformer.py** (Transformer Model)

**Size**: ~4 KB  
**Lines**: 150+  
**Purpose**: 7-day return sequence prediction using Transformer

**Architecture**:
```
Input (batch, 90, 21)
    ↓
Input Embedding (21 → 64 dimensions)
    ↓
Positional Encoding (add position information)
    ↓
Transformer Encoder (2 blocks × 4 heads)
    |  ├─ Multi-Head Self-Attention
    |  └─ Feed-Forward Network (64 → 256 → 64)
    ↓
Sequence-to-Sequence Decoder
    ├─ Context aggregation
    └─ Output projection
    ↓
Output (batch, 7, 1) - 7-day return forecast
```

**Classes**:
```python
class TimeSeriesTransformer(nn.Module):
    """Transformer for time series forecasting"""
    __init__(input_dim, d_model=64, nhead=4, num_layers=2, 
             dropout=0.1, output_dim=1, forecast_horizon=7)
    
    forward(x) -> (batch, 7, 1)
```

**Key Features**:
- Self-attention learns temporal dependencies
- Multi-head attention (4 parallel attention heads)
- Positional encoding for sequence order
- Output: 7-day return sequence

---

### 9. **models/xgboost_fusion.py** (XGBoost Classifier)

**Size**: ~2 KB  
**Lines**: 65  
**Purpose**: 3-class classification (Buy/Hold/Sell)

**Classification Logic**:
```
future_return = (Price[t+5] - Price[t]) / Price[t]

if future_return > 0.01:    → Class 2: BUY
elif future_return < -0.01: → Class 0: SELL
else:                       → Class 1: HOLD
```

**Class: XGBoostFusionModel**
```python
def __init__(self):
    self.model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        reg_alpha=0.1,      # L1 regularization
        reg_lambda=0.1      # L2 regularization
    )

def prepare_labels(df, horizon=5, threshold=0.01):
    """Generate buy/sell/hold labels"""
    
def train(X, y, eval_set=None):
    """Train with validation for early stopping"""
    
def predict(X) -> np.array:
    """Return class predictions (0, 1, or 2)"""
    
def predict_proba(X) -> np.array:
    """Return class probabilities"""
```

---

### 10. **models/ensemble.py** (Ensemble Model)

**Size**: ~2 KB  
**Lines**: 80  
**Purpose**: Combine LSTM, Transformer, XGBoost predictions

**Ensemble Strategy**:
```python
ensemble_prediction = (0.40 × LSTM + 0.60 × Transformer)
```

**Weights**:
- LSTM: 40% (reliable for near-term)
- Transformer: 60% (better for longer horizons)
- XGBoost: Used separately for classification

---

### 11. **training/train.py** (Main Training Pipeline)

**Size**: ~10 KB  
**Lines**: 300+  
**Purpose**: Full end-to-end training for LSTM & Transformer

**Pipeline Flow**:
```
1. Load CSV data
   ↓
2. Clean data (handle NaN, duplicates)
   ↓
3. Feature engineering (13 indicators + market corr + external)
   ↓
4. Scale features (StandardScaler)
   ↓
5. Create sequences (90-day windows, 7-day targets)
   ↓
6. Train/test split (80/20, time-based)
   ↓
7. Initialize model (LSTM or Transformer)
   ↓
8. Training loop
   |  ├─ Batch training (MSE loss)
   |  ├─ Validation (check convergence)
   |  └─ Learning rate scheduling
   ↓
9. Save model checkpoints
   ↓
10. Save scaler (for inference)
```

**Key Function**:
```python
def create_sequences(data: np.ndarray, seq_length: int, 
                    forecast_horizon: int, target_col_idx: int):
    """
    Input:  (sequence of 90 days × 21 features)
    Output: (prediction of next 7 days for target feature)
    """
    return X (n_samples, 90, 21), y (n_samples, 7, 1)

def train_pipeline(file_path: str):
    """Complete training for one ticker"""
```

---

### 12. **inference/predict.py** (Make Predictions)

**Size**: ~2 KB  
**Purpose**: Load models and make predictions

**Functions**:
```python
def predict_lstm(model, X):
    """Predict 1-day return"""
    
def predict_transformer(model, X):
    """Predict 7-day return sequence"""
    
def predict_xgboost(model, X):
    """Predict buy/sell/hold signal"""
    
def predict_ensemble(lstm_pred, transformer_pred):
    """Combine predictions"""
```

---

### 13. **inference/stock_predictor.py** (High-Level API)

**Size**: ~5 KB  
**Purpose**: Complete prediction pipeline

**Class: StockPredictor**
```python
def __init__(self, ticker):
    """Load all models and scalers for a ticker"""
    
def predict(days: int = 7):
    """
    1. Fetch latest data
    2. Preprocess & calculate features
    3. Create sequences
    4. Get predictions from all models
    5. Return ensemble prediction
    """
    
def forecast_return() -> float:
    """Return expected return (%)"""
    
def signal() -> str:
    """Return 'BUY', 'HOLD', or 'SELL'"""
    
def confidence() -> float:
    """Return confidence score (0-1)"""
```

---

### 14. **evaluation/metrics.py** (Evaluation Metrics)

**Size**: ~2 KB  
**Purpose**: Calculate performance metrics

**Regression Metrics** (LSTM, Transformer):
```python
def calculate_metrics(y_true, y_pred):
    return {
        'MSE': Mean Squared Error,
        'RMSE': Root Mean Squared Error,
        'MAE': Mean Absolute Error,
        'R2': R-squared score,
        'MAPE': Mean Absolute Percentage Error,
        'Directional_Accuracy': % correct direction predictions
    }
```

**Classification Metrics** (XGBoost):
```python
{
    'Accuracy': % correct predictions,
    'Precision': True positives / (True + False positives),
    'Recall': True positives / (True positives + False negatives),
    'F1_Score': Harmonic mean of precision & recall
}
```

---

### 15. **scripts/comprehensive_model_evaluation.py** (Main Evaluation)

**Size**: ~18 KB  
**Lines**: 700+  
**Purpose**: Complete model evaluation with graphs

**Class: ComprehensiveModelEvaluator**

**Methods**:
```python
def evaluate_lstm(ticker) -> dict:
    """Evaluate LSTM on test set, return metrics"""
    
def evaluate_transformer(ticker) -> dict:
    """Evaluate Transformer on test set"""
    
def evaluate_xgboost(ticker) -> dict:
    """Evaluate XGBoost on test set"""
    
def evaluate_ensemble(ticker) -> dict:
    """Evaluate ensemble predictions"""
    
def _plot_rmse_comparison():
    """Bar chart: RMSE across models/tickers"""
    
def _plot_r2_comparison():
    """Bar chart: R² scores"""
    
def _plot_directional_accuracy():
    """Bar chart: Directional accuracy %"""
    
def _plot_xgboost_metrics():
    """Stacked bar: XGBoost classification metrics"""
    
def _plot_model_performance_heatmap():
    """Heatmap: All metrics × models"""
    
def _plot_ticker_performance():
    """Heatmap: Performance across tickers"""
    
def _plot_ensemble_improvement():
    """Line chart: Ensemble advantage"""
    
def _plot_box_plots():
    """Box plots: Distribution of metrics"""
```

**Outputs**:
- 8 PNG graphs (300 DPI, publication quality)
- comprehensive_evaluation_report.txt (full metrics)
- model_summary.csv (summary statistics)
- detailed_comparison.csv (per-model breakdown)
- paper_tables.tex (LaTeX tables)

---

### 16. **scripts/retrain_for_evaluation.py** (Retraining Suite)

**Size**: ~15 KB  
**Lines**: 600+  
**Purpose**: Retrain models with current 21-feature set

**Functions**:
```python
def train_lstm_model(ticker):
    """Retrain LSTM with correct architecture (input_dim=21)"""
    
def train_transformer_model(ticker):
    """Retrain Transformer with correct architecture"""
    
def train_xgboost_model(ticker):
    """Retrain XGBoost classifier"""
    
def main():
    """Main orchestrator - retrains all models for all tickers"""
```

**Configuration**:
- Epochs: 50 (can be increased to 100)
- Batch size: 32
- Learning rate: 0.001
- Early stopping: Yes (on validation loss)
- Device: GPU if available, else CPU

---

## File Inventory by Size

### Smallest Files (Configuration & Setup)
1. `__init__.py` files - ~0 KB (empty marker files, 15+ copies)
2. `core/config.py` - ~1 KB
3. `core/logging.py` - ~1 KB
4. `core/exceptions.py` - ~1 KB

### Small Files (Utilities)
5. `preprocessing/cleaning.py` - ~3 KB
6. `preprocessing/scaling.py` - ~2 KB
7. `features/external_data.py` - ~3 KB
8. `models/lstm_attention.py` - ~2 KB
9. `models/ensemble.py` - ~2 KB
10. `models/xgboost_fusion.py` - ~2 KB
11. `inference/predict.py` - ~2 KB
12. `evaluation/metrics.py` - ~2 KB

### Medium Files (Core Functionality)
13. `models/transformer.py` - ~4 KB
14. `features/indicators.py` - ~4 KB
15. `app/schemas.py` - ~3 KB
16. `inference/stock_predictor.py` - ~5 KB

### Large Files (Training & Scripts)
17. `training/train.py` - ~10 KB (300+ lines)
18. `scripts/comprehensive_model_evaluation.py` - ~18 KB (700+ lines)
19. `scripts/retrain_for_evaluation.py` - ~15 KB (600+ lines)
20. `scripts/generate_paper_reports.py` - ~8 KB (320 lines)

### Huge Files (Main Application)
21. `app/main.py` - ~8 KB (350+ lines, FastAPI routes)
22. `app/routes.py` - ~10 KB (API handlers)
23. `data/realtime.py` - ~4 KB (yfinance wrapper)
24. `backtesting/backtest.py` - ~6 KB

### Data Files (Stock Data)
25. CSV files (5 tickers) - ~100-200 KB each
26. Scaler pickle files (5 tickers) - ~50 KB each
27. LSTM model files (5 tickers) - ~500 KB-1 MB each
28. Transformer model files (5 tickers) - ~300 KB-600 KB each
29. XGBoost model files (5 tickers) - ~50-100 KB each

### Output Files (Evaluation Results)
30. comprehensive_evaluation_report.txt - ~10 KB
31. PNG graphs (8 files) - ~100-300 KB each
32. CSV reports (2 files) - ~20-50 KB each
33. paper_tables.tex - ~5 KB

---

## Dependencies & Requirements

### Core Libraries
```txt
# Data Processing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0

# Deep Learning
torch>=1.10.0
torchvision>=0.11.0

# Gradient Boosting
xgboost>=1.5.0

# Data Fetching
yfinance>=0.1.70
requests>=2.26.0
beautifulsoup4>=4.9.0

# Web Framework
fastapi>=0.70.0
uvicorn>=0.15.0
pydantic>=1.8.0

# Database
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0

# Data & Features
ta>=0.10.0                  # Technical Analysis
textblob>=0.17.0            # Sentiment Analysis
transformers>=4.0.0         # FinBERT for sentiment
shap>=0.40.0                # Model explainability

# Utilities
joblib>=1.0.0               # Model serialization
python-dateutil>=2.8.0
pytz>=2021.1

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Optimization
optuna>=2.9.0               # Hyperparameter tuning
scipy>=1.7.0

# Authentication
python-jose>=3.0.0          # JWT tokens
bcrypt>=3.2.0               # Password hashing
```

---

## Data Flow & Architecture

### Training Pipeline
```
Raw CSV Data
    ↓
load_data() → Validation & Type Conversion
    ↓
clean_data() → Handle NaN, Duplicates
    ↓
add_technical_indicators() → 13 indicators
    ↓
add_market_correlation() → Market_Correlation
    ↓
ExternalDataSimulator.add_external_features() → Sentiment, Macro_Score
    ↓
data = 21 features per day
    ↓
StockScaler.fit_transform() → Normalize [0, 1]
    ↓
create_sequences() → (90, 21) input, (7, 1) target
    ↓
train_lstm_model() → Save lstm_attention_{ticker}.pth
train_transformer_model() → Save transformer_{ticker}.pth
train_xgboost_model() → Save xgboost_fusion_{ticker}.pkl
```

### Inference Pipeline
```
Latest Stock Data (Real-time)
    ↓
load_data() & clean_data()
    ↓
Feature Engineering (same as training)
    ↓
StockScaler.transform() → Normalize
    ↓
Create Sequence (last 90 days)
    ↓
LSTM Model → 1-day prediction
    ↓
Transformer Model → 7-day prediction
    ↓
XGBoost Model → Buy/Hold/Sell
    ↓
Ensemble (40% LSTM + 60% Transformer)
    ↓
Return {return_forecast, signal, confidence}
```

### API Flow
```
Client Request
    ↓
Authentication (JWT Token)
    ↓
Route Handler (FastAPI)
    ↓
StockPredictor.predict()
    ↓
JSON Response with predictions
```

---

## Key Components

### 1. Model Training System
- **Automatic**: `training/train.py` handles all preprocessing
- **Logging**: Detailed training progress logged to file
- **Checkpointing**: Best models saved during training
- **Early Stopping**: Stop if validation loss plateaus
- **Scheduling**: Dynamic learning rate adjustment

### 2. Feature Engineering Pipeline
- **13 Technical Indicators**: SMA, RSI, MACD, Bollinger Bands, etc.
- **Market Correlation**: Calculated with NIFTY-50
- **Sentiment Analysis**: News sentiment (simulated/real)
- **Macro Indicators**: Economic health indicators
- **Total**: 21 features per stock per day

### 3. Multi-Model Ensemble
- **LSTM** (40%): Good for short-term patterns
- **Transformer** (60%): Better for longer forecasts
- **XGBoost**: Independent classifier for buy/sell signals
- **Combined**: Weighted average for final prediction

### 4. Evaluation Framework
- **Regression Metrics**: RMSE, MAE, R², Directional Accuracy
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: 8 publication-quality graphs
- **Reports**: CSV, LaTeX, text formats

### 5. API Server
- **Framework**: FastAPI (async, fast)
- **Authentication**: JWT tokens
- **Routes**: Prediction, model info, historical data
- **Error Handling**: Comprehensive try/catch blocks
- **Logging**: All requests logged

---

## Model Architecture

### LSTM Attention (1-day forecasting)
```
Parameter                      Value
─────────────────────────────────────
Input Dimension               21 (features)
LSTM Hidden Units             128
LSTM Layers                   2 (bidirectional)
Sequence Length               90 (days)
Attention Mechanism           Soft attention
FC1 → FC2                     128 → 1
Dropout                       0.3 (30%)
─────────────────────────────────────
Output                        1 value (1-day return)
Target Column                 Log_Return
Forecast Horizon              1 day
```

### Transformer (7-day forecasting)
```
Parameter                      Value
─────────────────────────────────────
Input Dimension               21 (features)
Embedding Dimension (d_model) 64
Attention Heads               4
Transformer Layers            2 (encoder blocks)
Sequence Length               90 (days)
Dropout                       0.1 (10%)
─────────────────────────────────────
Output                        7 values (7-day returns)
Target Column                 Log_Return
Forecast Horizon              7 days
```

### XGBoost Classifier (buy/sell signal)
```
Parameter                      Value
─────────────────────────────────────
Algorithm                     Gradient Boosting
n_estimators                  500 (trees)
max_depth                     6
learning_rate                 0.05
subsample                     1.0 (default)
L1 Regularization (alpha)     0.1
L2 Regularization (lambda)    0.1
─────────────────────────────────────
Classes                       3 (Buy, Hold, Sell)
Threshold (Buy/Sell)          ±1% return
Early Stopping Rounds         10
```

---

## API Endpoints

### Main Routes (app/main.py)

```
GET  /health                   → Server health check
GET  /api/v1/tickers          → List available tickers
GET  /api/v1/predict/{ticker} → Get prediction for ticker
POST /api/v1/train/{ticker}   → Retrain model for ticker
GET  /api/v1/models           → List models metadata
GET  /api/v1/historical/{ticker}?days=30 → Historical data
```

### Authentication
```
POST /api/v1/auth/login       → Get JWT token
POST /api/v1/auth/register    → Register new user
```

### Schemas (app/schemas.py)
```python
class PredictionResponse:
    ticker: str
    prediction: float
    signal: str  # 'BUY', 'HOLD', 'SELL'
    confidence: float
    timestamp: datetime

class HistoricalData:
    date: date
    close: float
    prediction: float
    actual: float
    error: float
```

---

## Scripts & Utilities

### Training Scripts
- `backend/training/train.py` - Full training pipeline
- `backend/training/train_remaining.py` - Retrain subsets
- `backend/training/auto_retrain.py` - Automatic daily retraining
- `backend/training/tuner.py` - Hyperparameter optimization

### Evaluation Scripts
- `backend/scripts/comprehensive_model_evaluation.py` - Main evaluation (700+ lines)
- `backend/scripts/run_evaluation.py` - Evaluation orchestrator
- `backend/scripts/retrain_for_evaluation.py` - Retrain for evaluation
- `backend/scripts/generate_paper_reports.py` - CSV/LaTeX export

### Prediction Scripts
- `backend/scripts/predict_all.py` - Predict for all tickers
- `backend/scripts/verify_prediction.py` - Verify outputs
- `backend/scripts/demo.py` - Demo predictions

### Data Scripts
- `backend/data/realtime.py` - Fetch real-time data
- `backend/data/update_stock_data.py` - Update historical data
- `backend/scripts/fetch_latest_news.py` - Fetch news

### Verification Scripts
- `backend/verify_setup.py` - Verify environment
- `backend/scripts/check_setup.py` - Check dependencies
- `backend/scripts/test_imports.py` - Test imports
- `backend/scripts/verify_all.py` - Full verification

---

## Summary Statistics

| Category | Count | Total Size |
|----------|-------|-----------|
| Python Source Files | 50+ | ~150 KB |
| Trained Models | 15 | ~10 MB |
| Model Scalers | 10 | ~500 KB |
| Stock Data (CSV) | 5 | ~1 MB |
| Evaluation Results | 12 | ~5 MB |
| **Total** | **92** | **~16.5 MB** |

---

## How Everything Connects

```
┌─────────────────────────────────────────────────────────────┐
│  DATA PIPELINE                                              │
├─────────────────────────────────────────────────────────────┤
│  CSV Data → Cleaning → Feature Engineering → Scaling        │
│            → Sequences → Models → Predictions               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  MODEL TRAINING                                             │
├─────────────────────────────────────────────────────────────┤
│  LSTM (1-day) + Transformer (7-day) + XGBoost (signals)    │
│  All trained with same 21-feature set                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  INFERENCE & SERVING                                        │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Server → Routes → Prediction Logic → Response      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  EVALUATION & REPORTING                                     │
├─────────────────────────────────────────────────────────────┤
│  Metrics → Graphs → CSV Reports → LaTeX Tables → Papers    │
└─────────────────────────────────────────────────────────────┘
```

---

**Document Complete**: This inventory covers all backend components, files, functions, and the complete data flow of the InvestIQ system.

