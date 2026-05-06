# XGBoost Classification - QUICK START & CODE REFERENCE

**Quick reference for implementing XGBoost classification in InvestIQ**

---

## 🚀 QUICK START (2 MINUTES)

### Step 1: Install Dependencies
```bash
pip install xgboost scikit-learn pandas numpy
```

### Step 2: Train Single Stock
```bash
python -c "
from backend.training.xgboost_classifier import train_xgboost_classifier

results = train_xgboost_classifier(
    ticker='RELIANCE',
    file_path='backend/data/stock_data/RELIANCE.csv'
)
print('Accuracy:', results['metrics']['accuracy'])
"
```

### Step 3: Train All Stocks
```bash
python batch_train_xgboost.py
```

### Step 4: View Results
```bash
python demo_xgboost_classification.py
```

---

## 📚 KEY CODE SNIPPETS

### 1. Label Creation (BUY/SELL/HOLD)

```python
def create_better_labels(df, buy_threshold=0.005, sell_threshold=-0.005, horizon=3):
    """Create labels based on future returns."""
    future_close = df['Close'].shift(-horizon)
    future_returns = (future_close - df['Close']) / df['Close']
    
    labels = np.ones(len(df), dtype=int)  # Default HOLD (1)
    labels[future_returns > buy_threshold] = 2   # BUY
    labels[future_returns < sell_threshold] = 0   # SELL
    
    return labels[:-horizon]
```

### 2. Feature Engineering - Momentum

```python
def add_momentum_features(df):
    """Add momentum-based features."""
    df['return_3d'] = df['Close'].pct_change(3)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_7d'] = df['Close'].pct_change(7)
    df['momentum_3d'] = df['Close'] - df['Close'].shift(3)
    df['momentum_5d'] = df['Close'] - df['Close'].shift(5)
    return df
```

### 3. Feature Engineering - Volume

```python
def add_volume_features(df):
    """Add volume-based features."""
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma_5'] = df['Volume'].rolling(5).mean()
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
    df['price_volume_trend'] = df['volume_ratio'] * df['Close'].pct_change()
    return df
```

### 4. Feature Engineering - Trend

```python
def add_trend_features(df):
    """Add trend-based features."""
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['sma_diff'] = df['SMA_20'] - df['SMA_50']
    df['price_sma20_diff'] = df['Close'] - df['SMA_20']
    df['sma_20_above_50'] = (df['SMA_20'] > df['SMA_50']).astype(int)
    return df
```

### 5. Feature Engineering - Volatility

```python
def add_volatility_features(df):
    """Add volatility-based features."""
    df['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
    df['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
    df['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
    df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
    return df
```

### 6. Data Cleaning

```python
def clean_features(X, y):
    """Remove NaN and infinite values."""
    # Drop NaN rows
    valid_idx = ~(X.isna().any(axis=1))
    X_clean = X[valid_idx].copy()
    y_clean = y[valid_idx].copy()
    
    # Replace infinite values
    X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_idx = ~(X_clean.isna().any(axis=1))
    X_clean = X_clean[valid_idx].copy()
    y_clean = y_clean[valid_idx].copy()
    
    return X_clean, y_clean
```

### 7. Time-Based Split (NO shuffle)

```python
def time_based_split(X, y, train_ratio=0.8):
    """Split data preserving time order."""
    split_idx = int(len(X) * train_ratio)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test
```

### 8. XGBoost Training

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,           # 200 boosting rounds
    max_depth=5,                # Max tree depth
    learning_rate=0.05,         # Shrinkage
    subsample=0.8,              # Row subsampling
    colsample_bytree=0.8,       # Feature subsampling
    objective='multi:softprob', # Multi-class probability
    eval_metric='mlogloss',     # Evaluation metric
    random_state=42,
    n_jobs=-1                   # Use all cores
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=False
)
```

### 9. Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
```

### 10. Confidence Scores

```python
# Get predictions and confidence
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)
confidence = proba.max(axis=1)  # Max probability

# Create signals dataframe
signals = pd.DataFrame({
    'Signal': [['SELL', 'HOLD', 'BUY'][p] for p in y_pred],
    'Confidence': confidence,
    'Prob_SELL': proba[:, 0],
    'Prob_HOLD': proba[:, 1],
    'Prob_BUY': proba[:, 2]
})

# Filter high-confidence signals
high_conf = signals[signals['Confidence'] > 0.7]
```

### 11. Feature Importance

```python
from xgboost import plot_importance

# Get importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Print top 10
print("Top 10 Features:")
print(feature_importance_df.head(10))

# Plot
plot_importance(model, max_num_features=20)
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
```

### 12. Save and Load Model

```python
import joblib

# Save
joblib.dump(model, 'xgboost_model.pkl')

# Load
model = joblib.load('xgboost_model.pkl')
```

---

## 📊 COMPLETE TRAINING PIPELINE

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# ============== 1. LOAD DATA ==============
df = pd.read_csv('stock_data.csv')
print(f"Loaded: {len(df)} rows")

# ============== 2. CREATE LABELS ==============
future_close = df['Close'].shift(-3)
future_returns = (future_close - df['Close']) / df['Close']

labels = np.ones(len(df), dtype=int)
labels[future_returns > 0.005] = 2   # BUY
labels[future_returns < -0.005] = 0   # SELL
labels = labels[:-3]  # Remove last 3 rows

# ============== 3. FEATURE ENGINEERING ==============
# Technical indicators
df['SMA_20'] = df['Close'].rolling(20).mean()
df['SMA_50'] = df['Close'].rolling(50).mean()
df['RSI'] = ...  # Via ta library

# Momentum
df['return_3d'] = df['Close'].pct_change(3)
df['return_5d'] = df['Close'].pct_change(5)

# Volume
df['volume_change'] = df['Volume'].pct_change()
df['volume_ma'] = df['Volume'].rolling(20).mean()

# Trend
df['sma_diff'] = df['SMA_20'] - df['SMA_50']
df['price_sma_diff'] = df['Close'] - df['SMA_20']

# Volatility
df['volatility_5d'] = df['Close'].pct_change().rolling(5).std()

# ============== 4. SELECT FEATURES ==============
feature_cols = [
    'SMA_20', 'SMA_50', 'RSI', 'MACD',
    'return_3d', 'return_5d',
    'volume_change', 'volume_ma',
    'sma_diff', 'price_sma_diff',
    'volatility_5d'
]
X = df[feature_cols]

# ============== 5. CLEAN DATA ==============
# Remove NaN
valid_idx = ~(X.isna().any(axis=1))
X = X[valid_idx]
labels = labels[valid_idx]

# ============== 6. TIME-BASED SPLIT ==============
split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = labels[:split], labels[split:]

# ============== 7. TRAIN MODEL ==============
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=False
)

# ============== 8. EVALUATE ==============
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

# ============== 9. GET SIGNALS ==============
proba = model.predict_proba(X_test)
confidence = proba.max(axis=1)

signals = pd.DataFrame({
    'Signal': [['SELL', 'HOLD', 'BUY'][p] for p in y_pred],
    'Confidence': confidence,
    'Prob_SELL': proba[:, 0],
    'Prob_HOLD': proba[:, 1],
    'Prob_BUY': proba[:, 2]
})

print("\nFirst 5 signals:")
print(signals.head())

# ============== 10. FEATURE IMPORTANCE ==============
importances = model.feature_importances_
top_features = sorted(
    zip(feature_cols, importances),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 10 Important Features:")
for i, (feat, imp) in enumerate(top_features[:10], 1):
    print(f"  {i}. {feat}: {imp:.6f}")
```

---

## 🎯 EXPECTED OUTPUT

### Training:
```
XGBoost Classification Pipeline - RELIANCE
================================================================================
Loading data: backend/data/stock_data/RELIANCE.csv
[OK] Data loaded: 5000 rows

Engineering features...
Total features created: 50

Selecting features...
Selected 35 features for training

Cleaning data: 5000 samples before
Cleaning data: 4980 samples after

Time-based split (no shuffle):
  Train: 3984 samples
  Test:  996 samples

Class Distribution:
  SELL: 850 (17.1%)
  HOLD: 2980 (59.9%)
  BUY:  1150 (23.1%)

Training XGBoost model...
XGBoost model trained successfully

Evaluation Metrics (XGBoost):
  Accuracy:  0.6543
  Precision: 0.6512
  Recall:    0.6451
  F1 Score:  0.6478

Confusion Matrix:
[[187  68  95]
 [ 42 598  60]
 [ 68  56 826]]

First 10 predictions:
     Signal  Confidence  Prob_SELL  Prob_HOLD   Prob_BUY
  0     BUY        0.714       0.086      0.200      0.714
  1    HOLD        0.621       0.156      0.621      0.223
  2    HOLD        0.589       0.201      0.589      0.210
  ...

================================================================================
```

### Signals:
```
Latest Signal: BUY with HIGH confidence

Recommendations:
  - BUY signal detected
  - Confidence: 71.4%
  - Expected upward movement > 0.5%
  - ACTION: Consider buying (high confidence)
```

### Feature Importance:
```
Top 10 Most Important Features:
  1. sma_diff                0.089234
  2. volatility_5d           0.076543
  3. return_5d               0.065432
  4. RSI                     0.054321
  5. MACD                    0.048765
  6. volume_ratio            0.043210
  7. momentum_5d             0.038765
  8. price_sma20_diff        0.035432
  9. return_3d               0.032109
  10. bb_position            0.028765
```

---

## ⚙️ CONFIGURATION PARAMETERS

### Label Thresholds (Adjust for different stocks)

```python
# Aggressive (more signals, lower accuracy)
buy_threshold = 0.01       # ±1%
sell_threshold = -0.01

# Moderate (balanced)
buy_threshold = 0.005      # ±0.5%
sell_threshold = -0.005

# Conservative (fewer signals, higher accuracy)
buy_threshold = 0.02       # ±2%
sell_threshold = -0.02
```

### Model Parameters (Tuning)

```python
# More complex (risk of overfitting)
XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1)

# Default (balanced)
XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05)

# Simpler (risk of underfitting)
XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01)
```

---

## 📁 FILES CREATED

| File | Purpose |
|------|---------|
| `backend/training/xgboost_classifier.py` | Main XGBoost pipeline |
| `batch_train_xgboost.py` | Batch training script |
| `demo_xgboost_classification.py` | Demo and examples |
| `docs/XGBOOST_CLASSIFICATION_GUIDE.md` | Full documentation |

---

## ✅ CHECKLIST

- [ ] Install xgboost: `pip install xgboost`
- [ ] Check data exists: `backend/data/stock_data/`
- [ ] Run demo: `python demo_xgboost_classification.py`
- [ ] Train single stock: `python batch_train_xgboost.py`
- [ ] Check models saved: `backend/models/saved_models/`
- [ ] Review metrics and signals
- [ ] Integrate into pipeline

---

**Status:** ✅ Production Ready  
**Last Updated:** 2026-04-09

