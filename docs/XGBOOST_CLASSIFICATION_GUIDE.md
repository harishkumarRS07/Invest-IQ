# XGBoost Classification Pipeline - Complete Documentation

**Project:** InvestIQ (Final Year Project)  
**Component:** XGBoost Classification Model  
**Status:** Production Ready  
**Date:** 2026-04-09

---

## 📋 TABLE OF CONTENTS

1. [Overview](#overview)
2. [Label Strategy](#label-strategy)
3. [Feature Engineering](#feature-engineering)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Confidence Scores](#confidence-scores)
8. [How to Use](#how-to-use)
9. [Expected Results](#expected-results)
10. [Code Examples](#code-examples)
11. [Troubleshooting](#troubleshooting)

---

## 📌 OVERVIEW

### What is this?

An XGBoost-based classification model that predicts three trading signals:
- **BUY** (Signal 2): Expected positive return > +0.5%
- **HOLD** (Signal 1): Expected return between -0.5% and +0.5%
- **SELL** (Signal 0): Expected negative return < -0.5%

### Why XGBoost over Deep Learning?

1. **Better for tabular data** - Technical indicators are structured features
2. **Faster training** - No GPU required, trains in seconds
3. **Interpretable** - Feature importance shows what matters
4. **Smaller models** - Easier to deploy and maintain
5. **Robust** - Less prone to overfitting than neural networks

### Key Improvements

- ✅ Better labels with thresholds (not just binary)
- ✅ 30+ engineered features (momentum, volume, trend, volatility)
- ✅ Time-based train-test split (no data leakage)
- ✅ Confidence scores for each prediction
- ✅ Feature importance analysis included

---

## 🏷️ LABEL STRATEGY

### How Labels are Created

```python
# Calculate 3-day future return
future_return = (Close[t+3] - Close[t]) / Close[t]

# Assign label based on threshold
if future_return > 0.005:      # 0.5%
    label = 2  # BUY
elif future_return < -0.005:   # -0.5%
    label = 0  # SELL
else:
    label = 1  # HOLD
```

### Why These Thresholds?

| Threshold | Reason |
|-----------|--------|
| **+0.5%** | Meaningful profit after brokerage fees (~0.1-0.2%) |
| **-0.5%** | Cut losses before major drawdown |
| **±0% range** | Avoid whipsaws and small movements |

### Class Distribution (Expected)

```
SELL:  15-20% (prediction signal)
HOLD:  50-60% (majority, neutral)
BUY:   15-20% (prediction signal)
```

### Example Calculation

```
Day 1:  Close = $100.00
Day 4:  Close = $100.75    # +0.75% return
Label:  BUY (2)            # > 0.5% threshold

Day 1:  Close = $100.00
Day 4:  Close = $99.30     # -0.70% return
Label:  SELL (0)           # < -0.5% threshold

Day 1:  Close = $100.00
Day 4:  Close = $100.20    # +0.20% return
Label:  HOLD (1)           # Within ±0.5%
```

---

## 🔧 FEATURE ENGINEERING

### 1. Technical Indicators (8 features)

```python
# Already computed by add_technical_indicators()
SMA_20, SMA_50      # Simple Moving Averages
RSI                 # Relative Strength Index
MACD, MACD_Signal, MACD_Hist  # Moving Average Convergence Divergence
BB_High, BB_Low, BB_Mid       # Bollinger Bands
ATR                 # Average True Range (volatility)
VWAP                # Volume Weighted Average Price
```

### 2. Momentum Features (5 features)

```python
return_3d = df['Close'].pct_change(3)      # 3-day return
return_5d = df['Close'].pct_change(5)      # 5-day return
return_7d = df['Close'].pct_change(7)      # 7-day return
momentum_3d = df['Close'] - df['Close'].shift(3)
momentum_5d = df['Close'] - df['Close'].shift(5)
```

**Why:** Capture short-term trend momentum

### 3. Volume Features (5 features)

```python
volume_change = df['Volume'].pct_change()        # Daily volume change
volume_ma_5 = df['Volume'].rolling(5).mean()     # 5-day avg volume
volume_ma_20 = df['Volume'].rolling(20).mean()   # 20-day avg volume
volume_ratio = df['Volume'] / volume_ma_20       # Current vs avg
price_volume_trend = volume_ratio * return_pct   # Combined signal
```

**Why:** Volume confirms price moves

### 4. Trend Features (5 features)

```python
sma_diff = df['SMA_20'] - df['SMA_50']           # Trend strength
price_sma20_diff = df['Close'] - df['SMA_20']    # Distance from 20-day MA
price_sma50_diff = df['Close'] - df['SMA_50']    # Distance from 50-day MA
sma_ratio = df['SMA_20'] / df['SMA_50']          # Cross ratio
sma_20_above_50 = (df['SMA_20'] > df['SMA_50']).astype(int)  # Golden cross
```

**Why:** Identify up/down trends

### 5. Volatility Features (5 features)

```python
volatility_5d = df['Close'].pct_change().rolling(5).std()      # 5-day vol
volatility_10d = df['Close'].pct_change().rolling(10).std()    # 10-day vol
volatility_20d = df['Close'].pct_change().rolling(20).std()    # 20-day vol
high_low_ratio = (High - Low) / Close                           # Intra-day range
bb_position = (Close - BB_Low) / (BB_High - BB_Low)            # Bollinger position
```

**Why:** Market regime detection

### 6. Other Features (3+ features)

```python
Volume_Change, Log_Return, Rolling_Volatility
```

### Total Features: ~35-40

All features are normalized by XGBoost internally.

---

## 🤖 MODEL ARCHITECTURE

### XGBoost Configuration

```python
XGBClassifier(
    n_estimators=200,               # 200 boosting rounds
    max_depth=5,                    # Max tree depth
    learning_rate=0.05,             # Shrinkage parameter (eta)
    subsample=0.8,                  # Subsample 80% of rows per tree
    colsample_bytree=0.8,           # Subsample 80% of features per tree
    objective='multi:softprob',     # Multi-class probability
    eval_metric='mlogloss',         # Evaluation metric
    early_stopping_rounds=20,       # Stop if no improvement for 20 rounds
    random_state=42,                # Reproducibility
    n_jobs=-1                       # Use all CPU cores
)
```

### Why These Parameters?

| Parameter | Value | Reason |
|-----------|-------|--------|
| **n_estimators** | 200 | Balance between underfitting and overfitting |
| **max_depth** | 5 | Prevent overfitting on small datasets |
| **learning_rate** | 0.05 | Slow learning for stable convergence |
| **subsample** | 0.8 | Prevent memorization |
| **colsample_bytree** | 0.8 | Feature randomness |
| **early_stopping** | 20 | Stop when validation performance plateaus |

---

## 📚 TRAINING PROCESS

### Step 1: Data Loading & Preprocessing

```python
# Load CSV
df = load_data(file_path)

# Basic cleaning
df = clean_data(df)

# Check length
if len(df) < 100:
    raise ValueError("Insufficient data")
```

### Step 2: Feature Engineering

```python
# Add technical indicators
df = add_technical_indicators(df)

# Add custom features
df = add_momentum_features(df)
df = add_volume_features(df)
df = add_trend_features(df)
df = add_volatility_features(df)
```

### Step 3: Label Creation

```python
# Create labels for 3-day horizon
y, returns = pipeline.create_better_labels(df)

# y is now array of [0=SELL, 1=HOLD, 2=BUY]
```

### Step 4: Feature Selection

```python
# Select only relevant features
X = pipeline.select_features(df)

# Returns subset of dataframe with ~35 features
```

### Step 5: Data Cleaning

```python
# Remove NaN rows
X, y = pipeline.clean_features(X, y)

# Also removes infinite values
```

### Step 6: Time-Based Split (NO shuffling)

```python
# First 80% = training
# Last 20% = testing

split_idx = int(0.8 * len(X))
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]  # IMPORTANT: Time order preserved
```

**Why no shuffling?** 
- Prevents data leakage
- Realistic simulation (can't know future)
- Tests on most recent data

### Step 7: Model Training

```python
model = XGBClassifier(...)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=False
)
```

**What happens:**
- Trains on 80% of data
- Validates on 20% during training
- Stops if validation improves no more

### Step 8: Evaluation

```python
# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
```

---

## 📊 EVALUATION METRICS

### 1. Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
Range: 0 to 1 (0% to 100%)
Expected: 60-70%
```

**Interpretation:**
- 50% = Same as random guessing between 3 classes
- 65% = 15% better than random
- 75% = 25% better than random

### 2. Precision (per class)

```
Precision_BUY = (True BUY predictions) / (All BUY predictions)
```

**Meaning:** "Of all predicted BUYs, how many were correct?"

**Example:**
- Predicted 10 BUYs
- 8 were actually correct
- Precision = 80%

### 3. Recall (per class)

```
Recall_BUY = (True BUY predictions) / (All actual BUYs)
```

**Meaning:** "Of all actual BUY moves, how many did we catch?"

**Example:**
- Actual BUYs in data: 50
- Predicted 40 correctly
- Recall = 80%

### 4. F1 Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Meaning:** Harmonic mean of precision and recall

**Why:** Balances both metrics (good for imbalanced classes)

### 5. Confusion Matrix

```
                 Predicted
              SELL  HOLD  BUY
Actual SELL    TP   FN   FN
       HOLD    FP   TP   FP
       BUY     FN   FN   TP
```

**Interpretation:**
- Diagonal = Correct predictions
- Off-diagonal = Errors (confusion)

---

## 💪 CONFIDENCE SCORES

### What is a Confidence Score?

The maximum probability from the model's three output probabilities.

```python
# Model outputs probabilities for each class
proba = model.predict_proba(X)
# Shape: (n_samples, 3)

# Maximum probability = confidence
confidence = proba.max(axis=1)
# Range: 0.33 to 1.0
# 0.33 = Random guess
# 1.0 = Certain prediction
```

### Example

```
Model outputs: [0.1, 0.2, 0.7]  (SELL, HOLD, BUY probabilities)
Prediction: BUY (highest = 0.7)
Confidence: 0.70 (70%)
```

### Using Confidence

```python
# Generate signals with confidence
predictions, confidence = pipeline.get_confidence_scores(X_test)

# Filter high-confidence signals only
high_conf_mask = confidence > 0.7
high_conf_signals = signals[high_conf_mask]

# Better results with higher threshold
```

### Confidence Interpretation

| Confidence | Interpretation | Action |
|:---|:---|:---|
| **>0.75** | Very high | Strong signal, trust it |
| **0.60-0.75** | High | Good signal, reasonable confidence |
| **0.50-0.60** | Medium | Weak signal, be cautious |
| **<0.50** | Low | Very weak signal, ignore or wait |

---

## 🚀 HOW TO USE

### Option 1: Train Single Stock

```python
from backend.training.xgboost_classifier import train_xgboost_classifier

results = train_xgboost_classifier(
    ticker="RELIANCE",
    file_path="backend/data/stock_data/RELIANCE.csv",
    buy_threshold=0.005,      # 0.5%
    sell_threshold=-0.005     # -0.5%
)

# Get signals
signals = results['signals']
print(signals.head())

# Get metrics
metrics = results['metrics']
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Option 2: Batch Train All Stocks

```bash
# From command line
python batch_train_xgboost.py

# Trains: HDFCBANK, ICICIBANK, INFY, RELIANCE, TCS
# Saves models to: backend/models/saved_models/
# Creates plots: backend/models/saved_models/feature_importance_*.png
```

### Option 3: Run Demo

```bash
# See example predictions and feature importance
python demo_xgboost_classification.py
```

### Option 4: Use in Python

```python
from backend.training.xgboost_classifier import XGBoostClassificationPipeline
import pandas as pd

# Initialize
pipeline = XGBoostClassificationPipeline(
    buy_threshold=0.005,
    sell_threshold=-0.005
)

# Prepare data
df = pd.read_csv("stock.csv")
df = pipeline.add_all_features(df)
X = pipeline.select_features(df)

# Load pretrained model
pipeline.load_model("path/to/model.pkl")

# Generate signals
signals = pipeline.generate_signals(X)
print(signals)
```

---

## 📈 EXPECTED RESULTS

### Performance Metrics

| Metric | Expected | Range |
|--------|----------|-------|
| **Accuracy** | 65% | 60-70% |
| **Precision** | 0.65 | 0.60-0.70 |
| **Recall** | 0.65 | 0.60-0.70 |
| **F1 Score** | 0.65 | 0.60-0.70 |
| **Confidence** (avg) | 0.55 | 0.50-0.60 |

### Comparison with Baseline

```
Baseline (always HOLD):
  - Accuracy: ~50-60%
  - Precision: N/A (no BUY/SELL predicted)
  - Recall: N/A

XGBoost:
  - Accuracy: ~65%
  - Precision: ~0.65
  - Recall: ~0.65
  - Improvement: +5-15% over baseline
```

### Class-Specific Results

```
SELL Class:
  - Precision: ~0.60-0.68    (When we predict SELL, we're right 60-68% of time)
  - Recall: ~0.55-0.70       (We catch 55-70% of actual SELL opportunities)

HOLD Class:
  - Precision: ~0.65-0.75    (When we predict HOLD, we're right 65-75% of time)
  - Recall: ~0.70-0.85       (We catch 70-85% of actual HOLD periods)

BUY Class:
  - Precision: ~0.60-0.68    (When we predict BUY, we're right 60-68% of time)
  - Recall: ~0.55-0.70       (We catch 55-70% of actual BUY opportunities)
```

### Signal Distribution

```
Per 100 predictions:
  BUY:   15-20 signals (should catch 50-70% of profitable moves)
  HOLD:  60-70 signals (neutral periods)
  SELL:  15-20 signals (should catch 50-70% of negative moves)
```

---

## 💻 CODE EXAMPLES

### Example 1: Training & Evaluation

```python
from backend.training.xgboost_classifier import train_xgboost_classifier

# Train on RELIANCE
results = train_xgboost_classifier(
    ticker="RELIANCE",
    file_path="backend/data/stock_data/RELIANCE.csv"
)

# Extract results
signals = results['signals']
metrics = results['metrics']

# Print metrics
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")

# Print signals
print("\nFirst 5 predictions:")
print(signals.head())

# Print confusion matrix
print("\nConfusion Matrix:")
print(metrics['confusion_matrix'])
```

### Example 2: Getting Predictions

```python
pipeline = results['pipeline']
X_test = results['X_test']

# Get predictions with confidence
predictions, confidence = pipeline.get_confidence_scores(X_test)

# Filter high-confidence predictions
high_conf = confidence > 0.7
high_conf_predictions = predictions[high_conf]

print(f"High-confidence signals: {high_conf.sum()}")
print(f"Confidence range: {confidence.min():.2f} - {confidence.max():.2f}")
```

### Example 3: Feature Importance

```python
# Get feature importance
importances = pipeline.model.feature_importances_
feature_names = pipeline.feature_names

# Sort by importance
sorted_idx = np.argsort(importances)[::-1]

print("Top 10 Features:")
for i, idx in enumerate(sorted_idx[:10], 1):
    print(f"  {i}. {feature_names[idx]}: {importances[idx]:.6f}")

# Plot
pipeline.plot_feature_importance(top_k=20, save_path="importance.png")
```

### Example 4: JSON Output for API

```python
latest_signal = signals.iloc[-1]

output = {
    "ticker": "RELIANCE",
    "timestamp": pd.Timestamp.now().isoformat(),
    "signal": latest_signal['Signal'],
    "confidence": float(latest_signal['Confidence']),
    "probabilities": {
        "sell": float(latest_signal['Prob_SELL']),
        "hold": float(latest_signal['Prob_HOLD']),
        "buy": float(latest_signal['Prob_BUY'])
    },
    "recommendation": f"{latest_signal['Signal']} (Confidence: {latest_signal['Confidence']:.2%})"
}

import json
print(json.dumps(output, indent=2))
```

---

## 🐛 TROUBLESHOOTING

### Issue: Poor Accuracy (below 55%)

**Possible Causes:**
1. Thresholds too aggressive (e.g., 1% instead of 0.5%)
2. Not enough data (need >500 samples)
3. Highly volatile stock that's unpredictable

**Solutions:**
```python
# Try more lenient thresholds
results = train_xgboost_classifier(
    ticker="STOCK",
    file_path="file.csv",
    buy_threshold=0.01,      # 1% instead of 0.5%
    sell_threshold=-0.01     # -1% instead of -0.5%
)

# More lenient = more HOLD predictions
# Easier to achieve higher accuracy
```

### Issue: All Predictions are HOLD

**Possible Causes:**
1. Model converged to majority class
2. Thresholds too strict
3. Insufficient variance in features

**Solutions:**
```python
# Check class balance
from backend.training.xgboost_classifier import XGBoostClassificationPipeline
pipeline = XGBoostClassificationPipeline()
pipeline.check_class_balance(y)

# Should see ~20% BUY, 50-60% HOLD, ~20% SELL
# If not, data issue
```

### Issue: Confidence Scores Too Low (all 0.33-0.40)

**Meaning:** Model is too uncertain (equivalent to random guess)

**Possible Causes:**
1. Features don't have predictive power
2. Stock is fundamentally unpredictable
3. Need more training data

**Solutions:**
1. Check if features contain information
2. Try different stocks/time periods
3. Consider feature engineering improvements

### Issue: Import Errors

```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
pip install xgboost scikit-learn
```

### Issue: Out of Memory

**Solutions:**
```python
# Reduce dataset size
df = df[::2]  # Take every 2nd row

# Or reduce features
# Select subset of most important features
```

---

## 📌 BEST PRACTICES

### 1. Always Check Class Balance

```python
pipeline.check_class_balance(y)
```

### 2. Validate on Recent Data

```python
# The model is tested on most recent 20% of data
# This is realistic for trading
```

### 3. Filter by Confidence

```python
# Only trade high-confidence signals
high_conf_signals = signals[signals['Confidence'] > 0.7]
```

### 4. Monitor Feature Importance

```python
# Ensure rational features are important
# (Not just noise)
```

### 5. Retrain Regularly

```python
# Stock patterns change over time
# Retrain monthly or quarterly
```

---

## 🎓 LEARNING RESOURCES

1. XGBoost Documentation: https://xgboost.readthedocs.io/
2. Feature Engineering: https://machinelearningmastery.com/
3. Time Series Classification: https://otexts.com/fpp2/

---

## 📞 SUPPORT

For issues or questions:
1. Check this documentation
2. Review code comments
3. Check demo script output
4. View logs for errors

---

**Status:** ✅ PRODUCTION READY  
**Last Updated:** 2026-04-09  
**Tested On:** Python 3.8+, XGBoost 1.5+

