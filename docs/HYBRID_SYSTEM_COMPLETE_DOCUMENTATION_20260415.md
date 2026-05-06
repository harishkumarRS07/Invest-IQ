# InvestIQ Hybrid System Complete Documentation (2026-04-15)

## 1. Purpose and Scope

This document summarizes everything currently used in the upgraded hybrid stock prediction system, including:

- All core modules in training and inference
- Data, labels, features, models, and ensemble logic
- Validation and backtesting workflow
- API serving workflow and response structure
- Full 5-ticker retrain results
- Baseline vs latest percentage comparison

This is based on the latest full batch retrain output and the previous diagnostics baseline.

## 2. Key Files and What They Do

### 2.1 Training pipeline and modeling

- backend/training/improved_hybrid_model.py
  - Main end-to-end pipeline
  - 5-day binary labels with noise drop
  - Walk-forward training folds
  - XGBoost + LSTM + sentiment weighted ensemble
  - Final metrics and artifact saving

- backend/training/model.py
  - Binary XGBoost wrapper
  - Tuned hyperparameters
  - Feature importance extraction for top-k feature selection

- backend/training/lstm_model.py
  - Lightweight directional LSTM for sequence probability

- backend/training/feature_engineering.py
  - Technical features, market context features, sentiment aggregates, and interaction features

- backend/training/sentiment.py
  - FinBERT-ready sentiment scoring path
  - Deterministic mock fallback
  - sentiment_avg_3d and sentiment_trend creation

- backend/training/data_loader.py
  - Loads cleaned ticker data
  - Attaches aligned NIFTY and market context features

- backend/training/evaluation.py
  - Classification and trading metric evaluation helpers
  - Weighted ensemble signal evaluation

- backend/training/backtesting.py
  - Next-day PnL simulation for UP or DOWN signals with NO TRADE handling

- backend/training/train_improved_hybrid_models.py
  - Batch training driver for 5 tickers
  - Saves diagnostics plots and summary CSV

### 2.2 Runtime inference and API

- backend/inference/hybrid_predict.py
  - Loads saved model artifacts
  - Computes xgb_prob, lstm_prob, sentiment
  - Applies weighted score and thresholds for BUY/SELL/HOLD
  - Returns API-compatible output including probabilities and risk

- backend/app/routes.py
  - Prediction endpoints and batch signal endpoint
  - Hybrid or legacy mode selection and fallback behavior
  - Cache key includes inference mode

- backend/core/config.py
  - Config flags including INFERENCE_MODE and HYBRID_FALLBACK_TO_LEGACY

- backend/app/schemas.py
  - Request and response schemas used by API and frontend

### 2.3 Metrics files used for this report

- diagnostics/batch_summary_20260413_142527.csv (baseline)
- diagnostics/batch_summary_20260415_140313.csv (latest full retrain)

## 3. Data and Preprocessing Used

1. Ticker CSV files are loaded from backend/data/stock_data.
2. Data is cleaned via preprocessing.cleaning.load_data and clean_data.
3. Market index data is fetched and aligned by date.
4. Added market context columns:
   - nifty_return_1d
   - nifty_return_3d
   - market_volatility_proxy
   - sector_trend
   - Market_Correlation
5. Missing and infinite values are handled by replacement plus forward-fill and backward-fill.

## 4. Labeling Logic Used

Labeling is binary and noise-filtered:

- Forecast horizon: 5 days
- UP (1): future 5-day return > +1.0%
- DOWN (0): future 5-day return < -1.0%
- Else: dropped as noise (label -1)

This replaced shorter-horizon weaker labels.

## 5. Features Used

### 5.1 Core technical and return features

- RSI
- MACD
- MACD_Signal
- BB_High
- BB_Low
- Return_1D
- Return_3D
- Return_5D
- Return_Lag_1
- Return_Lag_2
- Return_Lag_3
- Volume_Change
- Volatility_10D

### 5.2 Market context features

- Market_Correlation
- nifty_return_1d
- nifty_return_3d
- market_volatility_proxy
- sector_trend

### 5.3 Sentiment features

- sentiment_score
- sentiment_avg_3d
- sentiment_trend

### 5.4 Interaction features

- rsi_x_volume_change
- macd_x_volatility
- return_3d_x_sentiment

Total candidate feature set is controlled and feature selection is applied.

## 6. Models and Parameters Used

### 6.1 Binary XGBoost (main classifier)

Configured parameters:

- objective: binary:logistic
- n_estimators: 400
- max_depth: 7
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- eval_metric: logloss
- early_stopping_rounds: 40
- class imbalance handling via scale_pos_weight

### 6.2 LSTM directional model (lightweight)

- seq_length: 15
- hidden_dim: 24
- num_layers: 1
- epochs: 8

### 6.3 Sentiment model path

- FinBERT-ready loader (ProsusAI/finbert) if enabled and available
- deterministic mock fallback in {-1, 0, 1} when unavailable

## 7. Ensemble Logic Used

Weighted score:

final_score = 0.5 * xgb_prob + 0.3 * lstm_prob + 0.2 * normalized_sentiment

Where normalized_sentiment maps from [-1, 1] to [0, 1].

Decision thresholds:

- BUY or UP if final_score > 0.60
- SELL or DOWN if final_score < 0.40
- Otherwise NO TRADE or HOLD

## 8. Validation and Backtest Workflow

1. Load and preprocess data
2. Engineer features
3. Build 5-day labels and drop noisy rows
4. Build walk-forward folds (expanding train, rolling val and test)
5. Per fold:
   - Train XGBoost on scaled features
   - Select top 18 features by XGBoost importance
   - Retrain XGBoost on selected features
   - Train LSTM on close-price sequences
   - Compute final weighted score
   - Apply thresholds to produce signals
   - Backtest next-day returns
6. Aggregate fold metrics per ticker
7. Save model artifacts and diagnostics
8. Save batch summary CSV

## 9. API and Serving Workflow

1. Client calls /predict or /signals/batch
2. Route checks INFERENCE_MODE:
   - hybrid: uses HybridPredictor
   - legacy: uses legacy predictor
3. HybridPredictor:
   - loads xgboost_binary, scaler, feature list, optional lstm artifacts, optional hybrid config
   - computes probabilities and sentiment
   - computes weighted ensemble score
   - outputs signal, confidence, probabilities, and indicators
4. If hybrid fails and fallback flag is true, route falls back to legacy predictor
5. Cache key includes mode and symbol

## 10. Baseline vs Latest Full Retrain Comparison (5 Tickers)

### 10.1 Average metrics

| Metric | Baseline | Latest | Delta |
|---|---:|---:|---:|
| Accuracy | 51.23% | 51.75% | +0.52 pp |
| Precision | 52.73% | 55.36% | +2.64 pp |
| Recall | 51.97% | 54.59% | +2.61 pp |
| F1 | 51.49% | 54.30% | +2.81 pp |
| Directional Accuracy | 49.83% | 53.59% | +3.76 pp |
| Win Rate | 49.19% | 56.54% | +7.35 pp |
| Total Return | -2.64% | +11.79% | +14.43 pp |
| Trade Coverage | 42.87% | 6.11% | -36.76 pp |
| Avg Trades per Ticker | 318.0 | 22.8 | -295.2 |

Interpretation:

- Signal quality and profitability improved materially.
- Accuracy improved modestly overall.
- Strategy became far more selective, with much lower trade frequency and coverage.

### 10.2 Per-ticker baseline to latest

| Ticker | Accuracy | Win Rate | Total Return | Trades | Coverage |
|---|---|---|---|---:|---|
| HDFCBANK | 52.70% -> 51.91% | 47.01% -> 59.40% | -14.93% -> 16.27% | 134 -> 26 | 17.65% -> 4.89% |
| ICICIBANK | 49.74% -> 48.60% | 46.92% -> 60.63% | -7.05% -> 23.42% | 260 -> 21 | 33.94% -> 8.63% |
| INFY | 51.69% -> 50.57% | 52.63% -> 50.79% | -2.51% -> 3.46% | 95 -> 12 | 11.50% -> 3.65% |
| RELIANCE | 51.04% -> 52.08% | 49.91% -> 56.02% | 9.18% -> 9.93% | 547 -> 48 | 66.95% -> 7.71% |
| TCS | 50.99% -> 55.58% | 49.46% -> 55.84% | 2.10% -> 5.84% | 554 -> 7 | 84.32% -> 5.67% |

### 10.3 XGBoost vs Hybrid averages in latest run

| Metric | XGBoost Avg | Hybrid Avg |
|---|---:|---:|
| Accuracy | 52.05% | 51.75% |
| Win Rate | 40.82% | 56.54% |
| Total Return | 40.15% | 11.79% |

Note:

- Hybrid improves win consistency and decision quality under strict thresholds.
- XGBoost standalone total return average is higher in this run, but with very different behavior and risk profile; evaluate alongside drawdown, stability, and consistency goals.

## 11. Artifacts Produced Per Ticker

Saved under diagnostics/<TICKER>:

- 01_confusion_matrix.png
- 02_roc_curve.png
- 03_pr_curve.png
- 04_confidence_dist.png
- 05_calibration.png
- 06_equity_curve.png
- 07_trade_distribution.png
- 08_confidence_vs_accuracy.png
- <TICKER>_evaluation_<timestamp>.json
- <TICKER>_threshold_evaluation.csv

Saved under diagnostics:

- batch_summary_<timestamp>.csv

Saved under backend/models/saved_models:

- xgboost_binary_<TICKER>.pkl
- xgboost_binary_scaler_<TICKER>.pkl
- xgboost_binary_features_<TICKER>.pkl
- hybrid_config_<TICKER>.pkl
- lstm_binary_<TICKER>.pth
- lstm_binary_scaler_<TICKER>.pkl

## 12. Commands Used

- Full batch retrain:
  - d:/InvestIQ-main/backend/venv/Scripts/python.exe backend/training/train_improved_hybrid_models.py

- Compare baseline vs latest summaries:
  - Read and compare diagnostics/batch_summary_20260413_142527.csv vs diagnostics/batch_summary_20260415_140313.csv

## 13. Current System Status

- System is running with all requested major upgrades integrated.
- Full 5-ticker retrain completed successfully.
- Profitability and win rate are improved versus baseline.
- Accuracy is improved slightly on average, but still below portfolio-wide 55-58% target.
- Trade selectivity is very high (large trade count reduction), which reduced false trades but also significantly reduced coverage.

## 14. Recommended Next Tuning Track

To improve coverage while preserving quality:

1. Per-ticker threshold calibration rather than fixed 0.60 and 0.40 for all.
2. Add minimum trade floor per fold when selecting thresholds.
3. Compare score bands such as 0.58 and 0.42 and 0.56 and 0.44 with drawdown guardrails.
4. Keep walk-forward evaluation unchanged for fair comparisons.
