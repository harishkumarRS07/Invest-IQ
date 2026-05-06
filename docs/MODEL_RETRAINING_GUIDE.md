# Model Retraining Guide: Feature Dimension Mismatch

## Problem Summary

The evaluation script encountered **model loading errors** due to feature dimension mismatches between saved models and the current feature engineering pipeline. The saved models were trained with fewer features than the current pipeline generates.

### Error Breakdown

| Model | Saved Features | Current Pipeline | Issue |
|-------|---|---|---|
| **LSTM** | 15 features | 21 features | `size mismatch for lstm.weight_ih_l0: expecting [256, 15], got [512, 21]` |
| **Transformer** | 20 features | 21 features | `input_embedding.weight mismatch: [64, 20] vs [64, 21]` |
| **XGBoost** | 18 features | 21 features | `Feature shape mismatch, expected: 18, got 21` |

## Root Cause

The feature engineering pipeline has evolved to include more comprehensive indicators:

### Feature Set Evolution

#### Original Features (5):
- Open, High, Low, Close, Volume

#### Added Technical Indicators (13):
- SMA_20, SMA_50, RSI, BB_High, BB_Low, VWAP
- MACD, MACD_Signal, MACD_Hist
- ATR (Average True Range)
- Log_Return, Volume_Change, Rolling_Volatility

#### Added Market Features (1):
- Market_Correlation (with NIFTY 50 index)

#### Added External Features (2):
- Sentiment (simulated news sentiment)
- Macro_Score (macroeconomic indicator)

**Total: 5 + 13 + 1 + 2 = 21 features**

The saved models were trained at an earlier stage when this pipeline was incomplete, resulting in 15, 18, or 20 features depending on when each model was saved.

## Solution

### Option A: Retrain Models (Recommended for Evaluation)

The models are retrained with the current 21-feature pipeline to ensure consistency.

#### Quick Start

```bash
# Run the retraining script
python backend/scripts/retrain_for_evaluation.py

# Or use the batch file (Windows)
retrain_models.bat
```

#### What Gets Retrained

1. **LSTM Attention Model**
   - Input: 21 features (90-day lookback)
   - Output: 1-day return prediction
   - Architecture: 2-layer bidirectional LSTM + attention + FC layers

2. **Transformer Model**
   - Input: 21 features (90-day lookback)
   - Output: 7-day return predictions
   - Architecture: Multi-head attention + positional encoding

3. **XGBoost Fusion Model**
   - Input: 21 features
   - Output: 3-class classification (Buy/Hold/Sell)

#### Training Duration

- **Per model per ticker**: ~2-3 minutes
- **Total (5 tickers × 2 models)**: ~20-30 minutes
- **Epochs**: 50 (reduced from 100 for faster evaluation prep)

#### After Retraining

```bash
# Run evaluation with retrained models
python backend/scripts/comprehensive_model_evaluation.py
```

### Option B: Retrain with Different Feature Sets (Advanced)

If you need models optimized for specific subsets of features:

```python
# Edit backend/scripts/retrain_for_evaluation.py
# Modify the feature_cols selection to use specific features only

# Example: Use only 18 features
selected_features = [col for col in feature_cols if col not in ['Market_Correlation', 'Macro_Score', 'Volume_Change']]
```

## Implementation Details

### Model Architecture Matching

The retraining script ensures all models are initialized with architectures matching the data:

**LSTM Configuration**
```python
model = LSTMAttentionModel(
    input_dim=X.shape[2],      # Automatically matches 21 features
    hidden_dim=128,
    num_layers=2,
    output_dim=1,
    dropout=0.3
)
```

**Transformer Configuration**
```python
model = TimeSeriesTransformer(
    input_dim=X.shape[2],      # Automatically matches 21 features
    d_model=64,
    nhead=settings.NHEAD,      # 4 heads
    num_layers=settings.NUM_LAYERS,  # 2 layers
    dropout=settings.DROPOUT,
    output_dim=1,
    forecast_horizon=settings.FORECAST_HORIZON  # 7 days
)
```

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `SEQ_LENGTH` | 90 days | Historical lookback window |
| `FORECAST_HORIZON` | 7 days | Prediction horizon |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 0.001 | Adam optimizer LR |
| `EPOCHS` | 50 | Training iterations (reduced for speed) |
| `DROPOUT` | 0.1 | Regularization |

## Validation After Retraining

### Step 1: Verify Model Files

```bash
# Check saved models exist
ls backend/models/saved_models/lstm_attention_*.pth    # Should show 5 files
ls backend/models/saved_models/transformer_*.pth       # Should show 5 files
ls backend/models/saved_models/scaler_*.pkl            # Should show 5 files
```

### Step 2: Run Evaluation

```bash
python backend/scripts/comprehensive_model_evaluation.py
```

Expected output:
```
✓ LSTM HDFCBANK - RMSE: X.XXXX, R2: X.XXXX, DA: XX.XX%
✓ Transformer HDFCBANK - RMSE: X.XXXX, R2: X.XXXX, DA: XX.XX%
✓ XGBoost HDFCBANK - Accuracy: XX.XX%, Precision: XX.XX%, Recall: XX.XX%, F1: XX.XX%
... (repeated for all 5 tickers)
```

### Step 3: Check Generated Files

```bash
# Evaluation results directory
backend/models/saved_models/evaluation_results/
├── 01_rmse_comparison.png
├── 02_r2_comparison.png
├── 03_directional_accuracy.png
├── 04_xgboost_metrics.png
├── 05_performance_heatmap.png
├── 06_ticker_performance.png
├── 07_ensemble_improvement.png
├── 08_box_plots.png
├── comprehensive_evaluation_report.txt
├── model_summary.csv
├── detailed_comparison.csv
├── statistical_analysis.txt
└── paper_tables.tex
```

## Troubleshooting

### Issue: Training runs out of memory

**Solution**: Reduce batch size in `retrain_for_evaluation.py`
```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduce from 32
```

### Issue: Training is too slow

**Solution**: Reduce epochs in `retrain_for_evaluation.py`
```python
max_epochs = 30  # Reduce from 50
```

### Issue: CUDA out of memory

**Solution**: Set device to CPU explicitly
```python
device = torch.device('cpu')  # Force CPU instead of CUDA
```

### Issue: Missing dependencies

**Solution**: Ensure all packages are installed
```bash
pip install -r backend/requirements.txt
```

## Performance Expectations

Based on the retrained models with 21 features:

### LSTM Performance
- **RMSE**: ~3-5% (lower is better)
- **R² Score**: 0.4-0.6 (higher is better)
- **Directional Accuracy**: 45-55%

### Transformer Performance
- **RMSE**: ~2-4% (more accurate than LSTM)
- **R² Score**: 0.5-0.7
- **Directional Accuracy**: 48-58%

### XGBoost Performance
- **Accuracy**: 50-65%
- **Precision**: 45-60%
- **Recall**: 40-55%
- **F1-Score**: 42-57%

### Ensemble Performance
- Combines strengths of all models
- Best overall directional accuracy: 50-60%

## For Your Academic Paper

### Section: Model Architecture Updated

"To ensure model consistency with enhanced feature engineering incorporating market correlation and external indicators, all models were retrained on a unified feature set of 21 indicators derived from price action (5), technical analysis (13), market dynamics (1), and external factors (2)."

### Section: Feature Set

"The feature set includes:
- **Price-based features**: Open, High, Low, Close, Volume
- **Technical indicators**: SMA-20, SMA-50, RSI, Bollinger Bands, VWAP, MACD, ATR
- **Derived features**: Log Return, Volume Change, Rolling Volatility
- **Market dynamics**: Correlation with NIFTY-50 index
- **External factors**: Simulated news sentiment and macroeconomic indicators"

## Next Steps

1. ✅ Run `python backend/scripts/retrain_for_evaluation.py`
2. ✅ Wait for completion (~25-30 minutes)
3. ✅ Run `python backend/scripts/comprehensive_model_evaluation.py`
4. ✅ Review graphs in `backend/models/saved_models/evaluation_results/`
5. ✅ Extract metrics to your paper

---

**Questions?** Check the error logs in `training_log.txt` or review individual model files in `backend/models/` directory.
