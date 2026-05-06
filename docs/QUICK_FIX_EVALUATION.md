## Model Evaluation Fix - Quick Start Guide

### Problem
Your saved models were trained with different feature counts (15, 18, 20) than the current pipeline (21 features). This caused loading errors during evaluation.

### Solution
Retrain models with the current 21-feature set.

---

## ⚡ Quick Fix (3 Steps)

### Step 1: Retrain Models
```bash
python backend/scripts/retrain_for_evaluation.py
```
**Duration**: ~25-30 minutes  
**What it does**: Retrains LSTM and Transformer for 5 tickers with 21 features

### Step 2: Wait for Completion
Monitor the console output. You'll see:
```
LSTM HDFCBANK - X shape: (5994, 90, 21), y shape: (5994, 1)
LSTM HDFCBANK - Train Loss: 0.001234 - Val Loss: 0.001567
...
✓ LSTM model saved to backend/models/saved_models/lstm_attention_HDFCBANK.pth
```

### Step 3: Run Evaluation
```bash
python backend/scripts/comprehensive_model_evaluation.py
```

---

## 📊 Expected Output

After evaluation completes:

✅ **Console Output**
```
✓ LSTM HDFCBANK - RMSE: 0.0234, R2: 0.5678, DA: 52.34%
✓ Transformer HDFCBANK - RMSE: 0.0198, R2: 0.6234, DA: 54.56%
✓ XGBoost HDFCBANK - Accuracy: 58.90%, Precision: 60.12%, Recall: 56.78%, F1: 58.43%
```

✅ **Generated Files**
```
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
└── paper_tables.tex
```

---

## 🐛 Troubleshooting

### If retraining is too slow?
Edit `backend/scripts/retrain_for_evaluation.py` and change:
```python
max_epochs = 30  # Reduce from 50
```

### If you get CUDA memory error?
The script automatically uses CPU if CUDA runs out of memory. To force CPU:
```bash
# Windows
set CUDA_VISIBLE_DEVICES=-1
python backend/scripts/retrain_for_evaluation.py

# Linux/Mac
export CUDA_VISIBLE_DEVICES=-1
python backend/scripts/retrain_for_evaluation.py
```

### If retraining fails partway through?
Delete the partially trained model files and restart:
```bash
del backend/models/saved_models/lstm_attention_*.pth
del backend/models/saved_models/transformer_*.pth
python backend/scripts/retrain_for_evaluation.py
```

---

## 📚 Technical Background

**Why this happened**: Your saved models were trained when the feature engineering pipeline was simpler:
- Old: 15-20 features
- Now: 21 features (added Market_Correlation, Sentiment, Macro_Score)

**What's being retrained**:
- LSTM Attention Model (input: 21 features → 90-day lookback)
- Transformer Model (input: 21 features → 7-day forecast)
- XGBoost stays as-is (separately handled if needed)

**Retraining parameters**:
- Batch size: 32
- Learning rate: 0.001
- Epochs: 50 (fast mode for evaluation)
- Train/test split: 80/20

---

## ✨ For Your Paper

After evaluation completes, use these files:

1. **Graphs** (PNG 300dpi):
   - `01_rmse_comparison.png` - Model prediction accuracy comparison
   - `02_r2_comparison.png` - R² score across tickers
   - `04_xgboost_metrics.png` - Classification metrics for XGBoost
   - `05_performance_heatmap.png` - Overall performance matrix

2. **Data** (CSV/LaTeX):
   - `model_summary.csv` - Summary statistics
   - `detailed_comparison.csv` - Detailed per-model metrics
   - `paper_tables.tex` - LaTeX tables ready for paper

3. **Reports**:
   - `comprehensive_evaluation_report.txt` - Full metrics breakdown
   - `statistical_analysis.txt` - Statistical interpretations

---

## 🚀 Next Steps

1. Run retraining: `python backend/scripts/retrain_for_evaluation.py`
2. Run evaluation: `python backend/scripts/comprehensive_model_evaluation.py`
3. Copy PNG graphs to `docs/paper_figures/`
4. Use CSV metrics in results section
5. Include LaTeX tables in appendix

---

**Total time**: ~45 minutes (30 min retraining + 10 min evaluation + 5 min cleanup)

Need help? See `MODEL_RETRAINING_GUIDE.md` for detailed explanations.
