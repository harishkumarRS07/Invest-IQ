# ⚡ QUICK START: Model Evaluation for Your Paper

## In 3 Simple Steps

### Step 1: Run Evaluation (Takes 5-15 minutes)
Double-click: **`run_model_evaluation.bat`**

Or in terminal:
```bash
python backend/scripts/run_evaluation.py
```

### Step 2: Check Results
Results folder: `backend/models/saved_models/evaluation_results/`

You'll get:
- ✅ 8 publication-ready graphs (PNG files)
- ✅ Comprehensive metrics report
- ✅ CSV files with all numbers
- ✅ LaTeX tables for your paper

### Step 3: Use in Your Paper
- **Copy PNG files** → Add to paper figures
- **Copy metrics** → Add to results section
- **Use tables** → Include in manuscript

---

## What Gets Generated

### Graphs (ready for publication)
```
01_rmse_comparison.png          - Compare model accuracy
02_r2_comparison.png            - R² scores across models
03_directional_accuracy.png     - Prediction direction accuracy
04_xgboost_metrics.png          - Buy/Sell/Hold accuracy
05_performance_heatmap.png      - Visual performance grid
06_ticker_performance.png       - Performance per stock
07_ensemble_improvement.png     - Ensemble vs individual models
08_box_plots.png                - Statistical distributions
```

### Reports
```
comprehensive_evaluation_report.txt  - All metrics in table format
model_summary.csv                    - Summary statistics
detailed_comparison.csv              - Per-model per-ticker metrics
statistical_analysis.txt             - How to interpret results
paper_tables.tex                     - LaTeX tables ready to use
```

---

## Models Evaluated

| Model | Type | Best For |
|-------|------|----------|
| **LSTM** | Deep Learning | Price prediction |
| **Transformer** | Deep Learning | Multi-step forecasting |
| **XGBoost** | Classification | Buy/Sell signals |
| **Ensemble** | Hybrid | Robust predictions |

---

## Stocks Tested

- HDFCBANK (HDFC Bank)
- ICICIBANK (ICICI Bank)
- INFY (Infosys)
- RELIANCE (Reliance Industries)
- TCS (Tata Consultancy Services)

---

## Key Metrics Your Paper Will Report

### For Price Prediction
- **RMSE**: How far off predictions are (lower is better)
- **R²**: How much variance is explained (higher is better)
- **Directional Accuracy**: % of correct price directions

### For Buy/Sell Signals
- **Accuracy**: % of correct predictions
- **Precision**: Reliability of signals
- **Recall**: Coverage of opportunities
- **F1-Score**: Overall classification quality

---

## Optional: Get More Detailed Plots

Double-click: **`run_prediction_plots.bat`**

This generates additional visualizations:
- LSTM predictions vs actual (full & zoomed)
- Transformer 7-day forecasts
- Error analysis and residuals

Perfect for appendix or detailed analysis!

---

## Typical Results (For Reference)

Your results should look something like:

| Model | Avg RMSE | Avg R² | Avg Directional Accuracy |
|-------|----------|--------|--------------------------|
| LSTM | 0.02-0.05 | 0.0-0.3 | 52-65% |
| Transformer | 0.02-0.05 | 0.0-0.3 | 52-65% |
| Ensemble | 0.02-0.05 | 0.0-0.3 | 53-66% |
| **XGBoost** | **N/A** | **N/A** | **Accuracy: 50-70%** |

(Stock prediction is hard - these are good results!)

---

## Troubleshooting

### ❌ "Models not found"
Train them first:
```bash
python backend/scripts/train_all.py
```

### ❌ "Data not found"
Ensure CSV files exist in:
```
backend/data/stock_data/
```

### ❌ "Import errors"
Make sure environment is set up:
```bash
cd backend
python setup_env.bat
python verify_setup.py
```

### ❌ Scripts won't run on Mac/Linux
Use Python directly:
```bash
python backend/scripts/run_evaluation.py
python backend/scripts/generate_prediction_plots.py
```

---

## How to Use in Your Paper

### Add a Figure
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{01_rmse_comparison.png}
\caption{Model Performance: RMSE Comparison Across Stock Tickers}
\end{figure}
```

### Add Results
"The LSTM model achieved RMSE of 0.025 and directional accuracy 
of 58% on HDFCBANK stock, outperforming the baseline by 8%."

*Copy exact values from `comprehensive_evaluation_report.txt`*

### Add a Table
Copy from `detailed_comparison.csv` or use `paper_tables.tex`

### Discuss Your Results
"The ensemble approach combined LSTM and Transformer predictions 
with weights of 40% and 60% respectively, achieving an average 
directional accuracy of 64% across all tickers."

---

## File Structure

```
InvestIQ-main/
├── run_model_evaluation.bat           ← DOUBLE-CLICK TO START
├── run_prediction_plots.bat           ← For detailed plots (optional)
├── MODEL_EVALUATION_GUIDE.md          ← Full documentation
├── QUICK_START.md                     ← This file
└── backend/
    ├── models/saved_models/
    │   └── evaluation_results/        ← Your results go HERE
    └── scripts/
        ├── run_evaluation.py
        ├── generate_prediction_plots.py
        └── generate_paper_reports.py
```

---

## Time Estimate

- ⏱️ **Full Evaluation**: 5-15 minutes
- ⏱️ **Prediction Plots**: 5-10 minutes (optional)
- ⏱️ **Using Results**: 30+ minutes (copy to paper, write up)

---

## Advanced Usage

### Run Individual Models
Create a script to run specific models:
```python
from backend.scripts.comprehensive_model_evaluation import ComprehensiveModelEvaluator

evaluator = ComprehensiveModelEvaluator()
evaluator.evaluate_lstm('HDFCBANK')
```

### Customize Graphs
Modify `comprehensive_model_evaluation.py` to change:
- Graph colors and styles
- Metrics displayed
- Output format

### Add More Metrics
Edit `evaluate.py` or `metrics.py` to add:
- More statistical measures
- Custom error metrics
- Additional visualizations

---

## Need Help?

1. **Check the comprehensive guide**: [MODEL_EVALUATION_GUIDE.md](MODEL_EVALUATION_GUIDE.md)
2. **Read the report**: `evaluation_results/statistical_analysis.txt`
3. **Look at output**: `evaluation_results/comprehensive_evaluation_report.txt`

---

## Next: What to Do After Evaluation

### Immediate (Use in Paper)
- [ ] Copy PNG files to paper
- [ ] Copy metrics to results table
- [ ] Add discussion of findings

### Secondary (For Appendix)
- [ ] Include prediction plots
- [ ] Add residual analysis
- [ ] Show error distributions

### Final (Polish Paper)
- [ ] Cite model architectures
- [ ] Explain evaluation methodology
- [ ] Discuss limitations
- [ ] Suggest future work

---

## Paper Writing Tips

**Don't say:** "The models had high accuracy"
**Do say:** "The ensemble model achieved 64% directional accuracy, 
representing 14% improvement over the 50% baseline."

**Don't say:** "LSTM was best"
**Do say:** "Across 5 stock tickers, the ensemble approach achieved 
the highest average R² of 0.28, with RMSE of 0.032."

**Don't say:** "Results are in the appendix"
**Do say:** "As shown in Figure X and Table Y, the Transformer model 
consistently outperformed baseline methods..."

---

**Ready to get started? Double-click `run_model_evaluation.bat` now!** 🚀

---
*InvestIQ Model Evaluation Suite v1.0*
*For academic research and publication*
