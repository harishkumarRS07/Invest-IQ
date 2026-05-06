# InvestIQ Model Evaluation Suite for Academic Papers

Complete guide for evaluating all InvestIQ models and generating publication-quality figures and metrics.

## Overview

This evaluation suite provides **comprehensive analysis** of 4 machine learning models:
- **LSTM Attention**: Bidirectional LSTM with attention mechanism
- **Transformer**: Multi-head attention transformer for time series
- **XGBoost Fusion**: Gradient boosting classification for buy/sell signals
- **Ensemble**: Weighted combination of LSTM and Transformer

Tested across **5 major Indian stocks**:
- HDFCBANK (HDFC Bank Limited)
- ICICIBANK (ICICI Bank Limited) 
- INFY (Infosys Limited)
- RELIANCE (Reliance Industries Limited)
- TCS (Tata Consultancy Services)

## Quick Start (TL;DR)

```bash
# 1. Run complete evaluation and generate all results
python backend/scripts/run_evaluation.py

# 2. Generate detailed prediction plots (optional, takes longer)
python backend/scripts/generate_prediction_plots.py

# 3. Find results in:
# backend/models/saved_models/evaluation_results/
```

## Detailed Guide

### Step 1: Run Main Evaluation

```bash
python backend/scripts/run_evaluation.py
```

This single command:
1. ✓ Evaluates LSTM on all 5 tickers
2. ✓ Evaluates Transformer on all 5 tickers
3. ✓ Evaluates XGBoost on all 5 tickers
4. ✓ Evaluates Ensemble on all 5 tickers
5. ✓ Generates 8 publication-quality graphs
6. ✓ Creates comprehensive report with all metrics
7. ✓ Exports results to CSV format
8. ✓ Generates LaTeX tables ready for paper

**Output Files:**
- `01_rmse_comparison.png` - RMSE across models
- `02_r2_comparison.png` - R² scores comparison
- `03_directional_accuracy.png` - Directional accuracy %
- `04_xgboost_metrics.png` - Classification metrics
- `05_performance_heatmap.png` - RMSE heatmap
- `06_ticker_performance.png` - Performance by stock
- `07_ensemble_improvement.png` - Ensemble vs individuals
- `08_box_plots.png` - Metric distributions
- `comprehensive_evaluation_report.txt` - Full metrics
- `model_summary.csv` - Summary statistics
- `detailed_comparison.csv` - Per-model per-ticker metrics
- `statistical_analysis.txt` - Metric interpretations
- `paper_tables.tex` - LaTeX tables
- `README.md` - Detailed documentation

### Step 2 (Optional): Generate Prediction Visualization Plots

```bash
python backend/scripts/generate_prediction_plots.py
```

Creates detailed prediction vs actual plots:
- `lstm_predictions_*.png` - LSTM predictions over time
- `transformer_predictions_*.png` - Transformer 7-step forecasts
- `residuals_*.png` - Error analysis plots

These show:
- Full time series of predictions vs actual
- Zoomed view of recent predictions
- Residual analysis (errors, distribution, Q-Q plots)

**Perfect for:**
- Showing model behavior in paper appendix
- Visual assessment of prediction quality
- Error distribution analysis

### Step 3: Individual Scripts (Advanced)

If you want to run components separately:

#### A. Comprehensive Evaluation (graphs + report)
```bash
python backend/scripts/comprehensive_model_evaluation.py
```

#### B. Generate CSV/LaTeX Reports
```bash
python backend/scripts/generate_paper_reports.py
```

#### C. Generate Prediction Plots
```bash
python backend/scripts/generate_prediction_plots.py
```

Each script can be run independently in any order.

## Metrics Explained for Your Paper

### Regression Metrics (LSTM, Transformer, Ensemble)

| Metric | Formula | Interpretation | Good Range |
|--------|---------|-----------------|------------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | Root Mean Squared Error (lower is better) | 0.001 - 0.1 |
| **MAE** | $\frac{1}{n}\sum\|y-\hat{y}\|$ | Mean Absolute Error (lower is better) | 0.001 - 0.08 |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Variance explained (higher is better) | 0.0 - 0.8 |
| **DA** | % correct direction | Directional Accuracy % (higher is better) | 50-70% |

**Interpretation Tips:**
- R² < 0% means worse than predicting the mean
- DA = 50% is random guessing baseline
- DA > 55% shows predictive value
- Different tickers have different difficulty levels

### Classification Metrics (XGBoost)

| Metric | Definition | Good Range |
|--------|------------|------------|
| **Accuracy** | (TP + TN) / Total | 40-70% |
| **Precision** | TP / (TP + FP) | 40-70% |
| **Recall** | TP / (TP + FN) | 40-70% |
| **F1-Score** | 2 × (P × R) / (P + R) | 40-70% |

**Note:** Stock prediction is a 3-class problem (Buy/Hold/Sell), making 50% baseline. Accuracy > 55% is meaningful.

## File Structure

```
backend/models/saved_models/evaluation_results/
├── comprehensive_evaluation_report.txt      # Full metrics table
├── model_summary.csv                        # Summary statistics per model
├── detailed_comparison.csv                  # Detailed per-model per-ticker results
├── statistical_analysis.txt                 # Metrics explanations
├── paper_tables.tex                         # LaTeX tables for paper
├── README.md                                # Duplicate of this guide
│
├── 01_rmse_comparison.png                   # RMSE comparison
├── 02_r2_comparison.png                     # R² comparison
├── 03_directional_accuracy.png              # Directional accuracy
├── 04_xgboost_metrics.png                   # XGBoost results
├── 05_performance_heatmap.png               # Performance heatmap
├── 06_ticker_performance.png                # Per-ticker performance
├── 07_ensemble_improvement.png              # Ensemble analysis
├── 08_box_plots.png                         # Statistical distributions
│
└── prediction_visualizations/               # Optional detailed plots
    ├── lstm_predictions_*.png               # LSTM predictions
    ├── transformer_predictions_*.png        # Transformer predictions
    └── residuals_*.png                      # Error analysis
```

## Usage in Your Paper

### 1. Add Figures

Copy PNG files directly into your paper:

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.9\textwidth]{01_rmse_comparison.png}
  \caption{RMSE Comparison Across Stock Tickers}
  \label{fig:rmse}
\end{figure}
```

### 2. Add Results Tables

Use metrics from CSV files:

```latex
\begin{table}[h]
  \centering
  \caption{Model Performance Summary}
  \input{detailed_comparison.csv}  % auto-generated LaTeX table
\end{table}
```

Or copy from `comprehensive_evaluation_report.txt`.

### 3. Reference Metrics

"The LSTM model achieved RMSE of X.XXX and directional accuracy of X% on HDFCBANK stock (see Table 1)."

Copy exact values from the comprehensive report.

### 4. Discuss Model Comparison

Use insights from:
- `07_ensemble_improvement.png` - Show ensemble benefits
- `05_performance_heatmap.png` - Show consistency
- `statistical_analysis.txt` - Explain metrics

Example: "The ensemble approach improved average R² by X% compared to individual models."

## Technical Details for Paper

### Data Characteristics
- **Period**: 2015-2024 (historical daily data)
- **Stocks**: 5 major Indian indices
- **Features**: 20+ technical indicators + market correlation + external data
- **Preprocessing**: Log normalization, MinMax scaling
- **Train/Test Split**: 80/20 chronological split
- **Sequence Length**: 60 trading days
- **Forecast Horizon**: 7 days

### Model Architectures

**LSTM Attention:**
```
Input (seq_len, input_dim) 
  ↓
Bidirectional LSTM (128 hidden, 2 layers)
  ↓
Layer Normalization
  ↓
Attention Mechanism
  ↓
Dense (128 → 1)
  ↓
Output (continuous value)
```

**Transformer:**
```
Input (seq_len, input_dim)
  ↓
Linear Embedding → d_model=64
  ↓
Positional Encoding
  ↓
Transformer Encoder (nhead=4, layers=2)
  ↓
Dense Decoder (64 → 7*horizon)
  ↓
Output (7-day forecast)
```

**XGBoost:**
```
Input: Technical features
  ↓
XGBClassifier (500 estimators)
  ↓
Output: Buy/Hold/Sell probabilities
```

**Ensemble:**
```
LSTM Output (0.4 weight)
  ⊕
Transformer Output (0.6 weight)
  ↓
Weighted Average
```

## Common Questions

### Q: Which model performs best?
A: Check `07_ensemble_improvement.png` and `05_performance_heatmap.png`. The ensemble should be comparable to the best individual model.

### Q: Why is R² negative sometimes?
A: Stock prediction is difficult. Negative R² means the model performs worse than simply predicting the mean. This is expected for unpredictable assets.

### Q: Why is directional accuracy only 55-60%?
A: The market is stochastic. Perfect predictions are impossible. >55% is meaningful improvement over 50% baseline.

### Q: Which ticker is easiest to predict?
A: Check `02_r2_comparison.png` and `06_ticker_performance.png`. Usually liquid stocks like RELIANCE or TCS are easier.

### Q: Can I use these models for real trading?
A: The evaluation shows historical performance. Real trading requires additional considerations (transaction costs, slippage, risk management).

### Q: How do I interpret XGBoost accuracy?
A: It's a 3-class problem (Buy/Hold/Sell). 50% baseline accuracy = random guessing. 55-65% is good performance.

## Troubleshooting

### Issue: Models not found
**Solution:** Make sure models are trained first:
```bash
python backend/scripts/train_all.py
```

### Issue: Data files not found
**Solution:** Ensure CSV files exist in:
```
backend/data/stock_data/
```

### Issue: CUDA out of memory
**Solution:** Scripts automatically use CPU. No GPU required.

### Issue: Plots look truncated
**Solution:** Ensure matplotlib backend is set to 'Agg':
```bash
python -c "import matplotlib; matplotlib.use('Agg')"
```

## Performance Benchmarking

### Expected Results

| Model | RMSE | R² | DA (%) |
|-------|------|-----|--------|
| LSTM | 0.01-0.05 | 0.0-0.4 | 50-65 |
| Transformer | 0.01-0.05 | 0.0-0.4 | 50-65 |
| Ensemble | 0.01-0.05 | 0.0-0.4 | 52-66 |
| XGBoost Accuracy | - | - | 50-70% |

### Runtime
- Full evaluation: 5-15 minutes (depending on system)
- Prediction plots: 5-10 minutes (optional)
- CSV generation: < 1 minute

## Statistical Rigor

For academic paper, consider adding:

1. **Confidence Intervals**
   - Run K-fold cross-validation
   - Report mean ± std deviation

2. **Statistical Tests**
   - Is ensemble significantly better? (paired t-test)
   - Significance of difference between models (ANOVA)

3. **Ablation Studies**
   - Feature importance analysis
   - Impact of sequence length
   - Effect of forecast horizon

4. **Robustness Analysis**
   - Different time periods
   - Market conditions (bull/bear)
   - Different feature sets

## Citing This Work

Example citation:
```
"Model evaluation was performed on 5 Indian stock tickers 
(HDFCBANK, ICICIBANK, INFY, RELIANCE, TCS) using historical 
data from 2015-2024. We implemented LSTM Attention, Transformer, 
and XGBoost models with comprehensive evaluation metrics including 
RMSE, R², directional accuracy, and classification metrics."
```

## Next Steps

1. ✓ Run `python backend/scripts/run_evaluation.py`
2. ✓ Review output in `evaluation_results/` folder
3. ✓ Copy PNG figures to your paper
4. ✓ Include metrics from CSV/report files
5. ✓ Use LaTeX tables for results section
6. ✓ Discuss findings in analysis section
7. ✓ Generate prediction plots if needed for appendix

## Support

For issues or questions about the evaluation:
1. Check `statistical_analysis.txt` for metric explanations
2. Review `comprehensive_evaluation_report.txt` for specific values
3. Examine generated plots for visual guidance
4. Check Python logs in terminal output

Good luck with your paper! 📊📈

---
*Generated by InvestIQ Model Evaluation Suite*
*For academic research purposes only*
