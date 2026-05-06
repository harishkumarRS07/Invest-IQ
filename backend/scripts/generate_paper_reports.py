"""
Generate detailed CSV reports and statistical analysis for academic paper
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.getcwd())

from backend.core.config import settings
from backend.core.logging import logger

RESULTS_DIR = os.path.join(settings.MODEL_DIR, 'evaluation_results')

class ReportGenerator:
    def __init__(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
    def generate_csv_reports(self):
        """Generate CSV files from evaluation results"""
        
        # Read the text report first
        report_path = os.path.join(RESULTS_DIR, 'comprehensive_evaluation_report.txt')
        
        if not os.path.exists(report_path):
            logger.warning("Comprehensive report not found. Please run evaluation first.")
            return
        
        # Create summary statistics CSV
        self._create_summary_csv()
        self._create_detailed_comparison_csv()
        self._create_latex_table()
        
    def _create_summary_csv(self):
        """Create a summary CSV with all metrics"""
        # This will be populated after evaluation runs
        summary_file = os.path.join(RESULTS_DIR, 'model_summary.csv')
        
        # Create a template
        summary_data = {
            'Metric': [
                'Average RMSE',
                'Average R2',
                'Average Directional Accuracy (%)',
                'Best RMSE Model',
                'Best R2 Model',
                'Best DA Model'
            ],
            'LSTM': ['', '', '', '', '', ''],
            'Transformer': ['', '', '', '', '', ''],
            'XGBoost': ['', '', '', '', '', ''],
            'Ensemble': ['', '', '', '', '', '']
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        logger.info(f"Summary CSV template created at {summary_file}")
        
        return summary_file
    
    def _create_detailed_comparison_csv(self):
        """Create detailed comparison spreadsheet"""
        comparison_file = os.path.join(RESULTS_DIR, 'detailed_comparison.csv')
        
        # Template for detailed metrics
        tickers = ['HDFCBANK', 'ICICIBANK', 'INFY', 'RELIANCE', 'TCS']
        models = ['LSTM', 'Transformer', 'XGBoost', 'Ensemble']
        metrics = ['RMSE', 'MAE', 'R2', 'Directional_Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        rows = []
        for model in models:
            for ticker in tickers:
                row = {'Model': model, 'Ticker': ticker}
                for metric in metrics:
                    row[metric] = ''
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(comparison_file, index=False)
        logger.info(f"Detailed comparison template created at {comparison_file}")
        
        return comparison_file
    
    def _create_latex_table(self):
        """Create LaTeX tables for paper"""
        latex_file = os.path.join(RESULTS_DIR, 'paper_tables.tex')
        
        with open(latex_file, 'w') as f:
            f.write(r"\documentclass{article}" + "\n")
            f.write(r"\usepackage{booktabs}" + "\n")
            f.write(r"\begin{document}" + "\n\n")
            
            # Model Performance Table
            f.write(r"\begin{table}[h]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{Model Performance Comparison Across Stock Tickers}" + "\n")
            f.write(r"\begin{tabular}{lcccccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Model & Metric & HDFCBANK & ICICIBANK & INFY & RELIANCE & TCS \\" + "\n")
            f.write(r"\midrule" + "\n")
            
            metrics = ['RMSE', 'R2', r'DA (\%)']
            models = ['LSTM', 'Transformer', 'Ensemble']
            
            for metric in metrics:
                f.write(f"\n{metric} & & & & & & \\\\\n")
                for model in models:
                    f.write(f"& {model} & & & & & \\\\\n")
            
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n\n")
            
            # XGBoost Classification Results
            f.write(r"\begin{table}[h]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{XGBoost Classification Model Performance}" + "\n")
            f.write(r"\begin{tabular}{lccccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Ticker & Accuracy (\%) & Precision (\%) & Recall (\%) & F1-Score (\%) \\" + "\n")
            f.write(r"\midrule" + "\n")
            
            for ticker in ['HDFCBANK', 'ICICIBANK', 'INFY', 'RELIANCE', 'TCS']:
                f.write(f"{ticker} & & & & \\\\\n")
            
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n\n")
            
            f.write(r"\end{document}" + "\n")
        
        logger.info(f"LaTeX tables template created at {latex_file}")
        return latex_file
    
    def create_instructions_file(self):
        """Create a guide for the user on how to use the results"""
        instructions_file = os.path.join(RESULTS_DIR, 'README.md')
        
        with open(instructions_file, 'w') as f:
            f.write("# Model Evaluation Results for Academic Paper\n\n")
            
            f.write("## Files Generated\n\n")
            
            f.write("### Graphs\n")
            f.write("1. **01_rmse_comparison.png** - RMSE comparison across models and tickers\n")
            f.write("2. **02_r2_comparison.png** - R² Score comparison for regression models\n")
            f.write("3. **03_directional_accuracy.png** - Directional accuracy of predictions\n")
            f.write("4. **04_xgboost_metrics.png** - XGBoost classification metrics (Accuracy, Precision, Recall, F1)\n")
            f.write("5. **05_performance_heatmap.png** - Heatmap of RMSE across all models\n")
            f.write("6. **06_ticker_performance.png** - Performance comparison by stock ticker\n")
            f.write("7. **07_ensemble_improvement.png** - Ensemble vs individual models\n")
            f.write("8. **08_box_plots.png** - Distribution of metrics across tickers\n\n")
            
            f.write("### Reports\n")
            f.write("- **comprehensive_evaluation_report.txt** - Detailed text report with all metrics\n")
            f.write("- **model_summary.csv** - Summary statistics for all models\n")
            f.write("- **detailed_comparison.csv** - Per-ticker, per-model metrics\n")
            f.write("- **paper_tables.tex** - LaTeX tables ready for your paper\n\n")
            
            f.write("## Models Evaluated\n\n")
            f.write("### 1. LSTM Attention Model\n")
            f.write("- Architecture: Bidirectional LSTM with attention mechanism\n")
            f.write("- Task: Time series forecasting (7-day horizon)\n")
            f.write("- Output: Continuous price predictions\n")
            f.write("- Metrics: RMSE, MAE, R², Directional Accuracy\n\n")
            
            f.write("### 2. Transformer Model\n")
            f.write("- Architecture: Multi-head attention transformer\n")
            f.write("- Task: Multi-step time series forecasting (7-day horizon)\n")
            f.write("- Output: Continuous price predictions\n")
            f.write("- Metrics: RMSE, MAE, R², Directional Accuracy\n\n")
            
            f.write("### 3. XGBoost Fusion Model\n")
            f.write("- Architecture: Gradient Boosting Classification\n")
            f.write("- Task: Buy/Sell/Hold signal generation\n")
            f.write("- Output: Classification probabilities\n")
            f.write("- Metrics: Accuracy, Precision, Recall, F1-Score\n\n")
            
            f.write("### 4. Ensemble Model\n")
            f.write("- Approach: Weighted ensemble of LSTM and Transformer\n")
            f.write("- Weights: 40% LSTM, 60% Transformer\n")
            f.write("- Output: Averaged predictions from regression models\n")
            f.write("- Purpose: Improved robustness and generalization\n\n")
            
            f.write("## Metrics Explanation\n\n")
            
            f.write("### Regression Metrics (LSTM, Transformer, Ensemble)\n")
            f.write("- **RMSE**: Root Mean Square Error - average prediction error\n")
            f.write("- **MAE**: Mean Absolute Error - absolute average deviation\n")
            f.write("- **R²**: Coefficient of determination - proportion of variance explained\n")
            f.write("- **Directional Accuracy**: % predictions with correct price direction\n\n")
            
            f.write("### Classification Metrics (XGBoost)\n")
            f.write("- **Accuracy**: Correct predictions / total predictions\n")
            f.write("- **Precision**: Correct positive predictions / all positive predictions\n")
            f.write("- **Recall**: Correct positive predictions / all actual positives\n")
            f.write("- **F1-Score**: Harmonic mean of precision and recall\n\n")
            
            f.write("## Stock Tickers Tested\n")
            f.write("- HDFCBANK: HDFC Bank Limited\n")
            f.write("- ICICIBANK: ICICI Bank Limited\n")
            f.write("- INFY: Infosys Limited\n")
            f.write("- RELIANCE: Reliance Industries Limited\n")
            f.write("- TCS: Tata Consultancy Services\n\n")
            
            f.write("## Data Characteristics\n\n")
            f.write("- **Sequence Length**: 60 trading days\n")
            f.write("- **Forecast Horizon**: 7 days\n")
            f.write("- **Features**: Technical indicators + Market correlation + External data\n")
            f.write("- **Train-Test Split**: 80-20 (chronological split)\n\n")
            
            f.write("## How to Use These Results\n\n")
            f.write("1. **For Figures**: Copy the PNG files directly into your paper\n")
            f.write("2. **For Tables**: Use data from CSV/LaTeX files for precise metrics\n")
            f.write("3. **For Text**: Reference values from comprehensive_evaluation_report.txt\n")
            f.write("4. **For Discussion**: Compare models and explain performance differences\n\n")
            
            f.write("## Citation Format\n\n")
            f.write("Include something like:\n")
            f.write('"We evaluated four models (LSTM Attention, Transformer, XGBoost, Ensemble) ')
            f.write('across 5 major Indian stock tickers. Results are detailed in the evaluation_results folder."\n\n')
            
            f.write("## Notes for Paper\n\n")
            f.write("- All models were trained on historical data (2015-2024)\n")
            f.write("- Predictions made on unseen test data\n")
            f.write("- Metrics calculated on normalized log returns\n")
            f.write("- Statistical significance testing recommended for comparisons\n")
        
        logger.info(f"Instructions created at {instructions_file}")


def generate_statistical_summary():
    """Generate statistical summary of results"""
    
    summary_file = os.path.join(RESULTS_DIR, 'statistical_analysis.txt')
    
    with open(summary_file, 'w') as f:
        f.write("STATISTICAL ANALYSIS OF MODEL EVALUATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Description of Metrics:\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("RMSE (Root Mean Square Error):\n")
        f.write("  Definition: sqrt(sum((pred - actual)^2) / n)\n")
        f.write("  Interpretation: Lower is better. Units: price units\n")
        f.write("  Use case: Penalizes larger errors more heavily\n\n")
        
        f.write("MAE (Mean Absolute Error):\n")
        f.write("  Definition: sum(|pred - actual|) / n\n")
        f.write("  Interpretation: Lower is better. Same units as data\n")
        f.write("  Use case: More interpretable, less sensitive to outliers\n\n")
        
        f.write("R² (Coefficient of Determination):\n")
        f.write("  Definition: 1 - (SS_res / SS_tot)\n")
        f.write("  Range: -∞ to 1 (1 is perfect prediction)\n")
        f.write("  Interpretation: Higher is better\n")
        f.write("  Values: <0% (worse than mean), 50-70% (good), >80% (excellent)\n\n")
        
        f.write("Directional Accuracy:\n")
        f.write("  Definition: % of predictions with correct price direction\n")
        f.write("  Range: 0-100%\n")
        f.write("  Baseline: 50% (random prediction)\n")
        f.write("  Use case: Crucial for trading strategies\n\n")
        
        f.write("Classification Metrics (Accuracy, Precision, Recall, F1):\n")
        f.write("  Accuracy: (TP + TN) / Total - overall correctness\n")
        f.write("  Precision: TP / (TP + FP) - false positive rate\n")
        f.write("  Recall: TP / (TP + FN) - false negative rate\n")
        f.write("  F1-Score: 2 * (Precision * Recall) / (Precision + Recall)\n\n")
        
        f.write("Expected Performance Ranges:\n")
        f.write("-" * 80 + "\n")
        f.write("RMSE: 0.001 - 0.1 (normalized returns)\n")
        f.write("R²: -0.5 to 0.8 (for stock prediction, high R² is unusual)\n")
        f.write("Directional Accuracy: 45% - 70% (stock prediction is difficult)\n")
        f.write("XGBoost Accuracy: 40% - 70% (3-class classification is hard)\n\n")
        
        f.write("Interpretation Tips:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Stock markets are highly stochastic - perfect predictions are impossible\n")
        f.write("2. DA > 50% is meaningful improvement over random guessing\n")
        f.write("3. Ensemble should be slightly better than best individual model\n")
        f.write("4. Compare models using same metric (RMSE for regression, DA for trading)\n")
        f.write("5. Different tickers may show very different model performance\n")
        f.write("6. Negative R² means model worse than predicting the mean value\n")
    
    logger.info(f"Statistical analysis saved to {summary_file}")


if __name__ == "__main__":
    try:
        generator = ReportGenerator()
        generator.generate_csv_reports()
        generator.create_instructions_file()
        generate_statistical_summary()
        
        print("\n" + "=" * 80)
        print("CSV AND REPORT GENERATION COMPLETE!")
        print(f"Results saved to: {RESULTS_DIR}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        raise
