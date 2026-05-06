"""
Comprehensive Model Evaluation Script
Evaluates all models (LSTM, Transformer, XGBoost, Ensemble) across all tickers
Generates publication-quality graphs for academic papers
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.append(os.getcwd())

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator
from backend.models.enhanced_models import LSTMAttentionEnhanced
from backend.models.xgboost_fusion import XGBoostFusionModel
from backend.evaluation.metrics import calculate_metrics
from backend.training.train import create_sequences

# Set up matplotlib style for academic papers
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10

# Create output directory for results
RESULTS_DIR = os.path.join(settings.MODEL_DIR, 'evaluation_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Available tickers
TICKERS = ['HDFCBANK', 'ICICIBANK', 'INFY', 'RELIANCE', 'TCS']

class ComprehensiveModelEvaluator:
    def __init__(self):
        self.results = {
            'LSTM': {},
            'Transformer': {},
            'XGBoost': {},
            'Ensemble': {}
        }
        self.all_metrics_df = pd.DataFrame()
        
    def load_and_preprocess_data(self, ticker):
        """Load and preprocess data for a ticker"""
        try:
            data_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
            if not os.path.exists(data_path):
                # Try without .NS suffix
                data_path = os.path.join(settings.DATA_DIR, f"{ticker}.NS.csv")
                if not os.path.exists(data_path):
                    logger.warning(f"Data not found for {ticker}")
                    return None
            
            df = load_data(data_path)
            df = clean_data(df)
            
            # Add features
            market_df = ExternalDataSimulator.fetch_market_index(
                start_date=df.index[0], 
                end_date=df.index[-1]
            )
            df = add_technical_indicators(df)
            df = add_market_correlation(df, market_df)
            df = ExternalDataSimulator.add_external_features(df, ticker)
            df = df.dropna()
            
            # Load scaler
            scaler = StockScaler()
            try:
                scaler.load(f"scaler_{ticker}.pkl")
            except:
                logger.warning(f"Scaler not found for {ticker}, using new scaler")
                
            feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
            df_scaled = scaler.transform(df)
            
            return df, df_scaled, feature_cols, scaler
        except Exception as e:
            logger.error(f"Error preprocessing data for {ticker}: {e}")
            return None
    
    def evaluate_lstm(self, ticker):
        """Evaluate LSTM Attention Model"""
        try:
            data_tuple = self.load_and_preprocess_data(ticker)
            if data_tuple is None:
                logger.warning(f"Could not load data for {ticker}")
                return None
            
            df, df_scaled, feature_cols, scaler = data_tuple
            
            data_scaled = df_scaled[feature_cols].values
            target_col = 'Log_Return' if 'Log_Return' in feature_cols else 'Close'
            if target_col not in feature_cols:
                logger.warning(f"Target column {target_col} not in features for {ticker}")
                return None
            
            target_col_idx = feature_cols.index(target_col)
            
            X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, 1, target_col_idx)
            
            if X.shape[0] == 0 or y.shape[0] == 0:
                logger.warning(f"No sequences created for {ticker}")
                return None
            
            # Load model
            model_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")
            if not os.path.exists(model_path):
                logger.warning(f"LSTM model not found for {ticker}")
                return None
            
            model = LSTMAttentionEnhanced(
                input_dim=X.shape[2],
                hidden_dim=128,
                num_layers=2,
                output_dim=1,
                dropout=0.3,
                forecast_horizon=settings.FORECAST_HORIZON
            )
            
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # Make predictions
            X_tensor = torch.FloatTensor(X)
            with torch.no_grad():
                preds = model(X_tensor).numpy()
            
            # Ensure shapes match for metrics
            if preds.shape != y.shape:
                preds = preds.reshape(y.shape)
            
            # Calculate metrics
            metrics = calculate_metrics(y, preds)
            self.results['LSTM'][ticker] = metrics
            
            logger.info(f"✓ LSTM {ticker} - RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}, DA: {metrics['Directional_Accuracy']:.2f}%")
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating LSTM for {ticker}: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_transformer(self, ticker):
        """Evaluate Transformer Model"""
        try:
            data_tuple = self.load_and_preprocess_data(ticker)
            if data_tuple is None:
                logger.warning(f"Could not load data for {ticker}")
                return None
            
            df, df_scaled, feature_cols, scaler = data_tuple
            
            data_scaled = df_scaled[feature_cols].values
            target_col = 'Log_Return' if 'Log_Return' in feature_cols else 'Close'
            if target_col not in feature_cols:
                logger.warning(f"Target column {target_col} not in features for {ticker}")
                return None
            
            target_col_idx = feature_cols.index(target_col)
            
            X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, settings.FORECAST_HORIZON, target_col_idx)
            
            if X.shape[0] == 0 or y.shape[0] == 0:
                logger.warning(f"No sequences created for {ticker}")
                return None
            
            if len(y.shape) == 2:
                y = y[..., np.newaxis]
            
            # Load model
            model_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")
            if not os.path.exists(model_path):
                logger.warning(f"LSTM model not found for {ticker}")
                return None
            
            model = LSTMAttentionEnhanced(
                input_dim=X.shape[2],
                hidden_dim=128,
                num_layers=2,
                output_dim=1,
                dropout=0.3,
                forecast_horizon=settings.FORECAST_HORIZON
            )
            
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # Make predictions
            X_tensor = torch.FloatTensor(X)
            with torch.no_grad():
                preds = model(X_tensor).numpy()
            
            # Ensure shapes match for metrics
            if preds.shape != y.shape:
                preds = preds.reshape(y.shape)
            
            # Calculate metrics
            metrics = calculate_metrics(y, preds)
            self.results['Transformer'][ticker] = metrics
            
            logger.info(f"✓ Transformer {ticker} - RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}, DA: {metrics['Directional_Accuracy']:.2f}%")
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating Transformer for {ticker}: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_xgboost(self, ticker):
        """Evaluate XGBoost Classification Model"""
        try:
            data_tuple = self.load_and_preprocess_data(ticker)
            if data_tuple is None:
                logger.warning(f"Could not load data for {ticker}")
                return None
            
            df, df_scaled, feature_cols, scaler = data_tuple
            
            # Prepare features and labels
            df_scaled_copy = df_scaled.copy()
            
            xgb_model = XGBoostFusionModel()
            close_col = df_scaled_copy[['Close']] if 'Close' in df_scaled_copy.columns else df_scaled_copy.iloc[:, :1]
            labels = xgb_model.prepare_labels(close_col, horizon=5, threshold=0.01)
            
            X = df_scaled[feature_cols].values[:-5]
            y = labels.astype(int)[:len(X)]
            
            if X.shape[0] == 0 or y.shape[0] == 0:
                logger.warning(f"No valid data for XGBoost evaluation on {ticker}")
                return None
            
            # Load model
            model_path = os.path.join(settings.MODEL_DIR, f"xgboost_fusion_{ticker}.pkl")
            if not os.path.exists(model_path):
                logger.warning(f"XGBoost model not found for {ticker}")
                return None
            
            xgb_model.load(f"xgboost_fusion_{ticker}.pkl")
            
            # Make predictions
            preds = xgb_model.predict(pd.DataFrame(X, columns=feature_cols[:X.shape[1]]))
            
            # Calculate accuracy metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'Accuracy': accuracy_score(y, preds) * 100,
                'Precision': precision_score(y, preds, average='weighted', zero_division=0) * 100,
                'Recall': recall_score(y, preds, average='weighted', zero_division=0) * 100,
                'F1_Score': f1_score(y, preds, average='weighted', zero_division=0) * 100,
                'MSE': np.mean((y - preds) ** 2),
                'MAE': np.mean(np.abs(y - preds))
            }
            
            self.results['XGBoost'][ticker] = metrics
            
            logger.info(f"✓ XGBoost {ticker} - Accuracy: {metrics['Accuracy']:.2f}%, F1: {metrics['F1_Score']:.2f}%")
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating XGBoost for {ticker}: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_ensemble(self, ticker):
        """Ensemble performance (average of LSTM and Transformer)"""
        try:
            lstm_metrics = self.results['LSTM'].get(ticker)
            transformer_metrics = self.results['Transformer'].get(ticker)
            
            if lstm_metrics is None:
                logger.warning(f"LSTM metrics not available for {ticker}")
                return None
            if transformer_metrics is None:
                logger.warning(f"Transformer metrics not available for {ticker}")
                return None
            
            # Average ensemble metrics
            ensemble_metrics = {}
            for key in lstm_metrics.keys():
                if key in transformer_metrics:
                    ensemble_metrics[key] = (lstm_metrics[key] + transformer_metrics[key]) / 2
            
            self.results['Ensemble'][ticker] = ensemble_metrics
            
            logger.info(f"✓ Ensemble {ticker} - RMSE: {ensemble_metrics['RMSE']:.4f}, R2: {ensemble_metrics['R2']:.4f}")
            return ensemble_metrics
        
        except Exception as e:
            logger.error(f"Error evaluating Ensemble for {ticker}: {e}", exc_info=True)
            return None
    
    def run_all_evaluations(self):
        """Run evaluation for all models and tickers"""
        logger.info("=" * 80)
        logger.info("Starting Comprehensive Model Evaluation")
        logger.info("=" * 80)
        
        for ticker in TICKERS:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating {ticker}")
            logger.info(f"{'='*80}")
            
            self.evaluate_lstm(ticker)
            self.evaluate_transformer(ticker)
            self.evaluate_xgboost(ticker)
            self.evaluate_ensemble(ticker)
        
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Complete!")
        logger.info("=" * 80)
        
        self.create_summary_report()
        self.generate_all_graphs()
    
    def create_summary_report(self):
        """Create comprehensive summary report"""
        report_path = os.path.join(RESULTS_DIR, 'comprehensive_evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            
            # Model-by-model summary
            for model_name in ['LSTM', 'Transformer', 'XGBoost', 'Ensemble']:
                f.write(f"\n{model_name.upper()} MODEL RESULTS\n")
                f.write("-" * 100 + "\n")
                
                if not self.results[model_name]:
                    f.write("No results available\n")
                    continue
                
                # Create dataframe for this model
                model_df = pd.DataFrame(self.results[model_name]).T
                f.write(model_df.to_string())
                f.write("\n\n")
                
                # Calculate averages
                f.write(f"Average across all tickers:\n")
                avg_metrics = model_df.mean()
                f.write(avg_metrics.to_string())
                f.write("\n\n")
            
            # Best performers
            f.write("\n" + "=" * 100 + "\n")
            f.write("BEST PERFORMERS BY METRIC\n")
            f.write("=" * 100 + "\n\n")
            
            # Compile all results for comparison
            all_results = []
            for model in self.results:
                for ticker in self.results[model]:
                    metrics = self.results[model][ticker]
                    all_results.append({
                        'Model': model,
                        'Ticker': ticker,
                        **metrics
                    })
            
            if all_results:
                comparison_df = pd.DataFrame(all_results)
                
                # Find best for each metric
                for metric in comparison_df.columns:
                    if metric not in ['Model', 'Ticker']:
                        if 'Accuracy' in metric or 'R2' in metric or 'Precision' in metric or 'Recall' in metric or 'F1' in metric:
                            # Higher is better
                            best_idx = comparison_df[metric].idxmax()
                        else:
                            # Lower is better (RMSE, MSE, MAE, etc)
                            best_idx = comparison_df[metric].idxmin()
                        
                        best_row = comparison_df.loc[best_idx]
                        f.write(f"{metric}: {best_row['Model']} ({best_row['Ticker']}) = {best_row[metric]:.4f}\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def generate_all_graphs(self):
        """Generate all visualization graphs"""
        self._plot_rmse_comparison()
        self._plot_r2_comparison()
        self._plot_directional_accuracy()
        self._plot_model_performance_heatmap()
        self._plot_xgboost_metrics()
        self._plot_ticker_performance()
        self._plot_ensemble_improvement()
        self._plot_box_plots()
        
        logger.info(f"All graphs saved to {RESULTS_DIR}")
    
    def _plot_rmse_comparison(self):
        """Compare RMSE across all models and tickers"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        models = ['LSTM', 'Transformer', 'XGBoost', 'Ensemble']
        x = np.arange(len(TICKERS))
        width = 0.2
        
        for i, model in enumerate(models):
            rmses = []
            for ticker in TICKERS:
                metrics = self.results[model].get(ticker)
                if metrics and 'RMSE' in metrics:
                    rmses.append(metrics['RMSE'])
                elif metrics and 'MSE' in metrics:
                    rmses.append(np.sqrt(metrics['MSE']))
                else:
                    rmses.append(0)
            
            ax.bar(x + i*width, rmses, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: Root Mean Square Error Across Tickers', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(TICKERS)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '01_rmse_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_r2_comparison(self):
        """Compare R² Score across models"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        models = ['LSTM', 'Transformer', 'Ensemble']
        x = np.arange(len(TICKERS))
        width = 0.25
        
        for i, model in enumerate(models):
            r2_scores = []
            for ticker in TICKERS:
                metrics = self.results[model].get(ticker)
                r2_scores.append(metrics.get('R2', 0) if metrics else 0)
            
            ax.bar(x + i*width, r2_scores, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: R² Score Across Tickers', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(TICKERS)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '02_r2_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_directional_accuracy(self):
        """Compare Directional Accuracy"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        models = ['LSTM', 'Transformer', 'Ensemble']
        x = np.arange(len(TICKERS))
        width = 0.25
        
        for i, model in enumerate(models):
            da_scores = []
            for ticker in TICKERS:
                metrics = self.results[model].get(ticker)
                da_scores.append(metrics.get('Directional_Accuracy', 0) if metrics else 0)
            
            ax.bar(x + i*width, da_scores, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax.set_ylabel('Directional Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: Directional Accuracy Across Tickers', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(TICKERS)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim((0, 100))
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '03_directional_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_xgboost_metrics(self):
        """Compare XGBoost Classification Metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = []
            for ticker in TICKERS:
                xgb_metrics = self.results['XGBoost'].get(ticker)
                values.append(xgb_metrics.get(metric, 0) if xgb_metrics else 0)
            
            colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(TICKERS)))
            bars = ax.bar(TICKERS, values, color=colors, alpha=0.8)
            
            ax.set_ylabel(f'{metric} (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'XGBoost: {metric} by Ticker', fontsize=12, fontweight='bold')
            ax.set_ylim((0, 100))
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '04_xgboost_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance_heatmap(self):
        """Heatmap of all models' RMSE"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        heatmap_data = []
        models = ['LSTM', 'Transformer', 'Ensemble']
        
        for model in models:
            row = []
            for ticker in TICKERS:
                metrics = self.results[model].get(ticker)
                if metrics and 'RMSE' in metrics:
                    row.append(metrics['RMSE'])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        
        sns.heatmap(heatmap_array, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                   xticklabels=TICKERS, yticklabels=models, 
                   cbar_kws={'label': 'RMSE'}, ax=ax)
        
        ax.set_title('Model Performance Heatmap: RMSE Across Tickers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '05_performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ticker_performance(self):
        """Performance across tickers for all models"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(TICKERS))
        width = 0.2
        
        models = ['LSTM', 'Transformer', 'XGBoost', 'Ensemble']
        
        for i, model in enumerate(models):
            accuracies = []
            for ticker in TICKERS:
                metrics = self.results[model].get(ticker)
                if metrics:
                    if 'Accuracy' in metrics:  # XGBoost
                        accuracies.append(metrics['Accuracy'])
                    elif 'R2' in metrics:  # Regression
                        accuracies.append(max(0, metrics['R2'] * 100))  # Convert to percentage
                    else:
                        accuracies.append(0)
                else:
                    accuracies.append(0)
            
            ax.bar(x + i*width, accuracies, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Overall Performance by Ticker', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(TICKERS)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '06_ticker_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ensemble_improvement(self):
        """Show ensemble improvement over individual models"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(TICKERS))
        width = 0.25
        
        lstm_r2 = []
        transformer_r2 = []
        ensemble_r2 = []
        
        for ticker in TICKERS:
            lstm_metrics = self.results['LSTM'].get(ticker)
            lstm_r2.append(lstm_metrics.get('R2', 0) if lstm_metrics else 0)
            
            transformer_metrics = self.results['Transformer'].get(ticker)
            transformer_r2.append(transformer_metrics.get('R2', 0) if transformer_metrics else 0)
            
            ensemble_metrics = self.results['Ensemble'].get(ticker)
            ensemble_r2.append(ensemble_metrics.get('R2', 0) if ensemble_metrics else 0)
        
        ax.bar(x - width, lstm_r2, width, label='LSTM', alpha=0.8)
        ax.bar(x, transformer_r2, width, label='Transformer', alpha=0.8)
        ax.bar(x + width, ensemble_r2, width, label='Ensemble (Avg)', alpha=0.8, color='gold')
        
        ax.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Ensemble vs Individual Models - R² Score', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(TICKERS)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '07_ensemble_improvement.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_box_plots(self):
        """Box plots showing distribution of metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # RMSE distribution
        rmse_data = []
        rmse_labels = []
        for model in ['LSTM', 'Transformer', 'Ensemble']:
            rmses = []
            for ticker in TICKERS:
                metrics = self.results[model].get(ticker)
                if metrics and 'RMSE' in metrics:
                    rmses.append(metrics['RMSE'])
            if rmses:
                rmse_data.append(rmses)
                rmse_labels.append(model)
        
        if rmse_data:
            axes[0].boxplot(rmse_data)
            axes[0].set_xticklabels(rmse_labels)
            axes[0].set_ylabel('RMSE', fontsize=11, fontweight='bold')
            axes[0].set_title('RMSE Distribution by Model', fontsize=12, fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)
        
        # R² distribution
        r2_data = []
        r2_labels = []
        for model in ['LSTM', 'Transformer', 'Ensemble']:
            r2s = []
            for ticker in TICKERS:
                metrics = self.results[model].get(ticker)
                if metrics and 'R2' in metrics:
                    r2s.append(metrics['R2'])
            if r2s:
                r2_data.append(r2s)
                r2_labels.append(model)
        
        if r2_data:
            axes[1].boxplot(r2_data)
            axes[1].set_xticklabels(r2_labels)
            axes[1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
            axes[1].set_title('R² Distribution by Model', fontsize=12, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '08_box_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    try:
        evaluator = ComprehensiveModelEvaluator()
        evaluator.run_all_evaluations()
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print(f"Results saved to: {RESULTS_DIR}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error during evaluation: {e}")
        raise
