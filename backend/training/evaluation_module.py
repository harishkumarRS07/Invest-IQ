"""
Comprehensive Evaluation and Visualization Module

Provides production-level evaluation metrics, visualizations, and analysis tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    ConfusionMatrixDisplay
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    calibration_curve = None
import os
import sys
from typing import Any, Dict, Tuple, Optional

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from core.logging import logger

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ProductionEvaluator:
    """Advanced evaluation and visualization utilities."""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(cm, display_labels=['DOWN', 'UP']).plot(ax=ax, cmap='Blues')
        plt.title('Confusion Matrix (Test Set)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Optional[str] = None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Optional[str] = None):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.grid()
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confidence_distribution(confidence: np.ndarray, y_true: np.ndarray, save_path: Optional[str] = None):
        """Plot confidence score distribution for correct vs incorrect predictions."""
        y_pred = (confidence > 0.5).astype(int)
        correct = y_true == y_pred
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(confidence[correct], bins=30, alpha=0.7, label='Correct Predictions', color='green')
        ax.hist(confidence[~correct], bins=30, alpha=0.7, label='Incorrect Predictions', color='red')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Score Distribution')
        ax.legend()
        ax.grid()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Optional[str] = None):
        """Plot calibration curve."""
        if calibration_curve is None:
            logger.warning("calibration_curve is unavailable; skipping calibration plot")
            return
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, 's-', label='Model Calibration', linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.grid()
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def generate_evaluation_report(
        results: Dict,
        ticker: str,
        save_dir: str = "evaluation_reports"
    ) -> Dict:
        """Generate comprehensive evaluation report."""
        os.makedirs(save_dir, exist_ok=True)
        
        report = {
            'ticker': ticker,
            'timestamp': pd.Timestamp.now(),
            'metrics': {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1'],
                'roc_auc': results['roc_auc'],
                'directional_accuracy': results.get('directional_accuracy'),
                'win_rate': results.get('win_rate'),
                'total_return': results.get('total_return'),
                'num_trades': results.get('num_trades'),
                'trade_coverage': results.get('trade_coverage'),
                'best_threshold_up': results.get('best_threshold_up'),
                'best_threshold_down': results.get('best_threshold_down'),
            }
        }
        
        # Save report as JSON
        import json
        report_path = os.path.join(save_dir, f"{ticker}_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved: {report_path}")
        
        return report
    
    @staticmethod
    def plot_all_diagnostics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: np.ndarray,
        save_dir: str = "diagnostics"
    ):
        """Generate all diagnostic plots."""
        os.makedirs(save_dir, exist_ok=True)
        
        ProductionEvaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_path=os.path.join(save_dir, "01_confusion_matrix.png")
        )
        
        ProductionEvaluator.plot_roc_curve(
            y_true, confidence,
            save_path=os.path.join(save_dir, "02_roc_curve.png")
        )
        
        ProductionEvaluator.plot_precision_recall_curve(
            y_true, confidence,
            save_path=os.path.join(save_dir, "03_pr_curve.png")
        )
        
        ProductionEvaluator.plot_confidence_distribution(
            confidence, y_true,
            save_path=os.path.join(save_dir, "04_confidence_dist.png")
        )
        
        ProductionEvaluator.plot_calibration_curve(
            y_true, confidence,
            save_path=os.path.join(save_dir, "05_calibration.png")
        )
        
        logger.info(f"All diagnostics saved to: {save_dir}")


class TradingMetricsCalculator:
    """Calculate trading-specific metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio."""
        annual_return = returns.mean() * 252
        annual_std = returns.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def calculate_win_rate(signals: np.ndarray, actual_returns: np.ndarray) -> float:
        """Calculate win rate of signals."""
        signal_returns = signals * actual_returns
        return (signal_returns > 0).sum() / len(signal_returns)
    
    @staticmethod
    def backtest_signals(
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        confidence_threshold: float = 0.6,
        confidence_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Backtest trading signals."""
        # Filter by confidence
        if confidence_scores is not None:
            valid_mask = confidence_scores >= confidence_threshold
            predictions_filtered = predictions[valid_mask]
            returns_filtered = actual_returns[valid_mask]
        else:
            predictions_filtered = predictions
            returns_filtered = actual_returns
        
        # Convert predictions to signals (-1, 1)
        signals = np.where(predictions_filtered == 1, 1, -1)
        signal_returns = signals * returns_filtered
        
        # Calculate metrics
        total_return = np.prod(1 + signal_returns) - 1
        annual_return = (1 + total_return) ** (252 / len(signal_returns)) - 1
        winning_trades = (signal_returns > 0).sum()
        losing_trades = (signal_returns < 0).sum()
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'signal_coverage': len(predictions_filtered) / len(predictions)
        }


if __name__ == "__main__":
    logger.info("Evaluation module loaded successfully")
