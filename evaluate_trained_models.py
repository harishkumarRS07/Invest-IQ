#!/usr/bin/env python
"""Comprehensive evaluation of trained XGBoost models"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator

# Standard features matching training
STANDARD_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Log_Return', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'SMA_20', 'SMA_50', 'BB_High', 'BB_Low', 'ATR', 'VWAP',
    'Volume_Change', 'Rolling_Volatility', 'Market_Correlation'
]

def load_and_prepare_data(ticker_csv):
    """Load and prepare data with all features"""
    df = load_data(ticker_csv)
    df = clean_data(df)
    
    if len(df) < 100:
        return None
    
    # Add all features
    market_df = ExternalDataSimulator.fetch_market_index()
    df = add_technical_indicators(df)
    df = add_market_correlation(df, market_df)
    
    return df

def create_adaptive_labels(df, forecast_horizon=3, min_movement=0.001, smoothing_window=3):
    """Create binary UP/DOWN labels with adaptive thresholds"""
    future_close = df['Close'].shift(-forecast_horizon)
    future_returns = (future_close - df['Close']) / df['Close']
    future_returns_smooth = future_returns.rolling(window=smoothing_window, center=True).mean()
    
    # Adaptive thresholds
    returns_clean = future_returns.dropna()
    buy_threshold = returns_clean.quantile(0.65)
    sell_threshold = returns_clean.quantile(0.35)
    
    # Create labels (0=SELL, 1=HOLD, 2=BUY)
    labels = np.ones(len(df), dtype=int)
    labels[future_returns > buy_threshold] = 2  # BUY
    labels[future_returns < sell_threshold] = 0  # SELL
    labels = labels[:-forecast_horizon]
    
    # Remove NA
    X = df[STANDARD_FEATURES].iloc[:-forecast_horizon].copy()
    y = labels.copy()
    
    valid_idx = ~(X.isna().any(axis=1))
    X = X[valid_idx].values
    y = y[valid_idx]
    
    return X, y, sell_threshold, buy_threshold

def evaluate_model(ticker, model_path, X_test, y_test):
    """Evaluate a single model"""
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"⚠️  Failed to load model: {e}")
        return None
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class metrics
    class_names = ['SELL', 'HOLD', 'BUY']
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            class_accuracy = accuracy_score(y_test[mask], y_pred[mask])
            class_metrics[class_name] = {
                'samples': mask.sum(),
                'accuracy': class_accuracy,
                'cm_row': cm[i]
            }
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_proba,
        'y_true': y_test,
        'class_metrics': class_metrics,
        'total_samples': len(y_test)
    }

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Target stocks
    stocks = ['HDFCBANK', 'ICICIBANK', 'INFY', 'RELIANCE', 'TCS']
    
    all_results = {}
    
    for ticker in stocks:
        print(f"\n{'='*80}")
        print(f"Evaluating {ticker}")
        print(f"{'='*80}\n")
        
        # Load data
        csv_file = f"{settings.DATA_DIR}/{ticker}.csv"
        if not os.path.exists(csv_file):
            print(f"⚠️  Data file not found: {csv_file}")
            continue
        
        df = load_and_prepare_data(csv_file)
        if df is None:
            print(f"⚠️  Failed to load data for {ticker}")
            continue
        
        # Prepare labels
        X, y, sell_thresh, buy_thresh = create_adaptive_labels(df)
        
        # Train/test split (20% test)
        n = len(X)
        test_idx = int(n * 0.8)
        X_test = X[test_idx:]
        y_test = y[test_idx:]
        
        print(f"Test set: {len(X_test)} samples")
        print(f"  SELL: {(y_test == 0).sum()}")
        print(f"  HOLD: {(y_test == 1).sum()}")
        print(f"  BUY:  {(y_test == 2).sum()}\n")
        
        # Load and evaluate model
        model_path = f"backend/models/saved_models/xgboost_classifier_{ticker}.pkl"
        results = evaluate_model(ticker, model_path, X_test, y_test)
        
        if results is None:
            continue
        
        all_results[ticker] = results
        
        # Print metrics
        print(f"📊 METRICS:")
        print(f"  Accuracy:             {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Precision (macro):    {results['precision_macro']:.4f}")
        print(f"  Precision (weighted): {results['precision_weighted']:.4f}")
        print(f"  Recall (macro):       {results['recall_macro']:.4f}")
        print(f"  Recall (weighted):    {results['recall_weighted']:.4f}")
        print(f"  F1 Score (macro):     {results['f1_macro']:.4f}")
        print(f"  F1 Score (weighted):  {results['f1_weighted']:.4f}")
        
        print(f"\n📈 CONFUSION MATRIX:")
        print(f"     Predicted")
        print(f"     SELL  HOLD  BUY")
        for i, label in enumerate(['SELL', 'HOLD', 'BUY']):
            cm_row = results['confusion_matrix'][i]
            print(f"{label}: {cm_row[0]:4d} {cm_row[1]:4d} {cm_row[2]:4d}")
        
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        for class_name, metrics in results['class_metrics'].items():
            print(f"  {class_name:6} - Samples: {metrics['samples']:4d}, Accuracy: {metrics['accuracy']:.4f}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    summary_df = pd.DataFrame({
        ticker: {
            'Accuracy': results['accuracy'],
            'Precision': results['precision_weighted'],
            'Recall': results['recall_weighted'],
            'F1 Score': results['f1_weighted'],
            'Total Samples': results['total_samples']
        }
        for ticker, results in all_results.items()
    }).T
    
    print(f"\n{summary_df.to_string()}")
    
    print(f"\nAVERAGES:")
    print(f"  Accuracy:  {summary_df['Accuracy'].mean():.4f}")
    print(f"  Precision: {summary_df['Precision'].mean():.4f}")
    print(f"  Recall:    {summary_df['Recall'].mean():.4f}")
    print(f"  F1 Score:  {summary_df['F1 Score'].mean():.4f}")
    
    # Create visualization
    print("\n" + "="*80)
    print("Creating visualization...")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Evaluation across All Stocks', fontsize=16, fontweight='bold')
    
    tickers = list(all_results.keys())
    accuracies = [all_results[t]['accuracy'] for t in tickers]
    precisions = [all_results[t]['precision_weighted'] for t in tickers]
    recalls = [all_results[t]['recall_weighted'] for t in tickers]
    f1_scores = [all_results[t]['f1_weighted'] for t in tickers]
    
    # Accuracy
    ax = axes[0, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(tickers, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Model Accuracy by Stock')
    ax.set_ylim(0, 0.5)
    ax.axhline(y=0.33, color='red', linestyle='--', linewidth=2, label='Random Baseline (33%)')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # All metrics
    ax = axes[0, 1]
    x = np.arange(len(tickers))
    width = 0.2
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#1f77b4', alpha=0.8, edgecolor='black')
    ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#ff7f0e', alpha=0.8, edgecolor='black')
    ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#2ca02c', alpha=0.8, edgecolor='black')
    ax.bar(x + 1.5*width, f1_scores, width, label='F1', color='#d62728', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('All Metrics by Stock')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.set_ylim(0, 0.4)
    ax.grid(axis='y', alpha=0.3)
    
    # Averages
    ax = axes[1, 0]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    avg_values = [
        np.mean(accuracies),
        np.mean(precisions),
        np.mean(recalls),
        np.mean(f1_scores)
    ]
    bars = ax.bar(metrics_names, avg_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                  alpha=0.8, edgecolor='black')
    ax.set_ylabel('Average Score', fontweight='bold')
    ax.set_title('Average Performance Across All Stocks')
    ax.set_ylim(0, 0.4)
    ax.axhline(y=0.33, color='red', linestyle='--', linewidth=2, label='Random Baseline')
    for bar, val in zip(bars, avg_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Trend
    ax = axes[1, 1]
    ax.plot(tickers, accuracies, marker='o', markersize=10, linewidth=2.5, label='Accuracy', color='#1f77b4')
    ax.plot(tickers, precisions, marker='s', markersize=10, linewidth=2.5, label='Precision', color='#ff7f0e')
    ax.plot(tickers, recalls, marker='^', markersize=10, linewidth=2.5, label='Recall', color='#2ca02c')
    ax.plot(tickers, f1_scores, marker='d', markersize=10, linewidth=2.5, label='F1 Score', color='#d62728')
    ax.axhline(y=0.33, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random Baseline')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Trend Across Stocks')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.15, 0.45)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_evaluation_comprehensive.png")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS & INSIGHTS")
    print("="*80)
    
    print(f"""
📊 KEY FINDINGS:

1. OVERALL PERFORMANCE:
   • Average Accuracy: {np.mean(accuracies):.2%}
   • Random Baseline: 33.33% (3-class problem)
   • Performance vs Baseline: {((np.mean(accuracies) - 0.33) / 0.33 * 100):+.1f}%

2. BEST PERFORMER: {tickers[np.argmax(accuracies)]} ({max(accuracies):.2%})
3. NEEDS WORK: {tickers[np.argmin(accuracies)]} ({min(accuracies):.2%})

4. METRIC ANALYSIS:
   • Precision vs Recall: {"Balanced" if abs(np.mean(precisions) - np.mean(recalls)) < 0.05 else "Imbalanced"}
   • F1 Score: {np.mean(f1_scores):.2%} (harmonic mean of precision & recall)

5. WHAT THIS MEANS:
   ⚠️  Models are performing slightly BELOW random guessing
   📈 Models need significant improvement through:
       - Feature engineering enhancements
       - Class balancing techniques
       - Hyperparameter optimization
       - Ensemble methods
       - Deep learning approaches (LSTM)

6. RECOMMENDATIONS:
   ✓ Use improved_hybrid_model.py with advanced features
   ✓ Switch to binary classification (UP/DOWN only)
   ✓ Apply LSTM + XGBoost ensemble
   ✓ Implement walk-forward validation
   ✓ Use confidence thresholding for trading
""")
    
    print("="*80)

if __name__ == "__main__":
    main()
