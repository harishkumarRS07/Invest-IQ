#!/usr/bin/env python
"""
XGBoost Classification Demo and Testing Script

Demonstrates:
1. Training XGBoost classifier
2. Making predictions
3. Generating trading signals
4. Feature importance analysis
5. Output formatting
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.logging import logger
from backend.training.xgboost_classifier import train_xgboost_classifier


def demo_xgboost_predictions():
    """
    Demo: Train model and show example predictions with confidence scores.
    """
    logger.info("\n" + "="*80)
    logger.info("XGBOOST CLASSIFICATION - DEMO")
    logger.info("="*80 + "\n")
    
    # Select a stock to demo
    ticker = "RELIANCE"
    file_path = f"backend/data/stock_data/{ticker}.csv"
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        logger.info("Available stocks in backend/data/stock_data/:")
        import glob
        for f in glob.glob("backend/data/stock_data/*.csv"):
            logger.info(f"  - {os.path.basename(f)}")
        return
    
    # Train model
    logger.info(f"Training XGBoost classifier for {ticker}...\n")
    results = train_xgboost_classifier(
        ticker=ticker,
        file_path=file_path,
        buy_threshold=0.002,    # 0.2% - UPDATED FROM 0.005
        sell_threshold=-0.002   # -0.2% - UPDATED FROM -0.005
    )
    
    if not results:
        logger.error("Training failed!")
        return
    
    # Extract results
    signals = results['signals']
    X_test = results['X_test']
    y_test = results['y_test']
    metrics = results['metrics']
    pipeline = results['pipeline']
    
    # Example predictions output
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE PREDICTIONS WITH CONFIDENCE SCORES")
    logger.info("="*80 + "\n")
    
    # Show last 20 predictions
    logger.info("Last 20 Predictions:\n")
    logger.info(f"{'Index':<6} {'Signal':<8} {'Confidence':<12} {'SELL':<8} {'HOLD':<8} {'BUY':<8}")
    logger.info("-" * 80)
    
    for idx in range(max(0, len(signals) - 20), len(signals)):
        sig = signals.iloc[idx]
        logger.info(
            f"{idx:<6} {sig['Signal']:<8} {sig['Confidence']:<12.4f} "
            f"{sig['Prob_SELL']:<8.4f} {sig['Prob_HOLD']:<8.4f} {sig['Prob_BUY']:<8.4f}"
        )
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80 + "\n")
    
    logger.info("Signal Distribution:")
    signal_counts = signals['Signal'].value_counts()
    for signal in ['BUY', 'HOLD', 'SELL']:
        count = signal_counts.get(signal, 0)
        pct = 100.0 * count / len(signals)
        logger.info(f"  {signal}: {count} ({pct:.1f}%)")
    
    logger.info(f"\nConfidence Statistics:")
    logger.info(f"  Mean Confidence: {signals['Confidence'].mean():.4f}")
    logger.info(f"  Std Confidence:  {signals['Confidence'].std():.4f}")
    logger.info(f"  Min Confidence:  {signals['Confidence'].min():.4f}")
    logger.info(f"  Max Confidence:  {signals['Confidence'].max():.4f}")
    
    # Model Performance
    logger.info("\n" + "="*80)
    logger.info("MODEL PERFORMANCE")
    logger.info("="*80 + "\n")
    
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    
    # Confusion matrix analysis
    logger.info("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    logger.info(f"  Predicted SELL: True={cm[0,0]}, False={cm[0,1]+cm[0,2]}")
    logger.info(f"  Predicted HOLD: True={cm[1,1]}, False={cm[1,0]+cm[1,2]}")
    logger.info(f"  Predicted BUY:  True={cm[2,2]}, False={cm[2,0]+cm[2,1]}")
    
    # Feature Importance
    logger.info("\n" + "="*80)
    logger.info("TOP 10 MOST IMPORTANT FEATURES")
    logger.info("="*80 + "\n")
    
    # Get feature importances
    importances = pipeline.model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': pipeline.feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"  {row['Feature']:<30} {row['Importance']:.6f}")
    
    # Trading Signal Example
    logger.info("\n" + "="*80)
    logger.info("TRADING SIGNAL EXAMPLE")
    logger.info("="*80 + "\n")
    
    # Get the most recent prediction
    latest_signal = signals.iloc[-1]
    
    example_output = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "ticker": ticker,
        "signal": latest_signal['Signal'],
        "confidence": float(latest_signal['Confidence']),
        "probabilities": {
            "sell": float(latest_signal['Prob_SELL']),
            "hold": float(latest_signal['Prob_HOLD']),
            "buy": float(latest_signal['Prob_BUY'])
        },
        "recommendation": f"SIGNAL: {latest_signal['Signal']} (Confidence: {latest_signal['Confidence']:.2%})"
    }
    
    logger.info("\nJSON Output Format:")
    import json
    logger.info(json.dumps(example_output, indent=2))
    
    # Action recommendations
    logger.info("\n" + "="*80)
    logger.info("ACTION RECOMMENDATIONS")
    logger.info("="*80 + "\n")
    
    if latest_signal['Confidence'] > 0.7:
        confidence_level = "HIGH"
    elif latest_signal['Confidence'] > 0.55:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"
    
    logger.info(f"Latest Signal: {latest_signal['Signal']} with {confidence_level} confidence\n")
    
    if latest_signal['Signal'] == 'BUY':
        logger.info("Recommendations:")
        logger.info("  - BUY signal detected")
        logger.info(f"  - Confidence: {latest_signal['Confidence']:.2%}")
        logger.info(f"  - Expected upward movement > 0.5%")
        if confidence_level == "HIGH":
            logger.info("  - ACTION: Consider buying (high confidence)")
        else:
            logger.info("  - ACTION: Consider buying with caution (lower confidence)")
    
    elif latest_signal['Signal'] == 'SELL':
        logger.info("Recommendations:")
        logger.info("  - SELL signal detected")
        logger.info(f"  - Confidence: {latest_signal['Confidence']:.2%}")
        logger.info(f"  - Expected downward movement < -0.5%")
        if confidence_level == "HIGH":
            logger.info("  - ACTION: Consider selling (high confidence)")
        else:
            logger.info("  - ACTION: Consider selling with caution (lower confidence)")
    
    else:  # HOLD
        logger.info("Recommendations:")
        logger.info("  - HOLD signal detected")
        logger.info(f"  - Confidence: {latest_signal['Confidence']:.2%}")
        logger.info(f"  - Expected movement between -0.5% and +0.5%")
        logger.info("  - ACTION: Hold current position or await better signal")
    
    # Model insights
    logger.info("\n" + "="*80)
    logger.info("MODEL INSIGHTS")
    logger.info("="*80 + "\n")
    
    logger.info("Feature Engineering Summary:")
    logger.info(f"  - Total features used: {len(pipeline.feature_names)}")
    logger.info(f"  - Feature categories:")
    logger.info(f"    * Technical Indicators: RSI, MACD, Bollinger Bands, ATR, VWAP, SMA")
    logger.info(f"    * Momentum Features: 3-day, 5-day, 7-day returns and momentum")
    logger.info(f"    * Volume Features: Volume changes, MA ratios, price-volume trend")
    logger.info(f"    * Trend Features: SMA differences, price positions vs SMAs")
    logger.info(f"    * Volatility Features: Historical volatility, Bollinger position, High-Low range")
    
    logger.info("\nModel Configuration:")
    logger.info(f"  - Algorithm: XGBoost (Multi-class Classification)")
    logger.info(f"  - Trees: 200")
    logger.info(f"  - Max Depth: 5")
    logger.info(f"  - Learning Rate: 0.05")
    logger.info(f"  - Class Labels: SELL (0), HOLD (1), BUY (2)")
    logger.info(f"  - BUY Threshold: > +0.5% return")
    logger.info(f"  - SELL Threshold: < -0.5% return")
    
    logger.info("\nTraining Strategy:")
    logger.info(f"  - Split: 80% train / 20% test (time-based, no shuffle)")
    logger.info(f"  - Forecast Horizon: 3 days")
    logger.info(f"  - Early Stopping: 20 rounds with no improvement")
    logger.info(f"  - Class Weights: Balanced")
    
    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE")
    logger.info("="*80 + "\n")


def compare_with_baseline():
    """
    Compare XGBoost predictions with simple baseline (always HOLD).
    """
    logger.info("\n" + "="*80)
    logger.info("BASELINE COMPARISON")
    logger.info("="*80 + "\n")
    
    logger.info("Baseline Model: Always predict HOLD\n")
    logger.info("Expected Results:")
    logger.info("  If HOLD class is 50-60% of data:")
    logger.info("    - Baseline Accuracy: 50-60%")
    logger.info("    - Baseline F1: Low (no other classes predicted)\n")
    
    logger.info("XGBoost Model should:")
    logger.info("  - Achieve higher accuracy (60-70%+)")
    logger.info("  - Better identify BUY and SELL opportunities")
    logger.info("  - Provide actionable trading signals")


if __name__ == "__main__":
    # Run demo
    demo_xgboost_predictions()
    
    # Show baseline comparison
    compare_with_baseline()
