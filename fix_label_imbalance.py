#!/usr/bin/env python
"""
XGBoost Label Imbalance Fix - Complete Training Script

OBJECTIVE:
Fix label imbalance by:
1. Reducing threshold from 0.005 to 0.002 (DONE in xgboost_fusion.py)
2. Displaying class distribution after label creation
3. Retraining all XGBoost models with new labels
4. Verifying diverse BUY/SELL/HOLD signals

EXPECTED OUTPUT:
- Class distribution showing BUY, SELL, HOLD all present
- No 100% HOLD labels (previous problem)
- Mixed signals in predictions
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators
from backend.models.xgboost_fusion import XGBoostFusionModel
from backend.training.xgboost_classifier import (
    XGBoostClassificationPipeline,
    train_xgboost_classifier
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def verify_label_distribution(ticker: str, file_path: str, 
                              buy_threshold: float = 0.002,
                              sell_threshold: float = -0.002,
                              horizon: int = 3) -> Dict:
    """
    TASK 2: Check class distribution immediately after label creation.
    
    Expected output:
    {0: 120, 1: 300, 2: 150}  # BUY exists, SELL exists, HOLD not dominant
    
    NOT:
    {1: 500}  # All HOLD (broken labels)
    """
    print(f"\n{'='*80}")
    print(f"[TASK 2] VERIFYING LABEL DISTRIBUTION FOR {ticker}")
    print(f"{'='*80}")
    
    try:
        # Load and clean data
        df = load_data(file_path)
        df = clean_data(df)
        
        if len(df) < horizon + 10:
            print(f"  ✗ Insufficient data ({len(df)} rows)")
            return {}
        
        # Create labels using CORRECTED threshold (0.002)
        print(f"\n  Using thresholds:")
        print(f"    BUY threshold:  > {buy_threshold:.4f} ({buy_threshold*100:.2f}%)")
        print(f"    SELL threshold: < {sell_threshold:.4f} ({sell_threshold*100:.2f}%)")
        print(f"    HOLD threshold: in between")
        
        # Calculate future returns
        future_close = df['Close'].shift(-horizon)
        future_returns = (future_close - df['Close']) / df['Close']
        
        # Create labels
        labels = np.ones(len(df), dtype=int)  # Default HOLD (1)
        labels[future_returns > buy_threshold] = 2  # BUY
        labels[future_returns < sell_threshold] = 0  # SELL
        
        # Remove last horizon rows
        labels = labels[:-horizon]
        
        # Get class distribution
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        # Print results
        print(f"\n  CLASS DISTRIBUTION:")
        print(f"  {'Signal':<10} {'Count':<10} {'Percentage':<12} {'Label #':<10}")
        print(f"  {'-'*42}")
        
        total = len(labels)
        signal_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
        
        for label_num in [0, 1, 2]:
            count = class_dist.get(label_num, 0)
            pct = 100.0 * count / total if total > 0 else 0
            signal = signal_names[label_num]
            print(f"  {signal:<10} {count:<10} {pct:>10.2f}%  {label_num:<10}")
        
        # Validation
        print(f"\n  VALIDATION:")
        has_buy = class_dist.get(2, 0) > 0
        has_sell = class_dist.get(0, 0) > 0
        has_hold = class_dist.get(1, 0) > 0
        hold_pct = 100.0 * class_dist.get(1, 0) / total if total > 0 else 0
        
        print(f"    Has BUY signals:  {'✓' if has_buy else '✗'}")
        print(f"    Has SELL signals: {'✓' if has_sell else '✗'}")
        print(f"    Has HOLD signals: {'✓' if has_hold else '✗'}")
        print(f"    HOLD not dominant (< 80%): {'✓' if hold_pct < 80 else '✗'}")
        
        all_good = has_buy and has_sell and has_hold and (hold_pct < 80)
        print(f"\n  STATUS: {'OK - Labels are balanced!' if all_good else 'FAILED - Labels still imbalanced'}")
        
        return class_dist
        
    except Exception as e:
        print(f"  ✗ Error checking labels: {e}")
        import traceback
        traceback.print_exc()
        return {}


def train_xgboost_for_ticker(ticker: str, file_path: str,
                             buy_threshold: float = 0.002,
                             sell_threshold: float = -0.002) -> Optional[Dict]:
    """
    TASK 3 & 4: Retrain XGBoost with new labels and improved hyperparameters.
    """
    print(f"\n{'#'*80}")
    print(f"[TASK 3 & 4] TRAINING XGBOOST FOR {ticker}")
    print(f"{'#'*80}\n")
    
    start_time = time.time()
    
    try:
        # Use the training function that handles everything
        results = train_xgboost_classifier(
            ticker=ticker,
            file_path=file_path,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        
        elapsed = time.time() - start_time
        
        if results:
            metrics = results.get('metrics', {})
            print(f"\n  TRAINING RESULTS FOR {ticker}:")
            print(f"    Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"    Precision: {metrics.get('precision', 0):.4f}")
            print(f"    Recall:    {metrics.get('recall', 0):.4f}")
            print(f"    F1 Score:  {metrics.get('f1', 0):.4f}")
            print(f"    Time:      {elapsed:.1f}s")
            print(f"\n  ✓ Training Complete!")
            
            return results
        else:
            print(f"  ✗ Training failed - no results")
            return None
            
    except Exception as e:
        print(f"  ✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_predictions(ticker: str) -> Optional[Dict]:
    """
    TASK 5: Verify signal output after training.
    
    Expected:
    - BUY signals appear (not all HOLD)
    - SELL signals appear
    - Confidence scores vary (not all 0.5)
    """
    print(f"\n{'*'*80}")
    print(f"[TASK 5] VERIFYING PREDICTIONS FOR {ticker}")
    print(f"{'*'*80}\n")
    
    try:
        from backend.inference.predict import Predictor
        
        # Create predictor
        predictor = Predictor()
        
        # Get recent data
        data_file = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(data_file):
            print(f"  ✗ Data file not found: {data_file}")
            return None
        
        # Make predictions on test data
        result = predictor.predict(data_file, ticker=ticker)
        
        print(f"  PREDICTION RESULT:")
        print(f"    Signal:     {result.get('signal', 'N/A')}")
        print(f"    Confidence: {result.get('signal_confidence', 0):.4f}")
        print(f"    Current:    {result.get('current_price', 0):.2f}")
        print(f"    Predicted:  {result.get('predicted_price', 0):.2f}")
        print(f"    Expected Return: {((result.get('predicted_price', 0) - result.get('current_price', 0)) / result.get('current_price', 1) * 100):.2f}%")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Execute complete label imbalance fix and retraining."""
    print("\n" + "="*80)
    print(" "*15 + "XGBOOST LABEL IMBALANCE FIX - COMPLETE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Configuration
    buy_threshold = 0.002    # 0.2%
    sell_threshold = -0.002  # -0.2%
    
    # Find all CSV files
    data_dir = settings.DATA_DIR
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    if not csv_files:
        print(f"\n✗ No CSV files found in {data_dir}")
        return False
    
    tickers = [os.path.basename(f).replace(".csv", "") for f in csv_files]
    print(f"\nTickers to train: {', '.join(tickers)}\n")
    
    # Results tracking
    results_summary = {
        'label_checks': {},
        'training_results': {},
        'predictions': {},
        'successful': [],
        'failed': []
    }
    
    # Process each ticker
    for idx, (ticker, csv_file) in enumerate(zip(tickers, csv_files), 1):
        print(f"\n{'#'*80}")
        print(f"[{idx}/{len(tickers)}] PROCESSING {ticker}")
        print(f"{'#'*80}\n")
        
        # TASK 2: Check class distribution
        label_dist = verify_label_distribution(
            ticker, csv_file, 
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        results_summary['label_checks'][ticker] = label_dist
        
        # TASK 3 & 4: Train XGBoost
        train_results = train_xgboost_for_ticker(
            ticker, csv_file,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        
        if train_results:
            results_summary['training_results'][ticker] = train_results
            results_summary['successful'].append(ticker)
            
            # TASK 5: Verify predictions
            pred_result = verify_predictions(ticker)
            results_summary['predictions'][ticker] = pred_result
        else:
            results_summary['failed'].append(ticker)
    
    # Final Summary
    print(f"\n\n{'='*80}")
    print(" "*20 + "FINAL SUMMARY")
    print("="*80)
    
    print(f"\nSuccessful: {len(results_summary['successful'])}/{len(tickers)}")
    for ticker in results_summary['successful']:
        print(f"  ✓ {ticker}")
    
    if results_summary['failed']:
        print(f"\nFailed: {len(results_summary['failed'])}/{len(tickers)}")
        for ticker in results_summary['failed']:
            print(f"  ✗ {ticker}")
    
    print(f"\n{'='*80}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Success indicator
    all_success = len(results_summary['successful']) == len(tickers)
    print(f"\nOVERALL STATUS: {'SUCCESS - All models retrained!' if all_success else 'PARTIAL - Some models failed'}")
    
    return all_success


if __name__ == "__main__":
    import traceback
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
