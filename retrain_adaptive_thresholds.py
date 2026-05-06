#!/usr/bin/env python
"""
Retrain XGBoost models with ADAPTIVE THRESHOLDS (percentile-based)

This fixes the class imbalance by using data-driven thresholds instead of fixed values.
"""

import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.training.xgboost_classifier import train_xgboost_classifier
from backend.core.logging import logger

def main():
    """Retrain all stock models with adaptive thresholds."""
    
    print("\n" + "="*80)
    print("RETRAINING XGBoost WITH ADAPTIVE PERCENTILE-BASED THRESHOLDS")
    print("="*80)
    
    # Find all CSV files
    data_dir = "backend/data/stock_data"
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        return
    
    results = []
    
    for csv_file in csv_files:
        ticker = os.path.basename(csv_file).replace(".csv", "")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training: {ticker}")
        logger.info(f"{'='*80}")
        
        try:
            # Train with adaptive thresholds (computed inside create_better_labels)
            result = train_xgboost_classifier(
                ticker=ticker,
                file_path=csv_file,
                buy_threshold=0.002,   # Not used - will use percentiles instead
                sell_threshold=-0.002  # Not used - will use percentiles instead
            )
            
            if result:
                results.append({
                    'ticker': ticker,
                    'status': 'SUCCESS',
                    'metrics': result.get('metrics', {})
                })
                
                # Show first predictions
                signals = result.get('signals')
                if signals is not None:
                    print(f"\n{ticker} - First 5 predictions:")
                    print(signals.head().to_string())
            else:
                results.append({'ticker': ticker, 'status': 'FAILED'})
                
        except Exception as e:
            logger.error(f"Error training {ticker}: {e}")
            results.append({'ticker': ticker, 'status': 'ERROR', 'error': str(e)})
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for r in results:
        status = r['status']
        ticker = r['ticker']
        print(f"{ticker:15s} : {status}")
        if 'metrics' in r and r['metrics']:
            acc = r['metrics'].get('accuracy', 0)
            f1 = r['metrics'].get('f1', 0)
            print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    print("="*80)
    print(f"\nCompleted: {len([r for r in results if r['status'] == 'SUCCESS'])}/{len(results)} successful")

if __name__ == "__main__":
    main()
