#!/usr/bin/env python
"""
Diagnostic: Check XGBoost Label Fix Status
Shows exactly what's wrong
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from backend.core.config import settings
from backend.preprocessing.cleaning import load_data, clean_data

print("\n" + "="*80)
print("DIAGNOSTIC: XGBoost Label Imbalance Fix Status")
print("="*80 + "\n")

# 1. Check if models exist
print("[1] CHECKING MODEL FILES")
print("-" * 80)
model_dir = settings.MODEL_DIR
pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
print(f"Models directory: {model_dir}")
print(f"XGBoost models found: {len(pkl_files)}")
for f in sorted(pkl_files):
    print(f"  - {os.path.basename(f)}")

if len(pkl_files) == 0:
    print("\n  ⚠ WARNING: NO XGBOOST MODELS FOUND!")
    print("  The fix_label_imbalance.py script did not train models properly.")
else:
    print(f"\n  ✓ Found {len(pkl_files)} models")

# 2. Check label distribution in DATA
print("\n[2] CHECKING LABEL DISTRIBUTION IN RAW DATA")
print("-" * 80)

data_files = sorted(glob.glob(os.path.join(settings.DATA_DIR, "*.csv")))

for csv_file in data_files[:1]:  # Check first stock only
    ticker = os.path.basename(csv_file).replace(".csv", "")
    print(f"\nTicker: {ticker}")
    
    try:
        df = load_data(csv_file)
        df = clean_data(df)
        
        if len(df) < 10:
            print(f"  ✗ Insufficient data ({len(df)} rows)")
            continue
        
        # Calculate future returns with BOTH thresholds
        horizon = 3
        future_close = df['Close'].shift(-horizon)
        future_returns = (future_close - df['Close']) / df['Close']
        
        # OLD threshold (0.005)
        print(f"\n  OLD THRESHOLD (0.005 = 0.5%):")
        labels_old = np.ones(len(df), dtype=int)
        labels_old[future_returns > 0.005] = 2
        labels_old[future_returns < -0.005] = 0
        labels_old = labels_old[:-horizon]
        
        unique_old, counts_old = np.unique(labels_old, return_counts=True)
        total = len(labels_old)
        for label, count in zip(unique_old, counts_old):
            pct = 100.0 * count / total
            signal = ['SELL', 'HOLD', 'BUY'][int(label)]
            print(f"    {signal:6s}: {count:5d} ({pct:6.2f}%)")
        
        # NEW threshold (0.002)
        print(f"\n  NEW THRESHOLD (0.002 = 0.2%):")
        labels_new = np.ones(len(df), dtype=int)
        labels_new[future_returns > 0.002] = 2
        labels_new[future_returns < -0.002] = 0
        labels_new = labels_new[:-horizon]
        
        unique_new, counts_new = np.unique(labels_new, return_counts=True)
        for label, count in zip(unique_new, counts_new):
            pct = 100.0 * count / total
            signal = ['SELL', 'HOLD', 'BUY'][int(label)]
            print(f"    {signal:6s}: {count:5d} ({pct:6.2f}%)")
        
        hold_pct_new = 100.0 * np.sum(labels_new == 1) / len(labels_new)
        if hold_pct_new > 95:
            print(f"\n  ⚠ WARNING: HOLD still > 95% with 0.002 threshold!")
            print(f"     This means data itself has very small daily returns (normal for stable stocks)")
            print(f"     Need to use EVEN LOWER threshold (e.g., 0.0005)")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

# 3. Check what predictions look like
print("\n[3] CHECKING ACTUAL PREDICTIONS")
print("-" * 80)

try:
    from backend.inference.predict import Predictor
    
    predictor = Predictor()
    
    for csv_file in data_files[:1]:  # Check first stock
        ticker = os.path.basename(csv_file).replace(".csv", "")
        print(f"\nTicker: {ticker}")
        
        try:
            result = predictor.predict(csv_file, ticker=ticker)
            
            print(f"  Signal:     {result.get('signal', 'N/A')}")
            print(f"  Confidence: {result.get('signal_confidence', 0):.4f}")
            print(f"  Current:    {result.get('current_price', 0):.2f}")
            print(f"  Predicted:  {result.get('predicted_price', 0):.2f}")
            
            pct = ((result.get('predicted_price', 0) - result.get('current_price', 1)) / result.get('current_price', 1)) * 100
            print(f"  Expected Return: {pct:.2f}%")
            
            # Check probabilities
            probs = result.get('probabilities', {})
            print(f"  Probabilities: BUY={probs.get('buy', 0):.4f}, SELL={probs.get('sell', 0):.4f}, HOLD={probs.get('hold', 0):.4f}")
            
        except Exception as e:
            print(f"  Error predicting: {e}")
            import traceback
            traceback.print_exc()
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# 4. Recommendations
print("\n[4] DIAGNOSIS & RECOMMENDATIONS")
print("-" * 80)

if len(pkl_files) == 0:
    print("\n❌ PROBLEM: No XGBoost models trained")
    print("   SOLUTION: Run: python fix_label_imbalance.py")
    print("   Make sure it completes without errors")
else:
    print("\n✓ XGBoost models exist")
    print("  Checking if they produce diverse signals...")
    
    # This will be shown in section [3] above
    
print("\n" + "="*80 + "\n")
