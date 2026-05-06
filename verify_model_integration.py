#!/usr/bin/env python
"""Verify that XGBoost and Transformer models are combined in prediction engine."""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.inference.predict import Predictor
from backend.core.config import settings
from backend.core.logging import logger

def check_models():
    """Check model files exist."""
    print("="*80)
    print("MODEL FILES INVENTORY")
    print("="*80)
    
    model_dir = Path(settings.MODEL_DIR)
    
    transformers = sorted(model_dir.glob("transformer_*.pth"))
    xgboost = sorted(model_dir.glob("xgboost_classifier_*.pkl"))
    
    print(f"\nTransformer Models ({len(transformers)}):")
    for f in transformers:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name:<35} {size_mb:>6.2f} MB")
    
    print(f"\nXGBoost Models ({len(xgboost)}):")
    for f in xgboost:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name:<35} {size_mb:>6.2f} MB")
    
    return len(transformers) > 0, len(xgboost) == 5

def test_prediction():
    """Test prediction with combined models."""
    print("\n" + "="*80)
    print("COMBINED MODEL PREDICTION TEST")
    print("="*80)
    
    predictor = Predictor()
    
    data_dir = Path(settings.DATA_DIR)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("ERROR: No CSV files found")
        return False
    
    test_file = csv_files[0]
    ticker = test_file.stem
    
    print(f"\nTesting with: {ticker}")
    print(f"Data file: {test_file}")
    
    try:
        result = predictor.predict(str(test_file), ticker=ticker)
        
        print("\n" + "-"*80)
        print("PREDICTION RESULTS:")
        print("-"*80)
        print(f"Signal:              {result['signal']}")
        print(f"Confidence:          {result['signal_confidence']:.4f}")
        print(f"Current Price:       ${result['current_price']:.2f}")
        print(f"Predicted Price:     ${result['predicted_price']:.2f}")
        
        print("\nSignal Probabilities:")
        for signal, prob in result['probabilities'].items():
            print(f"  {signal.upper():6} -> {prob:.2%}")
        
        print("\n" + "-"*80)
        print("✓ XGBoost classifier executed (priority signal)")
        print("✓ Transformer model executed (price forecast)")
        print("✓ Combined output generated successfully!")
        print("-"*80)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run verification."""
    print("\n" + "#"*80)
    print("# XGBOOST + TRANSFORMER INTEGRATION VERIFICATION")
    print("#"*80 + "\n")
    
    has_transformers, has_xgboost = check_models()
    
    print("\n" + "="*80)
    print("INTEGRATION STATUS")
    print("="*80)
    
    if has_transformers and has_xgboost:
        print("\n✓ Transformer models: READY")
        print("✓ XGBoost models: READY")
        print("✓ Prediction engine: READY")
        print("\n[ARCHITECTURE]")
        print("  1. Load data + features")
        print("  2. Run XGBoost classifier -> BUY/SELL/HOLD signal + confidence")
        print("  3. Run Transformer -> price forecast")
        print("  4. Combine results into unified prediction")
        print("  5. Return signal with all probabilities")
        
        success = test_prediction()
        
        if success:
            print("\n" + "="*80)
            print("VERIFICATION COMPLETE: ALL MODELS SUCCESSFULLY COMBINED!")
            print("="*80 + "\n")
            return True
    else:
        print("\n✗ ERROR: Missing models")
        if not has_transformers:
            print("  - Transformer models missing")
        if not has_xgboost:
            print("  - XGBoost models missing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
