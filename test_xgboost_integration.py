
#!/usr/bin/env python
"""
Test script for XGBoost integration in the prediction engine.
This script will:
1. Train XGBoost models for sample stocks
2. Test the prediction engine with XGBoost signals
3. Verify signal generation and confidence scores
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_xgboost_training():
    """Test XGBoost model training"""
    print("\n" + "="*80)
    print("TEST 1: XGBoost Model Training")
    print("="*80)
    
    from batch_train_xgboost import batch_train_xgboost_classifiers
    from backend.core.config import settings
    from backend.core.logging import logger
    
    print(f"\nTraining XGBoost models from: {settings.DATA_DIR}")
    
    try:
        batch_train_xgboost_classifiers(
            data_dir=settings.DATA_DIR,
            buy_threshold=0.002,      # UPDATED FROM 0.005
            sell_threshold=-0.002,    # UPDATED FROM -0.005
            skip_tickers=[]
        )
        print("[OK] XGBoost training completed")
        
        # Check if models were created
        model_dir = settings.MODEL_DIR
        xgb_models = list(Path(model_dir).glob("xgboost_classifier_*.pkl"))
        print(f"[OK] Created {len(xgb_models)} XGBoost models")
        return True
        
    except Exception as e:
        print(f"[FAILED] XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_engine():
    """Test the prediction engine with XGBoost signals"""
    print("\n" + "="*80)
    print("TEST 2: Prediction Engine with XGBoost Signals")
    print("="*80)
    
    from backend.inference.predict import Predictor
    from backend.core.config import settings
    from backend.core.logging import logger
    from pathlib import Path
    import json
    
    predictor = Predictor()
    
    # Find a stock CSV file
    data_dir = settings.DATA_DIR
    csv_files = list(Path(data_dir).glob("*.csv"))
    
    if not csv_files:
        print("[FAILED] No CSV files found in data directory")
        return False
    
    # Test with the first stock
    test_file = csv_files[0]
    ticker = test_file.stem
    
    print(f"\nTesting with: {ticker}")
    print(f"Data file: {test_file}")
    
    try:
        result = predictor.predict(str(test_file), ticker=ticker)
        
        print(f"\n[OK] Prediction successful!")
        print(f"\nPrediction Results:")
        print(f"  - Current Price: ${result.get('current_price', 'N/A'):.2f}")
        print(f"  - Predicted Price (1-day): ${result.get('predicted_price', 'N/A'):.2f}")
        print(f"  - Signal: {result.get('signal', 'N/A')}")
        print(f"  - Signal Confidence: {result.get('signal_confidence', 'N/A')}")
        print(f"  - Probabilities: {result.get('probabilities', {})}")
        
        # Verify required fields
        required_fields = ['signal', 'signal_confidence', 'current_price', 'predicted_price', 'probabilities']
        missing_fields = [f for f in required_fields if f not in result]
        
        if missing_fields:
            print(f"\n[WARNING] Missing fields: {missing_fields}")
            return False
        
        # Verify signal values
        valid_signals = ['BUY', 'SELL', 'HOLD']
        if result['signal'] not in valid_signals:
            print(f"\n[FAILED] Invalid signal: {result['signal']}")
            return False
        
        # Verify confidence is between 0 and 1
        if not (0 <= result['signal_confidence'] <= 1):
            print(f"\n[FAILED] Invalid confidence: {result['signal_confidence']}")
            return False
        
        print(f"\n[OK] All validation checks passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Prediction engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xgboost_signal_generation():
    """Test XGBoost signal generation vs Transformer signal generation"""
    print("\n" + "="*80)
    print("TEST 3: XGBoost vs Transformer Signal Generation")
    print("="*80)
    
    from backend.inference.predict import Predictor
    from backend.core.config import settings
    from pathlib import Path
    
    predictor = Predictor()
    
    # Find a stock CSV file
    data_dir = settings.DATA_DIR
    csv_files = list(Path(data_dir).glob("*.csv"))
    
    if not csv_files:
        print("[FAILED] No CSV files found in data directory")
        return False
    
    test_file = csv_files[0]
    ticker = test_file.stem
    
    print(f"\nComparing signals for: {ticker}")
    
    try:
        result = predictor.predict(str(test_file), ticker=ticker)
        
        # Check if XGBoost model was used
        predictor_model_status = "XGBoost model was used" if predictor.xgboost_model else "Transformer fallback was used"
        
        print(f"\nModel Status: {predictor_model_status}")
        print(f"Generated Signal: {result.get('signal', 'N/A')}")
        print(f"Signal Confidence: {result.get('signal_confidence', 'N/A')}")
        
        print(f"\n[OK] Signal generation test completed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Signal generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "#"*80)
    print("# XGBoost Integration Tests")
    print("#"*80)
    
    tests = [
        ("XGBoost Training", test_xgboost_training),
        ("Prediction Engine", test_prediction_engine),
        ("Signal Generation", test_xgboost_signal_generation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*80 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
