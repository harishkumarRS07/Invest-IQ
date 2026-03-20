#!/usr/bin/env python3
"""
Enhanced Training Runner - Robust model training with detailed error handling and logging.
Fixes potential issues and provides comprehensive diagnostics.
"""

import sys
import os
import traceback
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Suppress warnings during training
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def setup_environment():
    """Initialize the training environment."""
    print("Configuring training environment...")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Check GPU availability
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()
            print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ℹ Running on CPU (GPU not available)")
        return device
    except ImportError:
        print("  ⚠ PyTorch not available")
        return None

def validate_csv_file(file_path: str) -> bool:
    """Validate CSV file before training."""
    try:
        df = pd.read_csv(file_path, nrows=5)
        if df.empty:
            print(f"    ✗ CSV is empty")
            return False
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"    ✗ Missing required columns: {missing}")
            return False
        
        print(f"    ✓ CSV structure valid ({len(df)} initial rows)")
        return True
    except Exception as e:
        print(f"    ✗ CSV validation failed: {e}")
        return False

def train_model_safe(file_path: str, ticker: str):
    """Train a single model with comprehensive error handling."""
    from backend.core.config import settings
    from backend.core.logging import logger
    from backend.training.train import train_pipeline
    
    try:
        print(f"    → Validating data...")
        if not validate_csv_file(file_path):
            return False
        
        print(f"    → Loading and preprocessing...")
        result = train_pipeline(file_path)
        
        # Verify model was saved
        checkpoint_path = os.path.join(settings.MODEL_DIR, f"transformer_{ticker}.pth")
        if os.path.exists(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"    ✓ Model saved: {file_size:.1f} MB")
            return True
        else:
            print(f"    ⚠ Model created but checkpoint not found at {checkpoint_path}")
            return False
            
    except ValueError as e:
        error_msg = str(e)
        if "not enough values to unpack" in error_msg:
            print(f"    ✗ Data format error - check CSV structure: {e}")
        elif "could not convert string to float" in error_msg:
            print(f"    ✗ Data type error - non-numeric values found: {e}")
        else:
            print(f"    ✗ Value error: {e}")
        return False
        
    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg:
            print(f"    ✗ GPU error: {e}")
        elif "size mismatch" in error_msg:
            print(f"    ✗ Model architecture error - data shape mismatch: {e}")
        else:
            print(f"    ✗ Runtime error: {e}")
        return False
        
    except Exception as e:
        error_type = type(e).__name__
        print(f"    ✗ {error_type}: {e}")
        
        # Log full traceback for debugging
        logger.error(f"Training failed for {ticker}", exc_info=True)
        return False

def train_all_models():
    """Execute training for all stocks."""
    from backend.core.config import settings
    from backend.core.logging import logger
    
    tickers = ["HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK"]
    
    print("\nTRAINING PIPELINE")
    print("=" * 70)
    
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        file_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        
        if not os.path.exists(file_path):
            print(f"  ✗ Data file not found: {file_path}")
            results['skipped'].append(ticker)
            continue
        
        if train_model_safe(file_path, ticker):
            results['success'].append(ticker)
        else:
            results['failed'].append(ticker)
    
    return results

def main():
    """Main execution."""
    print("\n" + "🤖 INVESTIQ MODEL TRAINING SYSTEM".center(70))
    print("=" * 70)
    
    try:
        # Step 1: Setup
        device = setup_environment()
        
        # Step 2: Verify prerequisites
        print("\nVERIFYING SETUP")
        print("-" * 70)
        
        from backend.core.config import settings
        if os.path.exists(settings.DATA_DIR):
            csv_count = len([f for f in os.listdir(settings.DATA_DIR) if f.endswith('.csv')])
            print(f"  ✓ Data directory exists ({csv_count} CSV files)")
        else:
            print(f"  ✗ Data directory not found: {settings.DATA_DIR}")
            return False
        
        if os.path.exists(settings.MODEL_DIR):
            print(f"  ✓ Model directory exists")
        else:
            os.makedirs(settings.MODEL_DIR, exist_ok=True)
            print(f"  ✓ Model directory created")
        
        # Step 3: Train models
        results = train_all_models()
        
        # Step 4: Print results
        print("\n" + "=" * 70)
        print("TRAINING RESULTS")
        print("=" * 70)
        
        if results['success']:
            print(f"\n✓ Successfully trained ({len(results['success'])} models):")
            for ticker in results['success']:
                print(f"   - {ticker}")
        
        if results['failed']:
            print(f"\n✗ Failed ({len(results['failed'])} models):")
            for ticker in results['failed']:
                print(f"   - {ticker}")
        
        if results['skipped']:
            print(f"\n⊘ Skipped ({len(results['skipped'])} models):")
            for ticker in results['skipped']:
                print(f"   - {ticker}")
        
        # Summary statistics
        total_trained = len(results['success']) + len(results['failed'])
        if total_trained > 0:
            success_rate = (len(results['success']) / total_trained) * 100
            print(f"\nSuccess Rate: {success_rate:.0f}% ({len(results['success'])}/{total_trained})")
        
        print("\n" + "=" * 70)
        
        # Return success if at least one model trained successfully
        return len(results['success']) > 0
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
