#!/usr/bin/env python3
"""
Safe Training Runner - Executes model training with comprehensive error handling.
Ensures all dependencies are available, validates data, and logs all operations.
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_imports():
    """Verify all critical imports are available."""
    print("Step 1: Checking Critical Imports...")
    imports_ok = True
    
    required_packages = {
        'torch': 'PyTorch (Deep Learning)',
        'pandas': 'Pandas (Data Processing)',
        'numpy': 'NumPy (Numerical Computing)',
        'sklearn': 'Scikit-Learn (ML Utilities)',
        'ta': 'Technical Analysis Library',
        'transformers': 'HuggingFace Transformers',
    }
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name} ({package})")
        except ImportError as e:
            print(f"  ✗ {name} ({package}): {e}")
            imports_ok = False
    
    return imports_ok

def check_data_files():
    """Verify all required data files exist."""
    print("\nStep 2: Verifying Data Files...")
    
    from backend.core.config import settings
    
    tickers = ["HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK"]
    all_exist = True
    
    for ticker in tickers:
        file_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✓ {ticker}.csv ({file_size:.1f} KB)")
        else:
            print(f"  ✗ {ticker}.csv NOT FOUND: {file_path}")
            all_exist = False
    
    return all_exist

def train_with_error_handling():
    """Execute training pipeline with comprehensive error handling."""
    print("\nStep 3: Starting Model Training Pipeline...")
    print("=" * 70)
    
    from backend.core.config import settings
    from backend.core.logging import logger
    from backend.training.train import train_pipeline
    
    tickers = ["HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK"]
    
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Training {ticker}...")
        file_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        
        if not os.path.exists(file_path):
            print(f"  → SKIPPED: Data file not found")
            results['skipped'].append((ticker, "Data file not found"))
            continue
        
        try:
            print(f"  → Loading data from {file_path}...")
            result = train_pipeline(file_path)
            print(f"  ✓ SUCCESS: Training completed for {ticker}")
            results['success'].append(ticker)
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ ERROR: {error_msg}")
            print(f"  → Full traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    print(f"     {line}")
            
            results['failed'].append((ticker, error_msg))
            logger.error(f"Training failed for {ticker}: {e}", exc_info=True)
    
    return results

def print_summary(results):
    """Print training summary."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ Successful:  {len(results['success'])} models")
    for ticker in results['success']:
        print(f"   - {ticker}")
    
    if results['failed']:
        print(f"\n✗ Failed:     {len(results['failed'])} models")
        for ticker, error in results['failed']:
            print(f"   - {ticker}: {error[:60]}...")
    
    if results['skipped']:
        print(f"\n⊘ Skipped:    {len(results['skipped'])} models")
        for ticker, reason in results['skipped']:
            print(f"   - {ticker}: {reason}")
    
    total = len(results['success']) + len(results['failed'])
    if total > 0:
        success_rate = (len(results['success']) / total) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}% ({len(results['success'])}/{total})")
    
    print("\n" + "=" * 70)
    return len(results['failed']) == 0

def main():
    """Main execution function."""
    print("\n" + "🤖 INVESTIQ MODEL TRAINING SYSTEM".center(70))
    print("=" * 70)
    
    try:
        # Check prerequisites
        if not check_imports():
            print("\n⚠ Some required packages are missing!")
            print("Please run: pip install -r backend/requirements.txt")
            return False
        
        if not check_data_files():
            print("\n⚠ Some data files are missing!")
            return False
        
        # Run training
        results = train_with_error_handling()
        
        # Print summary
        all_success = print_summary(results)
        
        if all_success:
            print("\n✓ All models trained successfully!")
            return True
        else:
            print("\n⚠ Some models failed. Review errors above.")
            return False
            
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
