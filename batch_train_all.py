#!/usr/bin/env python
"""
Batch Training Script - Train All Remaining Models
"""
import os
import sys
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.training.train import train_pipeline
from backend.core.logging import logger

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🚀 PHASE 1 BATCH TRAINING - Remaining Models")
    logger.info("="*70)
    
    # Get all CSV files
    all_files = sorted(glob.glob("backend/data/stock_data/*.csv"))
    
    if not all_files:
        logger.error("❌ No stock data files found")
        sys.exit(1)
    
    # Filter already trained
    trained = ["INFY"]
    remaining = []
    
    for csv_file in all_files:
        ticker = os.path.basename(csv_file).replace(".csv", "")
        if ticker not in trained:
            remaining.append(csv_file)
    
    logger.info(f"\n📊 Found {len(remaining)} remaining stocks to train:")
    for f in remaining:
        logger.info(f"   • {os.path.basename(f)}")
    
    # Train each
    success_count = 0
    for i, csv_file in enumerate(remaining, 1):
        ticker = os.path.basename(csv_file).replace(".csv", "")
        logger.info(f"\n{'='*70}")
        logger.info(f"[{i}/{len(remaining)}] Training {ticker}...")
        logger.info(f"{'='*70}\n")
        
        try:
            train_pipeline(csv_file, days_ahead=3)
            success_count += 1
            logger.info(f"\n✅ {ticker} training completed!")
        except Exception as e:
            logger.error(f"\n❌ {ticker} training failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Successfully trained: {success_count}/{len(remaining)} models")
    
    if success_count == len(remaining):
        logger.info(f"✅ ALL MODELS TRAINED SUCCESSFULLY!")
    else:
        logger.warning(f"⚠️  {len(remaining) - success_count} models failed")
