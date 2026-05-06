#!/usr/bin/env python
"""
PHASE 1 Training Script - Run the corrected pipeline
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.training.train import train_pipeline
from backend.core.logging import logger

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🚀 PHASE 1 TRAINING - Corrected Pipeline")
    logger.info("="*70)
    
    # Train on INFY stock
    stock_file = "backend/data/stock_data/INFY.csv"
    
    if not os.path.exists(stock_file):
        logger.error(f"❌ Data file not found: {stock_file}")
        sys.exit(1)
    
    logger.info(f"📊 Starting training on {stock_file}...\n")
    
    try:
        train_pipeline(stock_file, days_ahead=3)
        logger.info("\n✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
