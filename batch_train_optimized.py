#!/usr/bin/env python
"""
PHASE 2: BATCH TRAINING SCRIPT - STABLE 100 EPOCH TRAINING
Trains all available stock models with optimized pipeline.

Features:
- Early stopping (20 epochs patience - stable training)
- Mixed precision training (CUDA)
- Learning rate scheduling (ReduceLROnPlateau: factor=0.5, patience=5)
- Gradient clipping (max_norm=1.0)
- Comprehensive validation logging
- Baseline comparison
- Best model checkpointing
- Stable loss curves
"""

import os
import sys
import glob
import time
from datetime import datetime
from typing import Optional

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.logging import logger
from backend.training.train_optimized import train_pipeline_optimized


def batch_train_optimized(
    data_dir: str = "backend/data/stock_data",
    use_mixed_precision: bool = True,
    skip_tickers: Optional[list] = None
):
    """
    Train all stock models with PHASE 2 optimizations.
    
    Args:
        data_dir: Directory containing stock CSV files
        use_mixed_precision: Enable mixed precision training
        skip_tickers: List of tickers to skip (already trained)
    """
    if skip_tickers is None:
        skip_tickers = []
    
    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 2: BATCH TRAINING - {len(csv_files)} STOCKS")
    logger.info(f"{'='*80}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mixed Precision: {use_mixed_precision}")
    logger.info(f"{'='*80}\n")
    
    # Filter out already trained stocks
    tickers_to_train = []
    for csv_file in csv_files:
        ticker = os.path.basename(csv_file).replace(".csv", "")
        if ticker not in skip_tickers:
            tickers_to_train.append((ticker, csv_file))
    
    if not tickers_to_train:
        logger.info("[OK] All stocks already trained!")
        return
    
    logger.info(f"Will train: {', '.join([t[0] for t in tickers_to_train])}\n")
    
    # Training results tracking
    results_summary = {
        'successful': [],
        'failed': [],
        'timing': {}
    }
    
    # Train each stock
    for idx, (ticker, csv_file) in enumerate(tickers_to_train, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"[{idx}/{len(tickers_to_train)}] TRAINING: {ticker}")
        logger.info(f"{'#'*80}\n")
        
        start_time = time.time()
        
        try:
            # Train with optimizations
            results = train_pipeline_optimized(
                file_path=csv_file,
                days_ahead=3,
                use_mixed_precision=use_mixed_precision
            )
            
            elapsed = time.time() - start_time
            results_summary['timing'][ticker] = elapsed
            results_summary['successful'].append(ticker)
            
            logger.info(f"\n[COMPLETED] {ticker} in {elapsed:.1f}s")
            logger.info(f"   Best Val Loss: {results.get('best_val_loss', 'N/A'):.6f}")
            logger.info(f"   Directional Accuracy: {results.get('best_directional_accuracy', 'N/A'):.2f}%")
            logger.info(f"   R2 Score: {results.get('best_r2', 'N/A'):.4f}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            results_summary['failed'].append((ticker, str(e)))
            logger.error(f"\n[FAILED] {ticker} after {elapsed:.1f}s")
            logger.error(f"   Error: {str(e)}")
    
    # Final summary
    logger.info(f"\n\n{'='*80}")
    logger.info(f"BATCH TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    
    logger.info(f"\n[SUCCESSFUL] ({len(results_summary['successful'])}):")
    for ticker in results_summary['successful']:
        timing = results_summary['timing'].get(ticker, 0)
        logger.info(f"   {ticker}: {timing:.1f}s")
    
    if results_summary['failed']:
        logger.info(f"\n[FAILED] ({len(results_summary['failed'])}):")
        for ticker, error in results_summary['failed']:
            logger.info(f"   {ticker}: {error}")
    
    total_time = sum(results_summary['timing'].values())
    logger.info(f"\nSTATISTICS:")
    logger.info(f"   Total Successful: {len(results_summary['successful'])}/{len(tickers_to_train)}")
    logger.info(f"   Total Failed: {len(results_summary['failed'])}/{len(tickers_to_train)}")
    logger.info(f"   Total Time: {total_time/60:.1f} minutes")
    logger.info(f"   Avg Time per Stock: {total_time/len(tickers_to_train):.1f}s")
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PHASE 2: Batch Training with Optimizations")
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated list of tickers to skip (e.g., 'INFY,TCS')"
    )
    parser.add_argument(
        "--mixed-precision",
        type=bool,
        default=True,
        help="Enable mixed precision training (default: True)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="backend/data/stock_data",
        help="Directory containing stock CSV files"
    )
    
    args = parser.parse_args()
    skip_list = [t.strip().upper() for t in args.skip.split(",")] if args.skip else []
    
    batch_train_optimized(
        data_dir=args.data_dir,
        use_mixed_precision=args.mixed_precision,
        skip_tickers=skip_list
    )
