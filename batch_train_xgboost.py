#!/usr/bin/env python
"""
Batch XGBoost Classification Training Script
Trains all available stocks with the improved XGBoost classifier.

Features:
- Better labels (BUY/SELL/HOLD with thresholds)
- Enhanced feature engineering (momentum, volume, trend, volatility)
- Time-based train-test split (no shuffling)
- Confidence scores for each prediction
- Feature importance analysis
- Comprehensive evaluation metrics
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
from backend.training.xgboost_classifier import train_xgboost_classifier


def batch_train_xgboost_classifiers(
    data_dir: str = "backend/data/stock_data",
    buy_threshold: float = 0.002,      # 0.2% - UPDATED FROM 0.005
    sell_threshold: float = -0.002,    # -0.2% - UPDATED FROM -0.005
    skip_tickers: Optional[list] = None
):
    """
    Train XGBoost classifiers for all stocks.
    
    Args:
        data_dir: Directory containing stock CSV files
        buy_threshold: Positive return threshold for BUY label
        sell_threshold: Negative return threshold for SELL label
        skip_tickers: List of tickers to skip
    """
    if skip_tickers is None:
        skip_tickers = []
    
    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"XGBOOST CLASSIFICATION - BATCH TRAINING")
    logger.info(f"{'='*80}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"BUY Threshold: >{buy_threshold:.3%}")
    logger.info(f"SELL Threshold: <{sell_threshold:.3%}")
    logger.info(f"{'='*80}\n")
    
    # Filter stocks to train
    tickers_to_train = []
    for csv_file in csv_files:
        ticker = os.path.basename(csv_file).replace(".csv", "")
        if ticker not in skip_tickers:
            tickers_to_train.append((ticker, csv_file))
    
    if not tickers_to_train:
        logger.info("No stocks to train.")
        return
    
    logger.info(f"Will train: {', '.join([t[0] for t in tickers_to_train])}\n")
    
    # Training results tracking
    results_summary = {
        'successful': [],
        'failed': [],
        'metrics': {}
    }
    
    # Train each stock
    for idx, (ticker, csv_file) in enumerate(tickers_to_train, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"[{idx}/{len(tickers_to_train)}] TRAINING: {ticker}")
        logger.info(f"{'#'*80}\n")
        
        start_time = time.time()
        
        try:
            # Train with optimized parameters
            results = train_xgboost_classifier(
                ticker=ticker,
                file_path=csv_file,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold
            )
            
            elapsed = time.time() - start_time
            
            if results:
                metrics = results.get('metrics', {})
                results_summary['successful'].append(ticker)
                results_summary['metrics'][ticker] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0),
                    'training_time': elapsed
                }
                
                logger.info(f"[OK] {ticker} completed in {elapsed:.1f}s")
            else:
                results_summary['failed'].append(ticker)
                logger.error(f"[FAILED] {ticker} - No results")
        
        except Exception as e:
            elapsed = time.time() - start_time
            results_summary['failed'].append(ticker)
            logger.error(f"[FAILED] {ticker} - Error: {e}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nSuccessful: {len(results_summary['successful'])}/{len(tickers_to_train)}")
    if results_summary['successful']:
        for ticker in results_summary['successful']:
            metrics = results_summary['metrics'][ticker]
            logger.info(f"  {ticker}:")
            logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")
            logger.info(f"    F1 Score:  {metrics['f1']:.4f}")
            logger.info(f"    Time:      {metrics['training_time']:.1f}s")
    
    if results_summary['failed']:
        logger.warning(f"\nFailed: {len(results_summary['failed'])}")
        for ticker in results_summary['failed']:
            logger.warning(f"  {ticker}")
    
    logger.info(f"\nTotal Time: {(time.time() - start_time):.1f}s")
    logger.info(f"{'='*80}\n")
    
    return results_summary


if __name__ == "__main__":
    # Run batch training
    summary = batch_train_xgboost_classifiers(
        buy_threshold=0.002,    # 0.2% - UPDATED FROM 0.005
        sell_threshold=-0.002   # -0.2% - UPDATED FROM -0.005
    )
