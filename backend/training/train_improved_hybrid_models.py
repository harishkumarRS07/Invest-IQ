#!/usr/bin/env python
"""
Quick Start: Train Hybrid LSTM + XGBoost Models

This script trains the improved hybrid model on all stocks and generates
comprehensive evaluation reports.

Usage:
    python train_improved_hybrid_models.py
    
Or with custom settings:
    python train_improved_hybrid_models.py --ticker HDFCBANK --seq_length 20
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from core.config import settings
from core.logging import logger
from training.improved_hybrid_model import (
    ProductionTrainingPipeline,
    AdvancedFeatureEngineer,
    HybridEnsembleModel
)
from training.evaluation_module import ProductionEvaluator, TradingMetricsCalculator
import pandas as pd
import numpy as np
from datetime import datetime


def save_trading_plots(results: Dict, ticker: str, diagnostics_dir: str) -> None:
    """Save trading-focused plots: equity, signal mix, and confidence buckets."""
    os.makedirs(diagnostics_dir, exist_ok=True)

    equity = np.asarray(results.get("equity_curve", np.array([1.0])))
    signals = np.asarray(results.get("signals", np.array([])), dtype=object)
    confidence = np.asarray(results.get("confidence", np.array([])), dtype=float)
    y_true = np.asarray(results.get("true_labels", np.array([])), dtype=int)

    # 1) Equity curve
    if equity.size > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(equity, color="tab:blue", linewidth=2)
        ax.set_title(f"{ticker} Equity Curve")
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, "06_equity_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2) Trade distribution
    if signals.size > 0:
        unique, counts = np.unique(signals, return_counts=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(unique, counts, color=["#2ca02c" if u == "UP" else "#d62728" if u == "DOWN" else "#7f7f7f" for u in unique])
        ax.set_title(f"{ticker} Signal Distribution")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, "07_trade_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3) Confidence vs accuracy by bucket
    if confidence.size > 0 and y_true.size == confidence.size:
        y_pred = (confidence >= 0.5).astype(int)
        bins = np.linspace(0.0, 1.0, 11)
        bucket_idx = np.digitize(confidence, bins) - 1
        bucket_centers = []
        bucket_acc = []
        for i in range(10):
            m = bucket_idx == i
            if np.any(m):
                bucket_centers.append((bins[i] + bins[i + 1]) / 2.0)
                bucket_acc.append(float(np.mean(y_true[m] == y_pred[m])))

        if bucket_centers:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(bucket_centers, bucket_acc, marker="o", linewidth=2, color="tab:purple")
            ax.set_title(f"{ticker} Confidence vs Accuracy")
            ax.set_xlabel("Confidence Bucket Center")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(diagnostics_dir, "08_confidence_vs_accuracy.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)


def save_threshold_table(results: Dict, ticker: str, diagnostics_dir: str) -> None:
    """Persist per-threshold evaluation table for each ticker."""
    threshold_rows = results.get("threshold_evaluations", [])
    if not threshold_rows:
        return

    rows = []
    for r in threshold_rows:
        rows.append(
            {
                "ticker": ticker,
                "up_threshold": r.get("up_threshold"),
                "down_threshold": r.get("down_threshold"),
                "accuracy": r.get("accuracy"),
                "win_rate": r.get("win_rate"),
                "num_trades": r.get("num_trades"),
                "trade_coverage": r.get("trade_coverage"),
                "total_return": r.get("total_return"),
                "directional_accuracy": r.get("directional_accuracy"),
            }
        )

    os.makedirs(diagnostics_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(diagnostics_dir, f"{ticker}_threshold_evaluation.csv"), index=False)


def train_single_stock(ticker: str, seq_length: int = 20, verbose: bool = True) -> Optional[Dict]:
    """Train model for a single stock."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {ticker} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    logger.info(f"{'='*80}")
    
    try:
        # Initialize pipeline
        pipeline = ProductionTrainingPipeline(ticker, seq_length=seq_length)
        
        # Load and preprocess
        file_path = f"{settings.DATA_DIR}/{ticker}.csv"
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            return None
        
        logger.info(f"Loading data from {file_path}...")
        df = pipeline.load_and_preprocess(file_path)
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Train with walk-forward validation
        logger.info("Training with walk-forward validation...")
        results = pipeline.train_with_walk_forward_validation(df)
        
        # Log results
        logger.info("\n" + "="*80)
        logger.info("TRAINING RESULTS")
        logger.info("="*80)
        logger.info(f"[OK] Accuracy:  {results['accuracy']:.2%}")
        logger.info(f"[OK] Precision: {results['precision']:.4f}")
        logger.info(f"[OK] Recall:    {results['recall']:.4f}")
        logger.info(f"[OK] F1-Score:  {results['f1']:.4f}")
        logger.info(f"[OK] ROC-AUC:   {results['roc_auc']:.4f}")
        logger.info(f"[OK] Directional Accuracy: {results['directional_accuracy']:.4f}")
        logger.info(f"[OK] Win Rate:             {results['win_rate']:.4f}")
        logger.info(f"[OK] Total Return:         {results['total_return']:.4f}")
        logger.info(f"[OK] Number of Trades:     {results['num_trades']}")
        logger.info(
            f"[OK] Best Threshold: up={results['best_threshold_up']:.2f}, "
            f"down={results['best_threshold_down']:.2f}"
        )
        if 'xgb_vs_hybrid' in results:
            comp = results['xgb_vs_hybrid']
            logger.info(
                "[OK] XGB vs Hybrid: "
                f"acc {comp['xgb_accuracy']:.4f}->{comp['hybrid_accuracy']:.4f}, "
                f"win {comp['xgb_win_rate']:.4f}->{comp['hybrid_win_rate']:.4f}, "
                f"ret {comp['xgb_total_return']:.4f}->{comp['hybrid_total_return']:.4f}"
            )
        
        # Generate diagnostics
        logger.info("\nGenerating evaluation plots...")
        diagnostics_dir = f"diagnostics/{ticker}"
        os.makedirs(diagnostics_dir, exist_ok=True)
        
        ProductionEvaluator.plot_all_diagnostics(
            results['true_labels'],
            results['predictions'],
            results['confidence'],
            save_dir=diagnostics_dir
        )
        save_trading_plots(results, ticker, diagnostics_dir)
        save_threshold_table(results, ticker, diagnostics_dir)
        logger.info(f"[OK] Diagnostics saved to {diagnostics_dir}/")
        
        # Generate report
        logger.info("Generating evaluation report...")
        ProductionEvaluator.generate_evaluation_report(results, ticker, save_dir=diagnostics_dir)
        logger.info(f"[OK] Report saved")
        
        return results
        
    except Exception as e:
        logger.error(f"[FAIL] Failed to train {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_all_stocks(seq_length: int = 20, skip_tickers: Optional[list] = None) -> Dict:
    """Train models for all stocks."""
    
    if skip_tickers is None:
        skip_tickers = []
    
    tickers = ["HDFCBANK", "ICICIBANK", "INFY", "RELIANCE", "TCS"]
    tickers = [t for t in tickers if t not in skip_tickers]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH TRAINING - {len(tickers)} Stocks")
    logger.info(f"{'='*80}")
    logger.info(f"Tickers: {', '.join(tickers)}")
    logger.info(f"Sequence Length: {seq_length}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results_all = {}
    success_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        results = train_single_stock(ticker, seq_length=seq_length, verbose=True)
        if results is not None:
            results_all[ticker] = results
            success_count += 1
    
    # Summary report
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total Stocks: {len(tickers)}")
    logger.info(f"Successfully Trained: {success_count}")
    logger.info(f"Failed: {len(tickers) - success_count}")
    
    if results_all:
        logger.info(
            f"\n{'Ticker':<12} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} "
            f"{'WinRate':<10} {'Return':<10} {'Trades':<8} {'BestThr':<12}"
        )
        logger.info("-" * 100)
        
        accuracies = []
        summary_rows = []
        for ticker, r in results_all.items():
            logger.info(
                f"{ticker:<12} {r['accuracy']:<8.2%} {r['precision']:<8.4f} {r['recall']:<8.4f} "
                f"{r['f1']:<8.4f} {r['win_rate']:<10.2%} {r['total_return']:<10.2%} "
                f"{r['num_trades']:<8d} {r['best_threshold_up']:.2f}/{r['best_threshold_down']:.2f}"
            )
            accuracies.append(r['accuracy'])
            summary_rows.append(
                {
                    "ticker": ticker,
                    "accuracy": r["accuracy"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "directional_accuracy": r["directional_accuracy"],
                    "win_rate": r["win_rate"],
                    "total_return": r["total_return"],
                    "num_trades": r["num_trades"],
                    "trade_coverage": r["trade_coverage"],
                    "best_threshold_up": r["best_threshold_up"],
                    "best_threshold_down": r["best_threshold_down"],
                    "xgb_accuracy": r.get("xgb_vs_hybrid", {}).get("xgb_accuracy"),
                    "xgb_win_rate": r.get("xgb_vs_hybrid", {}).get("xgb_win_rate"),
                    "xgb_total_return": r.get("xgb_vs_hybrid", {}).get("xgb_total_return"),
                    "hybrid_accuracy": r.get("xgb_vs_hybrid", {}).get("hybrid_accuracy"),
                    "hybrid_win_rate": r.get("xgb_vs_hybrid", {}).get("hybrid_win_rate"),
                    "hybrid_total_return": r.get("xgb_vs_hybrid", {}).get("hybrid_total_return"),
                }
            )
        
        logger.info("-" * 100)
        logger.info(f"{'Average Accuracy':<20} {np.mean(accuracies):.2%}")
        logger.info(f"{'Min Accuracy':<20} {np.min(accuracies):.2%}")
        logger.info(f"{'Max Accuracy':<20} {np.max(accuracies):.2%}")

        summary_dir = "diagnostics"
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info(f"[OK] Saved batch summary CSV: {summary_path}")
    
    logger.info(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results_all


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Train hybrid LSTM + XGBoost models"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Train a specific ticker (default: train all)"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=20,
        help="Sequence length for LSTM (default: 20)"
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="+",
        default=[],
        help="Skip these tickers (e.g., --skip HDFCBANK ICICIBANK)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run training
    if args.ticker:
        logger.info(f"Training single ticker: {args.ticker}")
        results = train_single_stock(args.ticker, seq_length=args.seq_length)
    else:
        logger.info("Training all tickers")
        results = train_all_stocks(seq_length=args.seq_length, skip_tickers=args.skip)
    
    logger.info("\n[OK] Training complete!")


if __name__ == "__main__":
    main()
