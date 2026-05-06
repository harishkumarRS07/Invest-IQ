#!/usr/bin/env python
"""
Quick Verification - Show the impact of threshold change

This script demonstrates:
1. OLD THRESHOLD (0.005 = 0.5%): Results in majority HOLD
2. NEW THRESHOLD (0.002 = 0.2%): Results in balanced classes
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.config import settings
from backend.preprocessing.cleaning import load_data, clean_data


def show_threshold_impact(ticker: str = "HDFCBANK", horizon: int = 3):
    """Show impact of OLD vs NEW threshold"""
    
    # Load data
    file_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
    
    print(f"\n{'='*80}")
    print(f"THRESHOLD IMPACT ANALYSIS - {ticker}")
    print(f"{'='*80}\n")
    
    df = load_data(file_path)
    df = clean_data(df)
    
    if len(df) < horizon + 10:
        print("Insufficient data")
        return
    
    # Calculate future returns
    future_close = df['Close'].shift(-horizon)
    future_returns = (future_close - df['Close']) / df['Close']
    
    # OLD THRESHOLD (0.005)
    print("OLD THRESHOLD: 0.005 (0.5%)")
    print("-" * 80)
    
    old_threshold = 0.005
    labels_old = np.ones(len(df), dtype=int)
    labels_old[future_returns > old_threshold] = 2
    labels_old[future_returns < -old_threshold] = 0
    labels_old = labels_old[:-horizon]
    
    unique_old, counts_old = np.unique(labels_old, return_counts=True)
    total_old = len(labels_old)
    
    print(f"{'Signal':<10} {'Count':<10} {'Percentage':<12}")
    print("-" * 32)
    for label, count in zip(unique_old, counts_old):
        pct = 100.0 * count / total_old
        signal = ['SELL', 'HOLD', 'BUY'][int(label)]
        print(f"{signal:<10} {count:<10} {pct:>10.2f}%")
    
    # NEW THRESHOLD (0.002)
    print(f"\n\nNEW THRESHOLD: 0.002 (0.2%)")
    print("-" * 80)
    
    new_threshold = 0.002
    labels_new = np.ones(len(df), dtype=int)
    labels_new[future_returns > new_threshold] = 2
    labels_new[future_returns < -new_threshold] = 0
    labels_new = labels_new[:-horizon]
    
    unique_new, counts_new = np.unique(labels_new, return_counts=True)
    total_new = len(labels_new)
    
    print(f"{'Signal':<10} {'Count':<10} {'Percentage':<12}")
    print("-" * 32)
    for label, count in zip(unique_new, counts_new):
        pct = 100.0 * count / total_new
        signal = ['SELL', 'HOLD', 'BUY'][int(label)]
        print(f"{signal:<10} {count:<10} {pct:>10.2f}%")
    
    # Analysis
    print(f"\n{'='*80}")
    print("IMPACT ANALYSIS")
    print(f"{'='*80}\n")
    
    hold_pct_old = 100.0 * np.sum(labels_old == 1) / len(labels_old)
    hold_pct_new = 100.0 * np.sum(labels_new == 1) / len(labels_new)
    
    buy_old = np.sum(labels_old == 2)
    buy_new = np.sum(labels_new == 2)
    
    sell_old = np.sum(labels_old == 0)
    sell_new = np.sum(labels_new == 0)
    
    print(f"HOLD percentage: {hold_pct_old:.1f}% → {hold_pct_new:.1f}%")
    print(f"BUY signals:     {buy_old} → {buy_new} (+{buy_new - buy_old})")
    print(f"SELL signals:    {sell_old} → {sell_new} (+{sell_new - sell_old})")
    
    if hold_pct_old > 95:
        print(f"\n✗ OLD THRESHOLD: Over 95% HOLD (BAD - model useless)")
    else:
        print(f"\n✓ OLD THRESHOLD: Acceptable distribution")
    
    if hold_pct_new < 80 and buy_new > 0 and sell_new > 0:
        print(f"✓ NEW THRESHOLD: Balanced distribution (GOOD - model useful)")
    else:
        print(f"✗ NEW THRESHOLD: Still imbalanced")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Test for each stock
    tickers = ["HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK"]
    
    for ticker in tickers:
        try:
            show_threshold_impact(ticker)
        except Exception as e:
            print(f"Error for {ticker}: {e}")
