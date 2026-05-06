#!/usr/bin/env python
"""
Retrain XGBoost with ADAPTIVE THRESHOLDS and STANDARD 19 FEATURES ONLY

This matches the features available in predict.py for seamless integration.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator


def train_adaptive_xgboost(ticker: str, csv_file: str):
    """Train XGBoost with adaptive percentile-based thresholds and 19 standard features."""
    
    print(f"\n{'='*80}")
    print(f"Training {ticker} with Adaptive Thresholds + Standard 19 Features")
    print(f"{'='*80}\n")
    
    # Load data
    df = load_data(csv_file)
    df = clean_data(df)
    
    if len(df) < 100:
        logger.error(f"Insufficient data: {len(df)} rows")
        return None
    
    # Add indicators
    market_df = ExternalDataSimulator.fetch_market_index()
    df = add_technical_indicators(df)
    df = add_market_correlation(df, market_df)
    
    # Define 19 STANDARD FEATURES (matching predict.py)
    standard_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Log_Return', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'SMA_20', 'SMA_50', 'BB_High', 'BB_Low', 'ATR', 'VWAP',
        'Volume_Change', 'Rolling_Volatility', 'Market_Correlation'
    ]
    
    # Keep only available features
    available_features = [col for col in standard_features if col in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Create labels with ADAPTIVE THRESHOLDS
    print("\n" + "="*80)
    print("ADAPTIVE THRESHOLD COMPUTATION")
    print("="*80)
    
    future_close = df['Close'].shift(-3)  # 3-day horizon
    future_returns = (future_close - df['Close']) / df['Close']
    returns_clean = future_returns.dropna()
    
    # Compute percentile-based thresholds
    buy_threshold = returns_clean.quantile(0.65)   # Top 35% = BUY
    sell_threshold = returns_clean.quantile(0.35)  # Bottom 35% = SELL
    
    print(f"Return statistics:")
    print(f"  Mean:    {returns_clean.mean():.6f} ({returns_clean.mean()*100:.4f}%)")
    print(f"  Std:     {returns_clean.std():.6f}")
    print(f"  Median:  {returns_clean.median():.6f}")
    print(f"\nAdaptive thresholds:")
    print(f"  SELL (35th pct): {sell_threshold:.6f} ({sell_threshold*100:.4f}%)")
    print(f"  BUY  (65th pct): {buy_threshold:.6f} ({buy_threshold*100:.4f}%)")
    
    # Create labels
    labels = np.ones(len(df), dtype=int)  # Default HOLD
    labels[future_returns > buy_threshold] = 2    # BUY
    labels[future_returns < sell_threshold] = 0   # SELL
    labels = labels[:-3]  # Remove last horizon rows
    
    # Prepare features
    X = df[available_features].iloc[:-3].copy()
    y = labels.copy()
    
    # Clean
    valid_idx = ~(X.isna().any(axis=1))
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Check distribution
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\nClass distribution:")
    for label, count in zip(unique, counts):
        pct = 100.0 * count / total
        signal = ['SELL', 'HOLD', 'BUY'][int(label)]
        print(f"  {signal}: {count:6d} ({pct:6.2f}%)")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"\nTraining: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    print(f"\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    
    # Save model
    model_path = f"backend/models/saved_models/xgboost_classifier_{ticker}.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Show sample predictions
    y_proba = model.predict_proba(X_test[:5])
    print(f"\nFirst 5 test predictions:")
    for i, proba in enumerate(y_proba):
        pred_label = y_pred[i]
        signal = ['SELL', 'HOLD', 'BUY'][int(pred_label)]
        print(f"  {i}: {signal} (SELL={proba[0]:.3f}, HOLD={proba[1]:.3f}, BUY={proba[2]:.3f})")
    
    return {'accuracy': accuracy, 'f1': f1}


def main():
    """Retrain all models."""
    print("\n" + "="*80)
    print("RETRAINING ALL MODELS WITH ADAPTIVE THRESHOLDS + 19 STANDARD FEATURES")
    print("="*80)
    
    data_dir = "backend/data/stock_data"
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    if not csv_files:
        print(f"No CSV files in {data_dir}")
        return
    
    results = []
    for csv_file in csv_files:
        ticker = os.path.basename(csv_file).replace(".csv", "")
        try:
            result = train_adaptive_xgboost(ticker, csv_file)
            if result:
                results.append({'ticker': ticker, 'status': 'SUCCESS', **result})
            else:
                results.append({'ticker': ticker, 'status': 'FAILED'})
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            results.append({'ticker': ticker, 'status': 'ERROR'})
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for r in results:
        status = r['status']
        ticker = r['ticker']
        print(f"{ticker:15s}: {status:10s}", end="")
        if 'accuracy' in r:
            print(f" | Acc: {r['accuracy']:.4f}, F1: {r['f1']:.4f}")
        else:
            print()
    
    successful = len([r for r in results if r['status'] == 'SUCCESS'])
    print(f"\nCompleted: {successful}/{len(results)} successful")

if __name__ == "__main__":
    main()
