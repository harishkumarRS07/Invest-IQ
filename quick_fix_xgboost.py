#!/usr/bin/env python
"""
QUICK FIX: Retrain XGBoost with matching features from predict.py

Problem: Models trained with 33 features, but predict.py only has 19
Solution: Retrain with ONLY the 19 standard features
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

sys.path.insert(0, os.getcwd())

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator


def train_xgboost_simple(ticker: str, csv_file: str, 
                        buy_threshold: float = 0.002,
                        sell_threshold: float = -0.002,
                        horizon: int = 3):
    """
    Train XGBoost using ONLY the 19 standard features available in predict.py
    
    Standard features (from Transformer pipeline):
    - Technical: RSI, MACD, MACD_Signal, SMA_20, SMA_50, BB_High, BB_Low, ATR, VWAP, MACD_Hist
    - Price: Open, High, Low, Close
    - Volume: Volume
    - Log return: Log_Return
    - Correlation: Market_Correlation  
    """
    
    print(f"\n{'='*80}")
    print(f"Training XGBoost for {ticker}")
    print(f"{'='*80}\n")
    
    # 1. Load data
    print(f"Loading {csv_file}...")
    df = load_data(csv_file)
    df = clean_data(df)
    
    if len(df) < horizon + 100:
        print(f"✗ Insufficient data ({len(df)} rows)")
        return None
    
    # 2. Add indicators
    print("Adding technical indicators...")
    market_df = ExternalDataSimulator.fetch_market_index()
    df = add_technical_indicators(df)
    df = add_market_correlation(df, market_df)
    
    # 3. Select ONLY standard features (19 features)
    standard_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Log_Return', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'SMA_20', 'SMA_50', 'BB_High', 'BB_Low', 'ATR', 'VWAP',
        'Volume_Change', 'Rolling_Volatility', 'Market_Correlation'
    ]
    
    # Keep only features that exist
    available_features = [col for col in standard_features if col in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    # 4. Create labels
    print(f"Creating labels (BUY > {buy_threshold:.4f}, SELL < {sell_threshold:.4f})...")
    
    future_close = df['Close'].shift(-horizon)
    future_returns = (future_close - df['Close']) / df['Close']
    
    labels = np.ones(len(df), dtype=int)  # Default HOLD
    labels[future_returns > buy_threshold] = 2   # BUY
    labels[future_returns < sell_threshold] = 0  # SELL
    labels = labels[:-horizon]
    
    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    print(f"\nClass Distribution:")
    for label, count in zip(unique, counts):
        pct = 100.0 * count / total
        signal = ['SELL', 'HOLD', 'BUY'][int(label)]
        print(f"  {signal}: {count:5d} ({pct:6.2f}%)")
    
    # 5. Prepare features
    X = df[available_features].iloc[:-horizon].copy()
    y = labels.copy()
    
    # Drop NaN
    valid_idx = ~(X.isna().any(axis=1))
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"\nData shape: {X.shape}")
    
    if len(X) < 100:
        print(f"✗ Not enough data after cleaning")
        return None
    
    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 7. Train XGBoost
    print(f"\nTraining XGBoost model...")
    
    # Calculate class weights to handle imbalance
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    
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
        verbosity=0,
        scale_pos_weight=None  # We'll use sample_weight instead
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    
    # 8. Evaluate
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
    
    # 9. Save model
    model_path = os.path.join(settings.MODEL_DIR, f"xgboost_classifier_{ticker}.pkl")
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    return {
        'ticker': ticker,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUICK FIX: Retrain XGBoost with Standard Features Only")
    print("="*80)
    
    tickers = ["HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK"]
    
    results = []
    
    for ticker in tickers:
        csv_file = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        
        if not os.path.exists(csv_file):
            print(f"\n✗ {ticker}: Data file not found")
            continue
        
        try:
            result = train_xgboost_simple(ticker, csv_file)
            if result:
                results.append(result)
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Trained: {len(results)}/{len(tickers)} stocks\n")
    
    for r in results:
        print(f"{r['ticker']}: Accuracy={r['accuracy']:.4f}, F1={r['f1']:.4f}")
    
    print(f"\n✓ All models retrained with standard features")
    print(f"  They now match the features available in predict.py")
    print(f"\nRun predictions:")
    print(f"  python backend/scripts/demo.py")
    print(f"\n" + "="*80 + "\n")
