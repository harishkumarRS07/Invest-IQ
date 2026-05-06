"""
Diagnostic script to identify evaluation issues
"""

import sys
import os
import torch
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator
from backend.models.enhanced_models import LSTMAttentionEnhanced
from backend.training.train import create_sequences

print("\n" + "=" * 80)
print("INVESTIQ EVALUATION DIAGNOSTIC")
print("=" * 80 + "\n")

TICKERS = ['HDFCBANK', 'ICICIBANK', 'INFY', 'RELIANCE', 'TCS']

# Test 1: Check data files
print("TEST 1: Checking data files...")
for ticker in TICKERS:
    data_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
    if os.path.exists(data_path):
        df = load_data(data_path)
        print(f"  ✓ {ticker}: {len(df)} rows")
    else:
        print(f"  ✗ {ticker}: NOT FOUND")

# Test 2: Check model files
print("\nTEST 2: Checking model files...")
models_to_check = [
    ('lstm_HDFCBANK.pth', 'LSTM'),
    ('xgboost_classifier_HDFCBANK.pkl', 'XGBoost'),
]

for model_file, model_name in models_to_check:
    model_path = os.path.join(settings.MODEL_DIR, model_file)
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ {model_name}: {size_mb:.2f} MB")
    else:
        print(f"  ✗ {model_name}: NOT FOUND")

# Test 3: Try loading and preprocessing one ticker
print("\nTEST 3: Testing data preprocessing for HDFCBANK...")
try:
    data_path = os.path.join(settings.DATA_DIR, "HDFCBANK.csv")
    df = load_data(data_path)
    df = clean_data(df)
    
    market_df = ExternalDataSimulator.fetch_market_index(
        start_date=df.index[0], 
        end_date=df.index[-1]
    )
    df = add_technical_indicators(df)
    df = add_market_correlation(df, market_df)
    df = ExternalDataSimulator.add_external_features(df, 'HDFCBANK')
    df = df.dropna()
    
    print(f"  ✓ Preprocessing successful: {len(df)} rows after preprocessing")
    
    scaler = StockScaler()
    try:
        scaler.load("scaler_HDFCBANK.pkl")
        print(f"  ✓ Scaler loaded successfully")
    except Exception as e:
        print(f"  ✗ Scaler error: {e}")
    
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
    print(f"  ✓ Features count: {len(feature_cols)}")
    
    df_scaled = scaler.transform(df)
    data_scaled = df_scaled[feature_cols].values
    
    target_col = 'Log_Return' if 'Log_Return' in feature_cols else 'Close'
    target_col_idx = feature_cols.index(target_col)
    
    X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, 1, target_col_idx)
    print(f"  ✓ Sequences created: X shape {X.shape}, y shape {y.shape}")
    
except Exception as e:
    print(f"  ✗ Preprocessing error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Try loading LSTM model
print("\nTEST 4: Testing LSTM model loading and inference...")
try:
    model_path = os.path.join(settings.MODEL_DIR, "lstm_HDFCBANK.pth")
    
    if not os.path.exists(model_path):
        print(f"  ✗ Model file not found: {model_path}")
    else:
        model = LSTMAttentionEnhanced(
            input_dim=X.shape[2],
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            dropout=0.3,
            forecast_horizon=settings.FORECAST_HORIZON
        )
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"  ✓ Model loaded successfully")
        
        # Test inference
        X_tensor = torch.FloatTensor(X[:10])  # First 10 samples
        with torch.no_grad():
            preds = model(X_tensor).numpy()
        
        print(f"  ✓ Inference successful: predictions shape {preds.shape}")
        print(f"  ✓ Predictions sample: {preds[:3].flatten()}")
        
except Exception as e:
    print(f"  ✗ LSTM error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Try loading LSTM model with forecast horizon
print("\nTEST 5: Testing LSTM model with forecast horizon...")
try:
    model_path = os.path.join(settings.MODEL_DIR, "lstm_HDFCBANK.pth")
    
    if not os.path.exists(model_path):
        print(f"  ✗ Model file not found: {model_path}")
    else:
        X_fh, y_fh = create_sequences(data_scaled, settings.SEQ_LENGTH, settings.FORECAST_HORIZON, target_col_idx)
        
        if len(y_fh.shape) == 2:
            y_fh = y_fh[..., np.newaxis]
        
        model = LSTMAttentionEnhanced(
            input_dim=X_fh.shape[2],
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            dropout=0.3,
            forecast_horizon=settings.FORECAST_HORIZON
        )
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"  ✓ Model loaded successfully")
        
        # Test inference
        X_tensor = torch.FloatTensor(X_fh[:10])
        with torch.no_grad():
            preds = model(X_tensor).numpy()
        
        print(f"  ✓ Inference successful: predictions shape {preds.shape}")
        print(f"  ✓ Predictions sample: {preds[:3, 0, 0]}")
        
except Exception as e:
    print(f"  ✗ LSTM Forecast error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check metrics calculation
print("\nTEST 6: Testing metrics calculation...")
try:
    from backend.evaluation.metrics import calculate_metrics
    
    # Create dummy predictions and actuals
    y_true = np.array([[0.01], [0.02], [-0.01], [0.03], [-0.02]])
    y_pred = np.array([[0.012], [0.018], [-0.012], [0.028], [-0.025]])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"  ✓ Metrics calculated successfully:")
    for key, value in metrics.items():
        print(f"    - {key}: {value}")
        
except Exception as e:
    print(f"  ✗ Metrics error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80 + "\n")
