import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator
from backend.models.enhanced_models import LSTMAttentionEnhanced
from backend.evaluation.metrics import calculate_metrics

sys.path.append(os.getcwd())

def evaluate_pipeline(file_path: str):
    ticker = os.path.basename(file_path).replace(".csv", "")
    logger.info(f"Starting evaluation for {ticker}...")
    
    # 1. Load Data (Same preprocessing as training)
    # Ideally we should split before preprocessing steps that leak info, but standardizing here is okay for now if using same scaler
    try:
        df = load_data(file_path)
    except Exception:
        return
    
    df = clean_data(df)
    market_df = ExternalDataSimulator.fetch_market_index(start_date=df.index[0], end_date=df.index[-1])
    df = add_technical_indicators(df)
    df = add_market_correlation(df, market_df)
    df = ExternalDataSimulator.add_external_features(df, ticker)
    df = df.dropna()
    
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
    # Ensure same feature order
    # Load Scaler to get feature columns
    scaler = StockScaler()
    try:
        scaler.load(f"scaler_{ticker}.pkl")
        feature_cols = scaler.feature_columns
    except Exception as e:
        logger.error(f"Could not load scaler for {ticker}: {e}")
        return

    # Transform
    df_scaled = scaler.transform(df)
    data_scaled = df_scaled[feature_cols].values
    
    # Create Sequences
    from backend.training.train import create_sequences
    target_col = 'Log_Return' if 'Log_Return' in feature_cols else 'Close'
    target_col_idx = feature_cols.index(target_col)
    
    X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, settings.FORECAST_HORIZON, target_col_idx)

    # Reshape y to (samples, horizon, 1) if it's (samples, horizon)
    if len(y.shape) == 2:
        y = y[..., np.newaxis]
    
    # Load Model
    input_dim = X.shape[2]
    model = LSTMAttentionEnhanced(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        dropout=0.3,
        forecast_horizon=settings.FORECAST_HORIZON
    )
    
    model_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")
    if not os.path.exists(model_path):
        logger.error(f"Model not found for {ticker}")
        return
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Predicting
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        preds = model(X_tensor).numpy() # (samples, horizon, 1)
        
    # Metrics
    metrics = calculate_metrics(y, preds)
    logger.info(f"Metrics for {ticker}: {metrics}")
    
    # Save Report
    with open(os.path.join(settings.MODEL_DIR, f"evaluation_{ticker}.txt"), "w") as f:
        f.write(f"Evaluation Report for {ticker}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
            
    # Plotting (Likely just first step of forecast vs actual for clarity)
    # y shape (N, 7, 1), preds shape (N, 7, 1)
    # Plot t+1 prediction vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y[:, 0, 0], label='Actual t+1')
    plt.plot(preds[:, 0, 0], label='Predicted t+1')
    plt.title(f"Prediction vs Actual for {ticker}")
    plt.legend()
    plt.savefig(os.path.join(settings.MODEL_DIR, f"plot_{ticker}.png"))
    plt.close()
    
    logger.info(f"Evaluation complete for {ticker}. Report and Plot saved.")

if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(settings.DATA_DIR, "*.csv"))
    for file_path in csv_files:
        evaluate_pipeline(file_path)
