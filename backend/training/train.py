import sys
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator
from backend.models.transformer import TimeSeriesTransformer

# Ensure project root is in path
sys.path.append(os.getcwd())

def create_sequences(data: np.ndarray, seq_length: int, forecast_horizon: int, target_col_idx: int):
    """
    Create sequences for Transformer.
    X: (samples, seq_length, features)
    y: (samples, forecast_horizon, 1) -> Predicting 'Close' or 'Log_Return' for next k days
    """
    sequences = []
    targets = []
    
    # We need data for i to i+seq_length (Input)
    # And targets for i+seq_length to i+seq_length+forecast_horizon (Output)
    
    num_samples = len(data) - seq_length - forecast_horizon + 1
    
    for i in range(num_samples):
        seq = data[i : i+seq_length] # Input Sequence
        
        # Target Sequence (next `forecast_horizon` steps)
        # We predict the target column values
        target = data[i+seq_length : i+seq_length+forecast_horizon, target_col_idx]
        
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(targets)

def train_pipeline(file_path: str):
    ticker = os.path.basename(file_path).replace(".csv", "")
    logger.info(f"Starting advanced training pipeline for {ticker}...")
    
    # 1. Load and Preprocess
    try:
        df = load_data(file_path)
    except Exception as e:
        logger.error(f"Failed to load {ticker}: {e}")
        return
        
    df = clean_data(df)

    # Use a uniform lookback window for all tickers before feature engineering.
    # This keeps training coverage consistent across symbols.
    if 'Date' in df.columns and not df['Date'].isnull().all():
        latest_date = df['Date'].max()
        window_start = latest_date - pd.DateOffset(years=25)
        df = df[df['Date'] >= window_start].copy()
        logger.info(
            f"Using 25-year window for {ticker}: {window_start.date()} to {latest_date.date()}"
        )
    
    # 2. Feature Engineering
    # Fetch Market Index (Only once per run ideally, but here per file for simplicity)
    # To avoid repeated API calls, we could fetch once outside, but let's try to fetch here.
    # In production, this should be cached.
    market_start = df['Date'].min() if 'Date' in df.columns else None
    market_end = df['Date'].max() if 'Date' in df.columns else None
    market_df = ExternalDataSimulator.fetch_market_index(start_date=market_start, end_date=market_end)
    
    df = add_technical_indicators(df)
    df = add_market_correlation(df, market_df)
    
    # Add External Data (Sentiment/Macro) - For training we simulate or fetch historic if available
    # For now, let's use the simulator for sentiment/macro as placeholders if real data isn't fully piped
    df = ExternalDataSimulator.add_external_features(df, ticker)
    
    # Drop rows with NaNs after feature engineering
    df = df.dropna()
    
    if len(df) < settings.SEQ_LENGTH + settings.FORECAST_HORIZON + 100:
        logger.warning(f"Insufficient data for {ticker} after preprocessing.")
        return

    # 3. Define Features and Target
    # We want to use all available numeric columns as features
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
    
    # Target: We usually predict Close price or Log Return. 
    # Let's predict 'Log_Return' for stationarity, or 'Close' if we want direct price.
    # Predicting 'Close' with Transformer is fine if scaled properly.
    # But 'Log_Return' is better for gradients. Let's stick to 'Log_Return' as primary target.
    target_col = 'Log_Return'
    if target_col not in feature_cols:
        # Fallback if Log_Return not created
        target_col = 'Close'
    
    target_col_idx = feature_cols.index(target_col)
    
    # 4. Scaling
    # Use StandardScaler for better convergence with Transformers
    scaler = StockScaler(scaler_type='standard')
    df_scaled = scaler.fit_transform(df, feature_cols)
    data_scaled = df_scaled[feature_cols].values
    
    # 5. Create Sequences (Multi-step)
    X, y = create_sequences(
        data_scaled, 
        settings.SEQ_LENGTH, 
        settings.FORECAST_HORIZON, 
        target_col_idx
    )
    
    # Reshape y to (samples, horizon, 1) if it's (samples, horizon)
    if len(y.shape) == 2:
        y = y[..., np.newaxis]
        
    # 6. Train/Val Split (Time-based, No Shuffle)
    train_size = int(len(X) * (1 - settings.TEST_SIZE))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Convert to Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=False) # Shuffle=False for time series? Actually shuffle=True is fine for training batches, but val should be sequential. Transformers handle Order via PE.
    # Ideally shuffle training windows to break correlation between batches? Yes.
    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)
    
    # 7. Model Initialization
    input_dim = X.shape[2]
    output_dim = 1 # Predicting 1 variable (Log_Return)
    
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=64,
        nhead=settings.NHEAD,
        num_layers=settings.NUM_LAYERS,
        dropout=settings.DROPOUT,
        output_dim=output_dim,
        forecast_horizon=settings.FORECAST_HORIZON
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    checkpoint_path = os.path.join(settings.MODEL_DIR, f"transformer_{ticker}.pth")
    
    # 8. Training Loop
    logger.info(f"Training on {device}...")
    
    for epoch in range(settings.EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X) # (batch, horizon, 1)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{settings.EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        # Step Scheduler
        scheduler.step(val_loss)
        
    # Save final model checkpoint after completing all epochs.
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Final model checkpoint saved to {checkpoint_path}")
            
    # Save Scaler and Metadata
    scaler.save(f"scaler_{ticker}.pkl")
    logger.info(f"Training completed for {ticker}.")

if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(settings.DATA_DIR, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {settings.DATA_DIR}")
    
    for file_path in csv_files:
        train_pipeline(file_path)
