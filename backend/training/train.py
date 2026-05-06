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
from backend.models.enhanced_models import LSTMAttentionEnhanced
from backend.utils.data_pipeline import (
    create_future_return_target,
    train_test_time_split,
    create_sequences_v2,
    validate_sequences,
    log_data_statistics
)

# Ensure project root is in path
sys.path.append(os.getcwd())


def create_sequences(data: np.ndarray, seq_length: int, forecast_horizon: int, target_col_idx: int):
    """
    DEPRECATED: Use create_sequences_v2 instead.
    Keeping for backward compatibility.
    """
    return create_sequences_v2(data, seq_length, forecast_horizon, target_col_idx, name="legacy")


def train_pipeline(file_path: str, days_ahead: int = 3):
    """
    PHASE 1: CORRECTED TRAINING PIPELINE
    
    Correct order with NO data leakage:
    1. Load data
    2. Clean data (strict)
    3. Add technical indicators
    4. Drop NaN rows
    5. Create future return target
    6. Time-based split (train/test)
    7. Fit scaler on TRAIN data only
    8. Transform both train and test
    9. Create sequences
    10. Log statistics
    11. Train model
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file with stock data
    days_ahead : int
        Days into future to predict (default 3)
    """
    ticker = os.path.basename(file_path).replace(".csv", "")
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 PHASE 1: Starting corrected training pipeline for {ticker}")
    logger.info(f"{'='*70}\n")
    
    # ========== STEP 1: LOAD DATA ==========
    try:
        logger.info("📥 STEP 1: Loading data...")
        df = load_data(file_path)
        logger.info(f"   Loaded: {df.shape} (rows, cols)")
    except Exception as e:
        logger.error(f"❌ Failed to load {ticker}: {e}")
        return
    
    # ========== STEP 2: CLEAN DATA (STRICT) ==========
    logger.info("🧹 STEP 2: Cleaning data (strict validation)...")
    df = clean_data(df, verbose=True)
    logger.info(f"   After cleaning: {df.shape}")
    
    # ========== STEP 3: FILTER TIME WINDOW ==========
    if 'Date' in df.columns and not df['Date'].isnull().all():
        logger.info("📅 STEP 3: Filtering time window...")
        latest_date = df['Date'].max()
        window_start = latest_date - pd.DateOffset(years=25)
        df = df[df['Date'] >= window_start].copy()
        logger.info(
            f"   Using 25-year window: {window_start.date()} → {latest_date.date()}"
        )
    
    # ========== STEP 4: ADD TECHNICAL INDICATORS ==========
    logger.info("📊 STEP 4: Adding technical indicators...")
    df = add_technical_indicators(df)
    logger.info(f"   Features after indicators: {len(df.columns)}")
    
    # ========== STEP 5: ADD MARKET CORRELATION (optional) ==========
    logger.info("🔗 STEP 5: Adding market correlation...")
    try:
        market_start = df['Date'].min() if 'Date' in df.columns else None
        market_end = df['Date'].max() if 'Date' in df.columns else None
        market_df = ExternalDataSimulator.fetch_market_index(
            start_date=market_start, 
            end_date=market_end
        )
        if not market_df.empty:
            df = add_market_correlation(df, market_df)
            logger.info("   Market correlation added ✓")
        else:
            logger.warning("   Market data unavailable, skipping correlation")
    except Exception as e:
        logger.warning(f"   Could not fetch market index: {e}")
    
    # ========== STEP 6: NO EXTERNAL FEATURES (PHASE 1) ==========
    logger.info("⚠️  STEP 6: Skipping synthetic external features (PHASE 1)...")
    logger.info("   Sentiment and Macro features disabled to prevent random noise")
    df = ExternalDataSimulator.add_external_features(df, ticker, use_real_data=False)
    
    # ========== STEP 7: DROP NaN ROWS ==========
    logger.info("🗑️  STEP 7: Dropping rows with NaN values...")
    len_before = len(df)
    df = df.dropna()
    len_after = len(df)
    logger.info(f"   Dropped {len_before - len_after} rows with NaN")
    logger.info(f"   Remaining: {len_after} rows")
    
    if len(df) < settings.SEQ_LENGTH + settings.FORECAST_HORIZON + 100:
        logger.error(f"❌ Insufficient data: {len(df)} rows (need at least {settings.SEQ_LENGTH + settings.FORECAST_HORIZON + 100})")
        return
    
    # ========== STEP 8: CREATE FUTURE RETURN TARGET ==========
    logger.info(f"🎯 STEP 8: Creating {days_ahead}-day future return target...")
    df = create_future_return_target(df, days_ahead=days_ahead, return_type='log')
    target_col = f'Future_Return_{days_ahead}d'
    
    # Drop rows where target is NaN (end of dataset)
    before_target_drop = len(df)
    df = df[df[target_col].notna()].copy()
    after_target_drop = len(df)
    logger.info(f"   Dropped {before_target_drop - after_target_drop} rows with missing targets")
    logger.info(f"   Final dataset: {after_target_drop} rows")
    
    # ========== STEP 9: DEFINE FEATURES ==========
    logger.info("🔧 STEP 9: Defining feature set...")
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', target_col]]
    logger.info(f"   Features: {len(feature_cols)} columns")
    logger.info(f"   Features: {feature_cols[:5]}... (showing first 5)")
    logger.info(f"   Target: {target_col}")
    
    target_col_idx = feature_cols.index('Log_Return') if 'Log_Return' in feature_cols else 0
    
    # ========== STEP 10: TIME-BASED SPLIT (NO SHUFFLE) ==========
    logger.info("✂️  STEP 10: Performing time-based train/test split...")
    train_df, test_df = train_test_time_split(df, test_size=settings.TEST_SIZE)
    
    # ========== STEP 11: FIT SCALER ON TRAIN ONLY ==========
    logger.info("⚖️  STEP 11: Fitting scaler on TRAINING data only...")
    scaler = StockScaler(scaler_type='standard')
    train_df_scaled = scaler.fit_transform(train_df, feature_cols)
    logger.info(f"   Scaler fitted on {len(train_df)} training samples ✓")
    
    # ========== STEP 12: TRANSFORM BOTH TRAIN AND TEST ==========
    logger.info("🔄 STEP 12: Transforming train and test data...")
    test_df_scaled = scaler.transform(test_df)
    logger.info(f"   Train data: {train_df_scaled.shape}")
    logger.info(f"   Test data:  {test_df_scaled.shape}")
    
    # Extract feature matrix
    X_train_data = train_df_scaled[feature_cols].values
    X_test_data = test_df_scaled[feature_cols].values
    
    # ========== STEP 13: CREATE SEQUENCES ==========
    logger.info("🔗 STEP 13: Creating training sequences...")
    X_train, y_train = create_sequences_v2(
        X_train_data,
        settings.SEQ_LENGTH,
        settings.FORECAST_HORIZON,
        target_col_idx,
        name="training"
    )
    
    logger.info("🔗 Creating test sequences...")
    X_test, y_test = create_sequences_v2(
        X_test_data,
        settings.SEQ_LENGTH,
        settings.FORECAST_HORIZON,
        target_col_idx,
        name="test"
    )
    
    # ========== STEP 14: VALIDATE SEQUENCES ==========
    logger.info("\n✅ STEP 14: Validating sequence integrity...")
    
    # Reshape y to (samples, horizon, 1) for model training
    y_train = y_train[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    
    is_valid_train, msg_train = validate_sequences(
        X_train, y_train, settings.SEQ_LENGTH, settings.FORECAST_HORIZON
    )
    
    if not is_valid_train:
        logger.error("❌ Training sequences failed validation!")
        return
    
    logger.info("✓ Training sequences valid!")
    
    is_valid_test, msg_test = validate_sequences(
        X_test, y_test, settings.SEQ_LENGTH, settings.FORECAST_HORIZON
    )
    
    if not is_valid_test:
        logger.error("❌ Test sequences failed validation!")
        return
    
    logger.info("✓ Test sequences valid!")
    
    # ========== STEP 15: LOG DATA STATISTICS ==========
    logger.info("\n📈 STEP 15: Data statistics...")
    log_data_statistics(train_df, feature_cols, target_col, prefix="TRAINING DATA")
    
    # ========== STEP 16: CONVERT TO TENSORS ==========
    logger.info("🔥 STEP 16: Converting to PyTorch tensors...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    logger.info(f"   Device: {device}")
    logger.info(f"   X_train: {X_train_t.shape}")
    logger.info(f"   y_train: {y_train_t.shape}")
    
    # ========== STEP 17: CREATE DATALOADERS ==========
    logger.info("📦 STEP 17: Creating data loaders...")
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)
    
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    # ========== STEP 18: INITIALIZE MODEL ==========
    logger.info("🚀 STEP 18: Initializing LSTM model...")
    input_dim = X_train.shape[2]
    output_dim = 1
    
    model = LSTMAttentionEnhanced(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=output_dim,
        dropout=0.3,
        forecast_horizon=settings.FORECAST_HORIZON
    ).to(device)
    
    logger.info(f"   Input dim: {input_dim}")
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    checkpoint_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")
    
    # ========== STEP 19: TRAINING LOOP ==========
    logger.info(f"\n{'='*70}")
    logger.info(f"🎓 STEP 19: TRAINING on {device}")
    logger.info(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(settings.EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        logger.info(f"Epoch {epoch+1:3d}/{settings.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"              → New best model saved! (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step(val_loss)
    
    # ========== FINAL RESULTS ==========
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ TRAINING COMPLETED for {ticker}")
    logger.info(f"{'='*70}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {checkpoint_path}")
    
    # Save Scaler
    scaler.save(f"scaler_{ticker}.pkl")
    logger.info(f"Scaler saved")
    
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"  • Dataset: {len(df)} rows")
    logger.info(f"  • Features: {len(feature_cols)}")
    logger.info(f"  • Target: {target_col} ({days_ahead}-day future return)")
    logger.info(f"  • Training samples: {len(X_train)}")
    logger.info(f"  • Test samples: {len(X_test)}")
    logger.info(f"  • Lookback window: {settings.SEQ_LENGTH} days")
    logger.info(f"  • Forecast horizon: {settings.FORECAST_HORIZON} days")
    logger.info(f"\n{'='*70}\n")


if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(settings.DATA_DIR, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {settings.DATA_DIR}")
    
    for file_path in csv_files:
        train_pipeline(file_path)
