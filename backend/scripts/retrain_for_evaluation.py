"""
Retrain All Models with Current Feature Pipeline
Ensures models match the current 21-feature set for accurate evaluation
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

sys.path.append(os.getcwd())

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator
from backend.models.enhanced_models import LSTMAttentionEnhanced
from backend.models.xgboost_fusion import XGBoostFusionModel
from backend.training.train import create_sequences
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

TICKERS = ['HDFCBANK', 'ICICIBANK', 'INFY', 'RELIANCE', 'TCS']

def train_lstm_model(ticker):
    """Train LSTM Attention model with correct architecture"""
    try:
        logger.info(f"Training LSTM for {ticker}...")
        
        # Load and preprocess data
        data_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        df = load_data(data_path)
        df = clean_data(df)
        
        # Feature engineering
        market_df = ExternalDataSimulator.fetch_market_index(
            start_date=df.index[0], 
            end_date=df.index[-1]
        )
        df = add_technical_indicators(df)
        df = add_market_correlation(df, market_df)
        df = ExternalDataSimulator.add_external_features(df, ticker)
        df = df.dropna()
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
        logger.info(f"Feature set for {ticker}: {len(feature_cols)} features - {feature_cols}")
        
        # Scale data
        scaler = StockScaler(scaler_type='standard')
        df_scaled = scaler.fit_transform(df, feature_cols)
        data_scaled = df_scaled[feature_cols].values
        
        # Create sequences (LSTM uses 1-day forecast)
        target_col_idx = feature_cols.index('Log_Return') if 'Log_Return' in feature_cols else 0
        X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, 1, target_col_idx)
        
        if X.shape[0] == 0:
            logger.warning(f"No sequences created for {ticker}")
            return False
        
        logger.info(f"LSTM {ticker} - X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device).reshape(-1, 1)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device).reshape(-1, 1)
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model with CORRECT architecture matching data
        input_dim = X.shape[2]  # This will be 21
        model = LSTMAttentionEnhanced(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            dropout=0.3,
            forecast_horizon=settings.FORECAST_HORIZON
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Training
        best_val_loss = float('inf')
        for epoch in range(settings.EPOCHS):
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
                for batch_X, batch_y in val_loader:
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM {ticker} Epoch {epoch+1}/50 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # Save model
        model_path = os.path.join(settings.MODEL_DIR, f"lstm_attention_{ticker}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"✓ LSTM model saved to {model_path}")
        
        # Save scaler
        scaler.save(f"scaler_{ticker}.pkl")
        logger.info(f"✓ Scaler saved for {ticker}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training LSTM for {ticker}: {e}", exc_info=True)
        return False

def train_transformer_model(ticker):
    """Train Transformer model with correct architecture"""
    try:
        logger.info(f"Training Transformer for {ticker}...")
        
        # Load and preprocess data
        data_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        df = load_data(data_path)
        df = clean_data(df)
        
        # Feature engineering
        market_df = ExternalDataSimulator.fetch_market_index(
            start_date=df.index[0], 
            end_date=df.index[-1]
        )
        df = add_technical_indicators(df)
        df = add_market_correlation(df, market_df)
        df = ExternalDataSimulator.add_external_features(df, ticker)
        df = df.dropna()
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
        logger.info(f"Feature set for {ticker}: {len(feature_cols)} features")
        
        # Scale data
        scaler = StockScaler(scaler_type='standard')
        df_scaled = scaler.fit_transform(df, feature_cols)
        data_scaled = df_scaled[feature_cols].values
        
        # Create sequences (Transformer uses 7-day forecast)
        target_col_idx = feature_cols.index('Log_Return') if 'Log_Return' in feature_cols else 0
        X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, settings.FORECAST_HORIZON, target_col_idx)
        
        if len(y.shape) == 2:
            y = y[..., np.newaxis]
        
        if X.shape[0] == 0:
            logger.warning(f"No sequences created for {ticker}")
            return False
        
        logger.info(f"Transformer {ticker} - X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model with CORRECT architecture (LSTM + FinBERT + XGBoost)
        input_dim = X.shape[2]  # This will be 21
        model = LSTMAttentionEnhanced(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            dropout=0.3,
            forecast_horizon=settings.FORECAST_HORIZON
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Training (use 50 epochs for faster evaluation prep)
        max_epochs = 50
        best_val_loss = float('inf')
        for epoch in range(max_epochs):
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
                for batch_X, batch_y in val_loader:
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM {ticker} Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # Save model
        model_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"✓ LSTM model saved to {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training LSTM for {ticker}: {e}", exc_info=True)
        return False

def train_xgboost_model(ticker):
    """Train XGBoost classification model"""
    try:
        logger.info(f"Training XGBoost for {ticker}...")
        
        # Load and preprocess data
        data_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        df = load_data(data_path)
        df = clean_data(df)
        
        # Feature engineering
        market_df = ExternalDataSimulator.fetch_market_index(
            start_date=df.index[0], 
            end_date=df.index[-1]
        )
        df = add_technical_indicators(df)
        df = add_market_correlation(df, market_df)
        df = ExternalDataSimulator.add_external_features(df, ticker)
        df = df.dropna()
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
        logger.info(f"Feature set for {ticker}: {len(feature_cols)} features")
        
        # Scale data
        scaler = StockScaler(scaler_type='standard')
        df_scaled = scaler.fit_transform(df, feature_cols)
        data_scaled = df_scaled[feature_cols].values
        
        # Prepare labels for classification
        xgb_model = XGBoostFusionModel()
        close_col = df_scaled[['Close']] if 'Close' in df_scaled.columns else df_scaled.iloc[:, :1]
        labels = xgb_model.prepare_labels(pd.DataFrame(close_col), horizon=5, threshold=0.01)
        
        # Use same features for training
        X = data_scaled[:-5]  # Remove last 5 rows to match label count
        
        if X.shape[0] == 0 or len(labels) == 0:
            logger.warning(f"No data for XGBoost training on {ticker}")
            return False
        
        # Ensure label count matches
        labels = labels[:X.shape[0]]
        
        logger.info(f"XGBoost {ticker} - X shape: {X.shape}, labels shape: {labels.shape}")
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = labels[:train_size]
        X_val = X[train_size:]
        y_val = labels[train_size:]
        
        # Convert to DataFrame for XGBoost
        X_train_df = pd.DataFrame(X_train, columns=feature_cols[:X_train.shape[1]])
        X_val_df = pd.DataFrame(X_val, columns=feature_cols[:X_val.shape[1]])
        
        # Train model with validation set for early stopping
        eval_set = [(X_val_df, y_val)]
        xgb_model.train(X_train_df, y_train, eval_set=eval_set)
        
        # Save model
        xgb_model.save(f"xgboost_fusion_{ticker}.pkl")
        logger.info(f"✓ XGBoost model saved for {ticker}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training XGBoost for {ticker}: {e}", exc_info=True)
        return False

def main():
    """Retrain all models"""
    print("\n" + "="*80)
    print("RETRAINING MODELS WITH CURRENT 21-FEATURE PIPELINE")
    print("="*80 + "\n")
    
    logger.info("Starting model retraining for evaluation...")
    
    results = {'LSTM': {}, 'Transformer': {}, 'XGBoost': {}}
    
    for ticker in TICKERS:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {ticker}")
        logger.info('='*80)
        
        # Train LSTM
        results['LSTM'][ticker] = train_lstm_model(ticker)
        
        # Train Transformer
        results['Transformer'][ticker] = train_transformer_model(ticker)
        
        # Train XGBoost
        results['XGBoost'][ticker] = train_xgboost_model(ticker)
        
        logger.info(f"✓ Completed {ticker}")
    
    print("\n" + "="*80)
    print("RETRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults:")
    print(f"  LSTM successes: {sum(1 for v in results['LSTM'].values() if v)}/{len(TICKERS)}")
    print(f"  Transformer successes: {sum(1 for v in results['Transformer'].values() if v)}/{len(TICKERS)}")
    print(f"  XGBoost successes: {sum(1 for v in results['XGBoost'].values() if v)}/{len(TICKERS)}")
    print("\nModels are now ready for evaluation with: python backend/scripts/comprehensive_model_evaluation.py")

if __name__ == "__main__":
    main()
