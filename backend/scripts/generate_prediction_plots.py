"""
Generate detailed prediction vs actual visualizations
Shows model predictions against actual values for visual assessment
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.append(os.getcwd())

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator
from backend.models.enhanced_models import LSTMAttentionEnhanced
from backend.training.train import create_sequences

# Set up matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

RESULTS_DIR = os.path.join(settings.MODEL_DIR, 'evaluation_results')
PREDICTION_DIR = os.path.join(RESULTS_DIR, 'prediction_visualizations')
os.makedirs(PREDICTION_DIR, exist_ok=True)

TICKERS = ['HDFCBANK', 'ICICIBANK', 'INFY', 'RELIANCE', 'TCS']

def plot_lstm_predictions(ticker):
    """Generate prediction vs actual plots for LSTM"""
    try:
        # Load data
        data_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(settings.DATA_DIR, f"{ticker}.NS.csv")
        
        if not os.path.exists(data_path):
            logger.warning(f"Data not found for {ticker}")
            return
        
        df = load_data(data_path)
        df = clean_data(df)
        
        # Add features
        market_df = ExternalDataSimulator.fetch_market_index(
            start_date=df.index[0], 
            end_date=df.index[-1]
        )
        df = add_technical_indicators(df)
        df = add_market_correlation(df, market_df)
        df = ExternalDataSimulator.add_external_features(df, ticker)
        df = df.dropna()
        
        # Scale
        scaler = StockScaler()
        try:
            scaler.load(f"scaler_{ticker}.pkl")
        except:
            logger.warning(f"Scaler not found for {ticker}")
            return
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
        df_scaled = scaler.transform(df)
        
        data_scaled = df_scaled[feature_cols].values
        target_col = 'Log_Return' if 'Log_Return' in feature_cols else 'Close'
        target_col_idx = feature_cols.index(target_col)
        
        X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, 1, target_col_idx)
        
        # Load model
        model_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")
        if not os.path.exists(model_path):
            logger.warning(f"LSTM model not found for {ticker}")
            return
        
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
        
        # Predictions
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            preds = model(X_tensor).numpy().flatten()
        
        y_actual = y.flatten()
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Full sequence
        ax1 = axes[0]
        time_steps = np.arange(len(y_actual))
        ax1.plot(time_steps, y_actual, label='Actual', linewidth=2, alpha=0.8, color='#2E86AB')
        ax1.plot(time_steps, preds, label='Predicted', linewidth=2, alpha=0.8, color='#A23B72', linestyle='--')
        ax1.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Log Returns', fontsize=12, fontweight='bold')
        ax1.set_title(f'LSTM Model: Predictions vs Actual - {ticker}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Zoomed in (last 100 points)
        ax2 = axes[1]
        start_idx = max(0, len(y_actual) - 100)
        time_zoom = np.arange(start_idx, len(y_actual))
        ax2.plot(time_zoom, y_actual[start_idx:], label='Actual', linewidth=2, alpha=0.8, color='#2E86AB', marker='o')
        ax2.plot(time_zoom, preds[start_idx:], label='Predicted', linewidth=2, alpha=0.8, color='#A23B72', marker='s', linestyle='--')
        ax2.set_xlabel('Time Steps (Last 100)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Log Returns', fontsize=12, fontweight='bold')
        ax2.set_title(f'LSTM Model: Recent Predictions (Zoomed)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PREDICTION_DIR, f'lstm_predictions_{ticker}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"LSTM prediction plot generated for {ticker}")
        
    except Exception as e:
        logger.error(f"Error generating LSTM plots for {ticker}: {e}")


def plot_transformer_predictions(ticker):
    """Generate prediction vs actual plots for Transformer"""
    try:
        # Load data - same as LSTM
        data_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(settings.DATA_DIR, f"{ticker}.NS.csv")
        
        if not os.path.exists(data_path):
            return
        
        df = load_data(data_path)
        df = clean_data(df)
        
        market_df = ExternalDataSimulator.fetch_market_index(
            start_date=df.index[0], 
            end_date=df.index[-1]
        )
        df = add_technical_indicators(df)
        df = add_market_correlation(df, market_df)
        df = ExternalDataSimulator.add_external_features(df, ticker)
        df = df.dropna()
        
        scaler = StockScaler()
        try:
            scaler.load(f"scaler_{ticker}.pkl")
        except:
            return
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
        df_scaled = scaler.transform(df)
        
        data_scaled = df_scaled[feature_cols].values
        target_col = 'Log_Return' if 'Log_Return' in feature_cols else 'Close'
        target_col_idx = feature_cols.index(target_col)
        
        X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, settings.FORECAST_HORIZON, target_col_idx)
        
        if len(y.shape) == 2:
            y = y[..., np.newaxis]
        
        # Load model
        model_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")
        if not os.path.exists(model_path):
            return
        
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
        
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            preds = model(X_tensor).numpy()
        
        # For 7-step forecast, plot step-1 predictions
        y_actual = y[:, 0, 0]  # First step actual
        preds_step1 = preds[:, 0, 0]  # First step predicted
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Full sequence
        ax1 = axes[0]
        time_steps = np.arange(len(y_actual))
        ax1.plot(time_steps, y_actual, label='Actual (Day+1)', linewidth=2, alpha=0.8, color='#2E86AB')
        ax1.plot(time_steps, preds_step1, label='Predicted (Day+1)', linewidth=2, alpha=0.8, color='#A23B72', linestyle='--')
        ax1.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Log Returns', fontsize=12, fontweight='bold')
        ax1.set_title(f'Transformer Model: Predictions vs Actual (7-Day Forecast) - {ticker}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Zoomed in
        ax2 = axes[1]
        start_idx = max(0, len(y_actual) - 100)
        time_zoom = np.arange(start_idx, len(y_actual))
        ax2.plot(time_zoom, y_actual[start_idx:], label='Actual (Day+1)', linewidth=2, alpha=0.8, color='#2E86AB', marker='o')
        ax2.plot(time_zoom, preds_step1[start_idx:], label='Predicted (Day+1)', linewidth=2, alpha=0.8, color='#A23B72', marker='s', linestyle='--')
        ax2.set_xlabel('Time Steps (Last 100)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Log Returns', fontsize=12, fontweight='bold')
        ax2.set_title(f'Transformer Model: Recent Predictions (Zoomed)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PREDICTION_DIR, f'transformer_predictions_{ticker}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Transformer prediction plot generated for {ticker}")
        
    except Exception as e:
        logger.error(f"Error generating Transformer plots for {ticker}: {e}")


def plot_residuals(ticker):
    """Plot residuals (prediction errors) for analysis"""
    try:
        # LSTM residuals
        data_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(settings.DATA_DIR, f"{ticker}.NS.csv")
        
        if not os.path.exists(data_path):
            return
        
        df = load_data(data_path)
        df = clean_data(df)
        
        market_df = ExternalDataSimulator.fetch_market_index(
            start_date=df.index[0], 
            end_date=df.index[-1]
        )
        df = add_technical_indicators(df)
        df = add_market_correlation(df, market_df)
        df = ExternalDataSimulator.add_external_features(df, ticker)
        df = df.dropna()
        
        scaler = StockScaler()
        try:
            scaler.load(f"scaler_{ticker}.pkl")
        except:
            return
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol']]
        df_scaled = scaler.transform(df)
        
        data_scaled = df_scaled[feature_cols].values
        target_col = 'Log_Return' if 'Log_Return' in feature_cols else 'Close'
        target_col_idx = feature_cols.index(target_col)
        
        X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, 1, target_col_idx)
        y_actual = y.flatten()
        
        # Get LSTM predictions
        model_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")
        if not os.path.exists(model_path):
            return
        
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
        
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            preds = model(X_tensor).numpy().flatten()
        
        # Calculate residuals
        residuals = y_actual - preds
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residual plot
        ax = axes[0, 0]
        ax.scatter(preds, residuals, alpha=0.5, s=20, color='#2E86AB')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax.set_title(f'{ticker}: Residual Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax = axes[0, 1]
        ax.hist(residuals, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Residual Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{ticker}: Distribution of Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Time series of residuals
        ax = axes[1, 0]
        time_steps = np.arange(len(residuals))
        ax.plot(time_steps, residuals, linewidth=1, alpha=0.7, color='#2E86AB')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.fill_between(time_steps, residuals, alpha=0.3, color='#A23B72')
        ax.set_xlabel('Time Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax.set_title(f'{ticker}: Residuals Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        ax = axes[1, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'{ticker}: Q-Q Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PREDICTION_DIR, f'residuals_{ticker}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Residual analysis plot generated for {ticker}")
        
    except Exception as e:
        logger.error(f"Error generating residual plots for {ticker}: {e}")


def generate_all_prediction_plots():
    """Generate all prediction visualization plots"""
    logger.info("Generating prediction vs actual plots...")
    
    for ticker in TICKERS:
        logger.info(f"Processing {ticker}...")
        plot_lstm_predictions(ticker)
        plot_transformer_predictions(ticker)
        plot_residuals(ticker)
    
    logger.info(f"All plots saved to {PREDICTION_DIR}")


if __name__ == "__main__":
    try:
        generate_all_prediction_plots()
        print(f"Prediction visualization complete! Results saved to: {PREDICTION_DIR}")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
