import sys
import os
import torch
import pandas as pd
import numpy as np
import os
from typing import Optional
from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator
from backend.features.realtime_price import fetch_latest_stock_data
from backend.models.transformer import TimeSeriesTransformer
from backend.core.exceptions import ModelNotTrainedException

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class Predictor:
    def __init__(self):
        self.scaler = StockScaler(scaler_type='standard') # Updated to standard
        self.model = None
        self.feature_cols = [] 

    def _load_model(self, ticker: str):
        try:
            # Paths
            model_path = os.path.join(settings.MODEL_DIR, f"transformer_{ticker}.pth")
            scaler_path = f"scaler_{ticker}.pkl"
            
            # Check existence
            if not os.path.exists(model_path):
                 # Try alternative
                alt_ticker = ticker.replace(".NS", "") if ".NS" in ticker else f"{ticker}.NS"
                alt_path = os.path.join(settings.MODEL_DIR, f"transformer_{alt_ticker}.pth")
                if os.path.exists(alt_path):
                    ticker = alt_ticker
                    model_path = alt_path
                    scaler_path = f"scaler_{ticker}.pkl"

            logger.info(f"Loading resources for {ticker}...")
            
            # Load Scaler
            try:
                self.scaler.load(scaler_path)
                self.feature_cols = self.scaler.feature_columns
            except Exception as e:
                logger.error(f"Scaler load failed: {e}")
                raise e

            # Load Model
            # We need input_dim from features
            input_dim = len(self.feature_cols)
            
            self.model = TimeSeriesTransformer(
                input_dim=input_dim,
                d_model=64,
                nhead=settings.NHEAD,
                num_layers=settings.NUM_LAYERS,
                dropout=settings.DROPOUT,
                output_dim=1,
                forecast_horizon=settings.FORECAST_HORIZON
            )
            
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model for {ticker}: {e}")
            self.model = None

    def predict(self, file_path: str, ticker: Optional[str] = None):
        if not ticker:
             ticker = os.path.basename(file_path).replace(".csv", "")
             
        self._load_model(ticker)
        
        if not self.model:
            raise ModelNotTrainedException(f"Model for {ticker} not found.")

        # Load and prep data
        df = load_data(file_path)
        
        # Real-time data injection
        try:
            live_df = fetch_latest_stock_data(ticker)
            if not live_df.empty:
                # Basic cleanup/merging logic (simplified)
                if 'Date' in df.columns:
                     df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                live_df['Date'] = pd.to_datetime(live_df['Date']).dt.tz_localize(None)
                
                last_date = df['Date'].iloc[-1]
                live_date = live_df['Date'].iloc[0]
                
                if live_date > last_date:
                    df = pd.concat([df, live_df], ignore_index=True)
                    logger.info(f"Added live data for {live_date}")
        except Exception as e:
            logger.warning(f"Live data fetch failed: {e}")

        # Feature Engineering Pipeline
        df = clean_data(df)
        market_df = ExternalDataSimulator.fetch_market_index(start_date=df.index[0], end_date=df.index[-1])
        df = add_technical_indicators(df)
        df = add_market_correlation(df, market_df)
        df = ExternalDataSimulator.add_external_features(df, ticker, deterministic=True)
        
        # Select features
        # Ensure we have enough data
        if len(df) < settings.SEQ_LENGTH:
            raise ValueError("Not enough data for prediction")
            
        # Get last sequence
        last_sequence_df = df.iloc[-settings.SEQ_LENGTH:]
        
        # Scale
        # Ensure columns match scaler
        # Missing columns handling?
        # For now assume pipelines match.
        try:
            df_scaled = self.scaler.transform(last_sequence_df)
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            # Likely new features vs old scaler.
            raise e
            
        seq_data = df_scaled[self.feature_cols].values
        
        # Inference
        input_tensor = torch.FloatTensor(seq_data).unsqueeze(0) # (1, seq, features)
        
        with torch.no_grad():
            preds = self.model(input_tensor).numpy() # (1, horizon, 1)
            
        # preds shape: (1, 7, 1)
        # We start with t+1 prediction
        pred_scaled = preds[0, :, 0] # (7,)
        
        # Inverse Transform
        # Log Return is target.
        # We need to construct a dummy row to inverse transform if scaler is multivariate.
        # But wait, did I create a separate scaler for target? No.
        # I need to inverse transform just the 'Log_Return' column.
        
        target_col = 'Log_Return'
        if target_col not in self.feature_cols:
             # Fallback
             target_col = 'Close'
             
        target_idx = self.feature_cols.index(target_col)
        
        # Inverse transform for each step
        pred_log_returns = []
        dummy_row = np.zeros((1, len(self.feature_cols)))
        
        for val in pred_scaled:
            dummy_row[0, target_idx] = val
            inv_val = self.scaler.scaler.inverse_transform(dummy_row)[0, target_idx]
            pred_log_returns.append(inv_val)
            
        # Current Price
        current_price = df['Close'].iloc[-1]
        
        # Convert Log Returns to Prices (Cumulative)
        # Price_t+k = Price_t * exp(sum(r_1...r_k))
        
        pred_prices = []
        cum_ret = 0
        for r in pred_log_returns:
            cum_ret += r
            price = current_price * np.exp(cum_ret)
            pred_prices.append(price)
            
        # Forecasts
        next_day_price = pred_prices[0]
        seven_day_price = pred_prices[-1]
        
        pct_change = (next_day_price - current_price) / current_price
        
        # Logic for Signal (Lowered threshold for realistic daily predictions)
        # Daily index movements are relatively small. A 0.5% signal is significant.
        if pct_change >= 0.005:
            signal = "BUY"
        elif pct_change <= -0.005:
            signal = "SELL"
        else:
            signal = "HOLD"
            
        # Confidence logic (Dynamic heuristics based on prediction magnitude)
        # A stronger directional push maps to higher signal confidence
        base_conf = 0.55
        magnitude_conf = min(abs(pct_change) * 15, 0.40) # cap bonus at 40%
        confidence = round(base_conf + magnitude_conf, 4)
        
        # Extract technical indicators from the last row
        last_row = df.iloc[-1]
        indicators = {
            "rsi": float(last_row.get('RSI', 0)),
            "macd": float(last_row.get('MACD', 0)),
            "macd_signal": float(last_row.get('MACD_Signal', 0)),
            "sma_20": float(last_row.get('SMA_20', 0)),
            "sma_50": float(last_row.get('SMA_50', 0)),
            "bb_high": float(last_row.get('BB_High', 0)),
            "bb_low": float(last_row.get('BB_Low', 0)),
            "vwap": float(last_row.get('VWAP', 0)),
            "atr": float(last_row.get('ATR', 0))
        }
        
        return {
            "current_price": current_price,
            "predicted_price": next_day_price,
            "7_day_forecast": pred_prices,
            "signal": signal,
            "signal_confidence": confidence,
            "risk_level": "Medium", # Placeholder or implement logic
            "reason": f"Model predicts {pct_change:.2%} return for next day.",
            "indicators": indicators
        }

if __name__ == "__main__":
    # Test
    csv_files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith('.csv')]
    if csv_files:
        p = Predictor()
        res = p.predict(os.path.join(settings.DATA_DIR, csv_files[0]))
        print(res)
