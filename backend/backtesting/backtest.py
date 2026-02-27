import pandas as pd
import numpy as np
import os
import glob
import torch
from backend.core.config import settings
from backend.core.logging import logger
from backend.models.transformer import TimeSeriesTransformer
from backend.preprocessing.scaling import StockScaler
from backend.training.train import create_sequences
from backend.preprocessing.cleaning import load_data, clean_data
from backend.features.indicators import add_technical_indicators
from backend.features.external_data import ExternalDataSimulator

class Backtester:
    def __init__(self, ticker: str, initial_capital: float = 100000.0, transaction_cost: float = 0.001):
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.shares = 0
        self.portfolio_values = []
        self.transaction_cost = transaction_cost
        self.history = []
        
        # Load Resources
        self.model_path = os.path.join(settings.MODEL_DIR, f"transformer_{ticker}.pth")
        self.scaler_path = f"scaler_{ticker}.pkl"
        self.scaler = StockScaler()
        try:
            self.scaler.load(self.scaler_path)
            self.feature_columns = self.scaler.feature_columns
        except:
            logger.error(f"Scaler not found for {ticker}")
            self.feature_columns = []

    def load_model(self, input_dim):
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=64,
            nhead=settings.NHEAD,
            num_layers=settings.NUM_LAYERS,
            dropout=settings.DROPOUT,
            output_dim=1,
            forecast_horizon=settings.FORECAST_HORIZON
        )
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def run(self, df):
        if not self.feature_columns:
            return
            
        # Prepare Data
        # Assume df is fully preprocessed including indicators
        
        # Scale
        df_scaled = self.scaler.transform(df)
        data_scaled = df_scaled[self.feature_columns].values
        
        target_col = 'Log_Return' # We predict this
        target_idx = self.feature_columns.index(target_col)
        
        # Sequences
        X, y = create_sequences(data_scaled, settings.SEQ_LENGTH, settings.FORECAST_HORIZON, target_idx)
        
        if len(X) == 0:
            logger.warning("No data to backtest.")
            return

        model = self.load_model(input_dim=X.shape[2])
        
        # Run Inference
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            preds = model(X_tensor).numpy() # (N, 7, 1)
        
        # Strategy
        # We have predictions for t+1...t+7.
        # Simple Strategy: Look at t+1 prediction.
        # If > 2% (0.02 log return), Buy.
        # If < -2%, Sell.
        
        # We need actual prices to simulate trade execution
        # X corresponds to windows. The prediction made at index i (using data up to time i) 
        # is for time i+1.
        # The price we execute at is Close price at time i (or Open at i+1?).
        # Let's assume we trade at Close of time i (next day Open is more realistic but data is Close).
        
        # Align predictions with dates
        # sequences start at index 0 of data (after removing seq_length)
        # Sequence i uses data[i : i+seq_len]. Last data point is at i+seq_len-1.
        # Prediction is for i+seq_len.
        # We trade at i+seq_len-1 (Close).
        
        start_idx = settings.SEQ_LENGTH - 1
        # Loop strictly through predictions
        
        for i in range(len(preds)):
            pred_return = preds[i, 0, 0] # Predicted log return for next day
            
            current_idx = start_idx + i
            if current_idx >= len(df) - 1:
                break
                
            current_price = df['Close'].iloc[current_idx]
            date = df.index[current_idx]
            
            action = "HOLD"
            threshold = 0.02
            
            if pred_return > threshold:
                action = "BUY"
            elif pred_return < -threshold:
                action = "SELL"
                
            self.execute_trade(action, current_price, date)
            
            # Update Portfolio Value
            total_value = self.capital + (self.shares * current_price)
            self.portfolio_values.append({'Date': date, 'Value': total_value})
            
        self.calculate_metrics()
        
    def execute_trade(self, action, price, date):
        if action == "BUY":
            # Buy max possible
            cost = price * (1 + self.transaction_cost)
            max_shares = int(self.capital / cost)
            if max_shares > 0:
                self.capital -= max_shares * cost
                self.shares += max_shares
                self.history.append({'Date': date, 'Action': 'BUY', 'Price': price, 'Shares': max_shares})
        elif action == "SELL":
            # Sell all
            if self.shares > 0:
                revenue = (self.shares * price) * (1 - self.transaction_cost)
                self.capital += revenue
                self.history.append({'Date': date, 'Action': 'SELL', 'Price': price, 'Shares': self.shares})
                self.shares = 0

    def calculate_metrics(self):
        if not self.portfolio_values:
            return
            
        df_res = pd.DataFrame(self.portfolio_values)
        df_res['Returns'] = df_res['Value'].pct_change()
        
        total_return = (df_res['Value'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        sharpe = df_res['Returns'].mean() / df_res['Returns'].std() * np.sqrt(252) if df_res['Returns'].std() != 0 else 0
        
        # Max Drawdown
        cum_max = df_res['Value'].cummax()
        drawdown = (df_res['Value'] - cum_max) / cum_max
        max_drawdown = drawdown.min() * 100
        
        win_rate = 0 # Calculate based on closed trades
        # Simplified win rate: % of positive daily returns? Or trade based.
        # Let's use trade based
        trades = pd.DataFrame(self.history)
        # Match Buys and Sells... complicated.
        # Let's stick to portfolio metrics.
        
        logger.info(f"Backtest Results for {self.ticker}:")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        
        result_file = os.path.join(settings.MODEL_DIR, f"backtest_{self.ticker}.txt")
        with open(result_file, "w") as f:
            f.write(f"Backtest Results for {self.ticker}\n")
            f.write(f"Total Return: {total_return:.2f}%\n")
            f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
            f.write(f"Max Drawdown: {max_drawdown:.2f}%\n")

if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(settings.DATA_DIR, "*.csv"))
    for file_path in csv_files:
        ticker = os.path.basename(file_path).replace(".csv", "")
        
        # Load data (Same pipeline)
        try:
            df = load_data(file_path)
            df = clean_data(df)
            df = add_technical_indicators(df)
            # Need market data... fetch it
            market_df = ExternalDataSimulator.fetch_market_index(start_date=df.index[0], end_date=df.index[-1])
            from backend.features.indicators import add_market_correlation
            df = add_market_correlation(df, market_df)
            df = ExternalDataSimulator.add_external_features(df, ticker)
            df = df.dropna()
            
            bt = Backtester(ticker)
            bt.run(df)
        except Exception as e:
            logger.error(f"Backtest failed for {ticker}: {e}")
