"""
XGBoost Classification Pipeline for Stock Prediction (PHASE 2)

Improves upon standard XGBoost by:
1. Better label strategy (BUY/SELL/HOLD with thresholds)
2. Enhanced features (momentum, volume, trend, volatility)
3. Proper data cleaning
4. Confidence scores
5. Feature importance analysis
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Any
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from xgboost import XGBClassifier, plot_importance
import joblib
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators
from backend.utils.data_pipeline import train_test_time_split


class XGBoostClassificationPipeline:
    """Complete XGBoost classification pipeline for stock prediction."""
    
    def __init__(self, 
                 buy_threshold: float = 0.002,
                 sell_threshold: float = -0.002,
                 forecast_horizon: int = 3,
                 random_state: int = 42):
        """
        Initialize pipeline.
        
        Args:
            buy_threshold: Positive return threshold for BUY label (default: 0.2%)
            sell_threshold: Negative return threshold for SELL label (default: -0.2%)
            forecast_horizon: Days ahead to predict
            random_state: Random seed for reproducibility
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.forecast_horizon = forecast_horizon
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
        
    def create_better_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """
        Create BUY/SELL/HOLD labels based on future returns with ADAPTIVE thresholds.
        
        Strategy (ADAPTIVE):
        - Compute percentiles from actual returns (data-driven, not fixed)
        - BUY (2): future_return > 65th percentile
        - SELL (0): future_return < 35th percentile
        - HOLD (1): otherwise (middle 30%)
        
        This ensures balanced ~35% BUY, ~30% HOLD, ~35% SELL distribution
        
        Returns:
            Tuple of (labels array, future_returns series)
        """
        # Calculate future close price
        future_close = df['Close'].shift(-self.forecast_horizon)
        
        # Calculate future return
        future_returns = (future_close - df['Close']) / df['Close']
        
        # ADAPTIVE THRESHOLDS: Use percentiles instead of fixed values
        returns_clean = future_returns.dropna()
        
        # Log return statistics
        logger.info("\n" + "="*80)
        logger.info("RETURN DISTRIBUTION ANALYSIS (for adaptive thresholds)")
        logger.info("="*80)
        logger.info(f"  Mean return:    {returns_clean.mean():.6f} ({returns_clean.mean()*100:.4f}%)")
        logger.info(f"  Std deviation:  {returns_clean.std():.6f} ({returns_clean.std()*100:.4f}%)")
        logger.info(f"  Min return:     {returns_clean.min():.6f} ({returns_clean.min()*100:.4f}%)")
        logger.info(f"  Max return:     {returns_clean.max():.6f} ({returns_clean.max()*100:.4f}%)")
        logger.info(f"  Median return:  {returns_clean.median():.6f} ({returns_clean.median()*100:.4f}%)")
        
        # Compute percentile-based thresholds
        buy_threshold_adaptive = returns_clean.quantile(0.65)   # Top 35% (BUY)
        sell_threshold_adaptive = returns_clean.quantile(0.35)  # Bottom 35% (SELL)
        
        logger.info(f"\n  ADAPTIVE THRESHOLDS:")
        logger.info(f"    SELL threshold (35th percentile): {sell_threshold_adaptive:.6f} ({sell_threshold_adaptive*100:.4f}%)")
        logger.info(f"    BUY threshold  (65th percentile): {buy_threshold_adaptive:.6f} ({buy_threshold_adaptive*100:.4f}%)")
        logger.info("="*80)
        
        # Create labels with adaptive thresholds
        labels = np.ones(len(df), dtype=int)  # Default to HOLD (1)
        labels[future_returns > buy_threshold_adaptive] = 2   # BUY
        labels[future_returns < sell_threshold_adaptive] = 0   # SELL
        
        # Remove last forecast_horizon rows (no valid labels)
        labels = labels[:-self.forecast_horizon]
        
        return labels, future_returns[:-self.forecast_horizon]
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        df = df.copy()
        
        # Multi-day returns
        df['return_3d'] = df['Close'].pct_change(3)
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_7d'] = df['Close'].pct_change(7)
        
        # Momentum (rate of change)
        df['momentum_3d'] = df['Close'] - df['Close'].shift(3)
        df['momentum_5d'] = df['Close'] - df['Close'].shift(5)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()
        
        # Volume changes
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma_5'] = df['Volume'].rolling(5).mean()
        df['volume_ma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
        
        # Price-volume trend
        df['price_volume_trend'] = (df['volume_ratio'] * df['Close'].pct_change())
        
        return df
    
    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features."""
        df = df.copy()
        
        # SMA differences (if not already present)
        if 'SMA_20' not in df.columns:
            df['SMA_20'] = df['Close'].rolling(20).mean()
        if 'SMA_50' not in df.columns:
            df['SMA_50'] = df['Close'].rolling(50).mean()
        
        df['sma_diff'] = df['SMA_20'] - df['SMA_50']
        df['price_sma20_diff'] = df['Close'] - df['SMA_20']
        df['price_sma50_diff'] = df['Close'] - df['SMA_50']
        
        # Trend strength (ratio)
        df['sma_ratio'] = df['SMA_20'] / (df['SMA_50'] + 1e-8)
        
        # Golden cross indicators
        df['sma_20_above_50'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        df = df.copy()
        
        # Historical volatility
        df['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        df['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        df['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        
        # Price range
        df['high_low_diff'] = df['High'] - df['Low']
        df['high_low_ratio'] = df['high_low_diff'] / df['Close']
        
        # Bollinger Bands (if not already present)
        if 'BB_High' not in df.columns:
            sma = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            df['BB_High'] = sma + (2 * std)
            df['BB_Low'] = sma - (2 * std)
            df['BB_Mid'] = sma
        
        df['bb_position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'] + 1e-8)
        
        return df
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all enhanced features."""
        logger.info("Engineering features...")
        
        # Add technical indicators first (if not present)
        if 'RSI' not in df.columns:
            df = add_technical_indicators(df)
        
        # Add custom features
        df = self.add_momentum_features(df)
        df = self.add_volume_features(df)
        df = self.add_trend_features(df)
        df = self.add_volatility_features(df)
        
        logger.info(f"Total features created: {len(df.columns)}")
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for XGBoost."""
        # Define feature columns to use
        feature_cols = [
            # Technical indicators
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_High', 'BB_Low', 'BB_Mid', 'ATR', 'VWAP',
            
            # Momentum
            'return_3d', 'return_5d', 'return_7d', 'momentum_3d', 'momentum_5d',
            
            # Volume
            'volume_change', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'price_volume_trend',
            
            # Trend
            'sma_diff', 'price_sma20_diff', 'price_sma50_diff', 'sma_ratio', 'sma_20_above_50',
            
            # Volatility
            'volatility_5d', 'volatility_10d', 'volatility_20d', 'high_low_ratio', 'bb_position',
            
            # Other
            'Volume_Change', 'Log_Return', 'Rolling_Volatility'
        ]
        
        # Keep only features that exist in dataframe
        available_cols = [col for col in feature_cols if col in df.columns]
        
        logger.info(f"Selected {len(available_cols)} features for training")
        self.feature_names = available_cols
        
        return df[available_cols]
    
    def clean_features(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Clean features by removing NaN and infinite values.
        
        Returns:
            Clean X and y, aligned
        """
        logger.info(f"Cleaning data: {len(X)} samples before")
        
        # Drop rows with NaN
        valid_idx = ~(X.isna().any(axis=1))
        X_clean = X[valid_idx].copy()
        y_clean = y[valid_idx].copy()
        
        # Replace infinite values
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_idx = ~(X_clean.isna().any(axis=1))
        X_clean = X_clean[valid_idx].copy()
        y_clean = y_clean[valid_idx].copy()
        
        logger.info(f"Cleaning data: {len(X_clean)} samples after")
        
        return X_clean, y_clean
    
    def check_class_balance(self, y: np.ndarray):
        """Print class distribution with percentages."""
        unique, counts = np.unique(y, return_counts=True)
        logger.info("\n" + "="*80)
        logger.info("CLASS DISTRIBUTION (Labels created with thresholds adjusted)")
        logger.info("="*80)
        total = len(y)
        for label, count in zip(unique, counts):
            pct = 100.0 * count / total
            signal_name = self.label_mapping[label]
            logger.info(f"  {signal_name:6s}: {count:6d} samples ({pct:6.2f}%)")
        logger.info("="*80)
    
    def time_based_split(self, X: pd.DataFrame, y: np.ndarray, 
                         train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into train/test using time-based split (no shuffle).
        
        Args:
            X: Features
            y: Labels
            train_ratio: Ratio of data for training (default: 0.8)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * train_ratio)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"\nTime-based split (no shuffle):")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Test:  {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                    X_test: Optional[pd.DataFrame] = None,
                    y_test: Optional[np.ndarray] = None) -> None:
        """
        Train XGBoost classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional validation features
            y_test: Optional validation labels
        """
        logger.info("\nTraining XGBoost model...")
        
        # Initialize model with optimized parameters
        self.model = XGBClassifier(
            n_estimators=200,           # Number of boosting rounds
            max_depth=5,                # Maximum depth of tree
            learning_rate=0.05,         # Shrinkage (eta)
            subsample=0.8,              # Subsample ratio per tree
            colsample_bytree=0.8,       # Feature subsample ratio
            objective='multi:softprob', # Multi-class probability
            eval_metric='mlogloss',     # Multi-class log loss
            random_state=self.random_state,
            n_jobs=-1,                  # Use all processors
            verbosity=0
        )
        
        # Prepare eval set for early stopping
        fit_kwargs: Dict[str, Any] = {'verbose': False}
        
        if X_test is not None and y_test is not None:
            eval_set = [(X_test, y_test)]
            fit_kwargs['eval_set'] = eval_set
            
            # Try to use early stopping with callbacks (XGBoost 2.0+)
            try:
                from xgboost import EarlyStoppingCallback  # type: ignore
                fit_kwargs['callbacks'] = [EarlyStoppingCallback(rounds=20, save_best=True)]
                logger.info("Using EarlyStoppingCallback for XGBoost 2.0+")
            except (ImportError, AttributeError):
                # Fallback: For older XGBoost versions or if callback not available
                logger.info("EarlyStoppingCallback not available, training without early stopping")
        
        # Train
        self.model.fit(X_train, y_train, **fit_kwargs)
        
        logger.info("XGBoost model trained successfully")
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary with all metrics
        """
        logger.info("\nEvaluating model...")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, 
                                      target_names=['SELL', 'HOLD', 'BUY'],
                                      zero_division=0)
        
        # Log results
        logger.info(f"\nEvaluation Metrics (XGBoost):")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\nClassification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def get_confidence_scores(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with confidence scores.
        
        Returns:
            predictions, confidence_scores (0-1)
        """
        proba = self.model.predict_proba(X_test)
        predictions = self.model.predict(X_test)
        confidence = proba.max(axis=1)  # Max probability = confidence
        
        return predictions, confidence
    
    def generate_signals(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with confidence.
        
        Returns:
            DataFrame with columns: Signal, Confidence, Probabilities
        """
        predictions, confidence = self.get_confidence_scores(X_test)
        proba = self.model.predict_proba(X_test)
        
        signals = pd.DataFrame({
            'Signal': [self.label_mapping[p] for p in predictions],
            'Confidence': confidence,
            'Prob_SELL': proba[:, 0],
            'Prob_HOLD': proba[:, 1],
            'Prob_BUY': proba[:, 2]
        })
        
        return signals
    
    def plot_feature_importance(self, top_k: int = 20, save_path: Optional[str] = None):
        """
        Plot top K most important features.
        
        Args:
            top_k: Number of top features to plot
            save_path: Optional path to save figure
        """
        logger.info(f"\nPlotting top {top_k} feature importances...")
        
        plt.figure(figsize=(10, 8))
        plot_importance(self.model, max_num_features=top_k, importance_type='weight')
        plt.title(f'XGBoost Feature Importance (Top {top_k})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved: {save_path}")
        
        return plt
    
    def save_model(self, path: str):
        """Save trained model using joblib for full serialization."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved: {path}")
    
    def load_model(self, path: str):
        """Load trained model from joblib pickle."""
        import joblib
        self.model = joblib.load(path)
        logger.info(f"Model loaded: {path}")


def train_xgboost_classifier(ticker: str, 
                             file_path: str,
                             buy_threshold: float = 0.002,
                             sell_threshold: float = -0.002) -> Dict:
    """
    Complete XGBoost classification training pipeline.
    
    Args:
        ticker: Stock ticker symbol
        file_path: Path to CSV file
        buy_threshold: BUY signal threshold (default: 0.2%)
        sell_threshold: SELL signal threshold (default: -0.2%)
    
    Returns:
        Dictionary with training results and model
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"XGBoost Classification Pipeline - {ticker}")
    logger.info(f"{'='*80}")
    
    # Initialize pipeline
    pipeline = XGBoostClassificationPipeline(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        forecast_horizon=3
    )
    
    # Load and preprocess data
    logger.info(f"\nLoading data: {file_path}")
    df = load_data(file_path)
    df = clean_data(df)
    
    if len(df) < 100:
        logger.error(f"Insufficient data: {len(df)} rows")
        return {}
    
    logger.info(f"Data loaded: {len(df)} rows")
    
    # Add features
    df = pipeline.add_all_features(df)
    
    # Create labels
    y, returns = pipeline.create_better_labels(df)
    
    # Select features
    X = pipeline.select_features(df)
    
    # Align X and y
    X = X[:-pipeline.forecast_horizon]  # Remove last rows for alignment
    
    # Clean data
    X, y = pipeline.clean_features(X, y)
    
    if len(X) < 50:
        logger.error(f"Insufficient clean data: {len(X)} rows")
        return {}
    
    # Check class balance
    pipeline.check_class_balance(y)
    
    # Time-based split
    X_train, X_test, y_train, y_test = pipeline.time_based_split(X, y, train_ratio=0.8)
    
    # Train model
    pipeline.train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = pipeline.evaluate_model(X_test, y_test)
    
    # Generate signals
    signals = pipeline.generate_signals(X_test)
    logger.info(f"\nFirst 10 predictions:\n{signals.head(10)}")
    
    # Save model
    model_path = f"backend/models/saved_models/xgboost_classifier_{ticker}.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pipeline.save_model(model_path)
    
    # Plot feature importance
    plot_path = f"backend/models/saved_models/feature_importance_{ticker}.png"
    pipeline.plot_feature_importance(top_k=20, save_path=plot_path)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"XGBoost Classification Complete - {ticker}")
    logger.info(f"{'='*80}\n")
    
    return {
        'ticker': ticker,
        'model': pipeline.model,
        'pipeline': pipeline,
        'metrics': metrics,
        'signals': signals,
        'X_test': X_test,
        'y_test': y_test
    }


if __name__ == "__main__":
    # Example usage
    ticker = "STOCK"
    file_path = "backend/data/stock_data/STOCK.csv"
    
    results = train_xgboost_classifier(ticker, file_path)
    
    if results:
        logger.info("Training completed successfully!")
