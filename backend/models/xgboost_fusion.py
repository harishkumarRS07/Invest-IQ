import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os
from backend.core.config import settings
from backend.core.logging import logger

class XGBoostFusionModel:
    def __init__(self):
        """
        Initialize XGBoost classifier with optimized hyperparameters.
        
        PHASE 2 UPDATES:
        - n_estimators: 500 → 200 (better generalization)
        - max_depth: 6 → 5 (prevent overfitting)
        - learning_rate: 0.05 (shrinkage)
        - subsample & colsample: 0.8 (regularization)
        - early_stopping_rounds: Increased to 20
        """
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,  # 0: Sell, 1: Hold, 2: Buy
            n_estimators=200,       # Updated: was 500
            max_depth=5,            # Updated: was 6
            learning_rate=0.05,     # Shrinkage
            subsample=0.8,          # Subsample per tree
            colsample_bytree=0.8,   # Feature subsample ratio
            reg_alpha=0.1,          # L1 Regularization
            reg_lambda=0.1,         # L2 Regularization
            eval_metric='mlogloss',
            early_stopping_rounds=20,  # Updated: was 10
            random_state=42,
            n_jobs=-1               # Use all processors
        )

    def prepare_labels(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.002):
        """
        Generate Buy/Sell/Hold labels based on future returns.
        Return = (Price[t+horizon] - Price[t]) / Price[t]
        Buy (2): Return > threshold (e.g., 0.2%)
        Sell (0): Return < -threshold (e.g., -0.2%)
        Hold (1): Otherwise
        
        UPDATED THRESHOLD: 0.01 (1%) -> 0.002 (0.2%)
        Reason: Stricter 1% threshold resulted in ALL HOLD labels
        Adjusted threshold creates realistic BUY/SELL/HOLD distribution
        """
        future_close = df['Close'].shift(-horizon)
        returns = (future_close - df['Close']) / df['Close']
        
        labels = np.ones(len(df), dtype=int)  # Default to HOLD (1)
        labels[returns > threshold] = 2  # Buy
        labels[returns < -threshold] = 0 # Sell
        
        # Drop last 'horizon' rows as they have no labels
        return labels[:-horizon]

    def train(self, X: pd.DataFrame, y: np.ndarray, eval_set=None):
        """Train the model with optional validation set for early stopping"""
        logger.info(f"Training XGBoost model with {len(X)} samples")
        
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y, verbose=False)
            
        logger.info("XGBoost training completed")

    def predict(self, X: pd.DataFrame):
        """Return class predictions"""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """Return class probabilities"""
        return self.model.predict_proba(X)

    def save(self, name: str = "xgboost_fusion.pkl"):
        path = os.path.join(settings.MODEL_DIR, name)
        joblib.dump(self.model, path)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, name: str = "xgboost_fusion.pkl"):
        path = os.path.join(settings.MODEL_DIR, name)
        self.model = joblib.load(path)
        logger.info(f"XGBoost model loaded from {path}")
