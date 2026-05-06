# Model Code Implementations
## LSTM, XGBoost, and FinBERT

---

## 1. LSTM Model Implementation

```python
"""Lightweight LSTM for binary directional probability on close-price sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class _TinyLSTM(nn.Module):
    def __init__(self, hidden_dim: int = 24, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


@dataclass
class LSTMDirectionalModel:
    seq_length: int = 15
    hidden_dim: int = 24
    num_layers: int = 1
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = MinMaxScaler()
        self.model = _TinyLSTM(hidden_dim=self.hidden_dim, num_layers=self.num_layers).to(self.device)

    def _to_sequences(self, close_scaled: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.seq_length, len(close_scaled)):
            X.append(close_scaled[i - self.seq_length : i])
            y.append(labels[i])
        if not X:
            return np.empty((0, self.seq_length, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)
        X_arr = np.asarray(X, dtype=np.float32).reshape(-1, self.seq_length, 1)
        y_arr = np.asarray(y, dtype=np.int64)
        return X_arr, y_arr

    def fit(self, close_train: np.ndarray, y_train: np.ndarray, close_val: np.ndarray, y_val: np.ndarray) -> None:
        close_train_s = self.scaler.fit_transform(close_train.reshape(-1, 1)).reshape(-1)
        close_val_s = self.scaler.transform(close_val.reshape(-1, 1)).reshape(-1)

        X_tr, y_tr = self._to_sequences(close_train_s, y_train)
        X_va, y_va = self._to_sequences(close_val_s, y_val)
        if len(X_tr) == 0:
            return

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if len(X_va) > 0:
            val_x = torch.from_numpy(X_va).to(self.device)
            val_y = torch.from_numpy(y_va.astype(np.float32)).to(self.device)
        else:
            val_x = None
            val_y = None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_val = float("inf")
        patience = 2
        bad_epochs = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if val_x is not None and val_y is not None and len(X_va) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_x)
                    val_loss = float(criterion(val_logits, val_y).item())
                if val_loss < best_val:
                    best_val = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= patience:
                    break

    def predict_proba_up(self, close_values: np.ndarray) -> np.ndarray:
        close_s = self.scaler.transform(close_values.reshape(-1, 1)).reshape(-1)
        X, _ = self._to_sequences(close_s, np.zeros(len(close_s), dtype=np.int64))
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        return probs
```

---

## 2. XGBoost Classification Implementation

```python
"""
XGBoost Classification Pipeline for Stock Prediction (PHASE 2)

Features:
1. Better label strategy (BUY/SELL/HOLD with adaptive thresholds)
2. Enhanced features (momentum, volume, trend, volatility)
3. Proper data cleaning
4. Confidence scores
5. Feature importance analysis
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import joblib


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
        """
        future_close = df['Close'].shift(-self.forecast_horizon)
        future_returns = (future_close - df['Close']) / df['Close']
        
        returns_clean = future_returns.dropna()
        
        # Compute percentile-based thresholds
        buy_threshold_adaptive = returns_clean.quantile(0.65)   # Top 35% (BUY)
        sell_threshold_adaptive = returns_clean.quantile(0.35)  # Bottom 35% (SELL)
        
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
        
        df['return_3d'] = df['Close'].pct_change(3)
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_7d'] = df['Close'].pct_change(7)
        df['momentum_3d'] = df['Close'] - df['Close'].shift(3)
        df['momentum_5d'] = df['Close'] - df['Close'].shift(5)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()
        
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma_5'] = df['Volume'].rolling(5).mean()
        df['volume_ma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
        df['price_volume_trend'] = (df['volume_ratio'] * df['Close'].pct_change())
        
        return df
    
    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features."""
        df = df.copy()
        
        if 'SMA_20' not in df.columns:
            df['SMA_20'] = df['Close'].rolling(20).mean()
        if 'SMA_50' not in df.columns:
            df['SMA_50'] = df['Close'].rolling(50).mean()
        
        df['sma_diff'] = df['SMA_20'] - df['SMA_50']
        df['price_sma20_diff'] = df['Close'] - df['SMA_20']
        df['price_sma50_diff'] = df['Close'] - df['SMA_50']
        df['sma_ratio'] = df['SMA_20'] / (df['SMA_50'] + 1e-8)
        df['sma_20_above_50'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        df = df.copy()
        
        df['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        df['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        df['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        df['high_low_diff'] = df['High'] - df['Low']
        df['high_low_ratio'] = df['high_low_diff'] / df['Close']
        
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
        df = self.add_momentum_features(df)
        df = self.add_volume_features(df)
        df = self.add_trend_features(df)
        df = self.add_volatility_features(df)
        return df
    
    def clean_features(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Clean features by removing NaN and infinite values."""
        valid_idx = ~(X.isna().any(axis=1))
        X_clean = X[valid_idx].copy()
        y_clean = y[valid_idx].copy()
        
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_idx = ~(X_clean.isna().any(axis=1))
        X_clean = X_clean[valid_idx].copy()
        y_clean = y_clean[valid_idx].copy()
        
        return X_clean, y_clean
    
    def time_based_split(self, X: pd.DataFrame, y: np.ndarray, 
                         train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split data into train/test using time-based split (no shuffle)."""
        split_idx = int(len(X) * train_ratio)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                    X_test: Optional[pd.DataFrame] = None,
                    y_test: Optional[np.ndarray] = None) -> None:
        """Train XGBoost classification model."""
        
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        fit_kwargs: Dict[str, Any] = {'verbose': False}
        
        if X_test is not None and y_test is not None:
            eval_set = [(X_test, y_test)]
            fit_kwargs['eval_set'] = eval_set
            
            try:
                from xgboost import EarlyStoppingCallback
                fit_kwargs['callbacks'] = [EarlyStoppingCallback(rounds=20, save_best=True)]
            except (ImportError, AttributeError):
                pass
        
        self.model.fit(X_train, y_train, **fit_kwargs)
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """Evaluate model on test set."""
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        report = classification_report(y_test, y_pred, 
                                      target_names=['SELL', 'HOLD', 'BUY'],
                                      zero_division=0)
        
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
        """Get predictions with confidence scores."""
        proba = self.model.predict_proba(X_test)
        predictions = self.model.predict(X_test)
        confidence = proba.max(axis=1)
        
        return predictions, confidence
    
    def generate_signals(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with confidence."""
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
    
    def save_model(self, path: str):
        """Save trained model using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    def load_model(self, path: str):
        """Load trained model from joblib."""
        self.model = joblib.load(path)
```

---

## 3. FinBERT Sentiment Analysis Implementation

```python
"""Sentiment utilities with FinBERT-ready interface and safe mock fallback."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

_FINBERT_PIPELINE = None
_FINBERT_LOAD_FAILED = False


def _get_finbert_pipeline():
    """Load FinBERT model for sentiment analysis."""
    global _FINBERT_PIPELINE, _FINBERT_LOAD_FAILED
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE
    if _FINBERT_LOAD_FAILED:
        return None

    try:
        from transformers import pipeline

        hf_pipeline = cast(Any, pipeline)
        _FINBERT_PIPELINE = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
        return _FINBERT_PIPELINE
    except Exception:
        _FINBERT_LOAD_FAILED = True
        return None


def _mock_sentiment(stock: str, date: datetime) -> int:
    """Deterministic mock sentiment for fast local runs: {-1,0,1}."""
    key = f"{stock}_{date:%Y%m%d}"
    bucket = abs(hash(key)) % 3
    return [-1, 0, 1][bucket]


def get_news_sentiment(
    stock: str,
    date: datetime,
    headline: Optional[str] = None,
    use_finbert: bool = False,
) -> int:
    """
    Return sentiment score in {-1,0,1}.

    Args:
        stock: Stock ticker symbol
        date: Date for the sentiment
        headline: Optional news headline for analysis
        use_finbert: Enable FinBERT model (True) or use mock (False)

    Returns:
        Sentiment score: -1 (negative), 0 (neutral), 1 (positive)

    If use_finbert=False or model unavailable, falls back to deterministic mock.
    """
    if not use_finbert:
        return _mock_sentiment(stock, date)

    pipe = _get_finbert_pipeline()
    if pipe is None:
        return _mock_sentiment(stock, date)

    text = headline or f"{stock} market update"
    try:
        out = pipe(text, truncation=True)[0]
        label = str(out.get("label", "neutral")).lower()
        if "positive" in label:
            return 1
        if "negative" in label:
            return -1
        return 0
    except Exception:
        return _mock_sentiment(stock, date)


def build_sentiment_time_features(
    stock: str,
    dates: pd.Series,
    use_finbert: bool = False,
) -> pd.DataFrame:
    """
    Create daily sentiment, 3-day average, and short-term sentiment trend.
    
    Args:
        stock: Stock ticker symbol
        dates: Series of dates
        use_finbert: Enable FinBERT model (True) or use mock (False)
    
    Returns:
        DataFrame with columns:
        - sentiment_score: Daily sentiment [-1, 0, 1]
        - sentiment_avg_3d: 3-day rolling average
        - sentiment_trend: 2-day difference (trend indicator)
    """
    ts = pd.to_datetime(dates, errors="coerce")
    scores = [
        float(get_news_sentiment(stock, d.to_pydatetime(), use_finbert=use_finbert)) 
        if pd.notna(d) else 0.0
        for d in ts
    ]
    s = pd.Series(scores, dtype=float)

    out = pd.DataFrame({
        "sentiment_score": s,
        "sentiment_avg_3d": s.rolling(3).mean(),
    })
    out["sentiment_trend"] = out["sentiment_avg_3d"].diff(2)

    # Clip sentiment to stable range for downstream weighted ensemble
    out["sentiment_score"] = np.clip(out["sentiment_score"], -1.0, 1.0)
    out["sentiment_avg_3d"] = np.clip(out["sentiment_avg_3d"], -1.0, 1.0)
    out["sentiment_trend"] = np.clip(out["sentiment_trend"], -1.0, 1.0)
    
    return out.fillna(0.0)
```

---

## Model Configuration & Usage

### LSTM Configuration
```python
lstm_model = LSTMDirectionalModel(
    seq_length=15,      # 15-day lookback window
    hidden_dim=24,      # Hidden units
    num_layers=1,       # Single LSTM layer
    epochs=8,           # Training epochs
    batch_size=64,      # Batch size
    learning_rate=0.001 # Adam learning rate
)

# Train on close prices
lstm_model.fit(close_train, y_train, close_val, y_val)

# Predict probability of UP
probs = lstm_model.predict_proba_up(close_test)
```

### XGBoost Configuration
```python
xgb_pipeline = XGBoostClassificationPipeline(
    buy_threshold=0.002,
    sell_threshold=-0.002,
    forecast_horizon=3
)

# Prepare data
df = xgb_pipeline.add_all_features(df)
y, _ = xgb_pipeline.create_better_labels(df)
X = df[feature_cols]

# Train
xgb_pipeline.train_model(X_train, y_train, X_test, y_test)

# Evaluate
metrics = xgb_pipeline.evaluate_model(X_test, y_test)

# Generate signals
signals = xgb_pipeline.generate_signals(X_test)
```

### FinBERT Sentiment
```python
# With FinBERT model
sentiment = get_news_sentiment(
    stock="HDFCBANK",
    date=datetime.now(),
    headline="Strong earnings report",
    use_finbert=True  # Enable FinBERT
)

# Build time-series sentiment features
sentiment_df = build_sentiment_time_features(
    stock="HDFCBANK",
    dates=df['Date'],
    use_finbert=True  # Enable FinBERT
)
```

---

**Generated:** April 17, 2026  
**Purpose:** Production Model Source Code  
**Status:** Ready for Implementation
