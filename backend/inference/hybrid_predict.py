"""Hybrid runtime inference adapter (XGBoost + optional LSTM + sentiment).

This module is designed for safe API wiring:
- Uses existing binary XGBoost artifacts when available
- Uses trained lightweight LSTM artifacts when available
- Falls back gracefully inside caller if artifacts are missing
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from backend.core.config import settings
from backend.core.logging import logger
from backend.features.indicators import add_market_correlation
from backend.features.external_data import ExternalDataSimulator
from backend.preprocessing.cleaning import clean_data, load_data
from backend.training.feature_engineering import engineer_features
from backend.training.lstm_model import LSTMDirectionalModel
from backend.training.sentiment import get_news_sentiment


class HybridPredictor:
    def __init__(self):
        self.xgb_model = None
        self.xgb_scaler = None
        self.feature_cols = None

        self.lstm_runtime: Optional[LSTMDirectionalModel] = None
        self._lstm_scaler = None

    def _model_paths(self, ticker: str) -> Dict[str, str]:
        model_dir = settings.MODEL_DIR
        return {
            "xgb": os.path.join(model_dir, f"xgboost_binary_{ticker}.pkl"),
            "xgb_scaler": os.path.join(model_dir, f"xgboost_binary_scaler_{ticker}.pkl"),
            "xgb_features": os.path.join(model_dir, f"xgboost_binary_features_{ticker}.pkl"),
            "hybrid_config": os.path.join(model_dir, f"hybrid_config_{ticker}.pkl"),
            "lstm": os.path.join(model_dir, f"lstm_binary_{ticker}.pth"),
            "lstm_scaler": os.path.join(model_dir, f"lstm_binary_scaler_{ticker}.pkl"),
        }

    def _load_artifacts(self, ticker: str) -> None:
        paths = self._model_paths(ticker)

        if not (os.path.exists(paths["xgb"]) and os.path.exists(paths["xgb_scaler"]) and os.path.exists(paths["xgb_features"])):
            raise FileNotFoundError(f"Missing binary XGBoost artifacts for {ticker}")

        self.xgb_model = joblib.load(paths["xgb"])
        self.xgb_scaler = joblib.load(paths["xgb_scaler"])
        self.feature_cols = joblib.load(paths["xgb_features"])
        self.hybrid_config = {
            "buy_threshold": 0.60,
            "sell_threshold": 0.40,
            "weights": {"xgb": 0.5, "lstm": 0.3, "sentiment": 0.2},
        }
        if os.path.exists(paths["hybrid_config"]):
            try:
                loaded = joblib.load(paths["hybrid_config"])
                if isinstance(loaded, dict):
                    self.hybrid_config.update(loaded)
            except Exception:
                logger.warning("Could not load hybrid config, using defaults")

        # Optional LSTM artifacts.
        self.lstm_runtime = None
        self._lstm_scaler = None
        if os.path.exists(paths["lstm"]) and os.path.exists(paths["lstm_scaler"]):
            state = torch.load(paths["lstm"], map_location="cpu")
            seq_length = int(state.get("seq_length", 15))
            hidden_dim = int(state.get("hidden_dim", 24))
            num_layers = int(state.get("num_layers", 1))

            lstm_runtime = LSTMDirectionalModel(
                seq_length=seq_length,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=1,
            )
            lstm_runtime.model.load_state_dict(state["model_state_dict"])
            lstm_runtime.model.eval()

            self.lstm_runtime = lstm_runtime
            self._lstm_scaler = joblib.load(paths["lstm_scaler"])

    @staticmethod
    def _risk_level(confidence: float) -> str:
        if confidence >= 0.70:
            return "Low"
        if confidence >= 0.55:
            return "Medium"
        return "High"

    @staticmethod
    def _weighted_score(xgb_prob: float, lstm_prob: float, sentiment: float, weights: Dict[str, float]) -> float:
        s_norm = float(np.clip((sentiment + 1.0) / 2.0, 0.0, 1.0))
        return float(
            (weights.get("xgb", 0.5) * xgb_prob)
            + (weights.get("lstm", 0.3) * lstm_prob)
            + (weights.get("sentiment", 0.2) * s_norm)
        )

    @staticmethod
    def _score_signal(score: float, buy_th: float, sell_th: float) -> str:
        if score > buy_th:
            return "BUY"
        if score < sell_th:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _signal_probabilities(signal: str, confidence: float) -> Dict[str, float]:
        c = float(min(max(confidence, 0.34), 0.95))
        rest = max(1.0 - c, 0.0)
        if signal == "BUY":
            return {"buy": round(c, 4), "hold": round(rest * 0.7, 4), "sell": round(rest * 0.3, 4)}
        if signal == "SELL":
            return {"buy": round(rest * 0.3, 4), "hold": round(rest * 0.7, 4), "sell": round(c, 4)}
        return {"buy": round(rest * 0.5, 4), "hold": round(c, 4), "sell": round(rest * 0.5, 4)}

    def _lstm_prob(self, close_series: np.ndarray) -> float:
        # No LSTM artifacts available => neutral.
        if self.lstm_runtime is None or self._lstm_scaler is None:
            return 0.5

        seq = self.lstm_runtime.seq_length
        if len(close_series) < seq + 1:
            return 0.5

        scaler = self._lstm_scaler
        close_scaled = scaler.transform(close_series.reshape(-1, 1)).reshape(-1)
        window = close_scaled[-seq:]

        x = torch.tensor(window.reshape(1, seq, 1), dtype=torch.float32)
        self.lstm_runtime.model.eval()
        with torch.no_grad():
            logit = self.lstm_runtime.model(x)
            prob = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
        return prob

    def predict(self, file_path: str, ticker: Optional[str] = None) -> Dict[str, Any]:
        ticker = ticker or os.path.basename(file_path).replace(".csv", "")
        self._load_artifacts(ticker)

        df = load_data(file_path)
        df = clean_data(df)

        try:
            market_df = ExternalDataSimulator.fetch_market_index(
                start_date=df["Date"].min() if "Date" in df.columns else None,
                end_date=df["Date"].max() if "Date" in df.columns else None,
            )
            if not market_df.empty:
                df = add_market_correlation(df, market_df)
        except Exception as exc:
            logger.warning(f"HybridPredictor market correlation skipped: {exc}")

        df = engineer_features(df, ticker)
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        if len(df) == 0:
            raise ValueError("No usable rows after preprocessing")

        if self.feature_cols is None:
            raise ValueError("Feature column metadata is missing for hybrid model")

        available_features = [f for f in self.feature_cols if f in df.columns]
        if not available_features:
            raise ValueError("No matching feature columns found for hybrid model")

        x_last = df.iloc[[-1]][available_features].to_numpy(dtype=float)
        x_last_scaled = self.xgb_scaler.transform(x_last)
        xgb_prob = float(self.xgb_model.predict_proba(x_last_scaled)[0, 1])

        close_vals = pd.to_numeric(df["Close"], errors="coerce").dropna().to_numpy(dtype=float)
        lstm_prob = self._lstm_prob(close_vals)

        date_for_sentiment = pd.Timestamp.now().to_pydatetime()
        if "Date" in df.columns and pd.notna(df.iloc[-1].get("Date")):
            date_for_sentiment = pd.Timestamp(df.iloc[-1]["Date"]).to_pydatetime()
        sentiment = int(get_news_sentiment(ticker, date_for_sentiment, use_finbert=False))

        score = self._weighted_score(
            xgb_prob=xgb_prob,
            lstm_prob=lstm_prob,
            sentiment=float(sentiment),
            weights=self.hybrid_config.get("weights", {}),
        )
        buy_th = float(self.hybrid_config.get("buy_threshold", 0.60))
        sell_th = float(self.hybrid_config.get("sell_threshold", 0.40))
        signal = self._score_signal(score=score, buy_th=buy_th, sell_th=sell_th)

        confidence = float(np.clip(abs(score - 0.5) * 2.0, 0.35, 0.95))
        current_price = float(df["Close"].iloc[-1])

        # Conservative one-day move estimate from model conviction.
        move = float(min(max(confidence * 0.01, 0.003), 0.02))
        if signal == "BUY":
            predicted_price = current_price * (1.0 + move)
        elif signal == "SELL":
            predicted_price = current_price * (1.0 - move)
        else:
            predicted_price = current_price

        # Simple 7-day flat-drift projection.
        seven_day = []
        price = current_price
        drift = move if signal == "BUY" else (-move if signal == "SELL" else 0.0)
        for _ in range(7):
            price = price * (1.0 + drift)
            seven_day.append(float(price))

        last_row = df.iloc[-1]
        indicators = {
            "rsi": float(last_row.get("RSI", 0.0)),
            "macd": float(last_row.get("MACD", 0.0)),
            "macd_signal": float(last_row.get("MACD_Signal", 0.0)),
            "sma_20": float(last_row.get("SMA_20", 0.0)),
            "sma_50": float(last_row.get("SMA_50", 0.0)),
            "bb_high": float(last_row.get("BB_High", 0.0)),
            "bb_low": float(last_row.get("BB_Low", 0.0)),
            "vwap": float(last_row.get("VWAP", 0.0)),
            "atr": float(last_row.get("ATR", 0.0)),
        }

        probs = self._signal_probabilities(signal, confidence)

        return {
            "current_price": current_price,
            "predicted_price": float(predicted_price),
            "7_day_forecast": seven_day,
            "signal": signal,
            "signal_confidence": confidence,
            "risk_level": self._risk_level(confidence),
            "reason": f"Hybrid weighted score={score:.3f} (xgb={xgb_prob:.3f}, lstm={lstm_prob:.3f}, sentiment={sentiment})",
            "indicators": indicators,
            "probabilities": probs,
            "xgb_prob": xgb_prob,
            "lstm_prob": lstm_prob,
            "sentiment_score": sentiment,
            "ensemble_score": score,
        }
