"""Binary XGBoost model wrapper with confidence-based signal generation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import xgboost as xgb


class BinaryXGBoostModel:
    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            "objective": "binary:logistic",
            "n_estimators": 400,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": 42,
            "early_stopping_rounds": 40,
        }
        self.model: Optional[xgb.XGBClassifier] = None

    @staticmethod
    def compute_scale_pos_weight(y_train: np.ndarray) -> float:
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        if pos == 0:
            return 1.0
        return float(neg / pos)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        params = dict(self.params)
        params["scale_pos_weight"] = self.compute_scale_pos_weight(y_train)
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained.")
        return self.model.predict_proba(X)[:, 1]

    def top_feature_names(self, feature_names: list[str], top_k: int = 18) -> list[str]:
        """Return top-k important feature names from fitted XGBoost model."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return feature_names[:top_k]

        importances = np.asarray(self.model.feature_importances_, dtype=float)
        if importances.size != len(feature_names):
            return feature_names[:top_k]

        ranked_idx = np.argsort(importances)[::-1]
        keep = [feature_names[int(i)] for i in ranked_idx[: max(1, min(top_k, len(feature_names)))]]
        return keep

    @staticmethod
    def generate_trade_signals(
        prob_up: np.ndarray,
        up_threshold: float,
        down_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return signal labels and trade decision mask."""
        signals = np.full(len(prob_up), "NO TRADE", dtype=object)
        signals[prob_up >= up_threshold] = "UP"
        signals[prob_up <= down_threshold] = "DOWN"
        trade_mask = signals != "NO TRADE"
        return signals, trade_mask
