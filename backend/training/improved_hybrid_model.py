"""Practical hybrid trading pipeline with walk-forward validation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from core.logging import logger
from training.data_loader import attach_market_context, load_stock_data
from training.evaluation import evaluate_weighted_ensemble
from training.feature_engineering import engineer_features, selected_feature_columns
from training.lstm_model import LSTMDirectionalModel
from training.model import BinaryXGBoostModel


class AdvancedFeatureEngineer:
    """Compatibility wrapper for feature engineering."""

    @staticmethod
    def engineer_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return engineer_features(df, ticker)


class SmartLabelEngineer:
    """Binary labels with stronger forward horizon and noise drop."""

    @staticmethod
    def create_binary_labels(
        df: pd.DataFrame,
        forecast_horizon: int = 5,
        up_threshold: float = 0.01,
        down_threshold: float = -0.01,
    ) -> Tuple[np.ndarray, pd.Series]:
        future_close = df["Close"].shift(-forecast_horizon)
        future_returns = (future_close - df["Close"]) / df["Close"]

        labels = np.full(len(df), -1, dtype=np.int32)
        labels[future_returns > up_threshold] = 1
        labels[future_returns < down_threshold] = 0

        return labels, future_returns


class HybridEnsembleModel:
    """Hybrid wrapper: XGBoost + lightweight LSTM + sentiment-aware weighted score."""

    def __init__(self, xgb_params: Optional[Dict[str, Any]] = None):
        self.binary_model = BinaryXGBoostModel(params=xgb_params)
        self.lstm_model = LSTMDirectionalModel(seq_length=15, hidden_dim=24, num_layers=1, epochs=8)
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        self.binary_model.fit(X_train, y_train, X_val, y_val)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.binary_model.predict_proba_up(X)

    @staticmethod
    def weighted_score(
        xgb_prob: float,
        lstm_prob: float,
        sentiment_score: float,
    ) -> float:
        sentiment_norm = float(np.clip((sentiment_score + 1.0) / 2.0, 0.0, 1.0))
        return float((0.5 * xgb_prob) + (0.3 * lstm_prob) + (0.2 * sentiment_norm))

    @staticmethod
    def score_to_signal(score: float, buy_threshold: float = 0.60, sell_threshold: float = 0.40) -> str:
        if score > buy_threshold:
            return "UP"
        if score < sell_threshold:
            return "DOWN"
        return "NO TRADE"


class ProductionTrainingPipeline:
    """End-to-end practical trading pipeline with walk-forward validation."""

    def __init__(self, ticker: str, seq_length: int = 20):
        self.ticker = ticker
        self.seq_length = seq_length  # Kept for backward compatibility only.
        self.experiment_dir = f"experiments/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.experiment_dir, exist_ok=True)

    def load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        _ = file_path  # Existing caller passes path; loader uses ticker + settings.
        logger.info("\n" + "=" * 80)
        logger.info(f"LOADING AND PREPROCESSING DATA FOR {self.ticker}")
        logger.info("=" * 80)

        df = load_stock_data(self.ticker)
        df = attach_market_context(df)
        df = AdvancedFeatureEngineer.engineer_features(df, self.ticker)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        logger.info(f"Data shape after preprocessing: {df.shape}")
        return df

    @staticmethod
    def _time_split(
        X: np.ndarray,
        y: np.ndarray,
        next_day_returns: np.ndarray,
        train_size: float = 0.7,
        val_size: float = 0.15,
    ) -> Tuple[np.ndarray, ...]:
        n = len(X)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))

        return (
            X[:train_end],
            y[:train_end],
            X[train_end:val_end],
            y[train_end:val_end],
            X[val_end:],
            y[val_end:],
            next_day_returns[val_end:],
        )

    @staticmethod
    def _walk_forward_slices(
        n_samples: int,
        min_train_frac: float = 0.55,
        val_frac: float = 0.15,
        test_frac: float = 0.10,
    ) -> List[Tuple[int, int, int, int]]:
        """Return (train_start, train_end, val_end, test_end) slices."""
        slices: List[Tuple[int, int, int, int]] = []
        if n_samples < 120:
            return slices

        train_end = int(n_samples * min_train_frac)
        val_size = max(20, int(n_samples * val_frac))
        test_size = max(20, int(n_samples * test_frac))

        while True:
            val_end = train_end + val_size
            test_end = val_end + test_size
            if test_end > n_samples:
                break
            slices.append((0, train_end, val_end, test_end))
            train_end += test_size

        return slices

    def _run_fold(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        next_ret: np.ndarray,
        close_clean: np.ndarray,
        sentiment_clean: np.ndarray,
        split: Tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        train_start, train_end, val_end, test_end = split

        X_train_df = X_df.iloc[train_start:train_end]
        X_val_df = X_df.iloc[train_end:val_end]
        X_test_df = X_df.iloc[val_end:test_end]
        y_train = y[train_start:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:test_end]
        next_ret_test = next_ret[val_end:test_end]

        close_train = close_clean[train_start:train_end]
        close_val = close_clean[train_end:val_end]
        close_test = close_clean[val_end:test_end]
        sentiment_test = sentiment_clean[val_end:test_end]

        model = HybridEnsembleModel()
        model.feature_names = list(X_df.columns)

        # Pass 1: all features to get importances.
        model.scaler.fit(X_train_df.to_numpy())
        X_train_s = model.scaler.transform(X_train_df.to_numpy())
        X_val_s = model.scaler.transform(X_val_df.to_numpy())
        X_test_s = model.scaler.transform(X_test_df.to_numpy())
        model.train_xgboost(X_train_s, y_train, X_val_s, y_val)

        # Pass 2: keep top 15-20 features and retrain.
        top_features = model.binary_model.top_feature_names(model.feature_names, top_k=18)
        if len(top_features) > 0 and len(top_features) < len(model.feature_names):
            model.feature_names = top_features
            X_train_df = X_train_df[top_features]
            X_val_df = X_val_df[top_features]
            X_test_df = X_test_df[top_features]

            model.scaler.fit(X_train_df.to_numpy())
            X_train_s = model.scaler.transform(X_train_df.to_numpy())
            X_val_s = model.scaler.transform(X_val_df.to_numpy())
            X_test_s = model.scaler.transform(X_test_df.to_numpy())
            model.train_xgboost(X_train_s, y_train, X_val_s, y_val)

        prob_up = model.predict_proba(X_test_s)

        # Train lightweight LSTM on close-price sequences.
        model.lstm_model.fit(close_train, y_train, close_val, y_val)
        lstm_prob_full = model.lstm_model.predict_proba_up(close_test)
        seq_offset = model.lstm_model.seq_length

        if len(lstm_prob_full) == 0:
            aligned_y = y_test
            aligned_prob_up = prob_up
            aligned_lstm_prob = np.full(len(prob_up), 0.5, dtype=float)
            aligned_next_ret = next_ret_test
            aligned_sentiment = sentiment_test
        else:
            aligned_y = y_test[seq_offset:]
            aligned_prob_up = prob_up[seq_offset:]
            aligned_lstm_prob = lstm_prob_full
            aligned_next_ret = next_ret_test[seq_offset:]
            aligned_sentiment = sentiment_test[seq_offset:]

        if len(aligned_y) == 0:
            return {
                "metrics": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "directional_accuracy": 0.0,
                    "trade_coverage": 0.0,
                    "win_rate": 0.0,
                    "total_return": 0.0,
                    "num_trades": 0,
                    "avg_return_per_trade": 0.0,
                    "max_drawdown": 0.0,
                    "confusion_matrix": np.array([[0, 0], [0, 0]]),
                    "equity_curve": np.array([1.0]),
                },
                "xgb_metrics": {
                    "accuracy": 0.0,
                    "win_rate": 0.0,
                    "total_return": 0.0,
                },
                "final_score": np.array([], dtype=float),
                "xgb_prob": np.array([], dtype=float),
                "lstm_prob": np.array([], dtype=float),
                "sentiment": np.array([], dtype=float),
                "y_true": np.array([], dtype=np.int32),
                "signals": np.array([], dtype=object),
                "model": model,
                "feature_names": model.feature_names,
            }

        final_score = np.array(
            [
                model.weighted_score(float(xp), float(lp), float(ss))
                for xp, lp, ss in zip(aligned_prob_up, aligned_lstm_prob, aligned_sentiment)
            ],
            dtype=float,
        )

        weighted_eval = evaluate_weighted_ensemble(
            y_true=aligned_y,
            final_score=final_score,
            next_day_returns=aligned_next_ret,
            buy_threshold=0.60,
            sell_threshold=0.40,
        )
        xgb_eval = evaluate_weighted_ensemble(
            y_true=aligned_y,
            final_score=aligned_prob_up,
            next_day_returns=aligned_next_ret,
            buy_threshold=0.60,
            sell_threshold=0.40,
        )

        return {
            "metrics": weighted_eval,
            "xgb_metrics": {
                "accuracy": float(xgb_eval["accuracy"]),
                "win_rate": float(xgb_eval["win_rate"]),
                "total_return": float(xgb_eval["total_return"]),
            },
            "final_score": final_score,
            "xgb_prob": aligned_prob_up,
            "lstm_prob": aligned_lstm_prob,
            "sentiment": aligned_sentiment,
            "y_true": aligned_y,
            "signals": weighted_eval["signals"],
            "model": model,
            "feature_names": model.feature_names,
        }

    def train_with_walk_forward_validation(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Dict[str, Any]:
        _ = train_size
        _ = val_size
        _ = test_size

        logger.info("\n" + "=" * 80)
        logger.info("TIME-BASED TRAIN/VAL/TEST SPLIT")
        logger.info("=" * 80)

        feature_cols = selected_feature_columns(df)
        if not feature_cols:
            raise ValueError("No selected feature columns found.")

        labels, future_returns = SmartLabelEngineer.create_binary_labels(df)
        next_day_returns = df["Close"].pct_change().shift(-1).to_numpy()

        X_df = df[feature_cols]
        valid_mask = (
            (labels != -1)
            & (~X_df.isna().any(axis=1).to_numpy())
            & (~future_returns.isna().to_numpy())
            & (~np.isnan(next_day_returns))
        )

        X_valid_df = X_df.loc[valid_mask].copy()
        y = labels[valid_mask]
        next_ret = next_day_returns[valid_mask]
        sentiment_col = "sentiment_avg_3d" if "sentiment_avg_3d" in df.columns else "sentiment_score"
        sentiment_series = pd.to_numeric(df[sentiment_col], errors="coerce").fillna(0.0)
        sentiment_clean = np.asarray(sentiment_series.loc[valid_mask], dtype=float)

        close_series = pd.to_numeric(pd.Series(df["Close"]), errors="coerce")
        close_clean = np.asarray(close_series.loc[valid_mask], dtype=float)

        splits = self._walk_forward_slices(len(X_valid_df))
        if not splits:
            # Graceful fallback for shorter history.
            n = len(X_valid_df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            if val_end + 1 >= n:
                raise ValueError("Insufficient data for walk-forward validation.")
            splits = [(0, train_end, val_end, n)]

        logger.info(f"Walk-forward folds: {len(splits)}")

        fold_results: List[Dict[str, Any]] = []
        for idx, split in enumerate(splits):
            logger.info(
                f"Fold {idx + 1}/{len(splits)}: "
                f"train=[{split[0]}:{split[1]}], val=[{split[1]}:{split[2]}], test=[{split[2]}:{split[3]}]"
            )
            fold_results.append(
                self._run_fold(
                    X_df=X_valid_df,
                    y=y,
                    next_ret=next_ret,
                    close_clean=close_clean,
                    sentiment_clean=sentiment_clean,
                    split=split,
                )
            )

        final_fold = fold_results[-1]
        model: HybridEnsembleModel = final_fold["model"]

        fold_metrics = [f["metrics"] for f in fold_results]
        acc_list = [float(m["accuracy"]) for m in fold_metrics]
        prec_list = [float(m["precision"]) for m in fold_metrics]
        rec_list = [float(m["recall"]) for m in fold_metrics]
        f1_list = [float(m["f1"]) for m in fold_metrics]
        da_list = [float(m["directional_accuracy"]) for m in fold_metrics]
        wr_list = [float(m["win_rate"]) for m in fold_metrics]
        ret_list = [float(m["total_return"]) for m in fold_metrics]
        trade_cov_list = [float(m["trade_coverage"]) for m in fold_metrics]

        final_score = np.asarray(final_fold["final_score"], dtype=float)
        aligned_prob_up = np.asarray(final_fold["xgb_prob"], dtype=float)
        aligned_lstm_prob = np.asarray(final_fold["lstm_prob"], dtype=float)
        aligned_sentiment = np.asarray(final_fold["sentiment"], dtype=float)
        aligned_y = np.asarray(final_fold["y_true"], dtype=np.int32)
        hybrid_signals = np.asarray(final_fold["signals"], dtype=object)

        if len(final_score) > 0:
            y_pred_aligned = (final_score >= 0.5).astype(np.int32)
            hybrid_trade_mask = hybrid_signals != "NO TRADE"
        else:
            y_pred_aligned = np.array([], dtype=np.int32)
            hybrid_trade_mask = np.array([], dtype=bool)

        try:
            roc_auc = float(roc_auc_score(aligned_y, final_score)) if len(final_score) > 0 else 0.5
        except ValueError:
            roc_auc = 0.5

        hybrid_confidence = np.asarray(np.abs(final_score - 0.5) * 2.0, dtype=float)
        structured_signals = [
            {
                "signal": str(sig),
                "confidence": float(conf),
                "xgb_prob": float(xp),
                "lstm_prob": float(lp),
                "sentiment": float(ss),
                "ensemble_score": float(es),
                "expected_direction": "UP" if es >= 0.5 else "DOWN",
                "trade_decision": bool(sig != "NO TRADE"),
            }
            for sig, conf, xp, lp, ss, es in zip(
                hybrid_signals,
                hybrid_confidence,
                aligned_prob_up,
                aligned_lstm_prob,
                aligned_sentiment,
                final_score,
            )
        ]

        threshold_table = [
            {
                "fold": i + 1,
                "up_threshold": 0.60,
                "down_threshold": 0.40,
                "accuracy": float(fr["metrics"]["accuracy"]),
                "win_rate": float(fr["metrics"]["win_rate"]),
                "num_trades": int(fr["metrics"]["num_trades"]),
                "trade_coverage": float(fr["metrics"]["trade_coverage"]),
                "total_return": float(fr["metrics"]["total_return"]),
                "directional_accuracy": float(fr["metrics"]["directional_accuracy"]),
            }
            for i, fr in enumerate(fold_results)
        ]

        xgb_acc = float(np.mean([f["xgb_metrics"]["accuracy"] for f in fold_results]))
        xgb_win = float(np.mean([f["xgb_metrics"]["win_rate"] for f in fold_results]))
        xgb_ret = float(np.mean([f["xgb_metrics"]["total_return"] for f in fold_results]))

        results: Dict[str, Any] = {
            "accuracy": float(np.mean(acc_list)),
            "precision": float(np.mean(prec_list)),
            "recall": float(np.mean(rec_list)),
            "f1": float(np.mean(f1_list)),
            "roc_auc": roc_auc,
            "directional_accuracy": float(np.mean(da_list)),
            "trade_coverage": float(np.mean(trade_cov_list)),
            "win_rate": float(np.mean(wr_list)),
            "total_return": float(np.mean(ret_list)),
            "num_trades": int(np.sum(hybrid_trade_mask)),
            "avg_return_per_trade": float(final_fold["metrics"]["avg_return_per_trade"]),
            "max_drawdown": float(final_fold["metrics"]["max_drawdown"]),
            "confusion_matrix": final_fold["metrics"]["confusion_matrix"],
            "predictions": y_pred_aligned,
            "confidence": final_score,
            "true_labels": aligned_y,
            "signals": hybrid_signals,
            "trade_decision": hybrid_trade_mask,
            "expected_direction": np.where(final_score >= 0.5, "UP", "DOWN"),
            "best_threshold_up": 0.60,
            "best_threshold_down": 0.40,
            "threshold_evaluations": threshold_table,
            "equity_curve": np.asarray(final_fold["metrics"]["equity_curve"], dtype=float),
            "xgb_probabilities": aligned_prob_up,
            "lstm_probabilities": aligned_lstm_prob,
            "sentiment_scores": aligned_sentiment,
            "structured_signals": structured_signals,
            "hybrid_win_rate": float(np.mean(wr_list)),
            "hybrid_total_return": float(np.mean(ret_list)),
            "hybrid_accuracy": float(np.mean(acc_list)),
            "hybrid_directional_accuracy": float(np.mean(da_list)),
            "xgb_vs_hybrid": {
                "xgb_accuracy": xgb_acc,
                "xgb_win_rate": xgb_win,
                "xgb_total_return": xgb_ret,
                "hybrid_accuracy": float(np.mean(acc_list)),
                "hybrid_win_rate": float(np.mean(wr_list)),
                "hybrid_total_return": float(np.mean(ret_list)),
            },
        }

        logger.info("\nSelected Threshold Pair:")
        logger.info(
            f"  up={results['best_threshold_up']:.2f}, down={results['best_threshold_down']:.2f} | "
            f"trades={results['num_trades']} | win_rate={results['win_rate']:.2%}"
        )

        logger.info("\nEvaluation Metrics:")
        logger.info(f"  Accuracy:             {results['accuracy']:.4f}")
        logger.info(f"  Precision:            {results['precision']:.4f}")
        logger.info(f"  Recall:               {results['recall']:.4f}")
        logger.info(f"  F1-Score:             {results['f1']:.4f}")
        logger.info(f"  Directional Accuracy: {results['directional_accuracy']:.4f}")
        logger.info(f"  Win Rate:             {results['win_rate']:.4f}")
        logger.info(f"  Total Return:         {results['total_return']:.4f}")
        logger.info(f"  Hybrid Win Rate:      {results['hybrid_win_rate']:.4f}")
        logger.info(f"  Hybrid Total Return:  {results['hybrid_total_return']:.4f}")
        logger.info(f"  Trade Coverage:       {results['trade_coverage']:.2%}")
        logger.info(f"  Confusion Matrix:\n{results['confusion_matrix']}")

        # Save model artifacts.
        model_dir = "backend/models/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model.binary_model.model, f"{model_dir}/xgboost_binary_{self.ticker}.pkl")
        joblib.dump(model.scaler, f"{model_dir}/xgboost_binary_scaler_{self.ticker}.pkl")
        joblib.dump(model.feature_names, f"{model_dir}/xgboost_binary_features_{self.ticker}.pkl")
        joblib.dump(
            {
                "buy_threshold": 0.60,
                "sell_threshold": 0.40,
                "weights": {"xgb": 0.5, "lstm": 0.3, "sentiment": 0.2},
            },
            f"{model_dir}/hybrid_config_{self.ticker}.pkl",
        )

        torch.save(
            {
                "model_state_dict": model.lstm_model.model.state_dict(),
                "seq_length": model.lstm_model.seq_length,
                "hidden_dim": model.lstm_model.hidden_dim,
                "num_layers": model.lstm_model.num_layers,
            },
            f"{model_dir}/lstm_binary_{self.ticker}.pth",
        )
        joblib.dump(model.lstm_model.scaler, f"{model_dir}/lstm_binary_scaler_{self.ticker}.pkl")

        return results


if __name__ == "__main__":
    ticker = "HDFCBANK"
    pipeline = ProductionTrainingPipeline(ticker)
    df = pipeline.load_and_preprocess(file_path="")
    out = pipeline.train_with_walk_forward_validation(df)

    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING COMPLETE FOR {ticker}")
    logger.info("=" * 80)
    logger.info(f"Accuracy:             {out['accuracy']:.2%}")
    logger.info(f"Directional Accuracy: {out['directional_accuracy']:.2%}")
    logger.info(f"Win Rate:             {out['win_rate']:.2%}")
    logger.info(f"Total Return:         {out['total_return']:.2%}")
