"""Evaluation helpers for threshold tuning and trading metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from training.backtesting import backtest_signals
from training.model import BinaryXGBoostModel


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def evaluate_threshold_pair(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    next_day_returns: np.ndarray,
    up_threshold: float,
    down_threshold: float,
) -> Dict:
    signals, trade_mask = BinaryXGBoostModel.generate_trade_signals(prob_up, up_threshold, down_threshold)

    # Classification uses base 0.5 cutoff to stay consistent and comparable.
    y_pred = (prob_up >= 0.5).astype(np.int32)
    cls = compute_classification_metrics(y_true, y_pred)

    bt = backtest_signals(signals, next_day_returns)
    directional_accuracy = float(accuracy_score(y_true[trade_mask], y_pred[trade_mask])) if np.any(trade_mask) else 0.0

    return {
        "up_threshold": up_threshold,
        "down_threshold": down_threshold,
        "signals": signals,
        "trade_mask": trade_mask,
        "trade_coverage": float(np.mean(trade_mask)),
        "directional_accuracy": directional_accuracy,
        "accuracy": cls["accuracy"],
        "precision": cls["precision"],
        "recall": cls["recall"],
        "f1": cls["f1"],
        "confusion_matrix": cls["confusion_matrix"],
        "total_return": bt["total_return"],
        "win_rate": bt["win_rate"],
        "num_trades": bt["num_trades"],
        "avg_return_per_trade": bt["avg_return_per_trade"],
        "max_drawdown": bt["max_drawdown"],
        "equity_curve": bt["equity_curve"],
    }


def choose_best_threshold(
    evaluations: List[Dict],
    min_trades: int,
) -> Dict:
    """Pick highest win-rate config among those with reasonable trade count."""
    eligible = [e for e in evaluations if e["num_trades"] >= min_trades]
    if not eligible:
        # Fall back to most trades if all are too sparse.
        return sorted(evaluations, key=lambda x: x["num_trades"], reverse=True)[0]

    return sorted(
        eligible,
        key=lambda x: (x["win_rate"], x["num_trades"], x["total_return"]),
        reverse=True,
    )[0]


def tune_thresholds(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    next_day_returns: np.ndarray,
    threshold_pairs: Sequence[Tuple[float, float]],
    min_trade_coverage: float = 0.05,
) -> Tuple[Dict, List[Dict]]:
    evaluations = [
        evaluate_threshold_pair(y_true, prob_up, next_day_returns, up_t, down_t)
        for up_t, down_t in threshold_pairs
    ]
    min_trades = max(1, int(len(y_true) * min_trade_coverage))
    best = choose_best_threshold(evaluations, min_trades=min_trades)

    # Coverage fallback: if fixed threshold pairs create no trades,
    # auto-relax using probability quantiles for practical usability.
    if best.get("num_trades", 0) == 0:
        # Use confidence-margin quantile to force actionable coverage.
        # This selects roughly the top 10% most-confident predictions on each side.
        margins = np.abs(prob_up - 0.5)
        margin_cutoff = float(np.quantile(margins, 0.90))
        up_fallback = min(0.99, 0.5 + margin_cutoff)
        down_fallback = max(0.01, 0.5 - margin_cutoff)
        if down_fallback > up_fallback:
            up_fallback, down_fallback = 0.5, 0.5

        relaxed = evaluate_threshold_pair(
            y_true=y_true,
            prob_up=prob_up,
            next_day_returns=next_day_returns,
            up_threshold=up_fallback,
            down_threshold=down_fallback,
        )
        relaxed["auto_generated"] = True
        evaluations.append(relaxed)
        best = relaxed

    return best, evaluations


def evaluate_weighted_ensemble(
    y_true: np.ndarray,
    final_score: np.ndarray,
    next_day_returns: np.ndarray,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.40,
) -> Dict[str, Any]:
    """Evaluate weighted ensemble score with NO TRADE band."""
    score = np.asarray(final_score, dtype=float)
    y = np.asarray(y_true, dtype=np.int32)

    signals = np.full(len(score), "NO TRADE", dtype=object)
    signals[score > buy_threshold] = "UP"
    signals[score < sell_threshold] = "DOWN"
    trade_mask = signals != "NO TRADE"

    y_pred = (score >= 0.5).astype(np.int32)
    cls = compute_classification_metrics(y, y_pred)
    directional_accuracy = float(accuracy_score(y[trade_mask], y_pred[trade_mask])) if np.any(trade_mask) else 0.0

    bt = backtest_signals(signals, next_day_returns)

    return {
        "signals": signals,
        "trade_mask": trade_mask,
        "trade_coverage": float(np.mean(trade_mask)) if len(trade_mask) else 0.0,
        "directional_accuracy": directional_accuracy,
        "accuracy": cls["accuracy"],
        "precision": cls["precision"],
        "recall": cls["recall"],
        "f1": cls["f1"],
        "confusion_matrix": cls["confusion_matrix"],
        "total_return": bt["total_return"],
        "win_rate": bt["win_rate"],
        "num_trades": bt["num_trades"],
        "avg_return_per_trade": bt["avg_return_per_trade"],
        "max_drawdown": bt["max_drawdown"],
        "equity_curve": bt["equity_curve"],
    }
