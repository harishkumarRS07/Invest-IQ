"""Simple next-day backtesting for UP/DOWN/NO TRADE signals."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def backtest_signals(signals: np.ndarray, next_day_returns: np.ndarray) -> Dict[str, Any]:
    """
    Simulate:
    - UP   => buy, PnL = +next_day_return
    - DOWN => sell, PnL = -next_day_return
    - NO TRADE => skip
    """
    trade_mask = signals != "NO TRADE"
    if not np.any(trade_mask):
        return {
            "total_return": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "avg_return_per_trade": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": np.array([1.0]),
            "trade_returns": np.array([]),
        }

    trade_signals = signals[trade_mask]
    trade_next_returns = next_day_returns[trade_mask]

    trade_returns = np.where(trade_signals == "UP", trade_next_returns, -trade_next_returns)
    equity_curve = np.cumprod(1.0 + trade_returns)

    wins = np.sum(trade_returns > 0)
    num_trades = len(trade_returns)
    win_rate = float(wins / num_trades) if num_trades > 0 else 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / np.maximum(running_max, 1e-12)
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) else 0.0

    return {
        "total_return": float(equity_curve[-1] - 1.0),
        "win_rate": win_rate,
        "num_trades": int(num_trades),
        "avg_return_per_trade": float(np.mean(trade_returns)) if num_trades > 0 else 0.0,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
        "trade_returns": trade_returns,
    }
