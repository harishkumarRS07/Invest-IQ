import sys
import os

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict
from backend.core.logging import logger


class PortfolioOptimizer:
    """
    Portfolio Optimization Engine.

    Strategy:
    1. Run unconstrained Mean-Variance Optimization (Maximum Sharpe Ratio).
    2. If MVO degenerates (one stock > 60%) — common with highly-correlated
       Indian blue-chips — fall back to Inverse-Volatility Weighting, which
       always gives accurate, naturally-balanced allocations.
    3. In either case enforce a 5% minimum so no bar disappears on the chart,
       then re-normalise so weights sum to exactly 1.
    """

    DEGENERACY_THRESHOLD = 0.60   # switch to inv-vol if any one stock > 60%
    MIN_WEIGHT           = 0.05   # minimum display weight (5%)

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    # ── Inverse-Volatility Weighting ──────────────────────────────────────────

    @staticmethod
    def _inverse_vol_weights(returns: pd.DataFrame) -> np.ndarray:
        """
        Allocate proportional to 1/volatility.
        Lower-volatility stocks get higher weight — robust for correlated assets.
        """
        vols = returns.std()
        vols = vols.replace(0, np.nan).dropna()
        inv_vol = 1.0 / vols
        weights = (inv_vol / inv_vol.sum()).values
        return weights.astype(float)

    # ── Minimum-floor enforcement ─────────────────────────────────────────────

    @staticmethod
    def _apply_floor(weights: np.ndarray, floor: float) -> np.ndarray:
        """
        Raise any weight below `floor` to `floor`.
        Excess is taken proportionally from stocks already above the floor.
        Re-normalises to sum = 1.
        """
        weights = weights.copy().astype(float)
        for _ in range(100):
            below = weights < floor
            if not below.any():
                break
            deficit = floor * int(below.sum()) - float(weights[below].sum())
            weights[below] = floor
            above = ~below
            above_sum = float(weights[above].sum())
            if above.any() and above_sum > 1e-9:
                reduction = deficit * (weights[above] / above_sum)
                weights[above] -= reduction
                weights[above] = np.clip(weights[above], floor, None)
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total > 1e-9:
            weights /= total
        return weights

    # ── Main optimize ─────────────────────────────────────────────────────────

    def optimize(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize portfolio weights.
        Returns a dict {ticker: weight} where all weights sum to 1 and
        each weight >= MIN_WEIGHT so every stock shows a real percentage.
        """
        returns    = price_data.pct_change().dropna()
        cov_matrix = returns.cov()
        num_assets = len(price_data.columns)
        tickers    = list(price_data.columns)

        if num_assets < 2:
            logger.warning("Portfolio optimization requires at least 2 assets.")
            eq = round(1.0 / max(num_assets, 1), 4)
            return {col: eq for col in tickers}

        # ── Step 1: unconstrained MVO ─────────────────────────────────────────
        def negative_sharpe(w):
            port_ret = float(np.dot(returns.mean(), w) * 252)
            port_vol = float(np.sqrt(w @ cov_matrix.values @ w) * np.sqrt(252))
            if port_vol < 1e-10:
                return 0.0
            return -(port_ret - self.risk_free_rate) / port_vol

        constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
        bounds      = tuple((0.0, 1.0) for _ in range(num_assets))
        init_guess  = np.full(num_assets, 1.0 / num_assets)

        mvo_weights = None
        try:
            res = minimize(
                negative_sharpe, init_guess,
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'ftol': 1e-12, 'maxiter': 2000},
            )
            if res.success:
                w = np.clip(res.x, 0.0, None)
                w /= w.sum()
                mvo_weights = w
        except Exception as e:
            logger.warning(f"MVO failed: {e}")

        # ── Step 2: degeneracy check ─────────────────────────────────────────
        use_inv_vol = True
        if mvo_weights is not None and float(mvo_weights.max()) <= self.DEGENERACY_THRESHOLD:
            use_inv_vol = False
            weights = mvo_weights
            logger.info("Using MVO weights (non-degenerate).")
        else:
            logger.info("MVO degenerate or failed -> using Inverse-Volatility weights.")

        if use_inv_vol:
            weights = self._inverse_vol_weights(returns)
            # Make sure columns match (inv_vol may drop zero-vol columns)
            if len(weights) != num_assets:
                weights = np.full(num_assets, 1.0 / num_assets)

        # ── Step 3: apply floor then renormalise ─────────────────────────────
        weights = self._apply_floor(weights, self.MIN_WEIGHT)

        allocation = {
            ticker: round(float(w), 4)
            for ticker, w in zip(tickers, weights)
        }
        logger.info(f"Final Portfolio Allocation: {allocation}")
        return allocation

    # ── Portfolio metrics ─────────────────────────────────────────────────────

    def get_portfolio_metrics(self, weights: List[float], returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate expected return, volatility, Sharpe ratio and max drawdown.
        """
        w = np.array(weights, dtype=float)
        cov = returns.cov().values

        portfolio_return     = float(np.dot(returns.mean(), w) * 252)
        portfolio_volatility = float(np.sqrt(w @ cov @ w) * np.sqrt(252))
        sharpe_ratio = (
            (portfolio_return - self.risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0 else 0.0
        )

        daily_ret    = pd.Series(returns.to_numpy() @ w, index=returns.index)
        cumulative   = (1 + daily_ret).cumprod()
        rolling_max  = cumulative.cummax()
        drawdowns    = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

        return {
            "expected_annual_return": round(portfolio_return, 6),
            "annual_volatility":      round(portfolio_volatility, 6),
            "sharpe_ratio":           round(sharpe_ratio, 4),
            "max_drawdown":           round(max_drawdown, 6),
        }
