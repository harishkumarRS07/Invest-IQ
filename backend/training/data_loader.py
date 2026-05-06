"""Data loading utilities for the practical trading pipeline."""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from core.config import settings
from core.logging import logger
from features.external_data import ExternalDataSimulator
from features.indicators import add_market_correlation
from preprocessing.cleaning import clean_data, load_data


def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load raw stock data and apply baseline cleaning."""
    file_path = f"{settings.DATA_DIR}/{ticker}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = load_data(file_path)
    df = clean_data(df, verbose=False)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def attach_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """Attach aligned market context features; continue safely on failure."""
    try:
        out = df.copy()
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
            out = out.dropna(subset=["Date"])
            out = out.sort_values("Date").reset_index(drop=True)

        market_df = ExternalDataSimulator.fetch_market_index(
            start_date=out["Date"].min() if "Date" in out.columns else None,
            end_date=out["Date"].max() if "Date" in out.columns else None,
        )
        if market_df.empty:
            return out

        market = market_df.copy()
        market = market.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
        if "Date" not in market.columns or "Close" not in market.columns:
            return out

        market["Date"] = pd.to_datetime(market["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
        market = market.dropna(subset=["Date", "Close"]).sort_values("Date")

        market["nifty_return_1d"] = pd.to_numeric(market["Close"], errors="coerce").pct_change(1)
        market["nifty_return_3d"] = pd.to_numeric(market["Close"], errors="coerce").pct_change(3)
        market["market_volatility_proxy"] = market["nifty_return_1d"].rolling(20).std()

        out = pd.merge(
            out,
            market[["Date", "nifty_return_1d", "nifty_return_3d", "market_volatility_proxy"]],
            on="Date",
            how="left",
        )

        out["sector_trend"] = out["Close"].pct_change(10) - out["nifty_return_1d"].rolling(10).sum()
        if "Log_Return" not in out.columns and "Close" in out.columns:
            out["Log_Return"] = np.log(pd.to_numeric(out["Close"], errors="coerce") / pd.to_numeric(out["Close"], errors="coerce").shift(1))
        out = add_market_correlation(out, market_df)

        for col in ["nifty_return_1d", "nifty_return_3d", "market_volatility_proxy", "sector_trend"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

        return out
    except Exception as exc:
        logger.warning(f"Could not add market correlation: {exc}")
    return df
