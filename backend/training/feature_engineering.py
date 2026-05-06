"""Feature engineering utilities for fast binary XGBoost trading signals."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from training.sentiment import build_sentiment_time_features


def engineer_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build only high-impact, low-cost features requested for this strategy."""
    df_feat = df.copy()

    close = df_feat["Close"]
    volume = df_feat["Volume"]

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    df_feat["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df_feat["MACD"] = ema12 - ema26
    df_feat["MACD_Signal"] = df_feat["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20,2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df_feat["BB_High"] = sma20 + 2 * std20
    df_feat["BB_Low"] = sma20 - 2 * std20

    # Requested return and lag features
    daily_ret = close.pct_change()
    df_feat["Return_1D"] = daily_ret
    df_feat["Return_3D"] = close.pct_change(3)
    df_feat["Return_5D"] = close.pct_change(5)
    df_feat["Volume_Change"] = volume.pct_change()
    df_feat["Return_Lag_1"] = daily_ret.shift(1)
    df_feat["Return_Lag_2"] = daily_ret.shift(2)
    df_feat["Return_Lag_3"] = daily_ret.shift(3)
    df_feat["Volatility_10D"] = daily_ret.rolling(10).std()

    # FinBERT-ready placeholder features with short rolling context.
    if "Date" in df_feat.columns:
        sentiment_df = build_sentiment_time_features(
            stock=ticker,
            dates=pd.to_datetime(df_feat["Date"], errors="coerce"),
            use_finbert=False,
        )
        df_feat["sentiment_score"] = sentiment_df["sentiment_score"].to_numpy(dtype=float)
        df_feat["sentiment_avg_3d"] = sentiment_df["sentiment_avg_3d"].to_numpy(dtype=float)
        df_feat["sentiment_trend"] = sentiment_df["sentiment_trend"].to_numpy(dtype=float)
    else:
        df_feat["sentiment_score"] = 0.0
        df_feat["sentiment_avg_3d"] = 0.0
        df_feat["sentiment_trend"] = 0.0

    if "market_volatility_proxy" not in df_feat.columns:
        df_feat["market_volatility_proxy"] = daily_ret.rolling(20).std()
    if "nifty_return_1d" not in df_feat.columns:
        df_feat["nifty_return_1d"] = 0.0
    if "nifty_return_3d" not in df_feat.columns:
        df_feat["nifty_return_3d"] = 0.0
    if "sector_trend" not in df_feat.columns:
        df_feat["sector_trend"] = 0.0

    # Requested interaction features.
    df_feat["rsi_x_volume_change"] = df_feat["RSI"] * df_feat["Volume_Change"]
    df_feat["macd_x_volatility"] = df_feat["MACD"] * df_feat["market_volatility_proxy"]
    df_feat["return_3d_x_sentiment"] = df_feat["Return_3D"] * df_feat["sentiment_avg_3d"]

    return df_feat


def selected_feature_columns(df: pd.DataFrame) -> List[str]:
    requested = [
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_High",
        "BB_Low",
        "Market_Correlation",
        "nifty_return_1d",
        "nifty_return_3d",
        "market_volatility_proxy",
        "sector_trend",
        "Volatility_10D",
        "Return_1D",
        "Return_3D",
        "Return_5D",
        "Volume_Change",
        "Return_Lag_1",
        "Return_Lag_2",
        "Return_Lag_3",
        "sentiment_score",
        "sentiment_avg_3d",
        "sentiment_trend",
        "rsi_x_volume_change",
        "macd_x_volatility",
        "return_3d_x_sentiment",
    ]
    return [c for c in requested if c in df.columns]
