"""Sentiment utilities with FinBERT-ready interface and safe mock fallback."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

_FINBERT_PIPELINE = None
_FINBERT_LOAD_FAILED = False


def _get_finbert_pipeline():
    global _FINBERT_PIPELINE, _FINBERT_LOAD_FAILED
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE
    if _FINBERT_LOAD_FAILED:
        return None

    try:
        from transformers import pipeline  # type: ignore

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
    """Create daily sentiment, 3-day average, and short-term sentiment trend."""
    ts = pd.to_datetime(dates, errors="coerce")
    scores = [
        float(get_news_sentiment(stock, d.to_pydatetime(), use_finbert=use_finbert)) if pd.notna(d) else 0.0
        for d in ts
    ]
    s = pd.Series(scores, dtype=float)

    out = pd.DataFrame({
        "sentiment_score": s,
        "sentiment_avg_3d": s.rolling(3).mean(),
    })
    out["sentiment_trend"] = out["sentiment_avg_3d"].diff(2)

    # Keep sentiment bounded to stable range for downstream weighted ensemble.
    out["sentiment_score"] = np.clip(out["sentiment_score"], -1.0, 1.0)
    out["sentiment_avg_3d"] = np.clip(out["sentiment_avg_3d"], -1.0, 1.0)
    out["sentiment_trend"] = np.clip(out["sentiment_trend"], -1.0, 1.0)
    return out.fillna(0.0)
