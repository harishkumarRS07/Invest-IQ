"""
Updated routes.py - adds Auth and enhanced Prediction endpoints
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
import os
import pandas as pd
from typing import Optional

from backend.app.schemas import (
    PredictionRequest, PredictionResponse, Indicators,
    TrainRequest, TrainResponse,
    SentimentRequest, SentimentResponse,
    PortfolioRequest, PortfolioResponse,
    RiskRequest, RiskResponse,
    RegisterRequest, LoginRequest, AuthResponse,
    BatchSignalRequest, BatchSignalResponse, SignalSummary,
)
from backend.app.auth import register_user, login_user, get_current_user
from backend.inference.predict import Predictor
from backend.training.train import train_pipeline
from backend.core.config import settings
from backend.core.logging import logger
from backend.core.exceptions import StockPredictorException

router = APIRouter()
predictor = Predictor()

# ─── Utilities ────────────────────────────────────────────────────────────────

def _require_auth(authorization: Optional[str]) -> dict:
    """Extract and validate Bearer token from the Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization[len("Bearer "):]
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

def _build_explanation(signal: str, pct_change: float, confidence: float, risk: str) -> str:
    direction = "rise" if pct_change > 0 else "fall"
    return (
        f"The AI model predicts the stock will {direction} by {abs(pct_change)*100:.2f}% "
        f"on the next trading day. Signal confidence is {confidence*100:.0f}%. "
        f"Risk level is {risk}. Signal: {signal}."
    )

def _extract_indicators(df: pd.DataFrame) -> Indicators:
    last = df.iloc[-1]
    def g(col):
        val = last.get(col)
        return float(val) if val is not None and pd.notna(val) else None
    return Indicators(
        rsi=g("RSI"),
        macd=g("MACD"),
        macd_signal=g("MACD_Signal"),
        sma_20=g("SMA_20"),
        sma_50=g("SMA_50"),
        bb_high=g("BB_High"),
        bb_low=g("BB_Low"),
        vwap=g("VWAP"),
        atr=g("ATR"),
    )

def _run_prediction(symbol: str, file_path: str = None):
    """Run prediction and return a standardized dict."""
    path = file_path or os.path.join(settings.DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        # Try with .NS suffix (Indian markets)
        alt = os.path.join(settings.DATA_DIR, f"{symbol}.NS.csv")
        if os.path.exists(alt):
            path = alt
        else:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    result = predictor.predict(path, ticker=symbol)
    return result

# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}

# ─── Auth ─────────────────────────────────────────────────────────────────────

@router.post("/auth/register", response_model=AuthResponse)
def register(request: RegisterRequest):
    result = register_user(request.email, request.password, request.name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/auth/login", response_model=AuthResponse)
def login(request: LoginRequest):
    result = login_user(request.email, request.password)
    if "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    return result

@router.get("/auth/me")
def get_me(authorization: Optional[str] = Header(None)):
    user = _require_auth(authorization)
    return {"email": user["sub"], "name": user.get("name")}

# ─── Prediction (Enhanced) ─────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    try:
        result = _run_prediction(request.symbol, request.file_path)
        pct = (result["predicted_price"] - result["current_price"]) / result["current_price"]
        explanation = _build_explanation(
            result["signal"], pct, result["signal_confidence"], result["risk_level"]
        )
        # Build confidence interval if missing
        ci = result.get("confidence_interval", (
            result["predicted_price"] * 0.97,
            result["predicted_price"] * 1.03,
        ))
        return PredictionResponse(
            symbol=request.symbol,
            current_price=result["current_price"],
            predicted_price=result["predicted_price"],
            seven_day_forecast=result.get("7_day_forecast", []),
            confidence_interval=ci,
            signal=result["signal"],
            signal_confidence=result["signal_confidence"],
            risk_level=result["risk_level"],
            indicators=result.get("_indicators", Indicators()),
            explanation=explanation,
        )
    except HTTPException:
        raise
    except StockPredictorException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ─── Batch Signals (Dashboard) ────────────────────────────────────────────────

import time as _time

_signals_cache: dict = {}       # symbol -> (timestamp, result_dict)
_CACHE_TTL = 600                # 10 minutes

@router.post("/signals/batch", response_model=BatchSignalResponse)
def batch_signals(request: BatchSignalRequest, authorization: Optional[str] = Header(None)):
    """Fetch trading signals for multiple tickers – used by the mobile Dashboard."""
    _require_auth(authorization)
    signals = []
    for symbol in request.symbols:
        try:
            now = _time.time()
            cached = _signals_cache.get(symbol)
            if cached and (now - cached[0]) < _CACHE_TTL:
                result = cached[1]
                logger.info(f"Cache hit for {symbol}")
            else:
                result = _run_prediction(symbol)
                _signals_cache[symbol] = (now, result)
                logger.info(f"Cache miss – ran prediction for {symbol}")

            current = result["current_price"]
            predicted = result["predicted_price"]
            pct = (predicted - current) / current
            explanation = _build_explanation(
                result["signal"], pct, result["signal_confidence"], result["risk_level"]
            )
            signals.append(SignalSummary(
                symbol=symbol,
                current_price=current,
                predicted_price=predicted,
                signal=result["signal"],
                signal_confidence=result["signal_confidence"],
                risk_level=result["risk_level"],
                pct_change=round(pct * 100, 2),
                indicators=result.get("_indicators", Indicators()),
                explanation=explanation,
            ))
        except Exception as e:
            logger.warning(f"Skipping {symbol}: {e}")
    return BatchSignalResponse(signals=signals)

# ─── Available Tickers ────────────────────────────────────────────────────────

@router.get("/tickers")
def list_tickers(authorization: Optional[str] = Header(None)):
    """List available stock tickers based on data files present."""
    _require_auth(authorization)
    data_dir = settings.DATA_DIR
    tickers = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                tickers.append(f.replace(".csv", ""))
    return {"tickers": sorted(tickers)}

# ─── Training ─────────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest, background_tasks: BackgroundTasks,
          authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="Training file not found")
    background_tasks.add_task(train_pipeline, request.file_path)
    return TrainResponse(status="accepted", message="Training started in background")

# ─── Sentiment ────────────────────────────────────────────────────────────────

from backend.features.sentiment import sentiment_analyzer
from backend.data.realtime import RealTimeDataFetcher
from backend.features.portfolio import PortfolioOptimizer
from backend.features.risk import RiskEngine

realtime_fetcher = RealTimeDataFetcher()
portfolio_optimizer = PortfolioOptimizer()

@router.post("/sentiment/analyze", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    try:
        if request.symbol:
            from backend.features.external_data import ExternalDataSimulator
            score = ExternalDataSimulator.fetch_live_sentiment(request.symbol)
        elif request.text:
            score = sentiment_analyzer.analyze(request.text)
        else:
            raise HTTPException(status_code=400, detail="Either text or symbol required")
        label = "Neutral"
        if score > 0.1: label = "Positive"
        if score < -0.1: label = "Negative"
        return SentimentResponse(symbol=request.symbol, sentiment_score=score, sentiment_label=label)
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Portfolio ────────────────────────────────────────────────────────────────

@router.post("/portfolio/optimize", response_model=PortfolioResponse)
def optimize_portfolio(request: PortfolioRequest, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    try:
        prices = pd.DataFrame()
        for symbol in request.symbols:
            df = realtime_fetcher.fetch_price_history(symbol, period=request.period)
            if not df.empty:
                prices[symbol] = df["Close"]
        if prices.empty:
            raise HTTPException(status_code=404, detail="No data found")
        allocation = portfolio_optimizer.optimize(prices)
        weights = [allocation.get(s, 0) for s in prices.columns]
        metrics = portfolio_optimizer.get_portfolio_metrics(weights, prices.pct_change().dropna())
        return PortfolioResponse(allocation=allocation, metrics=metrics)
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Risk ─────────────────────────────────────────────────────────────────────

@router.post("/risk/score", response_model=RiskResponse)
def get_risk_score(request: RiskRequest, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    try:
        df = realtime_fetcher.fetch_price_history(request.symbol, period="1y")
        if df.empty:
            raise HTTPException(status_code=404, detail="Symbol not found")
        metrics = RiskEngine.get_risk_profile(df)
        return RiskResponse(symbol=request.symbol, metrics=metrics)
    except Exception as e:
        logger.error(f"Risk error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
