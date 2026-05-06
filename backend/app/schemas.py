from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict, Any

# ─── Prediction ────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    symbol: str
    file_path: Optional[str] = None

class Indicators(BaseModel):
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    bb_high: Optional[float] = None
    bb_low: Optional[float] = None
    vwap: Optional[float] = None
    atr: Optional[float] = None

class Probabilities(BaseModel):
    sell: Optional[float] = None    # 0.0 – 1.0
    hold: Optional[float] = None    # 0.0 – 1.0
    buy: Optional[float] = None     # 0.0 – 1.0

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    seven_day_forecast: List[float] = []
    confidence_interval: Tuple[float, float]
    signal: str                 # BUY | SELL | HOLD
    signal_confidence: float    # 0.0 – 1.0
    risk_level: str             # Low | Medium | High
    indicators: Indicators = Indicators()
    probabilities: Optional[Probabilities] = None
    explanation: str = ""

# ─── Training ─────────────────────────────────────────────────
class TrainRequest(BaseModel):
    file_path: str

class TrainResponse(BaseModel):
    status: str
    message: str

# ─── Sentiment ────────────────────────────────────────────────
class SentimentRequest(BaseModel):
    text: Optional[str] = None
    symbol: Optional[str] = None

class SentimentResponse(BaseModel):
    symbol: Optional[str]
    sentiment_score: float
    sentiment_label: str

# ─── Portfolio ────────────────────────────────────────────────
class PortfolioRequest(BaseModel):
    symbols: List[str]
    period: str = "1y"

class PortfolioResponse(BaseModel):
    allocation: Dict[str, float]
    metrics: Dict[str, float]

# ─── Risk ─────────────────────────────────────────────────────
class RiskRequest(BaseModel):
    symbol: str

class RiskResponse(BaseModel):
    symbol: str
    metrics: Dict[str, float]

# ─── Auth ─────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str
    email: str
    name: str

class WatchlistUpdateRequest(BaseModel):
    symbols: List[str]

# ─── Batch Signals ─────────────────────────────────────────────
class BatchSignalRequest(BaseModel):
    symbols: List[str]

class SignalSummary(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    signal: str
    signal_confidence: float
    risk_level: str
    pct_change: float
    indicators: Indicators = Indicators()
    probabilities: Optional[Probabilities] = None
    explanation: str = ""

class BatchSignalResponse(BaseModel):
    signals: List[SignalSummary]

# ─── News ─────────────────────────────────────────────────────
class NewsArticle(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    publisher: Optional[str] = None
    link: Optional[str] = None
    providerPublishTime: Optional[Any] = None
    relatedTickers: Optional[List[str]] = []
    # Optionally store the computed sentiment for each article
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None

class NewsResponse(BaseModel):
    symbol: str
    articles: List[NewsArticle]
