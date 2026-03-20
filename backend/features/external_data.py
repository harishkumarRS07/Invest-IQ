import numpy as np
import pandas as pd
import yfinance as yf
import hashlib
from textblob import TextBlob
from backend.core.logging import logger
from typing import Optional, Union

class ExternalDataSimulator:
    """
    Handles external data sources (News Sentiment, Macroeconomics).
    Supports both Simulation (for training/backtesting) and Live Fetching (for inference).
    """
    
    @staticmethod
    def get_sentiment(ticker: str, date: Optional[pd.Timestamp] = None) -> float:
        """
        Simulate News Sentiment Score (-1.0 to 1.0).
        Used during training when historical news is hard to get.
        """
        return np.random.uniform(-1.0, 1.0)

    @staticmethod
    def _get_yf_ticker_string(ticker_symbol: str) -> str:
        """Helper to format string correctly."""
        if not ticker_symbol.endswith(".NS") and not ticker_symbol.endswith(".BO") and not ticker_symbol.startswith("^"):
             return f"{ticker_symbol}.NS"
        return ticker_symbol

    @staticmethod
    def fetch_live_news(ticker_symbol: str) -> list:
        """
        Fetch real-world news articles representing the current state of a stock ticker.
        Returns a list of structured dictionary articles natively decoded from Yahoo!
        """
        try:
            search_ticker = ExternalDataSimulator._get_yf_ticker_string(ticker_symbol)
            ticker_obj = yf.Ticker(search_ticker)
            news = ticker_obj.news
            
            if not news:
                logger.warning(f"No news found for {search_ticker}.")
                return []
                
            return news
            
        except Exception as e:
            logger.error(f"Failed to fetch live news for {ticker_symbol}: {e}")
            return []

    @staticmethod
    def fetch_live_sentiment(ticker_symbol: str) -> float:
        """
        Fetch REAL news sentiment using yfinance and FinBERT.
        Used during inference for real-time prediction.
        """
        try:
            logger.info(f"Fetching live sentiment for {ticker_symbol}...")
            
            news = ExternalDataSimulator.fetch_live_news(ticker_symbol)
            
            if not news:
                return 0.0
                
            # Collect all titles
            titles = []
            for article in news:
                content = article.get('content', {})
                if isinstance(content, dict):
                    title = content.get('title', '')
                    if title:
                        titles.append(title)
            
            if not titles:
                return 0.0
            
            # Use our improved SentimentAnalyzer
            from backend.features.sentiment import sentiment_analyzer
            avg_sentiment = sentiment_analyzer.analyze(titles)
            
            logger.info(f"Live Sentiment for {ticker_symbol}: {avg_sentiment:.4f}")
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"Failed to fetch live sentiment for {ticker_symbol}: {e}")
            return 0.0

    @staticmethod
    def get_macro_score(date: Optional[pd.Timestamp] = None) -> float:
        """
        Simulate Macroeconomic Health Score (0 to 100).
        0 = Recession, 100 = Boom.
        """
        # Macro data changes slowly. 
        # For simulation, we'll return a relatively stable random number 
        # or just a random number for robust model training demonstration.
        return np.random.uniform(40, 80)

    @staticmethod
    def add_external_features(df: pd.DataFrame, ticker: str, deterministic: bool = False) -> pd.DataFrame:
        """
        Enrich dataframe with simulated external features.
        """
        logger.info(f"Adding simulated external data for {ticker}...")
        
        # Generate synthetic data for the entire dataframe
        # Using numpy for speed
        n_rows = len(df)
        
        if deterministic:
            seed_source = f"{ticker}|external_features_v1"
            seed_bytes = hashlib.sha256(seed_source.encode("utf-8")).digest()[:8]
            seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
            rng = np.random.default_rng(seed)
            sentiments = rng.uniform(-1.0, 1.0, n_rows)
        else:
            # Randomized simulation for training robustness.
            sentiments = np.random.uniform(-1.0, 1.0, n_rows)
        
        # Macro: Slow moving random walk
        # Start at 60, move by small steps
        macro_scores = []
        current_macro = 60.0
        for _ in range(n_rows):
            if deterministic:
                change = rng.normal(0, 0.5)
            else:
                change = np.random.normal(0, 0.5)
            current_macro = np.clip(current_macro + change, 0, 100)
            macro_scores.append(current_macro)
            
        df = df.copy()
        df['Sentiment'] = sentiments
        df['Macro_Score'] = np.array(macro_scores)
        
        return df

    @staticmethod
    def fetch_market_index(
        ticker: str = "^NSEI",
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """
        Fetch market index data (e.g., NIFTY 50) for correlation analysis.
        """
        def _normalize_date(value: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
            if value is None:
                return None

            # Guard against integer index values like 0..N being passed as dates.
            if isinstance(value, (int, np.integer, float, np.floating)):
                return None

            try:
                parsed = pd.to_datetime(value, errors="coerce")
            except Exception:
                return None

            if pd.isna(parsed):
                return None
            return parsed

        def _fetch_index_history(index_ticker: str, start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> pd.DataFrame:
            index_obj = yf.Ticker(index_ticker)
            if start_ts is not None and end_ts is not None:
                return index_obj.history(start=start_ts, end=end_ts, interval="1d", auto_adjust=False)
            return index_obj.history(period="5y", interval="1d", auto_adjust=False)

        try:
            start_ts = _normalize_date(start_date)
            end_ts = _normalize_date(end_date)

            if start_ts is not None and end_ts is not None and start_ts > end_ts:
                start_ts, end_ts = end_ts, start_ts

            logger.info(f"Fetching market index data for {ticker}...")
            data = _fetch_index_history(ticker, start_ts, end_ts)

            if data is None or data.empty:
                fallback_ticker = "^BSESN" if ticker == "^NSEI" else None
                if fallback_ticker:
                    logger.warning(f"No data found for market index {ticker}. Trying fallback {fallback_ticker}.")
                    data = _fetch_index_history(fallback_ticker, start_ts, end_ts)

            if data is None or data.empty:
                logger.warning(f"No data found for market index {ticker}")
                return pd.DataFrame()

            return data
        except Exception as e:
            logger.error(f"Error fetching market index: {e}")
            return pd.DataFrame()
