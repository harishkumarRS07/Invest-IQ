import numpy as np
import pandas as pd
import sys
import os
import yfinance as yf
from typing import Optional, Union

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from core.logging import logger

class ExternalDataSimulator:
    """
    Handles external data sources (News Sentiment, Macroeconomics).
    
    PHASE 1 (Current): Only real features, no synthetic data
    - Remove random sentiment and macro scores
    - Focus on real market data and technical indicators
    - Sentiment/Macro will be added only when real data pipeline is established
    """
    
    @staticmethod
    def _get_yf_ticker_string(ticker_symbol: str) -> str:
        """Helper to format string correctly."""
        if not ticker_symbol.endswith(".NS") and not ticker_symbol.endswith(".BO") and not ticker_symbol.startswith("^"):
             return f"{ticker_symbol}.NS"
        return ticker_symbol

    @staticmethod
    def fetch_live_news(ticker_symbol: str) -> list:
        """
        Fetch real-world news articles for inference only.
        NOT used during training (to avoid data leakage).
        Returns a list of structured dictionary articles from Yahoo Finance.
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
        Used during inference ONLY, not during training.
        """
        try:
            logger.info(f"Fetching live sentiment for {ticker_symbol}...")
            
            news = ExternalDataSimulator.fetch_live_news(ticker_symbol)
            
            if not news:
                logger.warning(f"No news available for {ticker_symbol}, returning neutral sentiment")
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
                logger.warning(f"No titles found in news for {ticker_symbol}")
                return 0.0
            
            # Use real sentiment analyzer (optional - only if sentiment module is available)
            try:
                from backend.features.sentiment import sentiment_analyzer
                avg_sentiment = sentiment_analyzer.analyze(titles)
                logger.info(f"Live Sentiment for {ticker_symbol}: {avg_sentiment:.4f}")
                return avg_sentiment
            except ImportError:
                logger.warning("Sentiment analyzer not available, returning neutral sentiment")
                return 0.0
            
        except Exception as e:
            logger.error(f"Failed to fetch live sentiment for {ticker_symbol}: {e}")
            return 0.0

    @staticmethod
    def add_external_features(df: pd.DataFrame, ticker: str, use_real_data: bool = False) -> pd.DataFrame:
        """
        PHASE 1: DO NOT add external features during training.
        
        This method is intentionally simplified to prevent synthetic data injection.
        External features (sentiment, macro) will be added only during inference
        when real data is available and no leakage occurs.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock data
        ticker : str
            Stock ticker symbol
        use_real_data : bool
            If True, will attempt to fetch real sentiment data (for inference only)
            If False, returns dataframe unchanged (for training)
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with optional real features, NO synthetic data
        """
        logger.info(f"Skipping synthetic external features for {ticker} (PHASE 1: Real data only)")
        
        # During training, we don't add external features
        # This prevents the model from learning spurious correlations from random data
        # Features should be: Price, Volume, Technical Indicators only
        
        if use_real_data:
            # This would be used during inference only
            logger.info(f"Attempting to add REAL sentiment for {ticker} (inference only)...")
            sentiment = ExternalDataSimulator.fetch_live_sentiment(ticker)
            # Don't add historical sentiment during training
            # Only add to inference requests
            
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
