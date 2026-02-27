
import sys
import os
import yfinance as yf
# from textblob import TextBlob # Fallback or comparison
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# MOVED BACKEND IMPORT DOWN

def fetch_and_analyze_news():
    tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    
    print(f"{'Ticker':<15} | {'Sentiment':<10} | {'News Title'}", flush=True)
    print("-" * 120, flush=True)
    
    overall_results = {}
    
    # Pre-fetch news to isolate yfinance
    print("Pre-fetching news...", flush=True)
    news_cache = {}
    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...", flush=True)
            t = yf.Ticker(ticker)
            news_items = t.news
            news_cache[ticker] = news_items if news_items else []
            print(f"Fetched {len(news_cache[ticker])} items for {ticker}", flush=True)
        except Exception as e:
            print(f"Error pre-fetching {ticker}: {e}", flush=True)
            news_cache[ticker] = []

    print("Importing backend...", flush=True)
    from backend.features.sentiment import sentiment_analyzer
    # from backend.core.logging import logger
    print("Backend imported.", flush=True)

    with open("backend/scripts/final_results.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Ticker':<15} | {'Sentiment':<10} | {'News Title'}\n")
        f.write("-" * 120 + "\n")
        
        # Test sanity
        try:
            test_score = sentiment_analyzer.analyze("This is a great positive news.")
            f.write(f"DEBUG: Sanity Check Score: {test_score}\n")
        except Exception as e:
            f.write(f"DEBUG: Sanity Check Failed: {e}\n")

        for ticker in tickers:
            try:
                news = news_cache.get(ticker, [])
                
                if not news:
                    f.write(f"{ticker:<15} | {'N/A':<10} | No news found\n")
                    overall_results[ticker] = 0.0
                    continue
                    
                ticker_scores = []
                
                for article in news:
                    # Extract title from 'content' dictionary
                    content = article.get('content', {})
                    title = content.get('title', '') if isinstance(content, dict) else ''
                    
                    if not title: continue
                    
                    # Analyze with FinBERT
                    try:
                        score = sentiment_analyzer.analyze(title)
                        ticker_scores.append(score)
                        
                        # Truncate title for display
                        display_title = (title[:85] + '..') if len(title) > 85 else title
                        f.write(f"{ticker:<15} | {score:>.4f}      | {display_title}\n")
                    except Exception as e:
                         f.write(f"Error analyzing: {e}\n")

                avg_score = sum(ticker_scores) / len(ticker_scores) if ticker_scores else 0.0
                overall_results[ticker] = avg_score
                f.write("-" * 120 + "\n")

            except Exception as e:
                f.write(f"Error processing {ticker}: {e}\n")
                import traceback
                f.write(traceback.format_exc() + "\n")

        f.write("\nSummary of Average Sentiment:\n")
        for t, s in overall_results.items():
            f.write(f"{t:<15}: {s:.4f}\n")

if __name__ == "__main__":
    print("Script main entry point", flush=True)
    fetch_and_analyze_news()


