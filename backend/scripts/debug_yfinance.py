
import yfinance as yf

def debug_news():
    ticker = 'RELIANCE.NS'
    t = yf.Ticker(ticker)
    news = t.news
    print(f"News for {ticker}: {len(news)} items")
    if news:
        print(news[0])
    else:
        print("No news found.")
        # Try without .NS
        t2 = yf.Ticker('RELIANCE')
        news2 = t2.news
        print(f"News for RELIANCE: {len(news2)} items")

if __name__ == "__main__":
    debug_news()
