import yfinance as yf
import json

news_data = yf.Ticker('RELIANCE.NS').news

with open('test_news_utf8.json', 'w', encoding='utf-8') as f:
    json.dump(news_data, f, indent=2)
