import sys
import os

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from backend.inference.predict import Predictor
from backend.data.realtime import RealTimeDataFetcher

def verify():
    print("="*60)
    print("      VERIFYING PREDICTION ACCURACY      ")
    print("="*60)

    tickers = ["HDFCBANK.NS", "RELIANCE.NS"]
    predictor = Predictor()
    fetcher = RealTimeDataFetcher()

    for t in tickers:
        print(f"\nAnalyzing {t}...")
        
        # Get latest price
        current_price = fetcher.get_current_price(t)
        print(f" -> Current Price: {current_price}")
        
        if not current_price:
            print(" -> Failed to fetch price.")
            continue
            
        # Create temp csv
        hist = fetcher.fetch_price_history(t, period="2y")
        if hist.empty:
            print(" -> Failed to fetch history.")
            continue
            
        temp_csv = f"verify_{t}.csv"
        hist.to_csv(temp_csv)
        
        try:
            result = predictor.predict(temp_csv, ticker=t)
            
            pred_price = result['predicted_price']
            confidence = result['signal_confidence']
            signal = result['signal']
            
            print(f" -> Predicted Price (Next Day): {pred_price:.2f}")
            print(f" -> Signal: {signal} (Confidence: {confidence:.1%})")
            print(f" -> Risk: {result['risk_level']}")
            print(f" -> Reason: {result['reason']}")
            
            diff = pred_price - current_price
            print(f" -> Difference: {diff:.2f} ({diff/current_price:.1%})")
            
        except Exception as e:
            print(f" -> Prediction Error: {e}")
            
        # Cleanup
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

if __name__ == "__main__":
    verify()
