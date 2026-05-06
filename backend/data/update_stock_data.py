import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

# Define the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'stock_data')

# Map CSV filenames to yfinance tickers
# Assuming .BO as per the headers in the CSV files
TICKERS_MAP = {
    'HDFCBANK.csv': 'HDFCBANK.BO',
    'ICICIBANK.csv': 'ICICIBANK.BO',
    'INFY.csv': 'INFY.BO',
    'RELIANCE.csv': 'RELIANCE.BO',
    'TCS.csv': 'TCS.BO'
}

def update_stock_data():
    """
    Updates the stock data CSVs with the latest daily data from yfinance.
    Calculates and appends the Cumulative VWAP.
    """
    print(f"Starting stock data update at {datetime.now()}")
    
    for filename, ticker in TICKERS_MAP.items():
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping.")
            continue
            
        print(f"Processing {ticker} ({filename})...")
        
        try:
            # Load existing data
            df = pd.read_csv(file_path)

            # Normalize legacy CSVs that include an extra ticker-header row
            # and coerce numeric columns to proper dtypes.
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).copy()
            for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Ensure required numeric fields are valid for cumulative VWAP math.
            df['Volume'] = df['Volume'].fillna(0.0)
            df['VWAP'] = df['VWAP'].ffill().fillna(0.0)
            
            if df.empty:
                print(f"  {filename} is empty.")
                continue
                
            # Get the last date
            last_date = df['Date'].iloc[-1]
            print(f"  Last data date: {last_date.date()}")
            
            # Check if update is needed (if last date is before today)
            # Note: yfinance start date is inclusive, so we use last_date + 1 day
            start_date = last_date + timedelta(days=1)
            end_date = datetime.now()
            
            if start_date.date() >= end_date.date():
                print("  Data is already up to date.")
                continue
                
            # Fetch new data
            print(f"  Fetching data from {start_date.date()} to {end_date.date()}...")
            new_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if new_data is None or new_data.empty:
                print(f"  No new data found for {ticker}.")
                continue
                
            print(f"  Found {len(new_data)} new rows.")
            
            # Calculate cumulative values for VWAP from existing data
            # VWAP = Total_PV / Total_Volume
            # Total_PV = Last_VWAP * Total_Cumulative_Volume
            current_total_volume = float(df['Volume'].sum())
            last_vwap = float(df['VWAP'].iloc[-1])
            current_total_pv = last_vwap * current_total_volume
            
            new_rows = []
            
            # Process new data
            for index, row in new_data.iterrows():
                # Handle yfinance dict/scalar return
                # ... (extraction logic)
                # To be safe against MultiIndex which yf returns now:
                try:
                    # convert row to float
                    op = float(row['Open'])
                    hi = float(row['High'])
                    lo = float(row['Low'])
                    cl = float(row['Close'])
                    vo = float(row['Volume'])
                except:
                    # If validation implies it is a series
                    op = float(row['Open'].iloc[0])
                    hi = float(row['High'].iloc[0])
                    lo = float(row['Low'].iloc[0])
                    cl = float(row['Close'].iloc[0])
                    vo = float(row['Volume'].iloc[0])
                    
                date_str = index.strftime('%Y-%m-%d') if isinstance(index, pd.Timestamp) else str(index)
                
                typ_p = (hi + lo + cl) / 3.0
                pv = typ_p * vo
                
                current_total_pv += pv
                current_total_volume += vo
                
                if current_total_volume == 0: vwap = 0
                else: vwap = current_total_pv / current_total_volume
                
                new_rows.append({
                    'Date': date_str,
                    'Open': op,
                    'High': hi,
                    'Low': lo,
                    'Close': cl,
                    'Volume': int(vo),
                    'VWAP': vwap
                })
            
            if new_rows:
                # We append directly to the file to avoid messing up the weird header
                # Or we can rewrite the file properly. 
                # Given the user might have other tools reading this, 
                # appending to the file is safest if we respect calculation.
                # But the file might have that 2nd line. 
                # If we rewrite, we might lose that 2nd line. 
                # Is that line important? It looks like column names again but with Ticker.
                # 'Date,Open,High,Low,Close,Volume,VWAP'
                # ',RELIANCE.BO,RELIANCE.BO,RELIANCE.BO,RELIANCE.BO,RELIANCE.BO,'
                # Just skipping it logic-wise is fine, but writing back...
                # I will Append in 'a' mode.
                
                with open(file_path, 'a') as f:
                    # Ensure we start on a new line if not present
                    # But CSV writer handles it.
                    # Just formatting new_df to csv string without header
                    
                    new_df = pd.DataFrame(new_rows)
                    # Ensure columns order
                    new_df = new_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]
                    
                    # Write to string
                    csv_data = new_df.to_csv(header=False, index=False, lineterminator='\n')
                    f.write(csv_data)
                    
                print(f"  Appended {len(new_rows)} rows.")
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")

def refined_update():
    """Compatibility wrapper used by scheduler and retrain orchestration."""
    update_stock_data()


if __name__ == "__main__":
    refined_update()
