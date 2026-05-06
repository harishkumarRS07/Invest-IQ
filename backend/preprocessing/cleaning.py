import pandas as pd
import numpy as np
import sys
import os

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from core.logging import logger
from core.exceptions import DataNotFoundException, PreprocessingException

def load_data(file_path: str) -> pd.DataFrame:
    """Load stock data from CSV"""
    try:
        # User's data often has a 2nd row with metadata (Date is NaN or empty there)
        # We read it all as strings first to inspect
        df = pd.read_csv(file_path, dtype=str)
        
        # Check if the first row is metadata (e.g. Date is null or empty string)
        if pd.isna(df.iloc[0]['Date']) or df.iloc[0]['Date'].strip() == '':
            logger.info("Detected metadata row, dropping row 0")
            df = df.iloc[1:].reset_index(drop=True)
            
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise DataNotFoundException(f"File {file_path} not found")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise PreprocessingException(f"Error loading data: {e}")

def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    PHASE 1: STRICT DATA CLEANING
    
    Clean the dataset with quality validation:
    - Convert Date to datetime
    - Sort by Date
    - Convert to numeric types (catch bad values)
    - Forward fill then backward fill missing values
    - DROP rows where indicators are incomplete
    - Validate no NaN values remain in price columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw stock data
    verbose : bool
        Whether to log detailed statistics
        
    Returns:
    --------
    pd.DataFrame
        Clean dataset with guaranteed no NaN in price/volume columns
    """
    try:
        original_len = len(df)
        df = df.copy()
        
        # 1. Date Handling
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Drop rows where Date conversion failed
            if df['Date'].isnull().any():
                dropped = df['Date'].isnull().sum()
                logger.warning(f"Dropping {dropped} rows with invalid dates")
                df = df[df['Date'].notna()].reset_index(drop=True)
        
        # 2. Required columns validation
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise PreprocessingException(f"Missing required columns: {missing}")

        # 3. Convert to numeric (STRICT - catch bad data)
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Log NaN count before fill
        nan_before = df[required_cols].isnull().sum().sum()
        if nan_before > 0 and verbose:
            logger.warning(f"Found {nan_before} NaN values in price/volume columns before filling")

        # 4. Handle missing values (forward fill, then backward fill)
        df[required_cols] = df[required_cols].ffill(limit=5)  # Max 5 consecutive fills
        df[required_cols] = df[required_cols].bfill(limit=5)
        
        # 5. STRICT VALIDATION: Drop rows still containing NaN
        nan_after_fill = df[required_cols].isnull().any(axis=1).sum()
        if nan_after_fill > 0:
            logger.warning(f"Dropping {nan_after_fill} rows with NaN after filling")
            df = df[~df[required_cols].isnull().any(axis=1)].reset_index(drop=True)
        
        # 6. Validate data quality
        # Check for negative prices
        if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            logger.warning("Found non-positive prices, these may indicate bad data")
        
        # Check Volume is positive
        if (df['Volume'] <= 0).any():
            logger.warning("Found zero or negative volumes")
        
        # 7. Final Validation
        if df[required_cols].isnull().any().any():
            raise PreprocessingException("NaN values still present after cleaning! Data quality issue.")
        
        final_len = len(df)
        dropped_total = original_len - final_len
        
        if verbose:
            logger.info(f"Data cleaned: {original_len} -> {final_len} rows (dropped {dropped_total})")
            logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}" 
                       if 'Date' in df.columns else "")
            
        return df
        
    except Exception as e:
        logger.error(f"Error in clean_data: {e}")
        raise PreprocessingException(f"Error cleaning data: {e}")
