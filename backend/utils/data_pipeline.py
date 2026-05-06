"""
PHASE 1: CORRECTED DATA PIPELINE

This module implements the proper ML pipeline with no data leakage:

Correct Order:
1. Load data
2. Clean data (strict validation)
3. Add technical indicators
4. DROP NaN rows
5. Create future return target (3-day)
6. Time-based split (train/test)
7. Fit scaler on train data ONLY
8. Transform both train and test
9. Create sequences with proper alignment

Key Principles:
- No synthetic/random data during training
- Future returns calculated with proper shifting
- No future data leakage
- Scaler fit only on training data
"""

import numpy as np
import pandas as pd
from typing import Tuple
from backend.core.logging import logger


def create_future_return_target(
    df: pd.DataFrame, 
    days_ahead: int = 3,
    return_type: str = 'simple'
) -> pd.DataFrame:
    """
    Create future return target variable.
    
    CRITICAL: No future leakage - we look ahead `days_ahead` steps
    to predict future price movement as a classification/regression target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must have 'Close' column, already cleaned
    days_ahead : int
        How many days into the future to predict (default 3)
    return_type : str
        'simple': (close[t+n] - close[t]) / close[t]
        'log': log(close[t+n] / close[t])
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with new column 'Future_Return_{days_ahead}d'
        All rows where target cannot be computed are NaN (will be dropped)
    """
    df = df.copy()
    
    # Calculate future price (day t + days_ahead)
    future_close = df['Close'].shift(-days_ahead)
    
    if return_type == 'log':
        # Log return: more stationary, better for neural networks
        df[f'Future_Return_{days_ahead}d'] = np.log(future_close / df['Close'])
    else:
        # Simple return: (P_future - P_current) / P_current
        df[f'Future_Return_{days_ahead}d'] = (future_close - df['Close']) / df['Close']
    
    # At the end of the dataset, we cannot compute future returns
    # These will be NaN and will be dropped
    logger.info(f"Created {days_ahead}-day future return target")
    logger.info(f"Last {days_ahead} rows will have NaN target (expected)")
    
    return df


def validate_sequences(
    X: np.ndarray, 
    y: np.ndarray, 
    sequence_length: int,
    forecast_horizon: int,
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Validate sequence shapes and values before training.
    
    Parameters:
    -----------
    X : np.ndarray
        Input sequences (samples, seq_len, features)
    y : np.ndarray
        Target values (samples, horizon) or (samples, horizon, 1)
    sequence_length : int
        Length of input sequences
    forecast_horizon : int
        Length of forecast
    verbose : bool
        Print validation details
        
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, message)
    """
    checks = []
    
    # Shape checks
    if len(X.shape) != 3:
        checks.append(f"❌ X shape error: expected 3D, got {X.shape}")
    else:
        checks.append(f"✓ X shape correct: {X.shape}")
    
    if len(y.shape) not in [2, 3]:
        checks.append(f"❌ y shape error: expected 2D or 3D, got {y.shape}")
    else:
        checks.append(f"✓ y shape correct: {y.shape}")
    
    # First dimension (samples) should match
    if X.shape[0] != y.shape[0]:
        checks.append(f"❌ Mismatch: X has {X.shape[0]} samples, y has {y.shape[0]}")
    else:
        checks.append(f"✓ X and y have same number of samples: {X.shape[0]}")
    
    # Sequence length check
    if X.shape[1] != sequence_length:
        checks.append(f"❌ X seq length {X.shape[1]} != expected {sequence_length}")
    else:
        checks.append(f"✓ Sequence length correct: {sequence_length}")
    
    # Forecast horizon check
    if y.shape[1] != forecast_horizon:
        checks.append(f"❌ y horizon {y.shape[1]} != expected {forecast_horizon}")
    else:
        checks.append(f"✓ Forecast horizon correct: {forecast_horizon}")
    
    # NaN checks
    nan_in_X = np.isnan(X).sum()
    nan_in_y = np.isnan(y).sum()
    
    if nan_in_X > 0:
        checks.append(f"❌ Found {nan_in_X} NaN values in X!")
    else:
        checks.append(f"✓ No NaN values in X")
    
    if nan_in_y > 0:
        checks.append(f"❌ Found {nan_in_y} NaN values in y!")
    else:
        checks.append(f"✓ No NaN values in y")
    
    # Value range checks
    if not np.isfinite(X).all():
        checks.append(f"❌ X contains infinite values!")
    else:
        checks.append(f"✓ All X values finite")
    
    # Stats
    checks.append(f"X - min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}")
    checks.append(f"y - min: {y.min():.4f}, max: {y.max():.4f}, mean: {y.mean():.4f}")
    
    if verbose:
        for check in checks:
            logger.info(check)
    
    # Overall status
    is_valid = all('✓' in check for check in checks if '✓' in check or '❌' in check)
    message = '\n'.join(checks)
    
    return is_valid, message


def log_data_statistics(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    prefix: str = "DATA STATS"
) -> None:
    """
    Log detailed data statistics for debugging.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    feature_cols : list
        Feature column names
    target_col : str
        Target column name
    prefix : str
        Prefix for logs
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{prefix}")
    logger.info(f"{'='*60}")
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}" 
               if 'Date' in df.columns else "No date column")
    
    logger.info(f"\nFeatures ({len(feature_cols)}):")
    for col in feature_cols[:10]:  # First 10
        logger.info(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")
    if len(feature_cols) > 10:
        logger.info(f"  ... and {len(feature_cols) - 10} more")
    
    logger.info(f"\nTarget: {target_col}")
    if target_col in df.columns:
        logger.info(f"  min={df[target_col].min():.6f}")
        logger.info(f"  max={df[target_col].max():.6f}")
        logger.info(f"  mean={df[target_col].mean():.6f}")
        logger.info(f"  std={df[target_col].std():.6f}")
    
    logger.info(f"\nMissing values:")
    missing = df[feature_cols + [target_col]].isnull().sum()
    if missing.sum() > 0:
        logger.info(f"  {missing[missing > 0].to_dict()}")
    else:
        logger.info(f"  None ✓")
    
    logger.info(f"{'='*60}\n")


def train_test_time_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test using TIME-BASED splitting.
    
    This prevents data leakage by ensuring test data is chronologically
    AFTER training data (no future information in train set).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset (must be sorted by date)
    test_size : float
        Fraction of data for test (default 0.2 = 20%)
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Time-based split:")
    logger.info(f"  Train: {len(train_df)} samples ({100*(1-test_size):.1f}%)")
    logger.info(f"  Test:  {len(test_df)} samples ({100*test_size:.1f}%)")
    
    if 'Date' in df.columns:
        logger.info(f"  Train dates: {train_df['Date'].min()} to {train_df['Date'].max()}")
        logger.info(f"  Test dates:  {test_df['Date'].min()} to {test_df['Date'].max()}")
    
    return train_df, test_df


def create_sequences_v2(
    data: np.ndarray,
    seq_length: int,
    forecast_horizon: int,
    target_col_idx: int,
    name: str = "sequences"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PHASE 1: CORRECTED create_sequences function
    
    Create sequences for time series models with proper handling of:
    - Lookback window (past `seq_length` timesteps)
    - Forecast window (next `forecast_horizon` timesteps)
    - NO FUTURE LEAKAGE
    
    Example:
    --------
    seq_length=30, forecast_horizon=3
    
    For each sample i:
      Input X[i]:  data[i:i+30, :]           (30 timesteps, all features)
      Output y[i]: data[i+30:i+33, target]   (next 3 days for target feature)
    
    This ensures:
    - Input uses only past data
    - Target is genuinely future data
    - No overlap between input and output
    
    Parameters:
    -----------
    data : np.ndarray
        Preprocessed data (time_steps, features)
        Should already be scaled
    seq_length : int
        Number of timesteps to look back
    forecast_horizon : int
        Number of timesteps to forecast ahead
    target_col_idx : int
        Index of target column in data
    name : str
        Name for logging
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X: (num_sequences, seq_length, features)
        y: (num_sequences, forecast_horizon) - values of target column
    """
    sequences_X = []
    sequences_y = []
    
    # Calculate maximum number of valid sequences
    # We need: seq_length for input + forecast_horizon for output + no extra
    max_samples = len(data) - seq_length - forecast_horizon + 1
    
    if max_samples <= 0:
        raise ValueError(
            f"Insufficient data: {len(data)} timesteps "
            f"< seq_length({seq_length}) + forecast_horizon({forecast_horizon})"
        )
    
    # Create sequences
    for i in range(max_samples):
        # Input: past seq_length timesteps, all features
        input_seq = data[i : i + seq_length]
        
        # Output: next forecast_horizon timesteps, target column only
        target_seq = data[i + seq_length : i + seq_length + forecast_horizon, target_col_idx]
        
        sequences_X.append(input_seq)
        sequences_y.append(target_seq)
    
    X = np.array(sequences_X)
    y = np.array(sequences_y)
    
    logger.info(f"Created {name} sequences:")
    logger.info(f"  Input sequences (X): {X.shape}")
    logger.info(f"  Target sequences (y): {y.shape}")
    logger.info(f"  Lookback window: {seq_length} timesteps")
    logger.info(f"  Forecast horizon: {forecast_horizon} timesteps")
    
    return X, y
