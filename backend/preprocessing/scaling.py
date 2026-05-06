import pandas as pd
import joblib
from typing import Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from backend.core.config import settings
from backend.core.logging import logger

class StockScaler:
    """
    PHASE 1: PREVENT DATA LEAKAGE
    
    Scaler must be fit ONLY on training data.
    Validation and test data are transformed using training statistics.
    
    Correct workflow:
    1. Split data into train/test (time-based, no shuffle)
    2. Fit scaler on train data ONLY
    3. Transform both train and test using training statistics
    4. Train model on transformed data
    """
    
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Initialize scaler.
        
        Parameters:
        -----------
        scaler_type : str
            'minmax' (maps to [0, 1]) or 'standard' (zero mean, unit variance)
            Transformers typically work better with standard scaling
        """
        self.scaler_type = scaler_type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        TRAINING ONLY: Fit scaler on training data and transform.
        
        IMPORTANT: This should ONLY be called on training data!
        Do NOT call this on full dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training dataframe
        columns : list
            Columns to scale
            
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe with same index
        """
        self.feature_columns = columns
        
        # Validate input
        if df[columns].isnull().any().any():
            raise ValueError("NaN values detected in data before fitting scaler!")
        
        # Fit on this data
        self.scaler.fit(df[columns])
        self.is_fitted = True
        
        # Transform
        scaled_data = self.scaler.transform(df[columns])
        df_scaled = df.copy()
        df_scaled[columns] = scaled_data
        
        logger.info(f"Scaler FITTED on {len(df)} training samples with columns: {columns}")
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TEST/VAL ONLY: Transform new data using fitted scaler.
        
        IMPORTANT: Scaler must be fitted first (on training data).
        This applies training statistics to new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform (test or validation)
            
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted yet! Call fit_transform on training data first.")
        
        if not self.feature_columns:
            raise ValueError("No feature columns set")
        
        # Validate input
        if df[self.feature_columns].isnull().any().any():
            raise ValueError("NaN values detected in data before transformation!")
        
        scaled_data = self.scaler.transform(df[self.feature_columns])
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = scaled_data
        
        logger.info(f"Scaler TRANSFORMED {len(df)} samples using training statistics")
        return df_scaled
    
    def inverse_transform(self, data, column_indices: Optional[list] = None):
        """
        Inverse transform scaled data back to original scale.
        
        Useful for:
        - Converting model predictions back to original price scale
        - Evaluating MAPE/MAE on original scale
        
        Parameters:
        -----------
        data : array-like
            Scaled data (should match feature dimensions)
        column_indices : list
            Which columns to inverse transform (optional)
            
        Returns:
        --------
        array
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted yet!")
        
        # If we only have one column (e.g., just target), need to pad to match scaler dimensions
        # This is a bit hacky but works for most cases
        if hasattr(data, 'shape'):
            if len(data.shape) == 1:
                # Single column - need to create dummy columns for other features
                n_features = len(self.feature_columns)
                padded = np.zeros((len(data), n_features))
                if column_indices is not None:
                    padded[:, column_indices] = data.reshape(-1, 1)
                else:
                    padded[:, 0] = data
                data = padded
        
        return self.scaler.inverse_transform(data)

    def save(self, name: str = "scaler.pkl"):
        """Save fitted scaler to disk"""
        path = os.path.join(settings.MODEL_DIR, name)
        state = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted
        }
        joblib.dump(state, path)
        logger.info(f"Scaler saved to {path}")

    def load(self, name: str = "scaler.pkl"):
        """Load fitted scaler from disk"""
        path = os.path.join(settings.MODEL_DIR, name)
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.feature_columns = state['feature_columns']
        self.scaler_type = state.get('scaler_type', 'minmax')
        self.is_fitted = state.get('is_fitted', True)
        logger.info(f"Scaler loaded from {path}")

# Import numpy for inverse_transform
import numpy as np
