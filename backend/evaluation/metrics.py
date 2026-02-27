import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression and classification metrics.
    y_true, y_pred: numpy arrays of shape (n_samples, horizon) or (n_samples,)
    """
    metrics = {}
    
    # Flatten if multi-step for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Regression Metrics
    metrics['MSE'] = mean_squared_error(y_true_flat, y_pred_flat)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(y_true_flat, y_pred_flat)
    metrics['R2'] = r2_score(y_true_flat, y_pred_flat)
    
    # MAPE (Handle division by zero)
    mask = y_true_flat != 0
    metrics['MAPE'] = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    
    # Directional Accuracy
    # Compare sign of change? 
    # If y implies returns, sign of y is direction.
    direction_true = np.sign(y_true_flat)
    direction_pred = np.sign(y_pred_flat)
    metrics['Directional_Accuracy'] = np.mean(direction_true == direction_pred) * 100
    
    return metrics
