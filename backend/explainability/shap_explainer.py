import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from backend.core.config import settings
from backend.core.logging import logger
from backend.models.enhanced_models import LSTMAttentionEnhanced

class SHAPExplainer:
    def __init__(self, model, background_data):
        """
        model: PyTorch model
        background_data: numpy array or torch tensor of shape (samples, seq_len, features) used for baseline.
        """
        self.model = model
        self.background_data = torch.FloatTensor(background_data).to(next(model.parameters()).device)
        self.explainer = shap.DeepExplainer(model, self.background_data)
        
    def explain(self, X_sample, feature_names=None):
        """
        X_sample: numpy array (samples, seq_len, features) to explain
        """
        X_tensor = torch.FloatTensor(X_sample).to(next(self.model.parameters()).device)
        shap_values = self.explainer.shap_values(X_tensor)
        
        # shap_values is list of arrays (one for each output). 
        # For regression (output_dim=1), it might be a single array or list of 1.
        # DeepExplainer matches output. If model outputs (N, Horizon, 1), 
        # shap_values might be complex.
        
        # Let's simplify: Explain the first step of the forecast?
        # Or aggregate importance over time?
        
        # If output is (N, Horizon, 1), PyTorch DeepExplainer might fail if output is 3D.
        # It usually expects (N, OutputDim).
        # Our model outputs (N, Horizon, 1). We might need a wrapper to explain a specific horizon step.
        return shap_values

def generate_shap_plots(ticker, X_train, X_test, feature_names):
    try:
        # Load Model (LSTM + FinBERT + XGBoost architecture)
        input_dim = X_train.shape[2]
        model = LSTMAttentionEnhanced(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            dropout=0.3,
            forecast_horizon=settings.FORECAST_HORIZON
        )
        model_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Wrapper for SHAP to handle 3D output -> 1D (e.g. sum of forecasts or 1st step)
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                out = self.model(x) # (N, H, 1)
                return out[:, 0, 0] # Explain 1st step prediction
        
        wrapped_model = ModelWrapper(model)
        
        # Use a small background sample
        background = X_train[:100]
        explainer = shap.DeepExplainer(wrapped_model, torch.FloatTensor(background))
        
        # Explain test sample
        test_sample = X_test[:10]
        shap_values = explainer.shap_values(torch.FloatTensor(test_sample))
        
        # shap_values shape: (samples, seq_len, features)
        # We want to aggregate importance per feature (sum over interactions/time)
        
        # SHAP summary plot usually expects (samples, features).
        # We can sum absolute SHAP values over the time dimension (seq_len)
        
        # shap_values is usually a list for DeepExplainer? No, for regression it is array.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Aggregate over time: Mean Absolute SHAP value for each feature
        # Shape: (samples, seq_len, features)
        importances = np.mean(np.abs(shap_values), axis=1) # (samples, features)
        
        # Plot
        plt.figure()
        shap.summary_plot(importances, test_sample[:, -1, :], feature_names=feature_names, show=False)
        plt.savefig(os.path.join(settings.MODEL_DIR, f"shap_{ticker}.png"))
        plt.close()
        logger.info(f"SHAP plot saved for {ticker}")
        
    except Exception as e:
        logger.error(f"SHAP generation failed: {e}")
