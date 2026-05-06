"""
PHASE 2: ENHANCED MODEL CONFIGURATIONS
Updated models with improved dropout, regularization, and layer normalization.
"""

import torch
import torch.nn as nn
import math
from typing import cast


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer input."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = cast(torch.Tensor, self.pe)
        return x + pe[:x.size(0), :]


class TimeSeriesTransformerEnhanced(nn.Module):
    """
    PHASE 2: Enhanced Transformer for Time Series Forecasting
    
    Improvements:
    - Increased dropout (0.1 → 0.2)
    - Layer normalization
    - Residual connections
    - Better regularization
    - Gradient clipping compatible
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2, output_dim: int = 1,
                 forecast_horizon: int = 7):
        super(TimeSeriesTransformerEnhanced, self).__init__()
        
        self.model_type = 'TransformerEnhanced'
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',  # GELU for better performance
            norm_first=True  # Pre-layer norm for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Decoder with regularization
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim * forecast_horizon)
        )
        
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim

    def forward(self, src):
        """Forward pass with enhanced regularization."""
        # Embed and normalize
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.embedding_norm(src)
        src = self.embedding_dropout(src)
        
        # Add positional encoding
        src = src.permute(1, 0, 2)  # (L, N, E)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # (N, L, E)
        
        # Transformer
        output = self.transformer_encoder(src)  # (N, L, E)
        
        # Use last time step as context
        last_step_output = output[:, -1, :]  # (N, E)
        
        # Decode to forecast
        forecast = self.decoder(last_step_output)  # (N, forecast_horizon * output_dim)
        forecast = forecast.reshape(-1, self.forecast_horizon, self.output_dim)
        
        return forecast


class LSTMAttentionEnhanced(nn.Module):
    """
    PHASE 2: Enhanced LSTM with Attention
    
    Improvements:
    - Increased dropout (0.3 → 0.4 for multi-layer)
    - Improved attention mechanism
    - Layer normalization
    - Better initialization
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1,
                 dropout=0.3, forecast_horizon=7):
        super(LSTMAttentionEnhanced, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        
        # LSTM with higher dropout
        effective_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Layer normalization
        self.layer_norm_lstm = nn.LayerNorm(lstm_output_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm_attn = nn.LayerNorm(lstm_output_dim)
        
        # Fully connected layers with dropout
        self.fc_stack = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim * forecast_horizon)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, forecast_horizon, output_dim)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_output_dim)
        lstm_out = self.layer_norm_lstm(lstm_out)
        
        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm_attn(attn_out + lstm_out)  # Residual connection
        
        # Use last output as context
        context = attn_out[:, -1, :]  # (batch, lstm_output_dim)
        
        # Fully connected stack
        forecast = self.fc_stack(context)  # (batch, forecast_horizon * output_dim)
        forecast = forecast.reshape(-1, self.forecast_horizon, self.output_dim)
        
        return forecast


class EnhancedXGBoostPreprocessor:
    """
    PHASE 2: XGBoost-specific enhancements
    Pre-processes data for XGBoost with better feature engineering.
    """
    
    @staticmethod
    def create_lag_features(data: list, n_lags: int = 5) -> list:
        """
        Create lagged features for XGBoost.
        
        Args:
            data: Time series data
            n_lags: Number of lags to create
            
        Returns:
            List of samples with lag features
        """
        samples = []
        for i in range(n_lags, len(data)):
            sample = list(data[i - n_lags:i])
            samples.append(sample)
        return samples

    @staticmethod
    def create_rolling_stats(series: list, window: int = 5) -> list:
        """
        Create rolling statistics features (mean, std, min, max).
        
        Args:
            series: Time series data
            window: Rolling window size
            
        Returns:
            List of rolling statistics
        """
        stats = []
        for i in range(window, len(series)):
            window_data = series[i - window:i]
            stats.append({
                'mean': sum(window_data) / len(window_data),
                'std': (sum((x - sum(window_data) / len(window_data)) ** 2 
                       for x in window_data) / len(window_data)) ** 0.5,
                'min': min(window_data),
                'max': max(window_data)
            })
        return stats


# Alias for backward compatibility
TimeSeriesTransformer = TimeSeriesTransformerEnhanced
LSTMAttentionModel = LSTMAttentionEnhanced
