import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for Time Series Forecasting.
    Supports multivariate input and multi-step output.
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1, output_dim: int = 1, forecast_horizon: int = 7):
        super(TimeSeriesTransformer, self).__init__()
        
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        # Input embedding to project features to d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        # batch_first=True makes input (batch, seq, feature)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Decoder (or just Output Projection if using Encoder-only for regression)
        # Using Encoder-only for simpler forecasting unless we want autoregressive decoding
        # For multi-step forecasting of continuous values, a dense decoder or simple projection is often used.
        # Let's use a Decoder that maps the encoded sequence to the forecast horizon.
        
        # Option A: Encoder -> Flatten -> Dense -> Output (Can be heavy if seq_len is large)
        # Option B: Encoder -> Select last t steps -> Dense -> Output
        # Option C: Transformer Decoder (Autoregressive) - Needs target shifting during training.
        
        # Given "Support 7-day future forecasting", direct multi-step (Option A/B) is robust.
        # Let's use a standard implementation:
        # Use a linear layer to decode the LAST hidden state to `forecast_horizon` steps.
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim * forecast_horizon) 
            # Output shape: (batch, forecast_horizon * output_dim) -> reshape later
        )
        
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim

    def forward(self, src):
        """
        src: (batch_size, seq_len, input_dim)
        """
        # Embed and add position encoding
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.permute(1, 0, 2)).permute(1, 0, 2) # PE expects (L, N, E) but we work with batch_first=True in Transformer?
        # Check PE implementation: it returns (L, 1, E). 
        # If we permute src to (L, N, E), add PE, then permute back to (N, L, E).
        
        # Correction: My PE implementation is standard PyTorch tutorial which outputs (L, N, E).
        # TransformerEncoder with batch_first=True expects (N, L, E).
        # So:
        src = src.permute(1, 0, 2) # (L, N, E)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) # (N, L, E)
        
        output = self.transformer_encoder(src) # (N, L, E)
        
        # Take the embedding of the last time step as context for prediction
        last_step_output = output[:, -1, :] # (N, E)
        
        prediction = self.decoder(last_step_output) # (N, H * O)
        
        # Reshape to (N, Horizon, Output_Dim)
        prediction = prediction.view(-1, self.forecast_horizon, self.output_dim)
        
        return prediction
