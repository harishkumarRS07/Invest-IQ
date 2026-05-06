import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Stock Predictor AI"
    API_V1_STR: str = "/api/v1"
    
    # Data Paths
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stock_data")
    MODEL_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "saved_models")
    
    # Model Hyperparameters
    SEQ_LENGTH: int = 90  # Increased lookback window for better trend capture
    TEST_SIZE: float = 0.2
    
    # Training Params
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.0003

    # Inference mode flags
    INFERENCE_MODE: str = "legacy"  # "legacy" | "hybrid"
    HYBRID_FALLBACK_TO_LEGACY: bool = True
    
    # Advanced Model Params
    DROPOUT: float = 0.1
    NHEAD: int = 4
    NUM_LAYERS: int = 2
    FORECAST_HORIZON: int = 7
    
    class Config:
        case_sensitive = True

settings = Settings()

# Ensure model directory exists
os.makedirs(settings.MODEL_DIR, exist_ok=True)
