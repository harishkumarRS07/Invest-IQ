import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
except ImportError as e:
    print(f"Error importing torch: {e}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Error importing pandas: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"Error importing transformers: {e}")

try:
    from backend.features.sentiment import SentimentAnalyzer
    print("Successfully imported SentimentAnalyzer from backend.")
except ImportError as e:
    print(f"Error importing backend modules: {e}")
