# InvestIQ Model Training - Complete Guide

## Quick Start

Run the training with the enhanced version:

```bash
# Windows
run_training_v2.bat

# Or run directly with Python
python train_enhanced.py
```

## What Was Fixed

### 1. **Import Error in `ensemble.py`** ✓
   - **Issue**: Missing `Any` type import caused NameError
   - **Fix**: Added `Any` to typing imports
   - **File**: `backend/models/ensemble.py` (line 3)

### 2. **Transformer Model Bug** ✓
   - **Issue**: Positional encoding was applied twice, causing training instability
   - **Fix**: Removed duplicate positional encoding call
   - **File**: `backend/models/transformer.py` (forward method)

## Training System Components

### Core Pipeline
- **Data Loading**: Reads CSV files from `backend/data/stock_data/`
- **Preprocessing**: Cleans data, handles missing values
- **Feature Engineering**: Adds technical indicators and market correlation
- **Training**: Uses Transformer model with PyTorch
- **Validation**: Implements early stopping with 10-epoch patience
- **Checkpointing**: Saves best models to `backend/models/saved_models/`

### Supported Stocks
- HDFCBANK
- RELIANCE
- TCS
- INFY
- ICICIBANK

## Common Issues & Solutions

### Issue 1: "Module not found" errors
**Solution:**
```bash
pip install -r backend/requirements.txt
```

### Issue 2: GPU/CUDA errors
**Solution:**
The system automatically falls back to CPU. If you want GPU:
```bash
# Update PyTorch for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: "CSV file not found"
**Check:**
- CSV files are in: `backend/data/stock_data/`
- Files are named: `{TICKER}.csv` (e.g., HDFCBANK.csv)
- CSV has columns: Open, High, Low, Close, Volume

### Issue 4: "Out of Memory" errors
**Solution:**
Reduce batch size in `backend/core/config.py`:
```python
BATCH_SIZE: int = 16  # Reduced from 32
```

### Issue 5: "Insufficient data after preprocessing"
**Solution:**
Check if CSV has at least 150+ rows of data. Minimum requirement:
- SEQ_LENGTH: 90
- FORECAST_HORIZON: 7
- Minimum rows needed: ~250-300

## Training Configuration

Edit `backend/core/config.py` to adjust:

```python
# Model Architecture
SEQ_LENGTH = 90           # Lookback window (days)
FORECAST_HORIZON = 7     # Prediction horizon (days)
EPOCHS = 100             # Training epochs
BATCH_SIZE = 32          # Samples per batch
LEARNING_RATE = 0.001    # Optimizer learning rate

# Transformer
NHEAD = 4                # Attention heads
NUM_LAYERS = 2           # Transformer layers
DROPOUT = 0.1            # Dropout rate
```

## Expected Training Output

```
🤖 INVESTIQ MODEL TRAINING SYSTEM
======================================================================

Configuring training environment...
  ✓ GPU available: NVIDIA GeForce RTX 3080
  
VERIFYING SETUP
----------------------------------------------------------------------
  ✓ Data directory exists (5 CSV files)
  ✓ Model directory exists

TRAINING PIPELINE
======================================================================

[1/5] HDFCBANK
    → Validating data...
    ✓ CSV structure valid (8000+ initial rows)
    → Loading and preprocessing...
    ✓ Model saved: 45.2 MB

[2/5] RELIANCE
    ...
```

## Performance Expectations

- **Training time per model**: 5-15 minutes (depends on CPU/GPU)
- **Total training time**: 30-60 minutes for all 5 models
- **Model file size**: ~40-50 MB per model
- **GPU memory**: ~2-3 GB (with batch_size=32)
- **CPU memory**: ~4-6 GB

## Files Modified for Fixes

1. `backend/models/ensemble.py` - Added `Any` import
2. `backend/models/transformer.py` - Fixed positional encoding bug

## New Training Scripts

1. `train_enhanced.py` - Enhanced training with comprehensive error handling
2. `train_safe.py` - Safe training runner with detailed logging
3. `run_training_v2.bat` - Improved batch file for Windows

## Next Steps After Training

Once training completes successfully:

1. **Run inference**: 
   ```python
   from backend.inference.predict import Predictor
   predictor = Predictor()
   result = predictor.predict("HDFCBANK")
   ```

2. **Start API server**:
   ```bash
   python -m backend.app.main
   ```

3. **Run tests**:
   ```bash
   python -m pytest backend/tests/
   ```

## Debugging

For detailed logs:
```bash
# Check logs during/after training
tail -f backend/training.log
```

To enable verbose logging:
```python
# In backend/core/logging.py
logger.setLevel(logging.DEBUG)
```

## Support

If training still fails:
1. Check error message in console output
2. Review logs in `backend/logs/`
3. Verify CSV data integrity
4. Ensure all dependencies are installed: `pip install -r backend/requirements.txt --upgrade`

---

**Last Updated**: March 13, 2026
**Version**: 2.0 (Enhanced Training System)
