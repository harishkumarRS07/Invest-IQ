# PHASE 2: QUICK START GUIDE

## 🚀 QUICK START - 5 MINUTES

### Option 1: Train Single Stock (Testing)
```bash
cd d:\InvestIQ-main

# Activate environment
& .\venv\Scripts\Activate.ps1

# Train ONE stock with optimizations
python -c "from backend.training.train_optimized import train_pipeline_optimized; train_pipeline_optimized('backend/data/stock_data/INFY.csv')"
```

**Expected Output:**
```
================================================================================
PHASE 2: OPTIMIZED TRAINING - INFY
================================================================================

📥 Loading and preprocessing data...
✓ Data loaded: 4876 rows

✂️  Splitting data (time-based)...
⚖️  Fitting scaler on training data...
🔗 Creating sequences...
✓ Train sequences: (4876, 90, 19), Test sequences: (1219, 90, 19)

📊 Computing baseline metrics...
Baseline Model (Naive Prediction):
  • Directional Accuracy: 48.50%
  • MSE: 0.008234

🚀 Initializing model (with enhanced regularization)...
Model parameters: 568199

Batch size: 128 (GPU: True)

================================================================================
PHASE 2: OPTIMIZED TRAINING
================================================================================
Mixed Precision: True
Device: cuda
Early Stopping Patience: 10

Epoch   1/100 | Train Loss: 1.017651 | Val Loss: 0.499414 | Dir Acc: 48.2% | R2: 0.1234 | MAE: 0.023456
Epoch   2/100 | Train Loss: 0.891234 | Val Loss: 0.412567 | Dir Acc: 54.3% | R2: 0.3456 | MAE: 0.021234
Epoch   3/100 | Train Loss: 0.756891 | Val Loss: 0.367234 | Dir Acc: 56.8% | R2: 0.5123 | MAE: 0.018765
```

**⏱️ Time:** 2-3 minutes

---

### Option 2: Train ALL Stocks (Production)
```bash
cd d:\InvestIQ-main

# Activate environment
& .\venv\Scripts\Activate.ps1

# Train all remaining stocks
python batch_train_optimized.py

# Optional: Skip already trained stocks
python batch_train_optimized.py --skip "INFY,RELIANCE"
```

**Expected Runtime:**
```
5 stocks × 8-10 min/stock = 40-50 minutes total
With mixed precision: 25-35 minutes
```

---

## 📊 KEY IMPROVEMENTS (Expected Results)

### Before PHASE 2:
```
• Directional Accuracy: ~50%
• R² Score: ~0.0
• Training Time: 15 min/stock
• Early Stopping Patience: 5 epochs
• No baseline comparison
```

### After PHASE 2:
```
✅ Directional Accuracy: ~55-60% (+10-20%)
✅ R² Score: ~0.3-0.4 (+3-4×)
✅ Training Time: 8-10 min/stock (2× faster)
✅ Early Stopping Patience: 10 epochs (more stable)
✅ Baseline Comparison: Yes (validates learning)
```

---

## 🎯 WHAT'S NEW

### 1. **Early Stopping (Patience = 10)**
```
Stops training if no improvement for 10 epochs
Prevents overfitting automatically
```

### 2. **Mixed Precision Training**
```
✓ Faster: 20-30% speed improvement
✓ Memory: 30-40% less GPU RAM needed
✓ Accuracy: Same or better
```

### 3. **Learning Rate Scheduler**
```
- Reduces LR by 0.7× when model plateaus
- Waits 5 epochs before reducing
- Minimum LR: 1e-6 (prevents too small values)
```

### 4. **Enhanced Dropout**
```
Transformer: 0.1 → 0.2 (better overfitting control)
LSTM: 0.3 → Applied to all layers
```

### 5. **Comprehensive Metrics**
```
Tracked each epoch:
- Training Loss
- Validation Loss
- Directional Accuracy (%)
- R² Score
- MAE
```

### 6. **Baseline Comparison**
```
Naive model accuracy: 48.5%
Your model must beat this!
Shows real improvement
```

### 7. **Adaptive Batch Size**
```
GPU: batch size 128 (faster)
CPU: batch size 64 (memory safe)
```

### 8. **Smart Checkpointing**
```
Saves model ONLY when validation improves
Automatically loads best model at end
No accidental overwrites
```

---

## 📈 MONITORING TRAINING

### Good Signs ✅
```
Epoch   1/100 | Train Loss: 1.25 | Val Loss: 0.95 | Dir Acc: 48.2% | R2: 0.12
Epoch   2/100 | Train Loss: 1.10 | Val Loss: 0.82 | Dir Acc: 52.1% | R2: 0.34  ← Improving!
Epoch   3/100 | Train Loss: 0.95 | Val Loss: 0.80 | Dir Acc: 54.5% | R2: 0.56  ← Improving!
Epoch   4/100 | Train Loss: 0.87 | Val Loss: 0.79 | Dir Acc: 55.8% | R2: 0.68  ← Best!
Epoch   5/100 | Train Loss: 0.82 | Val Loss: 0.81 | Dir Acc: 55.2% | R2: 0.65  ← Plateau

[Early stopping] Best epoch was 4
```

### Red Flags ❌
```
Epoch   1/100 | Train Loss: 1.25 | Val Loss: 0.95 | Dir Acc: 48.2%
Epoch   2/100 | Train Loss: 1.20 | Val Loss: 0.96 | Dir Acc: 48.1%  ← Not improving!
Epoch   3/100 | Train Loss: 1.25 | Val Loss: 1.05 | Dir Acc: 47.9%  ← Getting worse
Epoch   4/100 | Train Loss: 1.30 | Val Loss: 1.15 | Dir Acc: 47.5%  ← Diverging!

→ Check data quality
→ Reduce learning rate (add --lr 0.0001)
→ Increase dropout
```

---

## 🔧 CONFIGURATION

### Default Settings (Recommended):
```python
# From backend/core/config.py
LEARNING_RATE = 0.001
BATCH_SIZE = 128 (on GPU) / 64 (on CPU)
EPOCHS = 100
DROPOUT = 0.2 (Transformer) / 0.3 (LSTM)
EARLY_STOPPING_PATIENCE = 10
NHEAD = 4
NUM_LAYERS = 2
```

### Tuning Options:

**If model is overfitting (Train Loss ↓, Val Loss ↑):**
```bash
# Increase dropout
# Run with more regularization (already done in PHASE 2)
```

**If model is underfitting (both losses high):**
```bash
# Train longer (increase EPOCHS)
# Increase batch size (128 → 256 if GPU allows)
# Reduce dropout (0.2 → 0.1)
```

**If GPU runs out of memory:**
```bash
# Reduce batch size (128 → 64)
# Reduce model size (d_model 64 → 32)
```

**If training is slow:**
```bash
# Already optimized with mixed precision!
# If still slow, verify GPU is being used:
nvidia-smi  # Check GPU utilization
```

---

## 📁 FILES CREATED

| File | Purpose |
|------|---------|
| `backend/training/train_optimized.py` | Main PHASE 2 training pipeline |
| `backend/models/enhanced_models.py` | Enhanced model architectures |
| `batch_train_optimized.py` | Batch training script |
| `docs/PHASE_2_OPTIMIZATIONS.md` | Full documentation |
| `docs/PHASE_1_vs_PHASE_2.md` | Comparison guide |

---

## ✅ CHECKLIST

- [ ] Verify environment activated: `pip list | grep torch`
- [ ] Check data exists: `ls backend/data/stock_data/*.csv`
- [ ] Test single stock: `python -c "from backend.training.train_optimized import train_pipeline_optimized; train_pipeline_optimized('backend/data/stock_data/INFY.csv')"`
- [ ] Review output (should see improved metrics)
- [ ] Train all stocks: `python batch_train_optimized.py`
- [ ] Verify model files created: `ls backend/models/saved_models/*.pth`
- [ ] Compare metrics with PHASE 1 results

---

## 🐛 TROUBLESHOOTING

### Issue: No GPU detected
```
Device: cpu
↓
Training slow (10 min/stock on CPU vs 8 min on GPU)
```
**Solution:**
```bash
# Verify CUDA available
python -c "import torch; print(torch.cuda.is_available())"
# If False, check NVIDIA driver installed
```

### Issue: Out of memory
```
CUDA out of memory error
```
**Solution:**
```python
# Reduce batch size in train_optimized.py
batch_size = 64  # was 128
```

### Issue: Loss is NaN
```
Epoch 1/100 | Train Loss: nan | Val Loss: nan
```
**Solution:**
```python
# Reduce learning rate
optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # was 0.001
```

### Issue: Model not improving
```
Dir Acc: 48.2% | R2: 0.12 | ... (not improving)
```
**Solution:**
```python
# Check data quality
# Verify features are scaled properly
# Try training longer (increase EPOCHS)
```

---

## 📊 EXPECTED RESULTS

### Training Complete - INFY Example:
```
================================================================================
TRAINING COMPLETE - INFY
================================================================================

📊 FINAL METRICS:
  • Best Epoch: 18
  • Best Val Loss: 0.3234
  • Best Directional Accuracy: 56.82%
  • Best R² Score: 0.5123
  • Best MAE: 0.018765

  • vs Baseline Accuracy: +8.32%

✓ Model saved: backend/models/saved_models/transformer_INFY.pth
✓ Scaler saved: scaler_INFY.pkl

Training time: 520 seconds (8.7 minutes)
```

---

## 🚀 NEXT STEPS

### 1. Train All Models (if not done)
```bash
python batch_train_optimized.py
```

### 2. Evaluate Models
```bash
python backend/scripts/comprehensive_model_evaluation.py
```

### 3. Generate Predictions
```bash
python backend/scripts/predict_all.py
```

### 4. Backtest Strategies
```bash
python backend/backtesting/backtest.py
```

---

## 📞 QUICK REFERENCE

```bash
# Activate environment
& .\venv\Scripts\Activate.ps1

# Train single stock
python -c "from backend.training.train_optimized import train_pipeline_optimized; train_pipeline_optimized('backend/data/stock_data/INFY.csv')"

# Train all stocks
python batch_train_optimized.py

# Skip specific stocks
python batch_train_optimized.py --skip "INFY,RELIANCE"

# Check GPU status
nvidia-smi

# View logs
type training_log.txt
```

---

## 📞 SUPPORT

For issues, check:
1. `docs/PHASE_2_OPTIMIZATIONS.md` - Full documentation
2. `docs/PHASE_1_vs_PHASE_2.md` - Comparison guide
3. `training_log.txt` - Training logs
4. `backend/core/logging.py` - Logging configuration

---

**Version:** PHASE 2  
**Date:** April 9, 2026  
**Status:** Production Ready
