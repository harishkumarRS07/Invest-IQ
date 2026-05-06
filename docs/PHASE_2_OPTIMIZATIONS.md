# PHASE 2: TRAINING OPTIMIZATIONS GUIDE

## Overview
PHASE 2 improves model accuracy and training stability with 8 key optimizations implemented in production-grade PyTorch code.

---

## 🎯 8 OPTIMIZATIONS IMPLEMENTED

### ✅ TASK 1: EARLY STOPPING
**What:** Stop training if validation loss doesn't improve for N epochs

**Implementation:**
```python
class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter on improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False
```

**Benefits:**
- Prevents overfitting
- Saves training time
- Loads best model automatically
- Patience = 10 epochs (configurable)

**Code Location:** `backend/training/train_optimized.py` (lines 20-44)

---

### ✅ TASK 2: LEARNING RATE SCHEDULER
**What:** Dynamically reduce learning rate when model plateaus

**Implementation:**
```python
# ReduceLROnPlateau: Reduce LR when validation loss plateaus
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6
)

# CosineAnnealingWarmRestarts: Periodic LR resets for better convergence
scheduler_warm_restart = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# Training loop
for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    scheduler_plateau.step(val_loss)  # Reduce LR if no improvement
    scheduler_warm_restart.step()      # Gradual warming/restart
```

**Benefits:**
- Better convergence
- Escapes local minima
- Reduces learning rate gradually
- Factor: 0.7× when no improvement for 5 epochs

**Code Location:** `backend/training/train_optimized.py` (lines 358-368)

---

### ✅ TASK 3: MIXED PRECISION TRAINING (CUDA)
**What:** Use float16 for faster training with lower memory usage

**Implementation:**
```python
from torch.cuda.amp import autocast, GradScaler

class OptimizedTrainer:
    def __init__(self, use_mixed_precision: bool = True):
        self.scaler = GradScaler()
    
    def train_epoch(self, model, train_loader):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            with autocast():  # Automatic mixed precision
                output = model(batch_X)
                loss = criterion(output, batch_y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()  # Update scaling factor
```

**Benefits:**
- 20-30% faster training
- Lower GPU memory usage
- Same accuracy as float32
- Automatic precision management

**Code Location:** `backend/training/train_optimized.py` (lines 158-187)

---

### ✅ TASK 4: IMPROVED BATCH SIZE
**What:** Adaptive batch size based on GPU availability

**Implementation:**
```python
# Adaptive batch size selection
batch_size = 128 if torch.cuda.is_available() else 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # Disable multi-worker for Windows compatibility
)
```

**Benefits:**
- 128 on GPU → Faster training, better gradient estimates
- 64 on CPU → Memory-efficient
- Automatic detection
- Larger batches reduce noise in gradients

**Code Location:** `backend/training/train_optimized.py` (line 355)

---

### ✅ TASK 5: ADD DROPOUT & REGULARIZATION
**What:** Prevent overfitting with dropout and L2 regularization

**Implementation:**

#### Transformer (PHASE 2 Enhanced):
```python
class TimeSeriesTransformerEnhanced(nn.Module):
    def __init__(self, dropout=0.2):  # Increased from 0.1
        # Embedding layer
        self.embedding_dropout = nn.Dropout(dropout)  # NEW
        
        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            dropout=dropout,  # 0.2 in all layers
            norm_first=True   # Pre-layer normalization
        )
        
        # Decoder with multiple dropout stages
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),  # Dropout after activation
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(dropout),  # Multiple layers
            nn.Linear(d_model // 2, output_dim * forecast_horizon)
        )
```

#### LSTM (PHASE 2 Enhanced):
```python
class LSTMAttentionEnhanced(nn.Module):
    def __init__(self, dropout=0.3):  # Increased from 0.2
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            dropout=dropout,      # Recurrent dropout
            bidirectional=True
        )
        
        # Attention with dropout
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            dropout=dropout       # Attention dropout
        )
        
        # FC layers with dropout
        self.fc_stack = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim * forecast_horizon)
        )
```

#### Optimizer with L2 Regularization:
```python
# AdamW with weight decay (L2 regularization)
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4  # L2 penalty term
)
```

**Benefits:**
- Dropout = 0.2 for Transformer (was 0.1)
- Dropout = 0.3 for LSTM (was 0.2)
- Layer normalization for stability
- Weight decay = 1e-4 prevents weight explosion
- Multiple dropout stages prevent co-adaptation

**Code Location:** 
- Transformer: `backend/models/enhanced_models.py` (lines 32-75)
- LSTM: `backend/models/enhanced_models.py` (lines 95-155)

---

### ✅ TASK 6: VALIDATION METRICS DURING TRAINING
**What:** Track comprehensive metrics every epoch

**Implementation:**
```python
class TrainingMetrics:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_directional_accuracy = []  # NEW
        self.val_r2_scores = []             # NEW
        self.val_mae = []                   # NEW
    
    def update(self, train_loss, val_loss, val_acc, r2, mae):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_directional_accuracy.append(val_acc)
        self.val_r2_scores.append(r2)
        self.val_mae.append(mae)
    
    def log_epoch(self, epoch, total_epochs):
        logger.info(
            f"Epoch {epoch+1:3d}/{total_epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Dir Acc: {val_acc:.1f}% | R2: {r2:.4f} | MAE: {mae:.6f}"
        )
```

**Metrics Tracked:**
1. **Training Loss** → Overall model performance
2. **Validation Loss** → Generalization ability
3. **Directional Accuracy** → % correct sign predictions
4. **R² Score** → Variance explained
5. **MAE** → Mean absolute error

**Example Output:**
```
Epoch   1/100 | Train Loss: 1.017651 | Val Loss: 0.499414 | Dir Acc: 48.2% | R2: 0.1234 | MAE: 0.023456
Epoch   2/100 | Train Loss: 0.891234 | Val Loss: 0.412567 | Dir Acc: 51.8% | R2: 0.2567 | MAE: 0.021234
Epoch   3/100 | Train Loss: 0.756891 | Val Loss: 0.367234 | Dir Acc: 54.3% | R2: 0.3891 | MAE: 0.018765
```

**Code Location:** `backend/training/train_optimized.py` (lines 47-90)

---

### ✅ TASK 7: BASELINE COMPARISON
**What:** Compare model against naive prediction

**Implementation:**
```python
class BaselinePredictor:
    @staticmethod
    def compute_baseline_accuracy(y_test, baseline_pred):
        """
        Naive baseline: use previous day return as prediction
        
        Returns: (accuracy, metrics_dict)
        """
        y_true_direction = np.sign(y_test[:, 0, 0])
        baseline_direction = np.sign(baseline_pred[:, 0])
        
        correct = np.sum(y_true_direction == baseline_direction)
        accuracy = 100.0 * correct / len(y_true_direction)
        
        # Compute additional metrics
        mse = mean_squared_error(y_test[:, 0, 0], baseline_pred[:, 0])
        mae = mean_absolute_error(y_test[:, 0, 0], baseline_pred[:, 0])
        r2 = r2_score(y_test[:, 0, 0], baseline_pred[:, 0])
        
        return accuracy, {'mse': mse, 'mae': mae, 'r2': r2, 'accuracy': accuracy}

# During training
baseline_acc = 48.5%
model_acc = 54.2%
improvement = +5.7%
```

**Output Example:**
```
Baseline Model (Naive Prediction):
  • Directional Accuracy: 48.50%
  • MSE: 0.008234
  • MAE: 0.062145
  • R²: -0.0234

Model Performance:
  • Best Directional Accuracy: 54.20%
  • vs Baseline Accuracy: +5.70%
  • Best R² Score: 0.3456
  • Best MAE: 0.043567
```

**Benefits:**
- Validates model is learning (not random)
- Shows real improvement
- Establishes minimum performance threshold
- Model must beat baseline to be useful

**Code Location:** `backend/training/train_optimized.py` (lines 92-124)

---

### ✅ TASK 8: SAVE BEST MODEL ONLY
**What:** Checkpoint only when validation improves

**Implementation:**
```python
def train(self, model, train_loader, test_loader, ...):
    best_model_state = None
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = self.train_epoch(model, train_loader, ...)
        val_loss = self.compute_validation_metrics(model, test_loader)
        
        # TASK 8: Save ONLY when validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, checkpoint_path)
            logger.info(f"✓ New best model saved! (val_loss: {val_loss:.6f})")
        
        # ... early stopping, scheduling ...
    
    # Load best model at end
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"✓ Loaded best model from checkpoint")
```

**Benefits:**
- Avoids overwriting with worse models
- Prevents disk waste
- Automatic rollback to best state
- Clear improvement tracking

**Code Location:** `backend/training/train_optimized.py` (lines 250-257)

---

## 📊 EXPECTED IMPROVEMENTS

### Before (PHASE 1):
```
Directional Accuracy: ~50%
R² Score: ~0.0
Training Stability: Low
Training Time: ~15 min per model
```

### After (PHASE 2):
```
Directional Accuracy: ~55-60%
R² Score: ~0.3-0.4
Training Stability: High
Training Time: ~8-10 min per model (mixed precision)
Improvement: +5-10% accuracy, 2× faster
```

---

## 🚀 USAGE

### Training Single Stock:
```bash
cd d:\InvestIQ-main
python -m backend.training.train_optimized
```

### Batch Training All Stocks:
```python
# Create batch_train_optimized.py
import glob
import os
from backend.training.train_optimized import train_pipeline_optimized

csv_files = sorted(glob.glob("backend/data/stock_data/*.csv"))

for i, csv_file in enumerate(csv_files, 1):
    ticker = os.path.basename(csv_file).replace(".csv", "")
    print(f"\n[{i}/{len(csv_files)}] Training {ticker}...")
    try:
        train_pipeline_optimized(csv_file, days_ahead=3, use_mixed_precision=True)
    except Exception as e:
        print(f"Failed: {e}")
```

---

## 🔧 CONFIGURATION TUNING

### Early Stopping:
```python
EarlyStopping(patience=10)  # Stop if no improvement for 10 epochs
# Increase patience for larger datasets
# Decrease for faster iteration
```

### Learning Rate Scheduler:
```python
# ReduceLROnPlateau parameters
ReduceLROnPlateau(
    optimizer,
    mode='min',           # Minimize validation loss
    factor=0.7,           # Multiply LR by 0.7
    patience=5,           # Wait 5 epochs before reducing
    min_lr=1e-6,          # Don't go below 1e-6
    verbose=False
)
```

### Dropout:
```python
# Transformer
TimeSeriesTransformerEnhanced(dropout=0.2)  # Higher = more regularization

# LSTM
LSTMAttentionEnhanced(dropout=0.3)          # Higher = more regularization
```

### Batch Size:
```python
# GPU: Use 128 for faster training (if GPU has 8GB+)
# GPU: Use 64 for memory-constrained devices
# CPU: Use 32 for CPU-only systems
batch_size = 128 if torch.cuda.is_available() else 64
```

---

## 📈 MONITORING TRAINING

### Key Indicators:
1. **Validation Loss Decreasing** ✓ → Model is learning
2. **Directional Accuracy > 50%** ✓ → Better than random
3. **Directional Accuracy > Baseline** ✓ → Model is improving
4. **Early Stopping Triggered** ✓ → Prevents overfitting
5. **Learning Rate Reducing** ✓ → Adaptive scheduling working

### Red Flags:
1. ❌ Validation loss increasing → Overfitting (increase dropout)
2. ❌ Directional accuracy ~50% → Model not learning (check data)
3. ❌ Training time very long → Check GPU usage
4. ❌ Loss NaN → Try reducing learning rate or batch size

---

## 🎯 NEXT STEPS

1. **Train all models with PHASE 2**: `batch_train_optimized.py`
2. **Compare metrics**: PHASE 1 vs PHASE 2
3. **Analyze convergence**: Plot loss curves
4. **Tune hyperparameters**: Based on validation metrics
5. **Evaluate on test set**: Final model assessment

---

## 📚 FILE LOCATIONS

| Component | Location |
|-----------|----------|
| Optimized Training | `backend/training/train_optimized.py` |
| Enhanced Models | `backend/models/enhanced_models.py` |
| Early Stopping | `backend/training/train_optimized.py` (lines 20-44) |
| Metrics Tracking | `backend/training/train_optimized.py` (lines 47-90) |
| Optimizer Setup | `backend/training/train_optimized.py` (lines 358-368) |
| Training Loop | `backend/training/train_optimized.py` (lines 210-257) |

---

## 💡 BEST PRACTICES FOR PRODUCTION

### GPU Training:
```python
# Ensure GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Mixed precision (when GPU supports it)
use_mixed_precision = torch.cuda.is_available()
trainer = OptimizedTrainer(device, use_mixed_precision=use_mixed_precision)
```

### Memory Management:
```python
# Clear unused variables
del X_train, X_test
torch.cuda.empty_cache()

# Use gradient accumulation for large batches
for i in range(accumulation_steps):
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    # Only step every N batches
```

### Checkpointing Strategy:
```python
# Save model only when validation improves
if val_loss < best_val_loss:
    torch.save(model.state_dict(), 'best_model.pth')

# Save additional info
checkpoint = {
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'best_loss': best_val_loss
}
torch.save(checkpoint, 'checkpoint.pt')
```

---

## 📞 TROUBLESHOOTING

| Issue | Solution |
|-------|----------|
| CUDA Out of Memory | Reduce batch size (64 → 32) |
| Loss NaN | Reduce learning rate (0.001 → 0.0001) |
| Slow Training | Enable mixed precision, increase batch size |
| Overfitting | Increase dropout (0.2 → 0.3) |
| Underfitting | Decrease dropout (0.3 → 0.1), train longer |
| No Improvement | Check data quality, verify features |

---

Generated: April 9, 2026
Phase: PHASE 2 - Training Optimizations
