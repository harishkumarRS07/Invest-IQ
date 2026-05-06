# PHASE 2: COMPLETE OPTIMIZED TRAINING EXAMPLE
"""
This is a reference implementation showing all 10 optimizations in context.
For actual training, use: python batch_train_optimized.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


class OptimizedTrainer:
    """PHASE 2: Trainer with all 10 optimizations."""
    
    def __init__(self, device: torch.device, use_mixed_precision: bool = True):
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None
    
    def compute_validation_metrics(self, model, dataloader, criterion):
        """TASK 8: Track comprehensive metrics."""
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # TASK 1: Fixed mixed precision
                if self.use_mixed_precision:
                    with torch.amp.autocast("cuda"):  # ✅ NEW API
                        output = model(batch_X)
                        loss = criterion(output, batch_y)
                else:
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                
                val_loss += loss.item()
                all_preds.append(output.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        val_loss /= len(dataloader)
        
        # TASK 8: Compute metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Directional accuracy
        pred_sign = np.sign(all_preds[:, 0, 0])
        true_sign = np.sign(all_targets[:, 0, 0])
        directional_acc = 100.0 * np.sum(pred_sign == true_sign) / len(true_sign)
        
        # R2 and MAE
        r2 = r2_score(all_targets[:, 0, 0], all_preds[:, 0, 0])
        mae = mean_absolute_error(all_targets[:, 0, 0], all_preds[:, 0, 0])
        
        return val_loss, directional_acc, r2, mae
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """TASK 7: Clean training loop."""
        model.train()  # ✅ 1. Train mode
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # ✅ 2. Prepare data
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # ✅ 3. Zero gradients
            optimizer.zero_grad()
            
            # ✅ 4. Forward pass with gradient clipping
            if self.use_mixed_precision:
                # TASK 1: Fixed mixed precision
                with torch.amp.autocast("cuda"):  # ✅ NEW API
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                
                # Scale loss
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                
                # TASK 5: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # ✅ 7. Optimizer step with scaler
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Without mixed precision
                output = model(batch_X)
                loss = criterion(output, batch_y)
                
                # ✅ 5. Backward pass
                loss.backward()
                
                # TASK 5: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # ✅ 7. Optimizer step
                optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def train(self, model, train_loader, test_loader, optimizer, scheduler,
              criterion, epochs, patience, baseline_acc):
        """TASK 7: Complete training loop with all optimizations."""
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        print(f"\n{'='*80}")
        print(f"PHASE 2: OPTIMIZED TRAINING - STABLE 100 EPOCHS")
        print(f"{'='*80}")
        print(f"Learning Rate: 0.0003 (ReduceLROnPlateau: factor=0.5, patience=5)")
        print(f"Gradient Clipping: max_norm=1.0")
        print(f"Early Stopping Patience: {patience}")
        print(f"{'='*80}\n")
        
        for epoch in range(epochs):
            # TASK 7: Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # TASK 7: Validate
            val_loss, dir_acc, r2, mae = self.compute_validation_metrics(
                model, test_loader, criterion
            )
            
            # TASK 8: Log metrics each epoch
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Dir Acc: {dir_acc:.1f}% | "
                  f"R2: {r2:.4f} | "
                  f"MAE: {mae:.6f}")
            
            # TASK 3: Learning rate scheduling
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # TASK 9: Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"   [Improved] Val loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter % 5 == 0:
                    print(f"   [No improvement] {patience_counter}/{patience}")
            
            # Log LR every 5 epochs
            if epoch % 5 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"   [LR] Current learning rate: {lr:.6f}")
            
            # TASK 2: Early stopping (patience=20 for stable training)
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Stopped at epoch {epoch+1}")
                break
        
        # TASK 9: Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"[OK] Loaded best model from checkpoint")
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }


# ============================================================================
# EXAMPLE CONFIGURATION (from backend/core/config.py)
# ============================================================================

class ExampleConfig:
    """PHASE 2 Configuration with all optimizations."""
    
    # TASK 3: Learning rate = 0.0003
    LEARNING_RATE = 0.0003  # ✅ Stable convergence
    
    # TASK 1: Epochs for full training
    EPOCHS = 100  # ✅ Allow enough time
    
    # TASK 4: Batch size
    BATCH_SIZE = 128  # ✅ GPU optimized
    
    # TASK 6: Dropout for regularization
    DROPOUT = 0.2  # ✅ Increased from 0.1
    
    # Training parameters
    FORECAST_HORIZON = 7
    SEQ_LENGTH = 90
    TEST_SIZE = 0.2


# ============================================================================
# EXAMPLE FACTORY FUNCTION (Pseudo-code)
# ============================================================================

def example_setup_optimized_training():
    """
    Complete example of setting up PHASE 2 optimized training.
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # TASK 10: CUDA verification
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dummy data for example
    X_train = np.random.randn(100, 90, 10)  # (samples, seq_len, features)
    y_train = np.random.randn(100, 7, 1)    # (samples, forecast_horizon, 1)
    X_test = np.random.randn(30, 90, 10)
    y_test = np.random.randn(30, 7, 1)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    # TASK 4: DataLoaders with optimized batch size
    batch_size = 128 if torch.cuda.is_available() else 64
    print(f"Batch size: {batch_size}")
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model (example)
    class SimpleModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, 64, num_layers=2, 
                               dropout=0.2, batch_first=True)  # TASK 6: Dropout
            self.fc = nn.Linear(64, 7)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out.unsqueeze(-1)
    
    model = SimpleModel(input_dim=10).to(device)
    
    # TASK 3: Optimizer with learning rate
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), 
                            lr=0.0003,          # ✅ TASK 3
                            weight_decay=1e-4)  # L2 regularization
    
    # TASK 3: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,        # Reduce by 50%
        patience=5,        # Wait 5 epochs
        min_lr=1e-6
    )
    
    # Create trainer
    trainer = OptimizedTrainer(device, use_mixed_precision=True)
    
    # TASK 2: Train with patience=20 for stable 100-epoch training
    results = trainer.train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=100,      # ✅ TASK 1: Full 100 epochs
        patience=20,     # ✅ TASK 2: Increased patience
        baseline_acc=50
    )
    
    print(f"\nTraining Complete!")
    print(f"Best Val Loss: {results['best_val_loss']:.6f}")
    print(f"Epochs Trained: {results['epochs_trained']}")


# ============================================================================
# KEY IMPLEMENTATION CHECKLIST
# ============================================================================

"""
BEFORE RUNNING TRAINING, VERIFY:

✅ TASK 1: Mixed Precision
   - from torch.cuda.amp import GradScaler, autocast
   - import torch.amp
   - Replace: with autocast() → with torch.amp.autocast("cuda")
   - NO FutureWarnings

✅ TASK 2: Early Stopping Patience
   - Configure patience=20 (not 10)
   - Expect training to reach 70-100 epochs

✅ TASK 3: Learning Rate
   - Set learning_rate = 0.0003 in config.py
   - ReduceLROnPlateau: factor=0.5, patience=5

✅ TASK 4: Batch Size
   - batch_size = 128 (GPU) / 64 (CPU)
   - Adaptive based on torch.cuda.is_available()

✅ TASK 5: Gradient Clipping
   - torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   - Called after backward() and before optimizer.step()

✅ TASK 6: Dropout Regularization
   - Transformer dropout=0.2
   - LSTM dropout=0.2-0.3

✅ TASK 7: Clean Training Loop
   - model.train() / model.eval()
   - optimizer.zero_grad()
   - forward + loss + backward + clip + step

✅ TASK 8: Metrics Tracking
   - Track: Train Loss, Val Loss, Dir Acc, R2, MAE
   - Log each epoch

✅ TASK 9: Best Model Checkpointing
   - Save when val_loss improves
   - Load at end of training

✅ TASK 10: CUDA Usage
   - device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   - model.to(device)
   - data.to(device)

EXPECTED RESULTS:
- No warnings or errors
- Training reaches 70-100 epochs
- Directional accuracy > 50%
- Smooth loss curves
- Models saved successfully
"""


if __name__ == "__main__":
    print("This is a reference implementation file.")
    print("For actual training, run: python batch_train_optimized.py")
    print("\nTo test this example:")
    print("  python -c 'from [filename] import example_setup_optimized_training; example_setup_optimized_training()'")
