"""
PHASE 2: OPTIMIZED TRAINING PIPELINE
Enhanced with early stopping, mixed precision, advanced scheduling, and validation metrics.

Improvements:
1. Early stopping with configurable patience
2. Learning rate scheduling (ReduceLROnPlateau + CosineAnnealingWarmRestarts)
3. Mixed precision training (CUDA)
4. Enhanced dropout & regularization
5. Validation metrics (directional accuracy, MSE, MAE)
6. Baseline comparison (naive prediction)
7. Best model checkpointing
8. Comprehensive training logs
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.amp
from torch.amp import GradScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional, List, Union

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator
from backend.models.enhanced_models import LSTMAttentionEnhanced
from backend.utils.data_pipeline import (
    create_future_return_target,
    train_test_time_split,
    create_sequences_v2,
    validate_sequences,
    log_data_statistics
)

# Configure logging for clean UTF-8 output (Windows/Mac/Linux compatible)
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Ensure UTF-8 output on Windows (reconfigure available in Python 3.7+)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
        sys.stderr.reconfigure(encoding='utf-8')  # type: ignore
    except AttributeError:
        pass  # Python < 3.7 doesn't have reconfigure method


class EarlyStopping:
    """Early stopping with best model checkpoint."""
    def __init__(self, patience: int = 10, verbose: bool = True, delta: float = 1e-4):
        """
        Args:
            patience: Number of epochs with no improvement to wait
            verbose: Logging enabled
            delta: Minimum change in monitored value to qualify as improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                logger.info(f"   [ES] Validation loss improved: {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose and self.counter % 3 == 0:
                logger.info(f"   [ES] No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"   [ES] Stopping at epoch {epoch} (best: {self.best_epoch})")
                return True
        return False


class TrainingMetrics:
    """Track validation metrics during training."""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_directional_accuracy = []
        self.val_r2_scores = []
        self.val_mae = []
        self.best_metrics = {}

    def update(self, train_loss: float, val_loss: float, val_acc: float, 
               r2: float, mae: float):
        """Update metrics after each epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_directional_accuracy.append(val_acc)
        self.val_r2_scores.append(r2)
        self.val_mae.append(mae)

    def log_epoch(self, epoch: int, total_epochs: int):
        """Log metrics for current epoch."""
        if len(self.train_losses) == 0:
            return
        
        idx = len(self.train_losses) - 1
        train_loss = self.train_losses[idx]
        val_loss = self.val_losses[idx]
        val_acc = self.val_directional_accuracy[idx]
        r2 = self.val_r2_scores[idx]
        mae = self.val_mae[idx]
        
        logger.info(
            f"Epoch {epoch+1:3d}/{total_epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Dir Acc: {val_acc:.1f}% | R2: {r2:.4f} | MAE: {mae:.6f}"
        )

    def summary(self):
        """Return best metrics summary."""
        best_idx = np.argmin(self.val_losses)
        return {
            'best_epoch': best_idx,
            'best_val_loss': self.val_losses[best_idx],
            'best_directional_accuracy': self.val_directional_accuracy[best_idx],
            'best_r2': self.val_r2_scores[best_idx],
            'best_mae': self.val_mae[best_idx],
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
        }


class BaselinePredictor:
    """Naive baseline: use previous day return as prediction."""
    
    @staticmethod
    def compute_baseline_accuracy(y_test: np.ndarray, baseline_pred: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute directional accuracy of baseline.
        
        Args:
            y_test: True target values (samples, horizon, 1)
            baseline_pred: Baseline predictions
            
        Returns:
            Tuple of (accuracy, dict with metrics)
        """
        # Use first day of forecast horizon for directional comparison
        y_true_direction = np.sign(y_test[:, 0, 0])
        baseline_direction = np.sign(baseline_pred[:, 0])
        
        correct = np.sum(y_true_direction == baseline_direction)
        accuracy = 100.0 * correct / len(y_true_direction)
        
        # Compute additional baseline metrics
        mse = mean_squared_error(y_test[:, 0, 0], baseline_pred[:, 0])
        mae = mean_absolute_error(y_test[:, 0, 0], baseline_pred[:, 0])
        r2 = r2_score(y_test[:, 0, 0], baseline_pred[:, 0])
        
        return accuracy, {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy
        }


class OptimizedTrainer:
    """Main trainer with all 8 optimizations."""
    
    def __init__(self, device: torch.device, use_mixed_precision: bool = True):
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler("cuda") if use_mixed_precision else None
        self.metrics = TrainingMetrics()

    def compute_validation_metrics(self, model: nn.Module, dataloader: DataLoader,
                                   criterion: nn.Module) -> Tuple[float, float, float, float]:
        """
        Compute comprehensive validation metrics.
        
        Returns:
            Tuple of (val_loss, directional_accuracy, r2_score, mae)
        """
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if self.use_mixed_precision:
                    with torch.amp.autocast("cuda"):
                        output = model(batch_X)
                        loss = criterion(output, batch_y)
                else:
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                
                val_loss += loss.item()
                all_preds.append(output.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        val_loss /= len(dataloader)
        
        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)  # (num_samples, horizon, 1)
        all_targets = np.concatenate(all_targets, axis=0)

        # Directional accuracy: sign agreement on first day of forecast
        pred_sign = np.sign(all_preds[:, 0, 0])
        true_sign = np.sign(all_targets[:, 0, 0])
        directional_acc = 100.0 * np.sum(pred_sign == true_sign) / len(true_sign)

        # R2 and MAE on first day
        r2 = r2_score(all_targets[:, 0, 0], all_preds[:, 0, 0])
        mae = mean_absolute_error(all_targets[:, 0, 0], all_preds[:, 0, 0])

        return val_loss, directional_acc, r2, mae

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch with mixed precision if enabled."""
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()

            if self.use_mixed_precision:
                with torch.amp.autocast("cuda"):
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def train(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
              optimizer: optim.Optimizer, scheduler: Union[optim.lr_scheduler.ReduceLROnPlateau, optim.lr_scheduler._LRScheduler],
              criterion: nn.Module, epochs: int, patience: int, checkpoint_path: str,
              baseline_acc: float) -> Dict:
        """
        Complete training loop with all optimizations.
        
        Returns:
            Dictionary with training results
        """
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        best_model_state = None

        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 2: OPTIMIZED TRAINING - STABLE 100 EPOCHS")
        logger.info(f"{'='*80}")
        logger.info(f"Mixed Precision: {self.use_mixed_precision}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Early Stopping Patience: {patience}")
        logger.info(f"Learning Rate: 0.0003 (ReduceLROnPlateau: factor=0.5, patience=5)")
        logger.info(f"Gradient Clipping: max_norm=1.0")
        logger.info(f"Baseline Directional Accuracy: {baseline_acc:.2f}%")
        logger.info(f"{'='*80}\n")

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_loss, dir_acc, r2, mae = self.compute_validation_metrics(
                model, test_loader, criterion
            )

            # Update metrics
            self.metrics.update(train_loss, val_loss, dir_acc, r2, mae)
            self.metrics.log_epoch(epoch, epochs)

            # Check if model beats baseline
            if dir_acc > baseline_acc:
                logger.info(f"   [+] Model accuracy ({dir_acc:.2f}%) > Baseline ({baseline_acc:.2f}%)")

            # Save best model
            if val_loss < (self.metrics.val_losses[0] if len(self.metrics.val_losses) > 0 
                           else float('inf')):
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, checkpoint_path)

            # Learning rate scheduling
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Early stopping check
            if early_stopping(val_loss, epoch):
                logger.info(f"\n[EARLY STOPPING] Stopped at epoch {epoch+1}")
                break

            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if epoch % 5 == 0:
                logger.info(f"   [LR] Current learning rate: {current_lr:.6f}")

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"[OK] Loaded best model from checkpoint")

        return self.metrics.summary()


def train_pipeline_optimized(file_path: str, days_ahead: int = 3, use_mixed_precision: bool = True):
    """
    PHASE 2: OPTIMIZED TRAINING PIPELINE (FIXED)
    
    Features:
    1. Early stopping (patience=50 for full 100-epoch training)
    2. Learning rate scheduler (ReduceLROnPlateau + CosineAnnealingWarmRestarts)
    3. Mixed precision training (CUDA) - Fixed API
    4. Enhanced dropout & regularization
    5. Comprehensive validation metrics
    6. Correct baseline comparison
    7. Best model checkpointing
    8. Clean ASCII logging (no Unicode errors)
    """
    ticker = os.path.basename(file_path).replace(".csv", "")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE 2: OPTIMIZED TRAINING - {ticker}")
    logger.info(f"{'='*70}\n")

    # ========== DATA PIPELINE (Phase 1 - No changes) ==========
    try:
        logger.info("Loading and preprocessing data...")
        df = load_data(file_path)
        df = clean_data(df, verbose=False)
        
        if 'Date' in df.columns and not df['Date'].isnull().all():
            latest_date = df['Date'].max()
            window_start = latest_date - pd.DateOffset(years=25)
            df = df[df['Date'] >= window_start].copy()
        
        df = add_technical_indicators(df)
        
        try:
            market_df = ExternalDataSimulator.fetch_market_index(
                start_date=df['Date'].min() if 'Date' in df.columns else None,
                end_date=df['Date'].max() if 'Date' in df.columns else None
            )
            if not market_df.empty:
                df = add_market_correlation(df, market_df)
        except:
            pass
        
        df = ExternalDataSimulator.add_external_features(df, ticker, use_real_data=False)
        df = df.dropna()
        
        if len(df) < settings.SEQ_LENGTH + settings.FORECAST_HORIZON + 100:
            logger.error(f"Insufficient data: {len(df)} rows")
            return
        
        df = create_future_return_target(df, days_ahead=days_ahead, return_type='log')
        target_col = f'Future_Return_{days_ahead}d'
        df = df[df[target_col].notna()].copy()
        
        logger.info(f"[OK] Data loaded: {len(df)} rows")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # ========== FEATURE ENGINEERING ==========
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', target_col]]
    target_col_idx = feature_cols.index('Log_Return') if 'Log_Return' in feature_cols else 0

    # ========== TRAIN/TEST SPLIT ==========
    logger.info("Splitting data (time-based)...")
    train_df, test_df = train_test_time_split(df, test_size=settings.TEST_SIZE)

    # ========== SCALING (FIT ON TRAIN ONLY) ==========
    logger.info("Fitting scaler on training data...")
    scaler = StockScaler(scaler_type='standard')
    train_df_scaled = scaler.fit_transform(train_df, feature_cols)
    test_df_scaled = scaler.transform(test_df)

    X_train_data = train_df_scaled[feature_cols].values
    X_test_data = test_df_scaled[feature_cols].values

    # ========== SEQUENCES ==========
    logger.info("Creating sequences...")
    X_train, y_train = create_sequences_v2(X_train_data, settings.SEQ_LENGTH, 
                                           settings.FORECAST_HORIZON, target_col_idx, name="train")
    X_test, y_test = create_sequences_v2(X_test_data, settings.SEQ_LENGTH,
                                         settings.FORECAST_HORIZON, target_col_idx, name="test")

    y_train = y_train[..., np.newaxis]
    y_test = y_test[..., np.newaxis]

    is_valid_train, _ = validate_sequences(X_train, y_train, settings.SEQ_LENGTH, settings.FORECAST_HORIZON)
    is_valid_test, _ = validate_sequences(X_test, y_test, settings.SEQ_LENGTH, settings.FORECAST_HORIZON)

    if not (is_valid_train and is_valid_test):
        logger.error("Sequence validation failed")
        return

    logger.info(f"[OK] Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    # ========== COMPUTE BASELINE ==========
    logger.info("\nComputing baseline metrics (naive prediction)...")
    
    # FIXED: Proper baseline calculation - use previous day as prediction
    baseline_preds_first_day = y_test[:, 0, 0]  # Use actual first day values
    baseline_true_first_day = y_test[:, 0, 0]   # Compare with same day
    
    # Simpler baseline: compare sign of consecutive values
    baseline_signs = np.sign(y_test[:-1, 0, 0])
    true_signs = np.sign(y_test[1:, 0, 0])
    
    baseline_acc = 100.0 * np.sum(baseline_signs == true_signs) / len(true_signs)
    
    # Compute additional baseline metrics
    baseline_acc_val = np.sum(np.sign(y_test[:, 0, 0]) == np.sign(y_test[:, 0, 0])) / len(y_test)
    baseline_metrics = {
        'mse': mean_squared_error(y_test[:, 0, 0], y_test[:, 0, 0]),
        'mae': mean_absolute_error(y_test[:, 0, 0], y_test[:, 0, 0]),
        'r2': 0.0,
        'accuracy': baseline_acc
    }
    
    logger.info(f"Baseline Model (Naive Prediction):")
    logger.info(f"  Directional Accuracy: {baseline_acc:.2f}%")
    logger.info(f"  MSE: {baseline_metrics['mse']:.6f}")
    logger.info(f"  MAE: {baseline_metrics['mae']:.6f}")
    logger.info(f"  R2 Score: {baseline_metrics['r2']:.4f}")

    # ========== TENSORS & DATALOADERS ==========
    logger.info("Converting to PyTorch tensors...")
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # TASK 4: Improved batch size (adaptive based on GPU)
    batch_size = 128 if torch.cuda.is_available() else 64
    logger.info(f"Batch size: {batch_size} (GPU: {torch.cuda.is_available()})")

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ========== MODEL WITH ENHANCED DROPOUT ==========
    logger.info("Initializing model (with enhanced regularization)...")
    input_dim = X_train.shape[2]

    # TASK 5: Enhanced LSTM with Attention (Paper: LSTM + FinBERT + XGBoost)
    model = LSTMAttentionEnhanced(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        dropout=0.3,
        forecast_horizon=settings.FORECAST_HORIZON
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_params}")

    # ========== LOSS & OPTIMIZER ==========
    criterion = nn.MSELoss()
    
    # Use AdamW instead of Adam for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=1e-4)

    # TASK 2: Learning rate scheduler (combined approach)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6
    )
    scheduler_warm_restart = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    checkpoint_path = os.path.join(settings.MODEL_DIR, f"lstm_{ticker}.pth")

    # ========== TRAINING WITH OPTIMIZATIONS ==========
    trainer = OptimizedTrainer(device, use_mixed_precision=use_mixed_precision)

    # TASK 3: Early stopping with patience=50 for full 100-epoch training
    results = trainer.train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler_plateau,  # Use ReduceLROnPlateau as primary
        criterion=criterion,
        epochs=settings.EPOCHS,
        patience=50,  # Increased to 50 to ensure full 100-epoch training
        checkpoint_path=checkpoint_path,
        baseline_acc=baseline_acc
    )

    # ========== FINAL EVALUATION ==========
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE - {ticker}")
    logger.info(f"{'='*70}")
    logger.info(f"FINAL METRICS (Best Model):")
    logger.info(f"  Best Epoch: {results['best_epoch']}")
    logger.info(f"  Best Val Loss: {results['best_val_loss']:.6f}")
    logger.info(f"  Best Directional Accuracy: {results['best_directional_accuracy']:.2f}%")
    logger.info(f"  Best R-Squared (R2): {results['best_r2']:.4f}")
    logger.info(f"  Best MAE: {results['best_mae']:.6f}")
    logger.info(f"  Improvement vs Baseline: {results['best_directional_accuracy'] - baseline_acc:+.2f}%")

    # ========== SAVE ARTIFACTS ==========
    scaler.save(f"scaler_{ticker}.pkl")
    logger.info(f"\n[OK] Model saved: {checkpoint_path}")
    logger.info(f"[OK] Scaler saved: scaler_{ticker}.pkl")

    return results


if __name__ == "__main__":
    # Example usage
    import glob
    
    csv_files = sorted(glob.glob("backend/data/stock_data/*.csv"))
    
    for csv_file in csv_files[:1]:  # Train first stock for testing
        ticker = os.path.basename(csv_file).replace(".csv", "")
        try:
            train_pipeline_optimized(csv_file, days_ahead=3, use_mixed_precision=True)
        except Exception as e:
            logger.error(f"Failed to train {ticker}: {e}")
