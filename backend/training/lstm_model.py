"""Lightweight LSTM for binary directional probability on close-price sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class _TinyLSTM(nn.Module):
    def __init__(self, hidden_dim: int = 24, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


@dataclass
class LSTMDirectionalModel:
    seq_length: int = 15
    hidden_dim: int = 24
    num_layers: int = 1
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = MinMaxScaler()
        self.model = _TinyLSTM(hidden_dim=self.hidden_dim, num_layers=self.num_layers).to(self.device)

    def _to_sequences(self, close_scaled: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.seq_length, len(close_scaled)):
            X.append(close_scaled[i - self.seq_length : i])
            y.append(labels[i])
        if not X:
            return np.empty((0, self.seq_length, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)
        X_arr = np.asarray(X, dtype=np.float32).reshape(-1, self.seq_length, 1)
        y_arr = np.asarray(y, dtype=np.int64)
        return X_arr, y_arr

    def fit(self, close_train: np.ndarray, y_train: np.ndarray, close_val: np.ndarray, y_val: np.ndarray) -> None:
        close_train_s = self.scaler.fit_transform(close_train.reshape(-1, 1)).reshape(-1)
        close_val_s = self.scaler.transform(close_val.reshape(-1, 1)).reshape(-1)

        X_tr, y_tr = self._to_sequences(close_train_s, y_train)
        X_va, y_va = self._to_sequences(close_val_s, y_val)
        if len(X_tr) == 0:
            return

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if len(X_va) > 0:
            val_x = torch.from_numpy(X_va).to(self.device)
            val_y = torch.from_numpy(y_va.astype(np.float32)).to(self.device)
        else:
            val_x = None
            val_y = None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_val = float("inf")
        patience = 2
        bad_epochs = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if val_x is not None and val_y is not None and len(X_va) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_x)
                    val_loss = float(criterion(val_logits, val_y).item())
                if val_loss < best_val:
                    best_val = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= patience:
                    break

    def predict_proba_up(self, close_values: np.ndarray) -> np.ndarray:
        close_s = self.scaler.transform(close_values.reshape(-1, 1)).reshape(-1)
        X, _ = self._to_sequences(close_s, np.zeros(len(close_s), dtype=np.int64))
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        return probs
