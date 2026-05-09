"""
LSTM Classifier Comparison (Reviewer Request)
==============================================
Trains an LSTM on the same 7 features (6 PCA + rainfall) using a sliding
lookback window, then reports metrics identical to Classifier_Comparison.py.
"""

import sys
from pathlib import Path

_repo = Path(__file__).resolve()
while _repo != _repo.parent:
    if (_repo / "pyproject.toml").exists():
        break
    _repo = _repo.parent
else:
    raise RuntimeError("Run from project clone; pyproject.toml not found.")
_src = _repo / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from landslide_forecast.config import (
    REPO_ROOT,
    RF_DATA_DIR,
    ASC_H5,
    DESC_H5,
    PCA_H5,
    RAINFALL_DIR,
    MINTPY_STACK_DIR,
)

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("LSTM CLASSIFIER — Same PCA Features, Sequence-Based Model")
print("=" * 80)

# ============================================================================
# Data Loading (identical to Classifier_Comparison.py)
# ============================================================================

asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
pca_file = str(PCA_H5)
rainfall_dir = str(RAINFALL_DIR)

print("\nLoading data...")

with h5py.File(asc_file, "r") as f:
    asc_data = f["timeseries"][:]
with h5py.File(desc_file, "r") as f:
    desc_data = f["timeseries"][:]

if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[: desc_data.shape[0]]

time_steps, height, width = asc_data.shape

with h5py.File(pca_file, "r") as f:
    asc_pca = f["ascending/scores"][:]
    desc_pca = f["descending/scores"][:]
min_samples = min(asc_pca.shape[0], desc_pca.shape[0])
pca_features = np.hstack([asc_pca[:min_samples], desc_pca[:min_samples]])

window_size = 7
threshold = 0.35
vertical = (asc_data + desc_data) / 2
horizontal = (asc_data - desc_data) / 2
displacement = np.sqrt(vertical**2 + horizontal**2)
cumulative_rolling = np.zeros_like(displacement)
for t in range(time_steps - window_size + 1):
    cumulative_rolling[t] = displacement[t : t + window_size].sum(axis=0)
labels = (cumulative_rolling > threshold).astype(int)
labels = labels.reshape(labels.shape[0], -1).sum(axis=1) > 0
labels = labels.astype(int)

rainfall_files = sorted(
    [
        os.path.join(rainfall_dir, f)
        for f in os.listdir(rainfall_dir)
        if f.endswith(".nc4")
    ]
)
rainfall_data = []
for f in rainfall_files:
    try:
        ds = xr.open_dataset(f)
        if "MWprecipitation" in ds:
            rainfall_data.append(
                ds["MWprecipitation"].mean(dim=["lon", "lat"]).to_dataframe()
            )
    except Exception:
        pass
rainfall_df = pd.concat(rainfall_data).reset_index()
rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])
aligned_rainfall = (
    rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"]
    .mean()
    .values
)

min_len = min(pca_features.shape[0], len(labels), len(aligned_rainfall))
pca_features = pca_features[:min_len]
labels = labels[:min_len]
aligned_rainfall = aligned_rainfall[:min_len]

X_flat = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])
y_flat = labels

print(f"Total samples: {len(y_flat)}, Features per step: {X_flat.shape[1]}")
print(f"Stable: {np.sum(y_flat == 0)}, Unstable: {np.sum(y_flat == 1)}")

# ============================================================================
# Create temporal sequences for LSTM
# ============================================================================

LOOKBACK = 10

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

X_seq, y_seq = [], []
for i in range(LOOKBACK, len(X_scaled)):
    X_seq.append(X_scaled[i - LOOKBACK : i])
    y_seq.append(y_flat[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"Sequences: {X_seq.shape[0]}, shape: {X_seq.shape} (samples, lookback, features)")

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# ============================================================================
# LSTM Model
# ============================================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

train_ds = TensorDataset(
    torch.FloatTensor(X_train),
    torch.FloatTensor(y_train),
)
test_ds = TensorDataset(
    torch.FloatTensor(X_test),
    torch.FloatTensor(y_test),
)

BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMClassifier(
    input_size=X_flat.shape[1],
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
).to(device)

pos_weight = torch.tensor(
    [np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)],
    dtype=torch.float32,
).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

EPOCHS = 150
best_loss = float("inf")
patience_counter = 0
EARLY_STOP_PATIENCE = 25

print(f"\nTraining LSTM for up to {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

model.load_state_dict(best_state)

# ============================================================================
# Evaluate
# ============================================================================

model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_batch.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
y_pred = (all_probs >= 0.5).astype(int)

acc = accuracy_score(all_labels, y_pred)
prec = precision_score(all_labels, y_pred, zero_division=0)
rec = recall_score(all_labels, y_pred, zero_division=0)
f1 = f1_score(all_labels, y_pred)
auc = roc_auc_score(all_labels, all_probs)

print("\n" + "=" * 80)
print("LSTM CLASSIFIER RESULTS")
print("=" * 80)
print(f"  Accuracy:  {acc * 100:.2f}%")
print(f"  Precision: {prec * 100:.2f}%")
print(f"  Recall:    {rec * 100:.2f}%")
print(f"  F1-Score:  {f1 * 100:.2f}%")
print(f"  AUC:       {auc * 100:.2f}%")
print("=" * 80)

print("\nFor Table (classifier_comparison):")
print(f"LSTM (2-layer) & {acc * 100:.2f}\\% & {prec * 100:.2f}\\% & {f1 * 100:.2f}\\% & {auc * 100:.2f}\\% \\\\")
print("=" * 80)
