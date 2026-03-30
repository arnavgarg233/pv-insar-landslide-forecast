"""
Model G: Fair RF Comparison (6 statistical features + rainfall = 7 total)
Same number of features as PCA (6 components + rainfall)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL G: Fair RF (6 Statistical Features + Rainfall)")
print("="*70)

# File paths
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
rainfall_dir = str(RAINFALL_DIR)

# Load InSAR data
print("\n[1/4] Loading InSAR data...")
with h5py.File(asc_file, 'r') as f:
    asc_data = f['timeseries'][:]
    
with h5py.File(desc_file, 'r') as f:
    desc_data = f['timeseries'][:]

if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]
    
print(f"  Data shape: {asc_data.shape}")

# Extract top 6 most meaningful statistical features
print("\n[2/4] Extracting 6 statistical features (for fair comparison)...")
time_steps, height, width = asc_data.shape
combined_data = (asc_data + desc_data) / 2
combined_flat = combined_data.reshape(time_steps, -1)

# Select 6 most informative features based on physical meaning
mean_per_time = np.mean(combined_flat, axis=1)
std_per_time = np.std(combined_flat, axis=1)
max_per_time = np.max(combined_flat, axis=1)
min_per_time = np.min(combined_flat, axis=1)
range_per_time = max_per_time - min_per_time
median_per_time = np.median(combined_flat, axis=1)

stat_features = np.column_stack([
    mean_per_time, std_per_time, max_per_time, 
    min_per_time, range_per_time, median_per_time
])

print(f"  Extracted {stat_features.shape[1]} statistical features")
print(f"  Features: Mean, Std, Max, Min, Range, Median")

# Generate labels
print("\n[3/4] Generating labels (window=7, threshold=0.35)...")
window_size = 7
threshold = 0.35

vertical = (asc_data + desc_data) / 2
horizontal = (asc_data - desc_data) / 2
displacement = np.sqrt(vertical**2 + horizontal**2)

cumulative_rolling = np.zeros_like(displacement)
for t in range(time_steps - window_size + 1):
    cumulative_rolling[t] = displacement[t:t + window_size].sum(axis=0)

labels = (cumulative_rolling > threshold).astype(int)
labels = labels.reshape(labels.shape[0], -1).sum(axis=1) > 0
labels = labels.astype(int)[:stat_features.shape[0]]

print(f"  Labels: Stable={np.sum(labels==0)}, Unstable={np.sum(labels==1)}")

# Load rainfall
print("\n[4/4] Loading rainfall and training...")
rainfall_files = sorted([os.path.join(rainfall_dir, f) for f in os.listdir(rainfall_dir) if f.endswith('.nc4')])
rainfall_data = []
for f in rainfall_files:
    try:
        ds = xr.open_dataset(f)
        if 'MWprecipitation' in ds:
            rainfall_data.append(ds['MWprecipitation'].mean(dim=['lon', 'lat']).to_dataframe())
    except:
        pass

rainfall_df = pd.concat(rainfall_data).reset_index()
rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])
aligned_rainfall = rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"].mean().values
aligned_rainfall = aligned_rainfall[:stat_features.shape[0]]

# Combine features: 6 stats + 1 rainfall = 7 total (matches PCA!)
all_features = np.hstack([stat_features, aligned_rainfall.reshape(-1, 1)])

print(f"  Total features: {all_features.shape[1]} (6 stats + 1 rainfall)")
print(f"  *** FAIR COMPARISON: Same as PCA (6 components + 1 rainfall) ***")

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(
    all_features, labels, test_size=0.2, random_state=42, stratify=labels
)

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"AUC Score: {auc:.4f} ({auc*100:.2f}%)")
print("="*70)

print("\n" + "="*70)
print("FAIR COMPARISON (Same # of features):")
print("="*70)
print("Model A (PCA - 6 components + rain):  Accuracy: 91.67%, F1: 93.02%, AUC: 95.29%")
print(f"Model G (RF - 6 stats + rain):        Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")
print("="*70)

# Show feature importances
importances = rf.feature_importances_
feature_names = ['Mean', 'Std', 'Max', 'Min', 'Range', 'Median', 'Rainfall']
sorted_idx = np.argsort(importances)[::-1]

print("\n" + "="*70)
print("FEATURE IMPORTANCES:")
print("="*70)
for i, idx in enumerate(sorted_idx):
    print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
print("="*70)

