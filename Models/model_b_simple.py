"""
Model B: RF Feature Importance Comparison (Simplified Version)
Compares Random Forest feature importance approach with PCA approach
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
print("MODEL B: Random Forest Feature Importance Approach")
print("="*70)

# File paths
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
rainfall_dir = str(RAINFALL_DIR)

# Load InSAR data
print("\n[1/5] Loading InSAR data...")
with h5py.File(asc_file, 'r') as f:
    asc_data = f['timeseries'][:]
    
with h5py.File(desc_file, 'r') as f:
    desc_data = f['timeseries'][:]

if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]
    
print(f"  Data shape: {asc_data.shape}")

# Extract simple features (time-series level, not pixel level)
print("\n[2/5] Extracting statistical features from time series...")
time_steps, height, width = asc_data.shape

# Combine asc and desc
combined_data = (asc_data + desc_data) / 2
# Flatten spatial dimensions
combined_flat = combined_data.reshape(time_steps, -1)  # (time, pixels)

# Calculate spatial statistics for each timestep (this matches PCA's approach)
features_list = []
feature_names = []

# For each timestep, calculate spatial statistics
mean_per_time = np.mean(combined_flat, axis=1)  # (time_steps,)
std_per_time = np.std(combined_flat, axis=1)
max_per_time = np.max(combined_flat, axis=1)
min_per_time = np.min(combined_flat, axis=1)
median_per_time = np.median(combined_flat, axis=1)
range_per_time = max_per_time - min_per_time

features_list.extend([mean_per_time, std_per_time, max_per_time, min_per_time, median_per_time, range_per_time])
feature_names.extend(['mean', 'std', 'max', 'min', 'median', 'range'])

# Velocity over time
time_idx = np.arange(time_steps)
velocity_per_pixel = np.polyfit(time_idx, combined_flat, 1)[0]
avg_velocity = np.mean(velocity_per_pixel)
features_list.append(np.full(time_steps, avg_velocity))
feature_names.append('avg_velocity')

# Percentiles
p25_per_time = np.percentile(combined_flat, 25, axis=1)
p75_per_time = np.percentile(combined_flat, 75, axis=1)
features_list.extend([p25_per_time, p75_per_time])
feature_names.extend(['percentile_25', 'percentile_75'])

all_features = np.column_stack(features_list)  # Shape: (time_steps, n_features)
print(f"  Extracted {all_features.shape[1]} features for {all_features.shape[0]} timesteps")

# Generate labels
print("\n[3/5] Generating labels (window=7, threshold=0.35)...")
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
labels = labels.astype(int)[:all_features.shape[0]]

print(f"  Labels: Stable={np.sum(labels==0)}, Unstable={np.sum(labels==1)}")

# Load rainfall
print("\n[4/5] Loading rainfall...")
rainfall_files = sorted([os.path.join(rainfall_dir, f) for f in os.listdir(rainfall_dir) if f.endswith('.nc4')])
print(f"  Found {len(rainfall_files)} rainfall files")
rainfall_data = []
for f in rainfall_files:
    try:
        ds = xr.open_dataset(f)
        if 'MWprecipitation' in ds:
            rainfall_data.append(ds['MWprecipitation'].mean(dim=['lon', 'lat']).to_dataframe())
    except:
        pass
print(f"  Successfully loaded {len(rainfall_data)} rainfall files")

rainfall_df = pd.concat(rainfall_data).reset_index()
rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])
aligned_rainfall = rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"].mean().values
aligned_rainfall = aligned_rainfall[:all_features.shape[0]]  # Match to time_steps

print(f"  Features shape: {all_features.shape}, Rainfall shape: {aligned_rainfall.shape}, Labels shape: {labels.shape}")

# Combine features
all_features_with_rain = np.hstack([all_features, aligned_rainfall.reshape(-1, 1)])
feature_names.append('rainfall')

print(f"  Total features with rainfall: {all_features_with_rain.shape}")

# Train initial RF to get importances
print("\n[5/5] Training and evaluating...")
X_train, X_test, y_train, y_test = train_test_split(
    all_features_with_rain, labels, test_size=0.2, random_state=42, stratify=labels
)

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Get top 6 features
importances = rf.feature_importances_
top_6_idx = np.argsort(importances)[::-1][:6]

print("\n  Top 6 features:")
for i, idx in enumerate(top_6_idx):
    print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# Train with top 6
X_train_top = X_train[:, top_6_idx]
X_test_top = X_test[:, top_6_idx]

rf_final = RandomForestClassifier(random_state=42, n_estimators=100)
rf_final.fit(X_train_top, y_train)

y_pred = rf_final.predict(X_test_top)
y_prob = rf_final.predict_proba(X_test_top)[:, 1]

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
print("COMPARISON:")
print("="*70)
print("Model A (PCA):           Accuracy: 91.67%, F1: 93.02%, AUC: 95.29%")
print(f"Model B (RF Importance): Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")
print("="*70)

