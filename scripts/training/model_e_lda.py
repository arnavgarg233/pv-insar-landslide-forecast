"""
Model E: LDA (Linear Discriminant Analysis) + Rainfall
Supervised dimensionality reduction that maximizes class separation
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL E: LDA (Linear Discriminant Analysis)")
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

time_steps, height, width = asc_data.shape

# Prepare data for LDA - extract statistical features first
print("\n[2/5] Extracting features for LDA input...")
combined_data = (asc_data + desc_data) / 2
combined_flat = combined_data.reshape(time_steps, -1)

# Extract statistical features (same as Model B)
mean_per_time = np.mean(combined_flat, axis=1)
std_per_time = np.std(combined_flat, axis=1)
max_per_time = np.max(combined_flat, axis=1)
min_per_time = np.min(combined_flat, axis=1)
median_per_time = np.median(combined_flat, axis=1)
range_per_time = max_per_time - min_per_time
time_idx = np.arange(time_steps)
velocity_per_pixel = np.polyfit(time_idx, combined_flat, 1)[0]
avg_velocity = np.mean(velocity_per_pixel)
p25_per_time = np.percentile(combined_flat, 25, axis=1)
p75_per_time = np.percentile(combined_flat, 75, axis=1)

stat_features = np.column_stack([
    mean_per_time, std_per_time, max_per_time, min_per_time, 
    median_per_time, range_per_time, 
    np.full(time_steps, avg_velocity),
    p25_per_time, p75_per_time
])

print(f"  Statistical features: {stat_features.shape}")

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
labels = labels.astype(int)[:stat_features.shape[0]]

print(f"  Labels: Stable={np.sum(labels==0)}, Unstable={np.sum(labels==1)}")

# Load rainfall
print("\n[4/5] Loading rainfall...")
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

# Apply LDA for dimensionality reduction
print("\n[5/5] Applying LDA and training classifier...")
# LDA can extract at most min(n_features, n_classes-1) components
# With 2 classes (stable/unstable), we can get at most 1 LDA component
# But we'll try to get more by using n_components parameter
n_lda_components = min(6, len(np.unique(labels)) - 1)  # At most n_classes - 1
print(f"  LDA will extract {n_lda_components} component(s)")

# Split data first to avoid leakage
X_train_stat, X_test_stat, y_train, y_test = train_test_split(
    stat_features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Fit LDA on training data only
lda = LinearDiscriminantAnalysis(n_components=n_lda_components)
X_train_lda = lda.fit_transform(X_train_stat, y_train)
X_test_lda = lda.transform(X_test_stat)

print(f"  LDA features shape (train): {X_train_lda.shape}")
print(f"  LDA features shape (test): {X_test_lda.shape}")

# Add rainfall to LDA features
rainfall_train = aligned_rainfall[:len(y_train)]
rainfall_test = aligned_rainfall[len(y_train):len(y_train)+len(y_test)]

# Need to align rainfall properly with train/test split
# Get indices from train_test_split
X_train_full, X_test_full, y_train_check, y_test_check, train_idx, test_idx = train_test_split(
    stat_features, labels, np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels
)

rainfall_train = aligned_rainfall[train_idx]
rainfall_test = aligned_rainfall[test_idx]

X_train_combined = np.hstack([X_train_lda, rainfall_train.reshape(-1, 1)])
X_test_combined = np.hstack([X_test_lda, rainfall_test.reshape(-1, 1)])

print(f"  Combined features (train): {X_train_combined.shape}")
print(f"  Combined features (test): {X_test_combined.shape}")

# Train Random Forest on LDA features
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train_combined, y_train)

y_pred = rf.predict(X_test_combined)
y_prob = rf.predict_proba(X_test_combined)[:, 1]

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
print("FULL COMPARISON:")
print("="*70)
print("Model A (PCA):               Accuracy: 91.67%, F1: 93.02%, AUC: 95.29%")
print("Model B (RF Importance):     Accuracy: 91.67%, F1: 93.62%, AUC: 97.56%")
print("Model D (Raw InSAR + Rain):  Accuracy: 66.67%, F1: 76.00%, AUC: 54.55%")
print(f"Model E (LDA):               Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")
print("="*70)

print("\n" + "="*70)
print("LDA DETAILS:")
print("="*70)
print(f"  Number of LDA components: {n_lda_components}")
print(f"  Explained variance ratio: {lda.explained_variance_ratio_}")
print("="*70)

