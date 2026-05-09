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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

def load_insar_data(file_path):
    """Load InSAR time series from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        timeseries = f['timeseries'][:]  # Shape: (time_steps, height, width)
        dates = f['date'][:]
        return timeseries, dates

def extract_raw_features(timeseries_data):
    """
    Extract raw statistical features from InSAR time series for each pixel.
    Returns features: mean, std, max, min, velocity, acceleration, etc.
    """
    time_steps, height, width = timeseries_data.shape
    num_pixels = height * width
    
    # Flatten spatial dimensions
    flat_data = timeseries_data.reshape(time_steps, -1).T  # Shape: (num_pixels, time_steps)
    
    features = []
    feature_names = []
    
    # Statistical features
    features.append(np.mean(flat_data, axis=1))
    feature_names.append('mean_displacement')
    
    features.append(np.std(flat_data, axis=1))
    feature_names.append('std_displacement')
    
    features.append(np.max(flat_data, axis=1))
    feature_names.append('max_displacement')
    
    features.append(np.min(flat_data, axis=1))
    feature_names.append('min_displacement')
    
    features.append(np.median(flat_data, axis=1))
    feature_names.append('median_displacement')
    
    # Velocity (linear trend)
    time_indices = np.arange(time_steps)
    velocities = np.array([np.polyfit(time_indices, pixel, 1)[0] for pixel in flat_data])
    features.append(velocities)
    feature_names.append('velocity')
    
    # Range (max - min)
    features.append(np.max(flat_data, axis=1) - np.min(flat_data, axis=1))
    feature_names.append('range')
    
    # Percentiles
    features.append(np.percentile(flat_data, 25, axis=1))
    feature_names.append('percentile_25')
    
    features.append(np.percentile(flat_data, 75, axis=1))
    feature_names.append('percentile_75')
    
    # Recent trend (last 25% of time series)
    recent_window = time_steps // 4
    recent_data = flat_data[:, -recent_window:]
    recent_trend = np.array([np.polyfit(np.arange(recent_window), pixel, 1)[0] for pixel in recent_data])
    features.append(recent_trend)
    feature_names.append('recent_velocity')
    
    # Stack all features
    feature_matrix = np.column_stack(features)
    
    return feature_matrix, feature_names

def calculate_cumulative_displacement_rolling(asc, desc, window_size):
    """Calculate cumulative displacement over rolling windows."""
    vertical = (asc + desc) / 2
    horizontal = (asc - desc) / 2
    displacement_magnitude = np.sqrt(vertical**2 + horizontal**2)
    time_steps, height, width = displacement_magnitude.shape
    cumulative_rolling = np.zeros_like(displacement_magnitude)

    for t in range(time_steps - window_size + 1):
        cumulative_rolling[t] = displacement_magnitude[t:t + window_size].sum(axis=0)

    return cumulative_rolling

def generate_labels(rolling_displacement, threshold):
    """
    Generate labels from rolling displacement by applying a threshold.
    Aggregates labels to ensure alignment with feature dimensions.
    """
    labels = (rolling_displacement > threshold).astype(int)
    # Flatten the labels and reduce to one label per sample
    labels = labels.reshape(labels.shape[0], -1).sum(axis=1) > 0  # Aggregate spatially
    return labels.astype(int)

def load_rainfall_data(rainfall_dir):
    """Load rainfall data from .nc4 files."""
    rainfall_files = [os.path.join(rainfall_dir, f) for f in os.listdir(rainfall_dir) if f.endswith('.nc4')]
    rainfall_data = []

    for file in rainfall_files:
        try:
            ds = xr.open_dataset(file)
            if 'MWprecipitation' in ds:
                daily_precip = ds['MWprecipitation'].mean(dim=['lon', 'lat']).to_dataframe()
                rainfall_data.append(daily_precip)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    rainfall_df = pd.concat(rainfall_data).reset_index()
    rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])
    return rainfall_df

def align_data(rainfall_df, displacement_labels):
    """Align rainfall data with displacement labels."""
    aligned_rainfall = rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"].mean().values
    aligned_rainfall = aligned_rainfall[:len(displacement_labels)]  # Ensure alignment
    return aligned_rainfall

def train_and_evaluate_model(X, Y):
    """Train and evaluate a Random Forest model."""
    # Ensure two classes
    if len(np.unique(Y)) < 2:
        print("Only one class present in the labels. Skipping configuration.")
        return None, None, None, None, None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    # Train Random Forest
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None

    return acc, f1, auc, confusion_matrix(y_test, y_pred), model

# File paths - UPDATE THESE TO YOUR PATHS
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
rainfall_dir = str(RAINFALL_DIR)

print("="*70)
print("MODEL B: Random Forest Feature Importance Approach")
print("="*70)

# Load data
print("\n[1/6] Loading InSAR data...")
asc_data, _ = load_insar_data(asc_file)
desc_data, _ = load_insar_data(desc_file)

if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]

print(f"  Ascending data shape: {asc_data.shape}")
print(f"  Descending data shape: {desc_data.shape}")

# Extract raw features
print("\n[2/6] Extracting raw statistical features from InSAR time series...")
asc_features, asc_feature_names = extract_raw_features(asc_data)
desc_features, desc_feature_names = extract_raw_features(desc_data)

# Combine ascending and descending features
combined_features = np.hstack([asc_features, desc_features])
feature_names = [f"asc_{name}" for name in asc_feature_names] + [f"desc_{name}" for name in desc_feature_names]

print(f"  Total raw features extracted: {combined_features.shape[1]}")
print(f"  Feature names: {feature_names}")

# Load rainfall
print("\n[3/6] Loading rainfall data...")
rainfall_df = load_rainfall_data(rainfall_dir)

# Parameters (same as Model A for fair comparison)
window_size = 7
threshold = 0.35

print(f"\n[4/6] Generating labels with window_size={window_size}, threshold={threshold}...")
# Calculate rolling displacement
rolling_displacement = calculate_cumulative_displacement_rolling(asc_data, desc_data, window_size)
labels = generate_labels(rolling_displacement, threshold)

# Ensure labels match feature dimensions
labels = labels[:combined_features.shape[0]]

# Align rainfall data
aligned_rainfall = align_data(rainfall_df, labels)

print(f"  Labels generated: {len(labels)} samples")
print(f"  Class distribution: Stable={np.sum(labels==0)}, Unstable={np.sum(labels==1)}")

# Step 1: Train initial RF to get feature importances
print("\n[5/6] Training initial RF to determine feature importances...")
# Combine all features including rainfall
initial_features = np.hstack([combined_features, aligned_rainfall.reshape(-1, 1)])
initial_feature_names = feature_names + ['rainfall']

X_train_init, X_test_init, y_train_init, y_test_init = train_test_split(
    initial_features, labels, test_size=0.2, random_state=42, stratify=labels
)

rf_initial = RandomForestClassifier(random_state=42, n_estimators=100)
rf_initial.fit(X_train_init, y_train_init)

# Get feature importances
importances = rf_initial.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n  Top 10 Most Important Features:")
for i in range(min(10, len(indices))):
    idx = indices[i]
    print(f"    {i+1}. {initial_feature_names[idx]}: {importances[idx]:.4f}")

# Step 2: Select top K features
K = 6  # Same number as PCA components for fair comparison
top_k_indices = indices[:K]
selected_features = initial_features[:, top_k_indices]
selected_feature_names = [initial_feature_names[i] for i in top_k_indices]

print(f"\n  Selected top {K} features:")
for name in selected_feature_names:
    print(f"    - {name}")

# Step 3: Train final model with selected features
print(f"\n[6/6] Training final RF with top {K} features...")
acc, f1, auc, cm, final_model = train_and_evaluate_model(selected_features, labels)

if acc is not None:
    print("\n" + "="*70)
    print("MODEL B PERFORMANCE (RF Feature Importance):")
    print("="*70)
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"AUC Score: {auc:.4f} ({auc*100:.2f}%)" if auc is not None else "AUC Score: N/A")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*70)
    
    print("\n" + "="*70)
    print("COMPARISON WITH MODEL A (PCA-based):")
    print("="*70)
    print("Model A (PCA):              Accuracy: 91.67%, F1: 93.02%, AUC: 95.29%")
    print(f"Model B (RF Importance):    Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")
    print("="*70)

