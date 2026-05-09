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

import os
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_displacement(cumulative_displacement):
    """Analyze cumulative displacement to determine a good threshold."""
    flattened_displacement = cumulative_displacement.reshape(-1)
    flattened_displacement = flattened_displacement[flattened_displacement > 0]  # Remove zeros

    print("Displacement Statistics:")
    print(f"Min: {np.min(flattened_displacement):.2f} cm")
    print(f"Max: {np.max(flattened_displacement):.2f} cm")
    print(f"Mean: {np.mean(flattened_displacement):.2f} cm")
    print(f"Median: {np.median(flattened_displacement):.2f} cm")
    print(f"90th Percentile: {np.percentile(flattened_displacement, 90):.2f} cm")

def load_insar_data(file_path):
    """Load InSAR time series from .h5 file."""
    with h5py.File(file_path, 'r') as f:
        timeseries = f['timeseries'][:]  # Shape: (time_steps, height, width)
        dates = f['date'][:]
        return timeseries, dates

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

def align_data(rainfall_df, displacement_labels, displacement_shape):
    """Align rainfall data with displacement labels."""
    aligned_rainfall = rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"].mean().values
    aligned_rainfall = aligned_rainfall[:displacement_labels.shape[0]]  # Ensure alignment

    displacement_labels = displacement_labels.reshape(displacement_labels.shape[0], -1).sum(axis=1) > 0
    return aligned_rainfall, displacement_labels

def train_and_evaluate_model(X, Y, include_time=False):
    """Train and evaluate a Random Forest model."""
    if include_time:
        X = np.hstack([X, np.arange(len(X)).reshape(-1, 1)])  # Add time as a feature

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None

    # Return metrics and confusion matrix
    return acc, f1, auc, confusion_matrix(y_test, y_pred)

# File paths
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
rainfall_dir = str(RAINFALL_DIR)

# Load data
asc_data, asc_dates = load_insar_data(asc_file)
desc_data, desc_dates = load_insar_data(desc_file)
if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]
elif desc_data.shape[0] > asc_data.shape[0]:
    desc_data = desc_data[:asc_data.shape[0]]

rainfall_df = load_rainfall_data(rainfall_dir)

# Hyperparameter search
best_config = None
best_acc = 0
best_f1 = 0
best_class_balance = None  # To store class balance of the best configuration

print("Starting hyperparameter search...\n")
print(f"{'Window Size':<15}{'Threshold':<15}{'Accuracy':<15}{'F1 Score':<15}{'AUC Score':<15}")
print("-" * 70)

for window_size in range(5, 30, 2):  # Test window sizes from 5 to 15
    for threshold in np.arange(0.05, 0.7, 0.05):  # Test thresholds from 0.05 to 0.25
        rolling_displacement = calculate_cumulative_displacement_rolling(asc_data, desc_data, window_size)
        landslide_labels = (rolling_displacement > threshold).astype(int)

        # Align rainfall and displacement data
        X, Y = align_data(rainfall_df, landslide_labels, rolling_displacement.shape)

        # Calculate class balance
        class_balance = np.bincount(Y)

        # Train and evaluate the model
        acc, f1, auc, cm = train_and_evaluate_model(X.reshape(-1, 1), Y, include_time=True)

        # Output metrics for this configuration
        print(f"{window_size:<15}{threshold:<15.2f}{acc:<15.2f}{f1:<15.2f}{(auc if auc is not None else 'N/A'):<15}")

        # Skip overfitting configurations
        if acc > 0.97 or f1 > 0.99:
            print(f"Skipping overfitting config: window_size={window_size}, threshold={threshold}, acc={acc:.2f}")
            continue

        # Save best configuration
        if acc > best_acc and f1 > best_f1:
            best_acc = acc
            best_f1 = f1
            best_config = (window_size, threshold, acc, f1, auc, cm)
            best_class_balance = class_balance

# Report best results
if best_config:
    window_size, threshold, acc, f1, auc, cm = best_config
    print("\nBest Configuration:")
    print(f"Window Size: {window_size}")
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}" if auc is not None else "AUC Score: N/A")
    print("Confusion Matrix:")
    print(cm)
    print(f"Initial Class Balance (No Landslide vs. Landslide): {best_class_balance}")

    # Visualize confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
else:
    print("No valid configuration found.")
