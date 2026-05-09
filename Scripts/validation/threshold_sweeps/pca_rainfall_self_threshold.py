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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import xarray as xr

# Functions to load data
def load_insar_data(file_path):
    """Load InSAR time series from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        timeseries = f['timeseries'][:]  # Shape: (time_steps, height, width)
        dates = f['date'][:]
        return timeseries, dates

def load_pca_features(pca_file):
    """Load PCA features from the provided HDF5 file."""
    with h5py.File(pca_file, 'r') as f:
        asc_scores = f['ascending/scores'][:]
        desc_scores = f['descending/scores'][:]
    combined_pca_features = np.hstack([asc_scores[:desc_scores.shape[0]], desc_scores])
    return combined_pca_features

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
    Aggregates labels to ensure alignment with PCA feature dimensions.
    """
    labels = (rolling_displacement > threshold).astype(int)
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

def align_data(rainfall_df, displacement_labels, displacement_shape):
    """Align rainfall data with displacement labels."""
    aligned_rainfall = rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"].mean().values
    aligned_rainfall = aligned_rainfall[:displacement_labels.shape[0]]  # Ensure alignment

    displacement_labels = displacement_labels.reshape(displacement_labels.shape[0], -1).sum(axis=1) > 0
    return aligned_rainfall, displacement_labels

def train_and_evaluate_model(X, Y):
    """Train and evaluate a Random Forest model."""
    # Ensure two classes
    if len(np.unique(Y)) < 2:
        print("Only one class present in the labels. Skipping configuration.")
        return None, None, None, None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Train Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None

    return acc, f1, auc, confusion_matrix(y_test, y_pred), model

# File paths
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
pca_file = str(PCA_H5)
rainfall_dir = str(RAINFALL_DIR)

# Load data
asc_data, _ = load_insar_data(asc_file)
desc_data, _ = load_insar_data(desc_file)
if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]

pca_features = load_pca_features(pca_file)
rainfall_df = load_rainfall_data(rainfall_dir)

# Hyperparameter search
best_config = None
best_acc = 0
best_f1 = 0
best_class_balance = None  # To store class balance of the best configuration
best_model = None

print("Starting hyperparameter search...\n")
print(f"{'Window Size':<15}{'Threshold':<15}{'Accuracy':<15}{'F1 Score':<15}{'AUC Score':<15}")
print("-" * 70)

for window_size in range(6, 9, 1):  # Adjust window sizes
    rolling_displacement = calculate_cumulative_displacement_rolling(asc_data, desc_data, window_size)
    for threshold in np.arange(0.05, 0.7, 0.05):  # Adjust thresholds
        # Generate labels
        labels = generate_labels(rolling_displacement, threshold)

        # Ensure labels match PCA feature dimensions
        labels = labels[:pca_features.shape[0]]

        # Align rainfall data
        aligned_rainfall, aligned_labels = align_data(rainfall_df, labels, rolling_displacement.shape)

        # Combine PCA features with rainfall data
        combined_features = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])

        # Train and evaluate the model
        acc, f1, auc, cm, model = train_and_evaluate_model(combined_features, aligned_labels)

        # Skip configurations with invalid metrics
        if acc > 0.99 or f1 > 0.99 or (auc is not None and auc > 0.99):
            print(f"Skipping overfitting config: window_size={window_size}, threshold={threshold}, acc={acc:.2f}")
            continue

        # Output metrics for this configuration
        print(f"{window_size:<15}{threshold:<15.2f}{acc:<15.2f}{f1:<15.2f}{(auc if auc is not None else 'N/A'):<15}")

        # Save best configuration
        if acc > best_acc and f1 > best_f1:
            best_acc = acc
            best_f1 = f1
            best_config = (window_size, threshold, acc, f1, auc, cm)
            best_model = model

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

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save the confusion matrix plot
    output_path = os.path.join(os.getcwd(), "confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to avoid overwriting
    print(f"Confusion matrix saved to: {output_path}")
else:
    print("No valid configuration found.")
