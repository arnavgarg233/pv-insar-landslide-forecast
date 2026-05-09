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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import os

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
    print(y_test.shape)
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

# Parameters
window_size = 7  # Fixed window size
threshold = 0.35  # Fixed threshold

# Calculate rolling displacement
rolling_displacement = calculate_cumulative_displacement_rolling(asc_data, desc_data, window_size)
labels = generate_labels(rolling_displacement, threshold)

# Ensure labels match PCA feature dimensions
labels = labels[:pca_features.shape[0]]

# Align rainfall data
aligned_rainfall = align_data(rainfall_df, labels)

# Combine PCA features with rainfall data
combined_features = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])

# Train and evaluate the model
acc, f1, auc, cm, model = train_and_evaluate_model(combined_features, labels)
print(labels.shape)
if acc is not None:
    print("\nModel Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}" if auc is not None else "AUC Score: N/A")
    print("Confusion Matrix:")
    print(cm)

# Generate risk map using model confidence scores
height, width = asc_data.shape[1], asc_data.shape[2]  # Spatial dimensions
pixelwise_probabilities = np.zeros((height, width))  # Initialize risk map

for i in range(height):
    for j in range(width):
        # Extract PCA features for the specific pixel
        pixel_index = i * width + j
        if pixel_index >= pca_features.shape[0]:  # Check PCA range
            continue

        pixel_pca_features = pca_features[pixel_index, :]  # Correct PCA feature extraction

        # Combine PCA features with average rainfall
        pixel_rainfall = np.mean(aligned_rainfall)  # Average rainfall over time
        pixel_features_combined = np.hstack([pixel_pca_features, pixel_rainfall])  # Combine features

        # Check feature alignment with the model
        if pixel_features_combined.shape[0] != model.n_features_in_:
            continue  # Skip if feature mismatch

        # Get the predicted probabilities for each class
        prob_no_landslide, prob_landslide = model.predict_proba([pixel_features_combined])[0]

        # Assign risk based on the class prediction
        if prob_landslide >= prob_no_landslide:
            pixelwise_probabilities[i, j] = prob_landslide * 100  # Landslide risk as percentage
        else:
            pixelwise_probabilities[i, j] = prob_no_landslide * 50  # Scale inversely for no landslide

# Normalize the risk map for better visualization
risk_map = pixelwise_probabilities
print(f"Risk map normalized: min={risk_map.min()}, max={risk_map.max()}")

# Plot and save the risk map
plt.figure(figsize=(10, 8))
cmap = plt.get_cmap("hot_r")
im = plt.imshow(risk_map, cmap=cmap, vmin=0, vmax=100)
plt.colorbar(im, label="Landslide Risk (%)")
plt.title("Confidence-Based Landslide Risk Map")
plt.tight_layout()

risk_map_output_path = os.path.join(os.getcwd(), "confidence_based_landslide_risk_map.png")
plt.savefig(risk_map_output_path, dpi=300)
plt.show()

print(f"Confidence-based risk map saved to: {risk_map_output_path}")