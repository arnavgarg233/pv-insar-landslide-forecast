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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

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

# Load data
asc_data, _ = load_insar_data(asc_file)
desc_data, _ = load_insar_data(desc_file)
if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]

pca_features = load_pca_features(pca_file)

# Set fixed parameters
window_size = 8 # Specify the fixed window size
threshold = 0.4  # Specify the fixed threshold

# Calculate rolling displacement
rolling_displacement = calculate_cumulative_displacement_rolling(asc_data, desc_data, window_size)

# Generate labels
labels = generate_labels(rolling_displacement, threshold)

# Ensure labels match PCA feature dimensions
labels = labels[:pca_features.shape[0]]

# Calculate class balance
class_balance = np.bincount(labels)

# Check class distribution condition
if len(class_balance) < 2 or min(class_balance) < 0.5 * max(class_balance):
    print(f"Skipping due to class imbalance: {class_balance}")
else:
    # Train and evaluate the model
    acc, f1, auc, cm, model = train_and_evaluate_model(pca_features, labels)

    if acc is not None:
        # Output results
        print("\nModel Performance:")
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
        output_path = os.path.join(os.getcwd(), "confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to: {output_path}")

       # Generate risk probabilities
       # # Generate risk probabilities
risk_probabilities = model.predict_proba(pca_features)[:, 1]  # Probability of landslide

# Ensure the number of risk probabilities matches the spatial dimensions
if risk_probabilities.size != asc_data.shape[1] * asc_data.shape[2]:
    print(f"Mismatch: Risk probabilities size {risk_probabilities.size} does not match spatial dimensions {asc_data.shape[1]}x{asc_data.shape[2]}")
    # Resize risk_probabilities to match the spatial dimensions
    risk_map = np.resize(risk_probabilities, (asc_data.shape[1], asc_data.shape[2]))
else:
    # Reshape to create the risk map
    risk_map = risk_probabilities.reshape(asc_data.shape[1], asc_data.shape[2]) * 100  # Scale to percentage

# Plot and save the risk map
plt.figure(figsize=(8, 6))
cmap = plt.get_cmap("hot_r")
cmap.set_under("blue")
im = plt.imshow(risk_map, cmap=cmap, vmin=0.01, vmax=100)
plt.colorbar(im, label="Landslide Risk (%)")
plt.title(f"Predicted Landslide Risk Map (Random Forest)")
plt.tight_layout()
risk_map_output_path = os.path.join(os.getcwd(), "predicted_landslide_risk_map.png")
plt.savefig(risk_map_output_path, dpi=300)
plt.close()
print(f"Risk map saved to: {risk_map_output_path}")
