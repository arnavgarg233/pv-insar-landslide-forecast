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

    return acc, f1, auc, confusion_matrix(y_test, y_pred)

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

# Hyperparameter search
best_config = None
best_acc = 0
best_f1 = 0
best_class_balance = None  # To store class balance of the best configuration

print("Starting hyperparameter search...\n")
print(f"{'Window Size':<15}{'Threshold':<15}{'Accuracy':<15}{'F1 Score':<15}{'AUC Score':<15}")
print("-" * 70)

# Updated section to enforce class distribution condition
for window_size in range(7,9,1):  # Test window sizes
    rolling_displacement = calculate_cumulative_displacement_rolling(asc_data, desc_data, window_size)
    for threshold in np.arange(0.05, 5, 0.05):  # Test thresholds
        # Generate labels
        labels = generate_labels(rolling_displacement, threshold)

        # Ensure labels match PCA feature dimensions
        labels = labels[:pca_features.shape[0]]

        # Calculate class balance
        class_balance = np.bincount(labels)

        # Check class distribution condition: smaller class count >= half of larger class count
        if len(class_balance) < 2 or min(class_balance) < 0.5 * max(class_balance):
            print(f"Skipping config due to class imbalance: {class_balance}")
            continue

        # Train and evaluate the model
        acc, f1, auc, cm = train_and_evaluate_model(pca_features, labels)

        # Skip configurations with invalid metrics
        if acc is None:
            continue

        # Output metrics for this configuration
        print(f"{window_size:<15}{threshold:<15.2f}{acc:<15.2f}{f1:<15.2f}{(auc if auc is not None else 'N/A'):<15}")

        # Skip overfitting configurations
        if acc > 0.99 or f1 > 0.99 or (auc is not None and auc > 0.99):
            print(f"Skipping overfitting config: window_size={window_size}, threshold={threshold}, acc={acc:.2f}")
            continue

        # Save best configuration
        if acc > best_acc and f1 > best_f1:
            best_acc = acc
            best_f1 = f1
            best_config = (window_size, threshold, acc, f1, auc, cm)
            best_class_balance = class_balance

# The rest of the code remains unchanged


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
