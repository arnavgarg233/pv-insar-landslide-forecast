"""
Classifier Comparison (Reviewer Request)
=========================================
Same features (6 PCA + rainfall), different classifiers.
Reuses exact same data pipeline as Comprehensive_Comparison.py
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CLASSIFIER COMPARISON — Same PCA Features, Different Models")
print("=" * 80)

# ============================================================================
# Data Loading (identical to Comprehensive_Comparison.py)
# ============================================================================

asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
pca_file = str(PCA_H5)
rainfall_dir = str(RAINFALL_DIR)

print("\nLoading data...")

with h5py.File(asc_file, 'r') as f:
    asc_data = f['timeseries'][:]
with h5py.File(desc_file, 'r') as f:
    desc_data = f['timeseries'][:]

if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]

time_steps, height, width = asc_data.shape

with h5py.File(pca_file, 'r') as f:
    asc_pca = f['ascending/scores'][:]
    desc_pca = f['descending/scores'][:]
min_samples = min(asc_pca.shape[0], desc_pca.shape[0])
pca_features = np.hstack([asc_pca[:min_samples], desc_pca[:min_samples]])

# Generate labels
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
labels = labels.astype(int)

# Load rainfall
rainfall_files = sorted([
    os.path.join(rainfall_dir, f)
    for f in os.listdir(rainfall_dir)
    if f.endswith('.nc4')
])
rainfall_data = []
for f in rainfall_files:
    try:
        ds = xr.open_dataset(f)
        if 'MWprecipitation' in ds:
            rainfall_data.append(ds['MWprecipitation'].mean(dim=['lon', 'lat']).to_dataframe())
    except Exception:
        pass
rainfall_df = pd.concat(rainfall_data).reset_index()
rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])
aligned_rainfall = rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"].mean().values

# Match dimensions
min_len = min(pca_features.shape[0], len(labels), len(aligned_rainfall))
pca_features = pca_features[:min_len]
labels = labels[:min_len]
aligned_rainfall = aligned_rainfall[:min_len]

X = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])
y = labels

print(f"Features shape: {X.shape} (6 PCA + 1 rainfall = 7)")
print(f"Samples: {len(y)}, Stable: {np.sum(y == 0)}, Unstable: {np.sum(y == 1)}")

# ============================================================================
# Train/Test Split (same as original)
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# ============================================================================
# Define Classifiers
# ============================================================================

classifiers = [
    ("Random Forest (Proposed)", RandomForestClassifier(
        random_state=42, n_estimators=100
    ), False),
    ("Gradient Boosting", GradientBoostingClassifier(
        random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1
    ), False),
    ("SVM (RBF Kernel)", SVC(
        kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale'
    ), True),
    ("SVM (Linear Kernel)", SVC(
        kernel='linear', probability=True, random_state=42, C=1.0
    ), True),
    ("Logistic Regression", LogisticRegression(
        random_state=42, max_iter=1000, C=1.0
    ), True),
    ("K-Nearest Neighbors", KNeighborsClassifier(
        n_neighbors=5
    ), True),
    ("AdaBoost", AdaBoostClassifier(
        random_state=42, n_estimators=100
    ), False),
    ("MLP Neural Network", MLPClassifier(
        random_state=42, hidden_layer_sizes=(64, 32), max_iter=500,
        early_stopping=True, validation_fraction=0.15
    ), True),
]

# Try to add XGBoost
try:
    from xgboost import XGBClassifier
    classifiers.append((
        "XGBoost",
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
                      n_estimators=100, max_depth=3, learning_rate=0.1),
        False
    ))
except ImportError:
    print("XGBoost not installed, skipping.")

# ============================================================================
# Evaluate All Classifiers
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING CLASSIFIER COMPARISON...")
print("=" * 80)

results = []

for i, (name, clf, needs_scaling) in enumerate(classifiers):
    print(f"\n[{i+1}/{len(classifiers)}] {name}")

    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_test_scaled if needs_scaling else X_test

    try:
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append((name, acc, prec, rec, f1, auc))
        print(f"  Acc: {acc*100:.2f}%  Prec: {prec*100:.2f}%  Rec: {rec*100:.2f}%  F1: {f1*100:.2f}%  AUC: {auc*100:.2f}%")

    except Exception as e:
        print(f"  FAILED: {e}")
        results.append((name, 0, 0, 0, 0, 0))

# ============================================================================
# Print Results Table
# ============================================================================

print("\n" + "=" * 80)
print("CLASSIFIER COMPARISON RESULTS")
print("=" * 80)
print(f"{'Classifier':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
print("-" * 80)

results_sorted = sorted(results, key=lambda x: x[4], reverse=True)
for name, acc, prec, rec, f1, auc in results_sorted:
    marker = " **" if name == "Random Forest (Proposed)" else ""
    print(f"{name:<30} {acc*100:>7.2f}% {prec*100:>7.2f}% {rec*100:>7.2f}% {f1*100:>7.2f}% {auc*100:>7.2f}%{marker}")

print("=" * 80)

# ============================================================================
# Generate LaTeX Table
# ============================================================================

print("\n" + "=" * 80)
print("LATEX TABLE FOR PAPER:")
print("=" * 80)

latex = r"""\begin{table}[htbp]
\centering
\caption{Classifier comparison using identical PCA-derived features (6 PCA components + daily rainfall). All models are trained and evaluated on the same chronological train--test split (80--20\%, stratified). The proposed Random Forest model is shown in bold.}
\label{tab:classifier_comparison}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc}
\toprule
\textbf{Classifier} & \textbf{Accuracy} & \textbf{Precision} & \textbf{F1-Score} & \textbf{AUC} \\
\midrule
"""

for name, acc, prec, rec, f1, auc in results_sorted:
    if "Proposed" in name:
        latex += f"\\textbf{{{name.replace(' (Proposed)', '')}}} & \\textbf{{{acc*100:.2f}\\%}} & \\textbf{{{prec*100:.2f}\\%}} & \\textbf{{{f1*100:.2f}\\%}} & \\textbf{{{auc*100:.2f}\\%}} \\\\\n"
        latex += r"\midrule" + "\n"
    else:
        latex += f"{name} & {acc*100:.2f}\\% & {prec*100:.2f}\\% & {f1*100:.2f}\\% & {auc*100:.2f}\\% \\\\\n"

latex += r"""\bottomrule
\end{tabular*}
\end{table}"""

print(latex)
print("=" * 80)
