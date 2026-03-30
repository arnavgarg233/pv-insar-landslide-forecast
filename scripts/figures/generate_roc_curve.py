"""
Generate publication-quality ROC curve with multiple classifiers + smoothing.
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
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Data Loading
# ============================================================================
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
pca_file = str(PCA_H5)
rainfall_dir = str(RAINFALL_DIR)

with h5py.File(asc_file, "r") as f:
    asc_data = f["timeseries"][:]
with h5py.File(desc_file, "r") as f:
    desc_data = f["timeseries"][:]

if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[: desc_data.shape[0]]

time_steps = asc_data.shape[0]

with h5py.File(pca_file, "r") as f:
    asc_pca = f["ascending/scores"][:]
    desc_pca = f["descending/scores"][:]
min_samples = min(asc_pca.shape[0], desc_pca.shape[0])
pca_features = np.hstack([asc_pca[:min_samples], desc_pca[:min_samples]])

window_size = 7
threshold = 0.35
vertical = (asc_data + desc_data) / 2
horizontal = (asc_data - desc_data) / 2
displacement = np.sqrt(vertical**2 + horizontal**2)
cumulative_rolling = np.zeros_like(displacement)
for t in range(time_steps - window_size + 1):
    cumulative_rolling[t] = displacement[t : t + window_size].sum(axis=0)
labels = (cumulative_rolling > threshold).astype(int)
labels = labels.reshape(labels.shape[0], -1).sum(axis=1) > 0
labels = labels.astype(int)

rainfall_files = sorted(
    [os.path.join(rainfall_dir, f) for f in os.listdir(rainfall_dir) if f.endswith(".nc4")]
)
rainfall_data = []
for f in rainfall_files:
    try:
        ds = xr.open_dataset(f)
        if "MWprecipitation" in ds:
            rainfall_data.append(
                ds["MWprecipitation"].mean(dim=["lon", "lat"]).to_dataframe()
            )
    except Exception:
        pass
rainfall_df = pd.concat(rainfall_data).reset_index()
rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])
aligned_rainfall = (
    rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"].mean().values
)

min_len = min(pca_features.shape[0], len(labels), len(aligned_rainfall))
pca_features = pca_features[:min_len]
labels = labels[:min_len]
aligned_rainfall = aligned_rainfall[:min_len]

X = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================================
# Helper: smooth a staircase ROC into a gentle curve
# ============================================================================
def smooth_roc(fpr_raw, tpr_raw, n_points=200):
    """Monotonic spline interpolation that preserves start/end anchors."""
    fpr_raw = np.concatenate([[0], fpr_raw, [1]])
    tpr_raw = np.concatenate([[0], tpr_raw, [1]])

    unique_mask = np.diff(fpr_raw) > 0
    unique_mask = np.concatenate([[True], unique_mask])
    fpr_u = fpr_raw[unique_mask]
    tpr_u = tpr_raw[unique_mask]

    if len(fpr_u) < 4:
        return fpr_u, tpr_u

    spline = make_interp_spline(fpr_u, tpr_u, k=3)
    fpr_smooth = np.linspace(0, 1, n_points)
    tpr_smooth = np.clip(spline(fpr_smooth), 0, 1)
    tpr_smooth = np.maximum.accumulate(tpr_smooth)
    return fpr_smooth, tpr_smooth


# ============================================================================
# Train classifiers and collect ROC data
# ============================================================================
classifiers = [
    ("Random Forest (Proposed)", RandomForestClassifier(random_state=42, n_estimators=100), False),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5), True),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3), False),
    ("SVM (RBF Kernel)", SVC(kernel="rbf", probability=True, random_state=42), True),
    ("AdaBoost", AdaBoostClassifier(random_state=42, n_estimators=100), False),
]

colors = ["#1a5276", "#c0392b", "#27ae60", "#8e44ad", "#e67e22"]
linewidths = [2.8, 1.6, 1.6, 1.6, 1.6]
linestyles = ["-", "--", "-.", ":", "--"]

fig, ax = plt.subplots(figsize=(7, 6))

for i, (name, clf, needs_scaling) in enumerate(classifiers):
    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_test_scaled if needs_scaling else X_test

    clf.fit(X_tr, y_train)
    y_prob = clf.predict_proba(X_te)[:, 1]

    fpr_raw, tpr_raw, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr_raw, tpr_raw)

    fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)

    ax.plot(
        fpr_s,
        tpr_s,
        color=colors[i],
        linewidth=linewidths[i],
        linestyle=linestyles[i],
        label=f"{name} (AUC = {roc_auc:.2f})",
    )

ax.plot([0, 1], [0, 1], color="#bdc3c7", linewidth=1.0, linestyle="--", label="Random Chance")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_aspect("equal")
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.25, linestyle="-")

plt.tight_layout()

output_path = str(RF_DATA_DIR / "ROC_Curve.jpg")
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"ROC curve saved to: {output_path}")
plt.close()
