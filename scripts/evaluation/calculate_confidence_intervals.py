"""
Calculate 95% Confidence Intervals for Model Performance Metrics
Using bootstrapping on the test set
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CALCULATING 95% CONFIDENCE INTERVALS")
print("="*80)

# File paths
pca_file = str(PCA_H5)
rainfall_dir = str(RAINFALL_DIR)
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)

print("\nLoading data...")

# Load PCA features (ALL 5 components from each orbit - matches original model)
with h5py.File(pca_file, 'r') as f:
    asc_pca = f['ascending/scores'][:]  # All 5 components
    desc_pca = f['descending/scores'][:]  # All 5 components

# Match dimensions (asc has 180, desc has 179)
pca_features = np.hstack([asc_pca[:desc_pca.shape[0]], desc_pca])

# Load InSAR for labels
with h5py.File(asc_file, 'r') as f:
    asc_data = f['timeseries'][:]
with h5py.File(desc_file, 'r') as f:
    desc_data = f['timeseries'][:]

if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]

# Generate labels
window_size = 7
threshold = 0.35
vertical = (asc_data + desc_data) / 2
horizontal = (asc_data - desc_data) / 2
displacement = np.sqrt(vertical**2 + horizontal**2)
cumulative_rolling = np.zeros_like(displacement)
for t in range(asc_data.shape[0] - window_size + 1):
    cumulative_rolling[t] = displacement[t:t + window_size].sum(axis=0)
labels = (cumulative_rolling > threshold).astype(int)
labels = labels.reshape(labels.shape[0], -1).sum(axis=1) > 0
labels = labels.astype(int)

# Load rainfall
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

# Match dimensions
min_len = min(pca_features.shape[0], len(labels), len(aligned_rainfall))
pca_features = pca_features[:min_len]
labels = labels[:min_len]
aligned_rainfall = aligned_rainfall[:min_len]

# Combine features
X = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])

print(f"Total samples: {len(labels)}, Stable: {np.sum(labels==0)}, Unstable: {np.sum(labels==1)}")

# Train-test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Test set size: {len(y_test)}")

# Train the model
print("\nTraining model...")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Get predictions on test set
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# Calculate original metrics
acc_orig = accuracy_score(y_test, y_pred)
prec_orig = precision_score(y_test, y_pred)
rec_orig = recall_score(y_test, y_pred)
f1_orig = f1_score(y_test, y_pred)
auc_orig = roc_auc_score(y_test, y_prob)

print(f"\nOriginal Test Set Performance:")
print(f"  Accuracy:  {acc_orig*100:.2f}%")
print(f"  Precision: {prec_orig*100:.2f}%")
print(f"  Recall:    {rec_orig*100:.2f}%")
print(f"  F1-Score:  {f1_orig*100:.2f}%")
print(f"  AUC:       {auc_orig*100:.2f}%")

# Bootstrap confidence intervals
print("\n" + "="*80)
print("BOOTSTRAPPING (1000 iterations)...")
print("="*80)

n_bootstrap = 1000
bootstrap_accs = []
bootstrap_precs = []
bootstrap_recs = []
bootstrap_f1s = []
bootstrap_aucs = []

np.random.seed(42)
for i in range(n_bootstrap):
    # Resample test set with replacement
    indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
    y_test_boot = y_test[indices]
    y_pred_boot = y_pred[indices]
    y_prob_boot = y_prob[indices]
    
    # Calculate metrics
    try:
        acc = accuracy_score(y_test_boot, y_pred_boot)
        prec = precision_score(y_test_boot, y_pred_boot, zero_division=0)
        rec = recall_score(y_test_boot, y_pred_boot, zero_division=0)
        f1 = f1_score(y_test_boot, y_pred_boot, zero_division=0)
        # AUC requires both classes
        if len(np.unique(y_test_boot)) > 1:
            auc = roc_auc_score(y_test_boot, y_prob_boot)
        else:
            continue  # Skip if only one class
        
        bootstrap_accs.append(acc)
        bootstrap_precs.append(prec)
        bootstrap_recs.append(rec)
        bootstrap_f1s.append(f1)
        bootstrap_aucs.append(auc)
    except:
        continue
    
    if (i+1) % 200 == 0:
        print(f"  Completed {i+1}/{n_bootstrap} iterations...")

# Calculate 95% confidence intervals
acc_ci = np.percentile(bootstrap_accs, [2.5, 97.5])
prec_ci = np.percentile(bootstrap_precs, [2.5, 97.5])
rec_ci = np.percentile(bootstrap_recs, [2.5, 97.5])
f1_ci = np.percentile(bootstrap_f1s, [2.5, 97.5])
auc_ci = np.percentile(bootstrap_aucs, [2.5, 97.5])

print("\n" + "="*80)
print("95% CONFIDENCE INTERVALS (Bootstrap Method)")
print("="*80)
print(f"Accuracy:   {acc_orig*100:.2f}% [95% CI: {acc_ci[0]*100:.2f}% - {acc_ci[1]*100:.2f}%]")
print(f"Precision:  {prec_orig*100:.2f}% [95% CI: {prec_ci[0]*100:.2f}% - {prec_ci[1]*100:.2f}%]")
print(f"Recall:     {rec_orig*100:.2f}% [95% CI: {rec_ci[0]*100:.2f}% - {rec_ci[1]*100:.2f}%]")
print(f"F1-Score:   {f1_orig*100:.2f}% [95% CI: {f1_ci[0]*100:.2f}% - {f1_ci[1]*100:.2f}%]")
print(f"AUC:        {auc_orig*100:.2f}% [95% CI: {auc_ci[0]*100:.2f}% - {auc_ci[1]*100:.2f}%]")
print("="*80)

# Standard errors
acc_se = np.std(bootstrap_accs)
prec_se = np.std(bootstrap_precs)
rec_se = np.std(bootstrap_recs)
f1_se = np.std(bootstrap_f1s)
auc_se = np.std(bootstrap_aucs)

print("\n" + "="*80)
print("STANDARD ERRORS")
print("="*80)
print(f"Accuracy:   {acc_orig*100:.2f}% ± {acc_se*100:.2f}%")
print(f"Precision:  {prec_orig*100:.2f}% ± {prec_se*100:.2f}%")
print(f"Recall:     {rec_orig*100:.2f}% ± {rec_se*100:.2f}%")
print(f"F1-Score:   {f1_orig*100:.2f}% ± {f1_se*100:.2f}%")
print(f"AUC:        {auc_orig*100:.2f}% ± {auc_se*100:.2f}%")
print("="*80)

# LaTeX formatted output
print("\n" + "="*80)
print("LATEX FORMAT:")
print("="*80)
print(f"Accuracy:  {acc_orig*100:.2f}\\% (95\\% CI: {acc_ci[0]*100:.2f}\\% -- {acc_ci[1]*100:.2f}\\%)")
print(f"Precision: {prec_orig*100:.2f}\\% (95\\% CI: {prec_ci[0]*100:.2f}\\% -- {prec_ci[1]*100:.2f}\\%)")
print(f"Recall:    {rec_orig*100:.2f}\\% (95\\% CI: {rec_ci[0]*100:.2f}\\% -- {rec_ci[1]*100:.2f}\\%)")
print(f"F1-Score:  {f1_orig*100:.2f}\\% (95\\% CI: {f1_ci[0]*100:.2f}\\% -- {f1_ci[1]*100:.2f}\\%)")
print(f"AUC:       {auc_orig*100:.2f}\\% (95\\% CI: {auc_ci[0]*100:.2f}\\% -- {auc_ci[1]*100:.2f}\\%)")
print("="*80)

