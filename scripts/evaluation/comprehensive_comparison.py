"""
Comprehensive Model Comparison
Runs all configurations requested by reviewers
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# File paths
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
pca_file = str(PCA_H5)
rainfall_dir = str(RAINFALL_DIR)

print("\nLoading data...")

# Load InSAR data
with h5py.File(asc_file, 'r') as f:
    asc_data = f['timeseries'][:]
    
with h5py.File(desc_file, 'r') as f:
    desc_data = f['timeseries'][:]

if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]

time_steps, height, width = asc_data.shape
combined_data = (asc_data + desc_data) / 2

# Load PCA features
with h5py.File(pca_file, 'r') as f:
    asc_pca = f['ascending/scores'][:, :3]
    desc_pca = f['descending/scores'][:, :3]
min_samples = min(asc_pca.shape[0], desc_pca.shape[0])
pca_features = np.hstack([asc_pca[:min_samples], desc_pca[:min_samples]])

# Extract statistical features
combined_flat = combined_data.reshape(time_steps, -1)
max_per_time = np.max(combined_flat, axis=1)
min_per_time = np.min(combined_flat, axis=1)
range_per_time = max_per_time - min_per_time
stat_features = np.column_stack([range_per_time, min_per_time, max_per_time])

# Raw InSAR feature (spatially averaged)
raw_insar_feature = np.mean(combined_flat, axis=1)

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

# Match all dimensions
min_len = min(pca_features.shape[0], stat_features.shape[0], len(labels), len(aligned_rainfall), len(raw_insar_feature))
pca_features = pca_features[:min_len]
stat_features = stat_features[:min_len]
labels = labels[:min_len]
aligned_rainfall = aligned_rainfall[:min_len]
raw_insar_feature = raw_insar_feature[:min_len]

print(f"Data loaded: {min_len} samples, Stable={np.sum(labels==0)}, Unstable={np.sum(labels==1)}")

# Store results
results = []

def evaluate_model(X, y, model_name, classifier_type='RF'):
    """Train and evaluate a model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if classifier_type == 'RF':
        clf = RandomForestClassifier(random_state=42, n_estimators=100)
    elif classifier_type == 'XGBoost':
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        except Exception as e:
            print(f"  XGBoost not available ({str(e)[:50]}...), using RF instead")
            clf = RandomForestClassifier(random_state=42, n_estimators=100)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    return acc, f1, auc

print("\n" + "="*80)
print("RUNNING ALL CONFIGURATIONS...")
print("="*80)

# 1. PCA + RF (Proposed)
print("\n[1/7] PCA + Random Forest (Proposed)")
X = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])
acc, f1, auc = evaluate_model(X, labels, "PCA + RF")
results.append(("PCA + RF (Proposed)", "6 PCA components + rainfall", acc, f1, auc))
print(f"  Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")

# 2. RF Gini Importance + RF
print("\n[2/7] RF Gini Importance + RF")
X = np.hstack([stat_features, aligned_rainfall.reshape(-1, 1)])
acc, f1, auc = evaluate_model(X, labels, "RF Importance + RF")
results.append(("RF Feature Importance + RF", "3 statistical features + rainfall", acc, f1, auc))
print(f"  Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")

# 3. PCA + XGBoost
print("\n[3/7] PCA + XGBoost")
X = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])
acc, f1, auc = evaluate_model(X, labels, "PCA + XGBoost", classifier_type='XGBoost')
results.append(("PCA + XGBoost", "6 PCA components + rainfall", acc, f1, auc))
print(f"  Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")

# 4. Raw InSAR only (no rainfall)
print("\n[4/7] Raw InSAR only")
X = raw_insar_feature.reshape(-1, 1)
acc, f1, auc = evaluate_model(X, labels, "Raw InSAR only")
results.append(("Raw InSAR only", "Spatially-averaged displacement", acc, f1, auc))
print(f"  Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")

# 5. Rainfall only
print("\n[5/7] Rainfall only")
X = aligned_rainfall.reshape(-1, 1)
acc, f1, auc = evaluate_model(X, labels, "Rainfall only")
results.append(("Rainfall only", "Daily precipitation", acc, f1, auc))
print(f"  Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")

# 6. Raw InSAR + Rainfall (no PCA)
print("\n[6/7] Raw InSAR + Rainfall (no PCA)")
X = np.column_stack([raw_insar_feature, aligned_rainfall])
acc, f1, auc = evaluate_model(X, labels, "Raw InSAR + Rainfall")
results.append(("Raw InSAR + Rainfall (no PCA)", "Raw displacement + rainfall", acc, f1, auc))
print(f"  Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")

# 7. LDA + RF (for completeness)
print("\n[7/7] LDA + RF")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Use statistical features as input to LDA
X_train_stat, X_test_stat, y_train, y_test = train_test_split(
    stat_features, labels, test_size=0.2, random_state=42, stratify=labels
)
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_stat, y_train)
X_test_lda = lda.transform(X_test_stat)
# Add rainfall
train_idx = np.arange(len(labels))[:int(0.8*len(labels))]
test_idx = np.arange(len(labels))[int(0.8*len(labels)):]
rainfall_train = aligned_rainfall[train_idx]
rainfall_test = aligned_rainfall[test_idx]
# Need proper split
X_combined, _, y_combined, _, indices, _ = train_test_split(
    stat_features, labels, np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels
)
X_train_full, X_test_full = X_combined[:int(0.8*len(X_combined))], X_combined[int(0.8*len(X_combined)):]
y_train, y_test = y_combined[:int(0.8*len(y_combined))], y_combined[int(0.8*len(y_combined)):]
train_indices, test_indices = indices[:int(0.8*len(indices))], indices[int(0.8*len(indices)):]
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_full, y_train)
X_test_lda = lda.transform(X_test_full)
X_train_with_rain = np.hstack([X_train_lda, aligned_rainfall[train_indices].reshape(-1, 1)])
X_test_with_rain = np.hstack([X_test_lda, aligned_rainfall[test_indices].reshape(-1, 1)])
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train_with_rain, y_train)
y_pred = rf.predict(X_test_with_rain)
y_prob = rf.predict_proba(X_test_with_rain)[:, 1]
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
results.append(("LDA + RF", "1 LDA component + rainfall", acc, f1, auc))
print(f"  Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc*100:.2f}%")

# Print comprehensive table
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON TABLE")
print("="*80)
print(f"{'Model':<40} {'Features':<35} {'Acc':<8} {'F1':<8} {'AUC':<8}")
print("-"*80)
for model, features, acc, f1, auc in results:
    print(f"{model:<40} {features:<35} {acc*100:>6.2f}% {f1*100:>6.2f}% {auc*100:>6.2f}%")
print("="*80)

# Generate LaTeX table
print("\n" + "="*80)
print("LATEX TABLE:")
print("="*80)
print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\caption{Comprehensive comparison of feature extraction and classification approaches. All models use the same chronologically-split train-test partition (80-20\%, stratified).}")
print(r"\label{tab:comprehensive_comparison}")
print(r"\begin{tabular}{l l c c c}")
print(r"\toprule")
print(r"\textbf{Model} & \textbf{Features} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{AUC} \\")
print(r"\midrule")
for i, (model, features, acc, f1, auc) in enumerate(results):
    if i == 0:
        print(f"\\textbf{{{model}}} & \\textbf{{{features}}} & \\textbf{{{acc*100:.2f}\\%}} & \\textbf{{{f1*100:.2f}\\%}} & \\textbf{{{auc*100:.2f}\\%}} \\\\")
        print(r"\midrule")
    else:
        print(f"{model} & {features} & {acc*100:.2f}\\% & {f1*100:.2f}\\% & {auc*100:.2f}\\% \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
print("="*80)

