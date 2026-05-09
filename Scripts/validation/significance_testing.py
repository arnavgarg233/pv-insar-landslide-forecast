"""
Statistical Significance Testing for Reviewer 4, Comment 2
===========================================================
1. Bootstrap test: AUC > 0.5 (random-chance baseline)
2. McNemar's tests: PCA+RF vs each alternative configuration
3. McNemar's tests: PCA+RF vs each alternative classifier
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
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Data Loading (identical pipeline to all existing scripts)
# ============================================================================
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
pca_file = str(PCA_H5)
rainfall_dir = str(RAINFALL_DIR)

print("=" * 80)
print("STATISTICAL SIGNIFICANCE TESTING")
print("=" * 80)
print("\nLoading data...")

with h5py.File(asc_file, "r") as f:
    asc_data = f["timeseries"][:]
with h5py.File(desc_file, "r") as f:
    desc_data = f["timeseries"][:]

if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[: desc_data.shape[0]]

time_steps, height, width = asc_data.shape

with h5py.File(pca_file, "r") as f:
    asc_pca = f["ascending/scores"][:, :3]
    desc_pca = f["descending/scores"][:, :3]
min_samples = min(asc_pca.shape[0], desc_pca.shape[0])
pca_features = np.hstack([asc_pca[:min_samples], desc_pca[:min_samples]])

combined_data = (asc_data + desc_data) / 2
combined_flat = combined_data.reshape(time_steps, -1)
max_per_time = np.max(combined_flat, axis=1)
min_per_time = np.min(combined_flat, axis=1)
range_per_time = max_per_time - min_per_time
stat_features = np.column_stack([range_per_time, min_per_time, max_per_time])
raw_insar_feature = np.mean(combined_flat, axis=1)

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

min_len = min(
    pca_features.shape[0],
    stat_features.shape[0],
    len(labels),
    len(aligned_rainfall),
    len(raw_insar_feature),
)
pca_features = pca_features[:min_len]
stat_features = stat_features[:min_len]
labels = labels[:min_len]
aligned_rainfall = aligned_rainfall[:min_len]
raw_insar_feature = raw_insar_feature[:min_len]

print(f"Data loaded: {min_len} samples, Stable={np.sum(labels==0)}, Unstable={np.sum(labels==1)}")

# ============================================================================
# Helper: McNemar's exact test (two-sided)
# ============================================================================
from scipy.stats import binom


def mcnemar_exact(y_true, y_pred_a, y_pred_b):
    """
    Exact McNemar's test (binomial).
    Returns (b, c, p_value) where b = A right & B wrong, c = A wrong & B right.
    """
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))  # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B correct
    n = b + c
    if n == 0:
        return b, c, 1.0
    p_value = 2 * binom.cdf(min(b, c), n, 0.5)
    p_value = min(p_value, 1.0)
    return b, c, p_value


# ============================================================================
# Build feature matrices for each configuration
# ============================================================================
X_pca_rain = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])
X_rf_feat_rain = np.hstack([stat_features, aligned_rainfall.reshape(-1, 1)])
X_raw_insar_only = raw_insar_feature.reshape(-1, 1)
X_rain_only = aligned_rainfall.reshape(-1, 1)
X_raw_insar_rain = np.column_stack([raw_insar_feature, aligned_rainfall])

# Single train-test split (same as all existing scripts)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca_rain, labels, test_size=0.2, random_state=42, stratify=labels
)
test_n = len(y_test)
print(f"Test set size: {test_n}")

# Need the same indices for all configs
split_indices = train_test_split(
    np.arange(min_len), labels, test_size=0.2, random_state=42, stratify=labels
)
train_idx, test_idx = split_indices[0], split_indices[1]

# ============================================================================
# PART 1: Train all CONFIGURATION variants and collect test predictions
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: CONFIGURATION COMPARISONS (McNemar's Test)")
print("=" * 80)

config_predictions = {}

# 1a. PCA + RF (Proposed)
rf_proposed = RandomForestClassifier(random_state=42, n_estimators=100)
rf_proposed.fit(X_pca_rain[train_idx], labels[train_idx])
config_predictions["PCA + RF (Proposed)"] = rf_proposed.predict(X_pca_rain[test_idx])
y_prob_proposed = rf_proposed.predict_proba(X_pca_rain[test_idx])[:, 1]

# 1b. RF Features + Rainfall + RF
rf_feat = RandomForestClassifier(random_state=42, n_estimators=100)
rf_feat.fit(X_rf_feat_rain[train_idx], labels[train_idx])
config_predictions["RF Features + Rainfall"] = rf_feat.predict(X_rf_feat_rain[test_idx])

# 1c. PCA + XGBoost
try:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_pca_rain[train_idx], labels[train_idx])
    config_predictions["PCA + XGBoost"] = xgb.predict(X_pca_rain[test_idx])
except ImportError:
    print("  XGBoost not available, skipping.")

# 1d. Raw InSAR Only
rf_raw = RandomForestClassifier(random_state=42, n_estimators=100)
rf_raw.fit(X_raw_insar_only[train_idx], labels[train_idx])
config_predictions["Raw InSAR Only"] = rf_raw.predict(X_raw_insar_only[test_idx])

# 1e. Rainfall Only
rf_rain = RandomForestClassifier(random_state=42, n_estimators=100)
rf_rain.fit(X_rain_only[train_idx], labels[train_idx])
config_predictions["Rainfall Only"] = rf_rain.predict(X_rain_only[test_idx])

# 1f. Raw InSAR + Rainfall (no PCA)
rf_raw_rain = RandomForestClassifier(random_state=42, n_estimators=100)
rf_raw_rain.fit(X_raw_insar_rain[train_idx], labels[train_idx])
config_predictions["Raw InSAR + Rainfall"] = rf_raw_rain.predict(X_raw_insar_rain[test_idx])

# 1g. LDA + RF
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda_train = lda.fit_transform(stat_features[train_idx], labels[train_idx])
X_lda_test = lda.transform(stat_features[test_idx])
X_lda_rain_train = np.hstack([X_lda_train, aligned_rainfall[train_idx].reshape(-1, 1)])
X_lda_rain_test = np.hstack([X_lda_test, aligned_rainfall[test_idx].reshape(-1, 1)])
rf_lda = RandomForestClassifier(random_state=42, n_estimators=100)
rf_lda.fit(X_lda_rain_train, labels[train_idx])
config_predictions["LDA + Rainfall + RF"] = rf_lda.predict(X_lda_rain_test)

y_true = labels[test_idx]
proposed_preds = config_predictions["PCA + RF (Proposed)"]

print(f"\n{'Configuration':<30} {'b':>4} {'c':>4} {'p-value':>10} {'Significant':>12}")
print("-" * 65)
for name, preds in config_predictions.items():
    if name == "PCA + RF (Proposed)":
        continue
    b, c, p = mcnemar_exact(y_true, proposed_preds, preds)
    sig = "Yes*" if p < 0.05 else "No"
    print(f"{name:<30} {b:>4} {c:>4} {p:>10.4f} {sig:>12}")

# ============================================================================
# PART 2: Train all CLASSIFIER variants and collect test predictions
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: CLASSIFIER COMPARISONS (McNemar's Test)")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_pca_rain[train_idx])
X_test_scaled = scaler.transform(X_pca_rain[test_idx])

classifiers = [
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3), False),
    ("SVM (RBF)", SVC(kernel="rbf", probability=True, random_state=42), True),
    ("SVM (Linear)", SVC(kernel="linear", probability=True, random_state=42), True),
    ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000), True),
    ("KNN", KNeighborsClassifier(n_neighbors=5), True),
    ("AdaBoost", AdaBoostClassifier(random_state=42, n_estimators=100), False),
    ("MLP", MLPClassifier(random_state=42, hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True), True),
]

try:
    from xgboost import XGBClassifier

    classifiers.append(
        ("XGBoost", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss", n_estimators=100, max_depth=3), False)
    )
except ImportError:
    pass

clf_predictions = {}
for name, clf, needs_scaling in classifiers:
    X_tr = X_train_scaled if needs_scaling else X_pca_rain[train_idx]
    X_te = X_test_scaled if needs_scaling else X_pca_rain[test_idx]
    clf.fit(X_tr, labels[train_idx])
    clf_predictions[name] = clf.predict(X_te)

print(f"\n{'Classifier':<30} {'b':>4} {'c':>4} {'p-value':>10} {'Significant':>12}")
print("-" * 65)
for name, preds in clf_predictions.items():
    b, c, p = mcnemar_exact(y_true, proposed_preds, preds)
    sig = "Yes*" if p < 0.05 else "No"
    print(f"{name:<30} {b:>4} {c:>4} {p:>10.4f} {sig:>12}")

# ============================================================================
# PART 3: Bootstrap significance test -- AUC > 0.5
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: BOOTSTRAP TEST -- AUC > 0.5 (random-chance baseline)")
print("=" * 80)

n_bootstrap = 10000
np.random.seed(42)
boot_aucs = []
for _ in range(n_bootstrap):
    idx = np.random.choice(test_n, size=test_n, replace=True)
    y_b = y_true[idx]
    p_b = y_prob_proposed[idx]
    if len(np.unique(y_b)) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_b, p_b))

boot_aucs = np.array(boot_aucs)
p_value_auc = np.mean(boot_aucs <= 0.5)
auc_ci_lo, auc_ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
auc_point = roc_auc_score(y_true, y_prob_proposed)

print(f"AUC point estimate: {auc_point*100:.2f}%")
print(f"AUC 95% CI: [{auc_ci_lo*100:.2f}%, {auc_ci_hi*100:.2f}%]")
print(f"Bootstrap p-value (H0: AUC <= 0.5): {p_value_auc:.6f}")
print(f"Significant at alpha=0.01: {'Yes' if p_value_auc < 0.01 else 'No'}")

# ============================================================================
# PART 4: Summary for LaTeX
# ============================================================================
print("\n" + "=" * 80)
print("LATEX-READY SUMMARY")
print("=" * 80)

print("\n--- Configuration McNemar's results (for paper text) ---")
for name, preds in config_predictions.items():
    if name == "PCA + RF (Proposed)":
        continue
    b, c, p = mcnemar_exact(y_true, proposed_preds, preds)
    print(f"{name}: $b={b}$, $c={c}$, $p={p:.3f}$" + (" *" if p < 0.05 else ""))

print("\n--- Classifier McNemar's results (for paper text) ---")
for name, preds in clf_predictions.items():
    b, c, p = mcnemar_exact(y_true, proposed_preds, preds)
    print(f"{name}: $b={b}$, $c={c}$, $p={p:.3f}$" + (" *" if p < 0.05 else ""))

print(f"\n--- AUC bootstrap (for paper text) ---")
print(f"AUC = {auc_point*100:.2f}\\% (95\\% CI: {auc_ci_lo*100:.2f}\\%--{auc_ci_hi*100:.2f}\\%)")
print(f"Bootstrap p-value against chance (AUC=0.5): p {'<' if p_value_auc < 0.001 else '='} {max(p_value_auc, 0.001):.3f}")
print("=" * 80)
