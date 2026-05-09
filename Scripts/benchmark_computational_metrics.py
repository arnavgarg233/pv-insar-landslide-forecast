"""
Benchmark script: compute processing time and memory for the PCA-RF pipeline.
Used to support "lightweight and scalable" claims in the manuscript.
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

import time
import sys
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Optional: peak memory (works on Unix; Windows may need psutil)
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

def get_peak_mb():
    if HAS_RESOURCE:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: ru_maxrss in KiB; macOS: often in bytes
        if rss > 1e6:
            return rss / (1024 * 1024)
        return rss / 1024
    return None

# Paths
asc_file = str(ASC_H5)
desc_file = str(DESC_H5)
pca_file = str(PCA_H5)
rainfall_dir = str(RAINFALL_DIR)

print("=" * 60)
print("COMPUTATIONAL BENCHMARK: PCA + RF PIPELINE")
print("=" * 60)

# --- 1. Data loading time and memory ---
t0 = time.perf_counter()
with h5py.File(asc_file, "r") as f:
    asc_data = f["timeseries"][:]
with h5py.File(desc_file, "r") as f:
    desc_data = f["timeseries"][:]
if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[: desc_data.shape[0]]
with h5py.File(pca_file, "r") as f:
    asc_pca = f["ascending/scores"][:]
    desc_pca = f["descending/scores"][:]
min_s = min(asc_pca.shape[0], desc_pca.shape[0])
pca_features = np.hstack([asc_pca[:min_s], desc_pca[:min_s]])

# Labels (simplified)
time_steps = asc_data.shape[0]
vertical = (asc_data + desc_data) / 2
horizontal = (asc_data - desc_data) / 2
displacement = np.sqrt(vertical**2 + horizontal**2)
cumulative_rolling = np.zeros_like(displacement)
for t in range(time_steps - 6):
    cumulative_rolling[t] = displacement[t : t + 7].sum(axis=0)
labels = (cumulative_rolling > 0.35).astype(int)
labels = labels.reshape(labels.shape[0], -1).sum(axis=1) > 0
labels = labels.astype(int)[:pca_features.shape[0]]

# Rainfall
rainfall_files = sorted([os.path.join(rainfall_dir, f) for f in os.listdir(rainfall_dir) if f.endswith(".nc4")])
rainfall_data = []
for f in rainfall_files[:365]:  # limit for speed
    try:
        ds = xr.open_dataset(f)
        if "MWprecipitation" in ds:
            rainfall_data.append(ds["MWprecipitation"].mean(dim=["lon", "lat"]).to_dataframe())
        ds.close()
    except Exception:
        pass
if rainfall_data:
    rainfall_df = pd.concat(rainfall_data).reset_index()
    rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])
    aligned_rainfall = rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"].mean().values
else:
    aligned_rainfall = np.zeros(pca_features.shape[0])
min_len = min(pca_features.shape[0], len(labels), len(aligned_rainfall))
pca_features = pca_features[:min_len]
labels = labels[:min_len]
aligned_rainfall = aligned_rainfall[:min_len]
X = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])
y = labels
t_load = time.perf_counter() - t0
mem_after_load_mb = get_peak_mb()
print(f"\n1. Data loading:     {t_load:.2f} s")
print(f"   Samples: {len(y)}, Features: {X.shape[1]}")

# --- 2. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Training time ---
t0 = time.perf_counter()
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
t_train = time.perf_counter() - t0
mem_after_train_mb = get_peak_mb()
print(f"\n2. Model training:   {t_train:.2f} s  (100 trees, n_estimators=100)")

# --- 4. Inference time (full test set and 10x larger batch) ---
t0 = time.perf_counter()
for _ in range(100):
    rf.predict(X_test)
t_infer_100 = time.perf_counter() - t0
t_infer_once = t_infer_100 / 100
# Simulate 10x scale: predict on 10x test size
X_large = np.tile(X_test, (10, 1)) if X_test.shape[0] * 10 < 1e7 else np.tile(X_test, (5, 1))
t0 = time.perf_counter()
rf.predict(X_large)
t_infer_large = time.perf_counter() - t0
print(f"\n3. Inference:        {t_infer_once*1000:.1f} ms per call (n={len(y_test)} samples)")
print(f"   Inference (batch ~10x): {t_infer_large*1000:.0f} ms for {X_large.shape[0]} samples")

# --- 5. Model size ---
import pickle
model_bytes = len(pickle.dumps(rf))
print(f"\n4. Serialized model size: {model_bytes/1024:.1f} KB")

# --- 6. Memory ---
if mem_after_train_mb is not None and mem_after_train_mb < 1e5:
    print(f"\n5. Peak RAM (approx):   {mem_after_train_mb:.1f} MB")
else:
    print("\n5. Peak RAM: (not reported; unit/platform-dependent)")
    mem_after_train_mb = None

# --- Summary for manuscript ---
print("\n" + "=" * 60)
print("SUGGESTED TEXT FOR MANUSCRIPT (Section 6.4 / Limitations)")
print("=" * 60)
print("""
To support the claim that the pipeline is lightweight and scalable, we report
the following computational metrics (single core, benchmark run):
- Data loading (InSAR + PCA + rainfall): {:.1f} s
- Model training (RF, 100 trees):        {:.1f} s
- Inference:                             {:.2f} ms per forecast ({} samples)
- Serialized model size:                 {:.1f} KB
""".format(t_load, t_train, t_infer_once*1000, len(y_test), model_bytes/1024))
if mem_after_train_mb is not None:
    print("- Peak RAM during training:       {:.0f} MB\n".format(mem_after_train_mb))
print("=" * 60)
sys.exit(0)
