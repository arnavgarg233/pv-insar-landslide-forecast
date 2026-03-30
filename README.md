# PV-InSAR Landslide Forecast: PCA + Random Forest (7-day horizon)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Seven-day landslide forecasting** from **PCA-derived Sentinel-1 InSAR** deformation features and **GPM IMERG** rainfall, using **Random Forest** and related classifiers — Palos Verdes Peninsula, California.

**Publication:** *Earth Systems and Environment* (Springer) — manuscript **ESEV-D-25-01196** (revised; **under peer review**).  
**Author:** Arnav Garg (corresponding), Palos Verdes Peninsula High School, Rancho Palos Verdes, CA, USA.

---

## Key results (paper)

- **Test accuracy:** 91.67% (chronological hold-out)
- **F1-score:** 93.02%
- **ROC-AUC:** 95.29% (95% CI: 87.00%–100.00%)
- **Lead time:** 7-day forecast horizon with forecast-oriented pseudo-labeling (anti-leakage)
- **Interpretability:** PCA modes linked to creep, seasonal signal, and transient response; RF feature importance

---

## Overview

This repository implements a multi-stage pipeline:

1. **InSAR time series** — SBAS-style displacement from Sentinel-1 (processed with MintPy-style workflow locally)
2. **PCA feature extraction** — unsupervised reduction of the displacement field to interpretable components
3. **Rainfall** — GPM IMERG (daily aggregation) aligned to the study area
4. **Pseudo-labels** — forecast-oriented labeling from future displacement thresholds
5. **Classification** — Random Forest (primary), with comparisons to LDA, LSTM, etc.

**Repository highlights:**

- ✅ `src/landslide_forecast/` — installable package with **centralized paths** (`config.py`; override via `LANDSLIDE_PROJECT_ROOT`)
- ✅ `scripts/` — **flare-pinn–style** layout: `training/`, `evaluation/`, `figures/`, `analysis/`, `tools/`
- ✅ `data_scripts/` — helpers for downloading / preparing ancillary data
- ✅ Chronological train/validation/test split; explicit distinction from random k-fold where documented in the paper
- ✅ Large rasters and `.h5` stacks are **gitignored**; this repo is **code + sample figures**, not the full data mirror

---

## Installation

### Requirements

- Python **3.10+**
- See `requirements.txt` (NumPy, pandas, xarray, h5py, scikit-learn, scipy, matplotlib, netCDF4)

### Quick start

```bash
git clone https://github.com/arnavgarg233/pv-insar-landslide-forecast.git
cd pv-insar-landslide-forecast

# Recommended (uv)
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e .

# Sanity check
python -c "from landslide_forecast.config import REPO_ROOT, RF_DATA_DIR; print(REPO_ROOT, RF_DATA_DIR)"
```

---

## Repository layout

```
pv-insar-landslide-forecast/
├── src/landslide_forecast/     # Package: config / paths
├── scripts/
│   ├── training/               # RF, LDA, model sweeps
│   ├── training/legacy/        # Older end-to-end scripts
│   ├── evaluation/             # Comparisons, significance, CIs
│   ├── evaluation/threshold_sweeps/
│   ├── figures/                # ROC, figure regeneration
│   ├── analysis/             # PC–rainfall, feature importance, uncertainty
│   ├── tools/                  # Benchmarks, exploratory utilities
│   └── README.md               # Script index
├── tools/                      # Repo maintenance (e.g. reorganize_repo.py)
├── data_scripts/               # Data download / prep
├── RandomForest MovmentLandslide threshodl/   # Local outputs: .h5, figures, TeX (large files not in git)
├── Plots/                      # Example plots (where committed)
└── Final visualizations/       # Example figures (where committed)
```

Run examples from the repo root:

```bash
python scripts/training/model_b_simple.py
python scripts/figures/generate_roc_curve.py
python scripts/analysis/pc_quantitative_support.py
```

---

## Data

Place processed artifacts locally (not tracked):

| Location | Contents |
|----------|-----------|
| `RandomForest MovmentLandslide threshodl/timeseries.h5` | Ascending InSAR stack |
| `RandomForest MovmentLandslide threshodl/pvtimeseriesd.h5` | Descending InSAR stack |
| `RandomForest MovmentLandslide threshodl/pca_results.h5` | PCA scores/components (optional if you regenerate PCA) |
| `Average Rainfall data/Rain/*.nc4` | GPM IMERG |

Raw MintPy stacks are **hundreds of GB** — keep them outside the repo. Set **`LANDSLIDE_PROJECT_ROOT`** if your clone path differs from the machine where `config` was first written.

---

## Citation

If you use this code before formal publication:

> Garg, A. *Seven-Day Landslide Forecasting from PCA-Derived InSAR Data with a Random Forest Classifier.* Submitted to *Earth Systems and Environment* (manuscript **ESEV-D-25-01196**).

A DOI will be added after acceptance.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE).
