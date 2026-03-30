# Seven-Day Landslide Forecasting from PCA-Derived InSAR Data with a Random Forest Classifier

Official code for **seven-day landslide instability forecasting** on the **Palos Verdes Peninsula, California**, using **Sentinel-1 SBAS (MintPy)**, **PCA-derived kinematic features**, **NASA GPM IMERG** rainfall, **pseudo-labeling**, and a **Random Forest** classifier (100 trees).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-under%20peer%20review-DBAB09)](https://github.com/arnavgarg233/7-day-landslide-forecasting)

---

## Publication

| | |
|--|--|
| **Journal** | *Earth Systems and Environment* (Springer Nature) |
| **Status** | Under peer review |
| **Manuscript** | **ESEV-D-25-01196R3** |
| **Author** | **Arnav Garg** (corresponding) · Palos Verdes Peninsula High School, Rancho Palos Verdes, CA, USA · [arnavgarg888@gmail.com](mailto:arnavgarg888@gmail.com) |

**Keywords:** InSAR · landslide forecasting · Principal Component Analysis · Random Forest · pseudo-labeling · rainfall triggers

If you use this repository or the methods described in the manuscript, please cite:

> Garg, A. *Seven-Day Landslide Forecasting from PCA-Derived InSAR Data with a Random Forest Classifier.* Under peer review, *Earth Systems and Environment* (manuscript **ESEV-D-25-01196R3**).

---

## Project overview

Landslide forecasting is limited by **sparse inventories** and **noisy, high-dimensional** InSAR time series. This project implements a full pipeline from **multi-temporal displacement** to a **binary 7-day ahead** instability forecast: **pseudo-labels** are defined from future displacement exceeding a **3.5 mm** threshold (tuned on a **chronological** validation block); **PCA** compresses the stack into **PC1** (long-term creep), **PC2** (seasonal shrink–swell), and **PC3** (transient accelerations), combined with **daily rainfall**. On an independent test period (including **2024 atmospheric rivers**), the main RF model reaches **91.67%** accuracy, **93.02%** F1, and **95.29%** ROC-AUC. **Gini importance** attributes about **48.6%** of predictive power to **PC2**; **quantitative checks** include PC2 vs 90-day mean rainfall (lag 1): **r = 0.15, p = 0.048, N = 179**; FFT peak for PC2 **~483 days** (annual band).

---

## Directory structure

**Configs**  
YAML and documentation for **paper-aligned hyperparameters** (forest size, forecast horizon, pseudo-label threshold). Paths to local data are centralized in `src/landslide_forecast/config.py`.

**Data**  
Version-controlled **figure inputs** for composites (`figure_inputs/`) and **acquisition scripts** for Sentinel-1 / ASF downloads (`acquisition/`). Large `.h5` stacks and GPM archives are **not** in git; see **Data layout (local)** below.

**Models**  
Random Forest and LDA **training scripts**, ablations, and **`legacy/`** end-to-end or plotting experiments from earlier iterations.

**Scripts**  
**Analysis** (PC–rainfall, feature importance, uncertainty), **validation** (comparisons, significance, bootstrap CIs, **threshold_sweeps/**), **visualization** (ROC, figure assembly), plus **benchmarks**, **rainfall plots**, **PCA** utilities, and **`reorganize_repo.py`** (historical migration helper).

**Utils**  
Placeholder for small non-package helpers; **shared paths and config** live in **`src/landslide_forecast/`**.

**src**  
Installable **`landslide_forecast`** package and **`launch.py`** entry to run a model script under `Models/`.

**final_results**  
Example outputs checked into the repo (e.g. ROC curve, mean rainfall figure).

**outputs**  
Local scratch and regenerated files (gitignored except `.gitkeep`).

---

## Key features

**Geodata and features**  
Multi-track InSAR time series, PCA loadings and scores, alignment with **GPM IMERG** daily precipitation.

**Modeling**  
Random Forest (primary), LDA and ablations; **chronological** train/validation/test splits for time-series leakage control.

**Evaluation**  
Accuracy, precision, recall, F1, ROC-AUC with bootstrap confidence intervals; classifier comparisons; threshold and rainfall sensitivity experiments.

**Reproducibility**  
Pinned defaults in `Configs/model_defaults.yaml`; override project root with **`LANDSLIDE_PROJECT_ROOT`** if needed.

---

## Usage

**Environment**

```bash
git clone https://github.com/arnavgarg233/7-day-landslide-forecasting.git
cd 7-day-landslide-forecasting
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
```

**Main training (default RF model)**

```bash
python src/launch.py
```

Equivalent:

```bash
python Models/model_b_simple.py
```

**Figures and analysis**

```bash
python Scripts/visualization/generate_roc_curve.py
python Scripts/analysis/pc_quantitative_support.py
python Scripts/visualization/combine_existing_figures.py
```

**Custom model script**

```bash
python src/launch.py --script Models/model_e_lda.py
```

---

## Requirements

- Python **3.10+**
- **h5py**, **NumPy**, **pandas**, **xarray**, **scikit-learn**, **SciPy**, **Matplotlib**, **netCDF4**  
  (see `requirements.txt` / `pyproject.toml`)

Optional: **full MintPy** stack and large InSAR archives on disk (not bundled).

---

## Data layout (local)

Place large files outside git or under ignored paths as in `.gitignore`:

| Path | Role |
|------|------|
| `RandomForest MovmentLandslide threshodl/timeseries.h5` | Ascending InSAR stack |
| `RandomForest MovmentLandslide threshodl/pvtimeseriesd.h5` | Descending stack |
| `RandomForest MovmentLandslide threshodl/pca_results.h5` | PCA results (optional if you recompute) |
| `Average Rainfall data/Rain/*.nc4` | GPM IMERG Final (daily) |

Raw Sentinel-1 SLCs and full MintPy processing: use **ASF** and local disks; see `Data/acquisition/` for download helpers.

---

## Test-set metrics (paper)

| Metric | Value |
|--------|--------|
| Accuracy | **91.67%** |
| Precision | **95.24%** |
| Recall | **90.91%** |
| F1 | **93.02%** |
| ROC-AUC | **95.29%** (95% CI **87.00%–100.00%**; bootstrap **n = 36**, **10,000** iterations) |

---

## License

[MIT](LICENSE).
