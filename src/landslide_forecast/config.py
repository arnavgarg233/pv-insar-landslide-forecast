"""Repository layout and data paths (override with LANDSLIDE_PROJECT_ROOT)."""

from __future__ import annotations

import os
from pathlib import Path

_LEGACY_RF_DIR = "RandomForest MovmentLandslide threshodl"


def repo_root() -> Path:
    env = os.environ.get("LANDSLIDE_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


REPO_ROOT: Path = repo_root()

# Processed InSAR / PCA artifacts (large .h5 files; not in git)
RF_DATA_DIR: Path = REPO_ROOT / _LEGACY_RF_DIR

# GPM IMERG daily files
RAINFALL_DIR: Path = REPO_ROOT / "Average Rainfall data" / "Rain"

# Optional: full MintPy stack (local only; gitignored)
MINTPY_STACK_DIR: Path = REPO_ROOT / "Ascending and Descending MintPy data"

ASC_H5: Path = RF_DATA_DIR / "timeseries.h5"
DESC_H5: Path = RF_DATA_DIR / "pvtimeseriesd.h5"
PCA_H5: Path = RF_DATA_DIR / "pca_results.h5"
UNCERTAINTY_H5: Path = RF_DATA_DIR / "uncertainty_propagation_results.h5"

# Small curated figures for the public repo (ROC, rainfall summary, etc.)
FINAL_RESULTS_DIR: Path = REPO_ROOT / "final_results"

# PNG inputs for `Scripts/visualization/combine_existing_figures.py` (paper-style composites)
FIGURE_INPUTS_DIR: Path = REPO_ROOT / "Data" / "figure_inputs"

# Ephemeral runs and regenerated assets (gitignored except .gitkeep)
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"

# RF / LDA training scripts and legacy notebooks-style code
MODELS_DIR: Path = REPO_ROOT / "Models"
