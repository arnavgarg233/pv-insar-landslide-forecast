"""
One-time: move RandomForest.../*.py into scripts/ + snake_case names, rewrite paths.
Run from repo root: python Scripts/reorganize_repo.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "RandomForest MovmentLandslide threshodl"

MOVES: list[tuple[str, str]] = [
    ("Model_B_Simple.py", "scripts/training/model_b_simple.py"),
    ("Model_B_RF_Feature_Importance.py", "scripts/training/model_b_rf_feature_importance.py"),
    ("Model_E_LDA.py", "scripts/training/model_e_lda.py"),
    ("Model_F_Pure_RF.py", "scripts/training/model_f_pure_rf.py"),
    ("Model_G_Fair_RF.py", "scripts/training/model_g_fair_rf.py"),
    ("Model_H_RF_3Features.py", "scripts/training/model_h_rf_3features.py"),
    ("Classifier_Comparison.py", "scripts/evaluation/classifier_comparison.py"),
    ("LSTM_Classifier_Comparison.py", "scripts/evaluation/lstm_classifier_comparison.py"),
    ("Comprehensive_Comparison.py", "scripts/evaluation/comprehensive_comparison.py"),
    ("Significance_Testing.py", "scripts/evaluation/significance_testing.py"),
    ("Calculate_Confidence_Intervals.py", "scripts/evaluation/calculate_confidence_intervals.py"),
    ("Combine_Existing_Figures.py", "scripts/figures/combine_existing_figures.py"),
    ("Regenerate_Figures_CORRECT.py", "scripts/figures/regenerate_figures_correct.py"),
    ("Regenerate_Figures_4_5.py", "scripts/figures/regenerate_figures_4_5.py"),
    ("Generate_ROC_Curve.py", "scripts/figures/generate_roc_curve.py"),
    ("Extract_Feature_Importance.py", "scripts/analysis/extract_feature_importance.py"),
    ("Extract_Feature_Importance_PC3.py", "scripts/analysis/extract_feature_importance_pc3.py"),
    ("Feature_Importance_Analysis.py", "scripts/analysis/feature_importance_analysis.py"),
    ("PC_quantitative_support.py", "scripts/analysis/pc_quantitative_support.py"),
    ("Uncertainty_Propagation.py", "scripts/analysis/uncertainty_propagation.py"),
    ("Benchmark_Computational_Metrics.py", "scripts/tools/benchmark_computational_metrics.py"),
    ("Model_Sweep_Features.py", "scripts/tools/model_sweep_features.py"),
    ("displacmentmatrix.py", "scripts/tools/displacement_matrix.py"),
    ("rain.py", "scripts/tools/rain.py"),
    ("pca.py", "scripts/tools/pca.py"),
    ("Final model training scripts/Final_traning.py", "scripts/training/legacy/final_training.py"),
    ("Final model training scripts/Regression_model.py", "scripts/training/legacy/regression_model.py"),
    ("Final model training scripts/Plotting_main.py", "scripts/training/legacy/plotting_main.py"),
    ("Final model training scripts/Plotting.py", "scripts/training/legacy/plotting.py"),
    ("Final model training scripts/Plotting copy.py", "scripts/training/legacy/plotting_copy.py"),
    ("Model testing scripts/Training_self_deciding_thresholdsPca.py", "scripts/evaluation/threshold_sweeps/training_self_deciding_thresholds_pca.py"),
    ("Model testing scripts/Pca_rainfall_self_threshold.py", "scripts/evaluation/threshold_sweeps/pca_rainfall_self_threshold.py"),
    ("Model testing scripts/Manual_threshold.py", "scripts/evaluation/threshold_sweeps/manual_threshold.py"),
    ("Model testing scripts/Self_rainfall_threshold.py", "scripts/evaluation/threshold_sweeps/self_rainfall_threshold.py"),
]

BOOTSTRAP = """import sys
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

"""

CONFIG_IMPORT = """from landslide_forecast.config import (
    REPO_ROOT,
    RF_DATA_DIR,
    ASC_H5,
    DESC_H5,
    PCA_H5,
    RAINFALL_DIR,
    MINTPY_STACK_DIR,
)

"""


def rewrite_paths(text: str) -> str:
    pairs: list[tuple[str, str]] = [
        (
            'asc_file = "/Volumes/Lexar/Landslide project/RandomForest MovmentLandslide threshodl/timeseries.h5"',
            "asc_file = str(ASC_H5)",
        ),
        (
            'desc_file = "/Volumes/Lexar/Landslide project/RandomForest MovmentLandslide threshodl/pvtimeseriesd.h5"',
            "desc_file = str(DESC_H5)",
        ),
        (
            'pca_file = "/Volumes/Lexar/Landslide project/RandomForest MovmentLandslide threshodl/pca_results.h5"',
            "pca_file = str(PCA_H5)",
        ),
        (
            'rainfall_dir = "/Volumes/Lexar/Landslide project/Average Rainfall data/Rain"',
            "rainfall_dir = str(RAINFALL_DIR)",
        ),
        (
            'asc_file = r"/Volumes/Lexar/Landslide project/RandomForest MovmentLandslide threshodl/timeseries.h5"',
            "asc_file = str(ASC_H5)",
        ),
        (
            'desc_file = r"/Volumes/Lexar/Landslide project/RandomForest MovmentLandslide threshodl/pvtimeseriesd.h5"',
            "desc_file = str(DESC_H5)",
        ),
        (
            'rainfall_dir = r"/Volumes/Lexar/Landslide project/Average Rainfall data/Rain"',
            "rainfall_dir = str(RAINFALL_DIR)",
        ),
        (
            'base_path = Path("/Volumes/Lexar/Landslide project")',
            "base_path = REPO_ROOT",
        ),
        (
            'base_path = Path("/Volumes/Lexar/Landslide project/RandomForest MovmentLandslide threshodl")',
            "base_path = RF_DATA_DIR",
        ),
        (
            'output_dir = Path("/Volumes/Lexar/Landslide project/RandomForest MovmentLandslide threshodl")',
            "output_dir = RF_DATA_DIR",
        ),
        (
            'mintpy_path = Path("/Volumes/Lexar/Landslide project/Ascending and Descending MintPy data")',
            "mintpy_path = MINTPY_STACK_DIR",
        ),
        (
            'output_path = "/Volumes/Lexar/Landslide project/RandomForest MovmentLandslide threshodl/ROC_Curve.jpg"',
            'output_path = str(RF_DATA_DIR / "ROC_Curve.jpg")',
        ),
        # Windows legacy paths
        (
            r'ASC_FILE = r"D:\Landslide project\RandomForest MovmentLandslide threshodl\timeseries.h5"',
            "ASC_FILE = str(ASC_H5)",
        ),
        (
            r'DESC_FILE = r"D:\Landslide project\RandomForest MovmentLandslide threshodl\pvtimeseriesd.h5"',
            "DESC_FILE = str(DESC_H5)",
        ),
        (
            r'asc_file = r"D:\Landslide project\RandomForest MovmentLandslide threshodl\timeseries.h5"',
            "asc_file = str(ASC_H5)",
        ),
        (
            r'desc_file = r"D:\Landslide project\RandomForest MovmentLandslide threshodl\pvtimeseriesd.h5"',
            "desc_file = str(DESC_H5)",
        ),
        (
            r'pca_file = r"D:\Landslide project\RandomForest MovmentLandslide threshodl\pca_results.h5"',
            "pca_file = str(PCA_H5)",
        ),
        (
            r'rainfall_dir = r"D:\Landslide project\Average Rainfall data\Rain"',
            "rainfall_dir = str(RAINFALL_DIR)",
        ),
        (
            r'RAINFALL_DIR = r"D:\Landslide project\Average Rainfall data\Rain"',
            "RAINFALL_DIR = str(RAINFALL_DIR)",
        ),
        (
            r'RAINFALL_DIR = r"D:\Landslide project\Average Rainfall data\Rainfall"',
            "RAINFALL_DIR = str(RAINFALL_DIR)",
        ),
        (
            r'rainfall_dir = r"D:\Landslide project\Rainfall data\Rainfall"',
            "rainfall_dir = str(RAINFALL_DIR)",
        ),
        (
            r'pca_file = r"D:\Landslide project\XGboost\Randomforest training workflow\pca_results.h5"',
            "pca_file = str(PCA_H5)",
        ),
        (
            r'OUTPUT_DIR = r"D:\Landslide project\RandomForest MovmentLandslide threshod"',
            "OUTPUT_DIR = str(RF_DATA_DIR)",
        ),
        (
            r'OUTPUT_DIR = r"D:\Landslide project\Plots"',
            'OUTPUT_DIR = str(REPO_ROOT / "Plots")',
        ),
    ]
    t = text
    for old, new in pairs:
        t = t.replace(old, new)
    return t


def insert_bootstrap_and_import(text: str) -> str:
    if "landslide_forecast.config" in text:
        return text
    if "pyproject.toml" in text and "_repo /" in text and "sys.path.insert" in text:
        return text
    m = re.match(r"^(\s*\"\"\".*?\"\"\"\s*\n)", text, re.DOTALL)
    if m:
        rest = text[m.end() :]
        if "landslide_forecast.config" in rest[:1200]:
            return text
        return m.group(1) + "\n" + BOOTSTRAP + CONFIG_IMPORT + rest
    if "landslide_forecast.config" in text[:1200]:
        return text
    return BOOTSTRAP + CONFIG_IMPORT + text


def dedupe_pathlib_imports(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    pathlib_done = False
    for line in lines:
        if line.strip() == "from pathlib import Path":
            if pathlib_done:
                continue
            pathlib_done = True
        out.append(line)
    return "\n".join(out)


def process_file(path: Path) -> None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    new = rewrite_paths(raw)
    new = insert_bootstrap_and_import(new)
    new = dedupe_pathlib_imports(new)
    path.write_text(new, encoding="utf-8")


def main() -> None:
    marker = ROOT / "Models" / "model_b_simple.py"
    if marker.exists():
        print("Already reorganized (Models/model_b_simple.py exists). Skipping.")
        return

    for rel_old, rel_new in MOVES:
        old = SRC / rel_old
        new = ROOT / rel_new
        if not old.exists():
            print("skip missing:", old)
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        print("moved", rel_old, "->", rel_new)
        if new.suffix == ".py":
            process_file(new)

    print("done.")


if __name__ == "__main__":
    main()
