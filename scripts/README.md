# Scripts (flare-pinn–style layout)

Run from the **repository root** after `uv pip install -r requirements.txt` or `uv pip install -e .` so `landslide_forecast` imports resolve. Each script prepends `src/` to `sys.path` if needed.

| Folder | Role |
|--------|------|
| `training/` | Random Forest / LDA baselines (paper Models B–H, sweeps) |
| `training/legacy/` | Older end-to-end training / plotting helpers |
| `evaluation/` | Classifier comparisons, significance tests, confidence intervals |
| `evaluation/threshold_sweeps/` | Threshold / pseudo-label experiments |
| `figures/` | Regenerate publication figures, ROC, composites |
| `analysis/` | PCA–rainfall checks, feature importance, uncertainty |
| `tools/` | Benchmarks, displacement matrix, exploratory `pca.py` / `rain.py` |

**Examples**

```bash
cd /path/to/pv-insar-landslide-forecast
python scripts/training/model_b_simple.py
python scripts/figures/generate_roc_curve.py
python scripts/analysis/pc_quantitative_support.py
```

Override data location with `LANDSLIDE_PROJECT_ROOT` (see `src/landslide_forecast/config.py`).
