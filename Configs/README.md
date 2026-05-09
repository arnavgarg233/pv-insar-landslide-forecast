# Configs

Hyperparameters and reproducibility defaults for the main Random Forest / pseudo-labeling pipeline. The training scripts read most settings from code; this folder documents the **paper-aligned** values for reference and for future YAML-driven runs.

| File | Role |
|------|------|
| `model_defaults.yaml` | Forest size, forecast horizon, displacement threshold for pseudo-labels, and random seed |

Edit paths and data locations in `src/landslide_forecast/config.py` or set `LANDSLIDE_PROJECT_ROOT` for a non-default checkout location.
