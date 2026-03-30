# Scripts

Analysis, validation, visualization, benchmarks, and legacy repository maintenance.

| Path | Purpose |
|------|---------|
| `analysis/` | PC–rainfall diagnostics, feature importance, uncertainty propagation |
| `validation/` | Classifier comparisons, significance tests, confidence intervals, `threshold_sweeps/` |
| `visualization/` | ROC generation and figure composites |
| `*.py` (root of `Scripts/`) | Rainfall plotting, PCA utilities, displacement matrices, model sweeps, benchmarks |
| `reorganize_repo.py` | Historical one-time migration helper |

Run from the repository root with the package installed (`uv pip install -e .`).
