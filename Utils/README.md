# Utils

Shared, importable helpers for this project live in the **`landslide_forecast`** package under `src/landslide_forecast/` (path resolution, HDF5 locations, rainfall and MintPy directories).

Use `uv pip install -e .` from the repository root, then:

```python
from landslide_forecast.config import REPO_ROOT, RF_DATA_DIR, FINAL_RESULTS_DIR
```

Add small reusable functions here only when they do not belong in the package (e.g. one-off notebook helpers).
