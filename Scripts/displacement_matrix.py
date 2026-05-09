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

import numpy as np
import h5py

# File paths for ascending and descending InSAR data
ASC_FILE = str(ASC_H5)
DESC_FILE = str(DESC_H5)

def calculate_displacement(asc_data, desc_data):
    """
    Calculate displacement using ascending and descending InSAR data.

    Parameters:
    - asc_data: np.ndarray, ascending InSAR data with shape (time_steps, height, width).
    - desc_data: np.ndarray, descending InSAR data with shape (time_steps, height, width).

    Returns:
    - displacement: np.ndarray, 3D array of displacement values with shape (time_steps, height, width).
    """
    # Ensure both datasets have the same shape
    if asc_data.shape != desc_data.shape:
        raise ValueError("Ascending and descending data shapes do not match!")

    # Calculate displacement: average of ascending and descending data
    displacement = (asc_data + desc_data) / 2

    return displacement

def main():
    # Load ascending and descending InSAR data
    with h5py.File(ASC_FILE, 'r') as f:
        asc_data = f['timeseries'][:]  # Shape: (time_steps, height, width)

    with h5py.File(DESC_FILE, 'r') as f:
        desc_data = f['timeseries'][:]  # Shape: (time_steps, height, width)

    # Ensure both datasets have the same number of time steps
    if asc_data.shape[0] > desc_data.shape[0]:
        asc_data = asc_data[:desc_data.shape[0]]  # Crop ascending data to match descending data
    elif desc_data.shape[0] > asc_data.shape[0]:
        desc_data = desc_data[:asc_data.shape[0]]  # Crop descending data to match ascending data

    # Calculate displacement
    displacement = calculate_displacement(asc_data, desc_data)

    # Output the 3D numpy array
    print("Displacement array shape:", displacement.shape)  # Should be (179, height, width)
    print("Displacement array (first time step):\n", displacement[0])  # Example output for the first time step

    # Save the displacement array to a file
    output_file = "displacement_3d.npy"
    np.save(output_file, displacement)
    print(f"Displacement array saved to {output_file}")

if __name__ == "__main__":
    main()