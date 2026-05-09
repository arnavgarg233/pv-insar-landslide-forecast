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
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import h5py

# Parameters
RAINFALL_DIR = str(RAINFALL_DIR)  # Path to directory containing rainfall NetCDF files
ASC_FILE = str(ASC_H5)
DESC_FILE = str(DESC_H5)

def align_rainfall_to_insar(rainfall_df, insar_timestamps):
    """
    Align rainfall data with InSAR timestamps using interpolation or cropping.

    Parameters:
    - rainfall_df: pd.DataFrame, rainfall data with "time" and "MWprecipitation" columns.
    - insar_timestamps: np.ndarray, timestamps from InSAR data.

    Returns:
    - aligned_rainfall: np.ndarray, rainfall values aligned with InSAR time steps.
    """
    # Ensure 'time' is the index and sorted
    rainfall_df = rainfall_df.set_index("time").sort_index()

    # Convert InSAR timestamps to datetime
    insar_timestamps = pd.to_datetime(insar_timestamps)

    # Align rainfall data with InSAR timestamps
    aligned_rainfall = rainfall_df.reindex(insar_timestamps, method="nearest")["MWprecipitation"]

    # Explicitly crop to match the exact number of InSAR time steps
    aligned_rainfall = aligned_rainfall.iloc[:len(insar_timestamps)].values

    if len(aligned_rainfall) != len(insar_timestamps):
        raise ValueError(
            f"Rainfall data time steps ({len(aligned_rainfall)}) still do not match InSAR time steps ({len(insar_timestamps)})."
        )

    return aligned_rainfall

def load_rainfall_data(rainfall_dir, insar_timestamps):
    """
    Load rainfall data from .nc4 files, aggregate it, and align it with InSAR time steps.

    Parameters:
    - rainfall_dir: str, path to the directory containing .nc4 files.
    - insar_timestamps: np.ndarray, timestamps from InSAR data.

    Returns:
    - aligned_rainfall: np.ndarray, rainfall data aligned with InSAR time steps.
    """
    rainfall_files = [os.path.join(rainfall_dir, f) for f in os.listdir(rainfall_dir) if f.endswith('.nc4')]
    rainfall_data = []

    for file in rainfall_files:
        try:
            # Open the .nc4 file
            ds = xr.open_dataset(file)

            # Check if 'MWprecipitation' variable exists
            if 'MWprecipitation' in ds:
                # Aggregate rainfall data over latitude and longitude
                daily_precip = ds['MWprecipitation'].mean(dim=['lon', 'lat']).to_dataframe()
                rainfall_data.append(daily_precip)
            else:
                print(f"Warning: The file {file} does not contain the 'MWprecipitation' variable.")
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not rainfall_data:
        raise ValueError("No valid rainfall data found in the provided .nc4 files.")

    # Combine all rainfall data into a single DataFrame
    rainfall_df = pd.concat(rainfall_data).reset_index()
    rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])  # Ensure time column is in datetime format

    # Align rainfall data with InSAR timestamps
    aligned_rainfall = align_rainfall_to_insar(rainfall_df, insar_timestamps)

    return aligned_rainfall

def plot_mean_rainfall_over_time(rainfall_data, insar_timestamps):
    """
    Plot the mean rainfall over time.

    Parameters:
    - rainfall_data: np.ndarray, rainfall data aligned with InSAR time steps.
    - insar_timestamps: np.ndarray, timestamps from InSAR data.
    """
    # Check the shape of rainfall_data
    print(f"Shape of rainfall_data: {rainfall_data.shape}")  # Debugging

    # Calculate mean rainfall over time (mean across pixels for each timestamp)
    mean_rainfall = np.mean(rainfall_data, axis=0)  # Shape: (num_timestamps,)

    # Check the shape of mean_rainfall
    print(f"Shape of mean_rainfall: {mean_rainfall.shape}")  # Debugging

    # Convert InSAR timestamps to datetime
    insar_timestamps = pd.to_datetime(insar_timestamps)

    # Check the length of insar_timestamps
    print(f"Length of insar_timestamps: {len(insar_timestamps)}")  # Debugging

    # Plot the mean rainfall over time
    plt.figure(figsize=(12, 6))
    plt.plot(insar_timestamps, mean_rainfall, label='Mean Rainfall', color='blue')
    plt.title('Mean Rainfall Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Mean Rainfall (mm)', fontsize=12)
    plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # 1) Load ascending and descending InSAR data
    with h5py.File(ASC_FILE, 'r') as f:
        asc_data = f['timeseries'][:]  # Shape: (time_steps, height, width)

    with h5py.File(DESC_FILE, 'r') as f:
        desc_data = f['timeseries'][:]  # Shape: (time_steps, height, width)
        desc_timestamps = f['date'][:]  # Assuming timestamps are stored in 'date'

    if asc_data.shape[0] > desc_data.shape[0]:
        asc_data = asc_data[:desc_data.shape[0]]  # Crop ascending data to match descending data
    if asc_data.shape != desc_data.shape:
        raise ValueError("Asc & Desc data shapes differ!")

    # Convert timestamps from bytes (if necessary)
    desc_timestamps = [timestamp.decode('utf-8') if isinstance(timestamp, bytes) else timestamp for timestamp in desc_timestamps]

    # 2) Load and align rainfall data
    rainfall_data = load_rainfall_data(RAINFALL_DIR, desc_timestamps)  # Align with descending data timestamps

    # 3) Plot mean rainfall over time
    plot_mean_rainfall_over_time(rainfall_data, desc_timestamps)

if __name__ == "__main__":
    main()