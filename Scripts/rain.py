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
    FINAL_RESULTS_DIR,
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
from datetime import datetime

# Parameters
RAINFALL_DIR = str(RAINFALL_DIR)  # Path to directory containing rainfall NetCDF files
OUTPUT_DIR = str(FINAL_RESULTS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist

def load_rainfall_data(rainfall_dir):
    """
    Load rainfall data from .nc4 files and aggregate it.

    Parameters:
    - rainfall_dir: str, path to the directory containing .nc4 files.

    Returns:
    - rainfall_df: pd.DataFrame, rainfall data with "time" and "MWprecipitation" columns.
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

    return rainfall_df

def check_time_range(rainfall_df):
    """
    Check the time range of the rainfall data and warn if it does not extend beyond 2021.

    Parameters:
    - rainfall_df: pd.DataFrame, rainfall data with "time" and "MWprecipitation" columns.
    """
    min_time = rainfall_df["time"].min()
    max_time = rainfall_df["time"].max()

    print(f"Rainfall data time range: {min_time} to {max_time}")

    # Check if the data extends beyond 2021
    if max_time < datetime(2022, 1, 1):
        print("Warning: Rainfall data does not extend beyond 2021. Additional data is required.")
    else:
        print("Rainfall data extends beyond 2021.")

def plot_mean_rainfall_over_time(rainfall_df, output_dir):
    """
    Plot the mean rainfall over time and save the plot as an image.

    Parameters:
    - rainfall_df: pd.DataFrame, rainfall data with "time" and "MWprecipitation" columns.
    - output_dir: str, directory to save the plot.
    """
    # Calculate mean rainfall over time
    mean_rainfall = rainfall_df.groupby("time")["MWprecipitation"].mean()

    # Plot the mean rainfall over time
    plt.figure(figsize=(24, 6))
    plt.plot(mean_rainfall.index, mean_rainfall.values, label='Mean Rainfall', color='blue')
    plt.title('Mean Rainfall Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Mean Rainfall (mm)', fontsize=12)
    plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "mean_rainfall_over_time.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free up memory
    print(f"Plot saved to: {output_file}")

def main():
    # Load rainfall data
    print("Loading rainfall data...")
    rainfall_df = load_rainfall_data(RAINFALL_DIR)

    # Check the time range of the rainfall data
    print("Checking time range of rainfall data...")
    check_time_range(rainfall_df)

    # Plot mean rainfall over time and save the plot
    print("Plotting and saving mean rainfall over time...")
    plot_mean_rainfall_over_time(rainfall_df, OUTPUT_DIR)

if __name__ == "__main__":
    main()