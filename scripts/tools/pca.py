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
import h5py
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # For parallel processing
from tqdm import tqdm  # For progress bars

# Parameters
WINDOW_SIZE = 10  # Size of the rolling window for PCA
N_COMPONENTS = 5  # Number of principal components to extract
OUTPUT_DIR = str(RF_DATA_DIR)
ASC_FILE = str(ASC_H5)
DESC_FILE = str(DESC_H5)
RAINFALL_DIR = str(RAINFALL_DIR)  # Path to directory containing rainfall NetCDF files

# Helper function to align rainfall data with InSAR timestamps
def align_rainfall_to_insar(rainfall_df, insar_timestamps):
    rainfall_df = rainfall_df.set_index("time").sort_index()
    insar_timestamps = pd.to_datetime(insar_timestamps)
    aligned_rainfall = rainfall_df.reindex(insar_timestamps, method="nearest")["MWprecipitation"]
    aligned_rainfall = aligned_rainfall.iloc[:len(insar_timestamps)].values
    if len(aligned_rainfall) != len(insar_timestamps):
        raise ValueError("Rainfall data time steps do not match InSAR time steps.")
    return aligned_rainfall

# Helper function to load rainfall data
def load_rainfall_data(rainfall_dir, insar_timestamps):
    rainfall_files = [os.path.join(rainfall_dir, f) for f in os.listdir(rainfall_dir) if f.endswith('.nc4')]
    rainfall_data = []
    for file in tqdm(rainfall_files, desc="Loading rainfall files"):  # Progress bar for loading files
        try:
            ds = xr.open_dataset(file)
            if 'MWprecipitation' in ds:
                daily_precip = ds['MWprecipitation'].mean(dim=['lon', 'lat']).to_dataframe()
                rainfall_data.append(daily_precip)
            else:
                print(f"Warning: The file {file} does not contain the 'MWprecipitation' variable.")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    if not rainfall_data:
        raise ValueError("No valid rainfall data found in the provided .nc4 files.")
    rainfall_df = pd.concat(rainfall_data).reset_index()
    rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])
    return align_rainfall_to_insar(rainfall_df, insar_timestamps)

# Function to perform PCA on a single window
def perform_pca(window_data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(window_data)
    return pca.components_  # Shape: (n_components, num_pixels)

# Function to calculate cosine similarity between two sets of principal components
def calculate_cosine_sim(pc1, pc2):
    return cosine_similarity(pc1, pc2).mean()

# Function to perform rolling PCA and calculate cosine similarity
def rolling_pca_cosine_similarity(combined_data, window_size, n_components):
    time_steps, num_pixels = combined_data.shape
    cosine_sim_matrix = np.zeros((time_steps - window_size, time_steps - window_size))

    # Perform PCA for each window and store principal components
    pca_results = []
    for i in tqdm(range(time_steps - window_size), desc="Performing PCA on windows"):  # Progress bar for PCA
        window_data = combined_data[i:i + window_size, :]
        pca_results.append(perform_pca(window_data, n_components))

    # Calculate cosine similarity between principal components in parallel
    def compute_similarity(i, j):
        return calculate_cosine_sim(pca_results[i], pca_results[j])

    # Use joblib to parallelize the computation
    results = Parallel(n_jobs=-1)(
        delayed(compute_similarity)(i, j)
        for i in tqdm(range(time_steps - window_size), desc="Calculating cosine similarities")  # Progress bar for cosine similarity
        for j in range(time_steps - window_size)
    )

    # Reshape results into the similarity matrix
    cosine_sim_matrix = np.array(results).reshape((time_steps - window_size, time_steps - window_size))

    return cosine_sim_matrix

# Main function
def main():
    # Load ascending and descending InSAR data
    with h5py.File(ASC_FILE, 'r') as f:
        asc_data = f['timeseries'][:]  # Shape: (time_steps, height, width)
    with h5py.File(DESC_FILE, 'r') as f:
        desc_data = f['timeseries'][:]  # Shape: (time_steps, height, width)
        desc_timestamps = f['date'][:]  # Assuming timestamps are stored in 'date'

    # Crop ascending data to match descending data
    if asc_data.shape[0] > desc_data.shape[0]:
        asc_data = asc_data[:desc_data.shape[0]]
    if asc_data.shape != desc_data.shape:
        raise ValueError("Asc & Desc data shapes differ!")

    # Convert timestamps from bytes (if necessary)
    desc_timestamps = [timestamp.decode('utf-8') if isinstance(timestamp, bytes) else timestamp for timestamp in desc_timestamps]

    # Load and align rainfall data
    print("Loading and aligning rainfall data...")
    rainfall_data = load_rainfall_data(RAINFALL_DIR, desc_timestamps)

    # Combine InSAR and rainfall data
    time_steps, height, width = asc_data.shape
    combined_data = (asc_data + desc_data) / 2  # Average of ascending and descending data
    combined_data = combined_data.reshape(time_steps, -1)  # Flatten spatial dimensions
    combined_data = np.hstack([combined_data, rainfall_data.reshape(-1, 1)])  # Add rainfall data

    # Perform rolling PCA and calculate cosine similarity
    print("Performing rolling PCA and calculating cosine similarity...")
    cosine_sim_matrix = rolling_pca_cosine_similarity(combined_data, WINDOW_SIZE, N_COMPONENTS)

    # Plot the cosine similarity heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(cosine_sim_matrix, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('Window Index')
    plt.ylabel('Window Index')
    plt.title('Rolling PCA Cosine Similarity Heatmap')
    plt.savefig(os.path.join(OUTPUT_DIR, 'rolling_pca_cosine_similarity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Heatmap saved successfully!")

if __name__ == "__main__":
    main()