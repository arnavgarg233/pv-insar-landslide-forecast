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

from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pandas as pd
import xarray as xr

# Parameters
N_COMPONENTS = 5  # Number of PCA components
OUTPUT_DIR = str(RF_DATA_DIR)
ASC_FILE = str(ASC_H5)
DESC_FILE = str(DESC_H5)
RAINFALL_DIR = str(RAINFALL_DIR)  # Path to directory containing rainfall NetCDF files


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


def calculate_cumulative_displacement_rolling(asc, desc, window_size):
    """Calculate cumulative displacement over rolling windows."""
    vertical = (asc + desc) / 2
    horizontal = (asc - desc) / 2
    displacement_magnitude = np.sqrt(vertical**2 + horizontal**2)
    time_steps, height, width = displacement_magnitude.shape
    cumulative_rolling = np.zeros_like(displacement_magnitude)

    for t in range(time_steps - window_size + 1):
        cumulative_rolling[t] = displacement_magnitude[t:t + window_size].sum(axis=0)

    return cumulative_rolling[: time_steps - window_size + 1]


def generate_pixelwise_labels(rolling_displacement, threshold):
    """
    Generate binary labels for each pixel based on cumulative displacement.
    """
    labels = (rolling_displacement > threshold).astype(int)
    return labels.reshape(-1, labels.shape[0])  # Shape: (num_pixels, time_steps)


def perform_pca_per_pixel(data, n_components=5):
    """
    Perform PCA for each pixel across the temporal dimension.

    Parameters:
    - data: np.ndarray, InSAR data with shape (num_pixels, time_steps).
    - n_components: int, number of principal components to extract.

    Returns:
    - pca_features: np.ndarray, PCA-transformed features for all pixels.
    - explained_variance: np.ndarray, variance explained by each principal component.
    """
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(data)  # Shape: (num_pixels, n_components)
    explained_variance = pca.explained_variance_ratio_
    return pca_features, explained_variance


def tune_hyperparameters(asc_data, desc_data, rainfall_data):
    """
    Tune the threshold and window_size hyperparameters to find the best configuration.
    """
    best_acc = 0
    best_f1 = 0
    best_config = None

    print(f"{'Window Size':<15}{'Threshold':<15}{'Accuracy':<15}{'F1 Score':<15}{'AUC':<15}")
    print("=" * 70)

    for window_size in range(6, 8, 1):  # Test window sizes from 5 to 30 (step 2)
        for threshold in np.arange(0.05, 0.3, 0.05):  # Test thresholds from 0.05 to 0.7 (step 0.05)
            # Calculate rolling displacement
            rolling_displacement = calculate_cumulative_displacement_rolling(asc_data, desc_data, window_size)

            # Generate labels based on the threshold
            labels = generate_pixelwise_labels(rolling_displacement, threshold)
            labels = labels.sum(axis=1) > 0  # Convert to binary labels (1 if any time step > threshold, else 0)

            # Reshape data for PCA
            time_steps, height, width = asc_data.shape
            combined_data = (asc_data + desc_data) / 2
            flattened_data = combined_data.reshape(time_steps, -1).T  # Shape: (num_pixels, time_steps)

            # Align rainfall data with PCA time steps
            if rainfall_data.shape[0] != flattened_data.shape[1]:
                raise ValueError(
                    f"Rainfall time steps ({rainfall_data.shape[0]}) do not match InSAR time steps ({flattened_data.shape[1]})."
                )
            rainfall_repeated = np.tile(rainfall_data, (flattened_data.shape[0], 1))  # Shape: (num_pixels, time_steps)

            # Combine InSAR and rainfall data
            combined_with_rainfall = np.hstack([flattened_data, rainfall_repeated])

            # Perform PCA
            pca_features, _ = perform_pca_per_pixel(combined_with_rainfall, n_components=N_COMPONENTS)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                pca_features, labels, test_size=0.2, random_state=42, stratify=labels
            )

            # Skip configurations where the smaller class is less than half the larger class
            class_counts = np.bincount(y_train)
            if len(class_counts) == 2 and min(class_counts) < 0.5 * max(class_counts):
                continue

            # Train Random Forest model
            from Regression_model import train_risk_classifier
            model = train_risk_classifier(X_train, y_train, use_xgboost=False)

            # Evaluate the model
            y_test_pred = model.predict_proba(X_test)[:, 1] >= 0.5
            acc = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            # Output metrics for this configuration
            print(f"{window_size:<15}{threshold:<15.2f}{acc:<15.2f}{f1:<15.2f}{(auc if auc is not None else 'N/A'):<15}")

            # Save the best configuration
            if acc > best_acc and f1 > best_f1:
                best_acc = acc
                best_f1 = f1
                best_config = (window_size, threshold, acc, f1, auc)

    # Report the best results
    if best_config:
        window_size, threshold, acc, f1, auc = best_config
        print("\nBest Configuration:")
        print(f"Window Size: {window_size}")
        print(f"Threshold: {threshold}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
    else:
        print("No valid configuration found.")

    return best_config
def generate_risk_map_with_probabilities(asc_data, desc_data, rainfall_data, best_config):
    """
    Generate a risk map using the best configuration, leveraging probabilities.

    Parameters:
    - asc_data: np.ndarray, ascending InSAR data.
    - desc_data: np.ndarray, descending InSAR data.
    - rainfall_data: np.ndarray, rainfall data aligned with InSAR time steps.
    - best_config: tuple, the best configuration (window_size, threshold, acc, f1, auc).

    Returns:
    - risk_map: np.ndarray, probability-based risk map of the study area.
    """
    window_size, threshold, _, _, _ = best_config

    # Calculate rolling displacement
    rolling_displacement = calculate_cumulative_displacement_rolling(asc_data, desc_data, window_size)

    # Generate binary labels based on the threshold
    labels = generate_pixelwise_labels(rolling_displacement, threshold)
    labels = labels.sum(axis=1) > 0  # Convert to binary labels (1 if any time step > threshold, else 0)

    # Prepare the dataset for prediction
    time_steps, height, width = asc_data.shape
    combined_data = (asc_data + desc_data) / 2
    flattened_data = combined_data.reshape(time_steps, -1).T  # Shape: (num_pixels, time_steps)

    # Align rainfall data with PCA time steps
    rainfall_repeated = np.tile(rainfall_data, (flattened_data.shape[0], 1))  # Shape: (num_pixels, time_steps)

    # Combine InSAR and rainfall data
    combined_with_rainfall = np.hstack([flattened_data, rainfall_repeated])

    # Perform PCA
    pca_features, _ = perform_pca_per_pixel(combined_with_rainfall, n_components=N_COMPONENTS)

    # Train the model on the entire dataset
    from Regression_model import train_risk_classifier
    model = train_risk_classifier(pca_features, labels, use_xgboost=False)

    # Predict probabilities for the entire dataset
    probabilities = model.predict_proba(pca_features)[:, 1]  # Probability predictions

    # Reshape probabilities into the spatial dimensions
    risk_map = probabilities.reshape(height, width) * 100.0  # Scale to percentage

    return risk_map

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

    # 3) Tune hyperparameters
    print("[Main] Tuning threshold and window size...")
    best_config = tune_hyperparameters(asc_data, desc_data, rainfall_data)

    if best_config:
        print("[Main] Best configuration found. Generating risk map...")
        # 4) Generate risk map using probabilities
        risk_map = generate_risk_map_with_probabilities(asc_data, desc_data, rainfall_data, best_config)
        risk_map = risk_map[50:, :]
        # Save and visualize the risk map
        # Save and visualize the risk map

    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib import cm
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    import numpy.ma as ma  # For masking water areas

    # Reduce risk values by 20%, ensuring no negatives
    risk_map = np.maximum(risk_map - 20, 0)

    # 🚀 **Step 3: Define a White-to-Green Gradient for Land**
    colors = ["#ffffff", "#d4f7c5", "#a0e18e", "#4db560", "#006400"]  # White to deep green
    cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)  # Smooth gradient

    # Save and visualize the risk map
    plt.figure(figsize=(12, 8), facecolor="lightgray")  # Light gray background for outer area

    # Normalize risk values (0-100%) for **smooth** color mapping
    norm = Normalize(vmin=0, vmax=100)

    # 🚀 **Step 4: Set the Background (Water) to Light Blue**
    ax = plt.gca()
    ax.set_facecolor("#b3e0ff")  # ✅ Force non-land areas (water) to be light blue

    # Plot the **masked risk map** so only land areas get the green gradient
    im = plt.imshow(risk_map, cmap=cmap, norm=norm, interpolation="nearest")  # No artificial smoothing

    # Define the tick positions and labels for the color bar
    tick_positions = [0, 25, 50, 75, 100]  # Positions for No Risk → Very High Risk
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, ticks=tick_positions)  # ✅ Set fixed tick positions
    cbar.set_label("Landslide Risk", fontsize=12)
    cbar.ax.set_yticklabels(["No Risk", "Low Risk", "Moderate", "High Risk", "Very High Risk"])  # ✅ Correct tick labels
    cbar.ax.tick_params(labelsize=10)

    # Title, labels, and grid
    plt.title("Palos Verdes Landslide Risk Map", fontsize=14, fontweight="bold")
    plt.xlabel("", fontsize=12)
    plt.ylabel("", fontsize=12)
    plt.grid(visible=True, color="white", linestyle="--", linewidth=0.5)

    # Save the enhanced risk map
    output_png = os.path.join(OUTPUT_DIR, "fixed_water_smooth_risk_map.png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="lightgray")
    plt.close()
    print(f"[Main] Fixed landslide risk map saved to: {output_png}")
    print("Risk Map Probabilities (Raw numpy array):")
    print(risk_map)  # Print the raw numpy array

        # Save the raw numpy array to a file
    output_npy = os.path.join(OUTPUT_DIR, "risk_map_probabilities.npy")
    np.save(output_npy, risk_map)
    print(f"[Main] Raw risk map probabilities saved to: {output_npy}")

        # Save the raw numpy array as a text file (optional)
    output_txt = os.path.join(OUTPUT_DIR, "risk_map_probabilities.txt")
    np.savetxt(output_txt, risk_map, fmt="%.4f")
    print(f"[Main] Raw risk map probabilities saved to: {output_txt}")

if __name__ == "__main__":
    main()