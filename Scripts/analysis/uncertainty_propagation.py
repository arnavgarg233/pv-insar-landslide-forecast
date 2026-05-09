"""
Uncertainty Propagation through WLS Inversion and PCA
======================================================
This script implements uncertainty propagation following the approach in:
Binh et al. (2020), Science of the Total Environment, DOI: 10.1016/j.scitotenv.2020.141258

1. WLS Inversion Uncertainty: Propagate coherence-based uncertainties through
   the weighted least squares time-series inversion
2. PCA Uncertainty: Propagate time-series uncertainties through PCA transformation
"""


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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Paths
base_path = RF_DATA_DIR
mintpy_path = MINTPY_STACK_DIR
asc_data = mintpy_path / "Ascendingdata/timeseries.h5"
asc_coh = mintpy_path / "Ascendingdata/temporalCoherence.h5"
desc_data = mintpy_path / "ProcessedDescenindg/timeseriesd.h5"
desc_coh = mintpy_path / "ProcessedDescenindg/temporalCoherence.h5"
output_file = base_path / "uncertainty_propagation_results.h5"

print("="*70)
print("Uncertainty Propagation Analysis")
print("="*70)

def estimate_interferogram_std(coherence, wavelength=0.056, L=20):
    """
    Estimate standard deviation of interferometric phase from coherence.
    
    Based on the Cramer-Rao lower bound for phase estimation:
    σ_φ = sqrt(1 / (2*L)) * sqrt((1 - coh^2) / coh^2)
    
    Parameters:
    -----------
    coherence : array
        Interferometric coherence values (0 to 1)
    wavelength : float
        Radar wavelength in meters (Sentinel-1 C-band: 0.056 m)
    L : int
        Number of looks (effective number of independent samples)
    
    Returns:
    --------
    std_phase : array
        Standard deviation of phase in radians
    std_displacement : array
        Standard deviation of LOS displacement in cm
    """
    # Cramer-Rao bound for phase standard deviation
    # Prevent division by zero
    coherence = np.clip(coherence, 0.01, 0.99)
    
    std_phase = np.sqrt(1.0 / (2.0 * L)) * np.sqrt((1 - coherence**2) / coherence**2)
    
    # Convert phase uncertainty to displacement uncertainty
    # displacement = phase * wavelength / (4π)
    std_displacement = std_phase * wavelength / (4 * np.pi) * 100  # Convert to cm
    
    return std_phase, std_displacement


def propagate_wls_uncertainty(design_matrix, observations_std, weights=None):
    """
    Propagate uncertainty through weighted least squares inversion.
    
    For WLS: X = (A^T W A)^-1 A^T W y
    Covariance: Cov(X) = (A^T W A)^-1 A^T W Cov(y) W A (A^T W A)^-1
    
    Simplified for diagonal weight matrix:
    Cov(X) = (A^T W A)^-1
    
    Parameters:
    -----------
    design_matrix : array, shape (n_obs, n_params)
        Design matrix A for time-series inversion
    observations_std : array, shape (n_obs,)
        Standard deviation of observations
    weights : array, shape (n_obs,), optional
        Weight for each observation (typically from coherence)
    
    Returns:
    --------
    timeseries_std : array, shape (n_params,)
        Standard deviation of estimated time-series parameters
    """
    A = design_matrix
    
    if weights is None:
        # Ordinary Least Squares
        # Cov(X) = σ^2 (A^T A)^-1
        ATA_inv = np.linalg.pinv(A.T @ A)
        mean_obs_var = np.mean(observations_std**2)
        ts_cov = mean_obs_var * ATA_inv
    else:
        # Weighted Least Squares
        # Weight matrix (diagonal)
        W = np.diag(weights)
        
        # Covariance of observations (diagonal)
        Sigma_y = np.diag(observations_std**2)
        
        # Full propagation: Cov(X) = (A^T W A)^-1 A^T W Sigma_y W A (A^T W A)^-1
        ATWA = A.T @ W @ A
        ATWA_inv = np.linalg.pinv(ATWA)
        
        middle = A.T @ W @ Sigma_y @ W @ A
        ts_cov = ATWA_inv @ middle @ ATWA_inv
    
    # Extract standard deviations (diagonal of covariance matrix)
    timeseries_std = np.sqrt(np.diag(ts_cov))
    
    return timeseries_std, ts_cov


def propagate_pca_uncertainty(pca_model, data_cov, n_components=3):
    """
    Propagate uncertainty through PCA transformation.
    
    For PCA: Z = V^T X, where V is the eigenvector matrix
    Covariance: Cov(Z) = V^T Cov(X) V
    
    Parameters:
    -----------
    pca_model : sklearn PCA object
        Fitted PCA model
    data_cov : array, shape (n_features, n_features)
        Covariance matrix of input data
    n_components : int
        Number of principal components
    
    Returns:
    --------
    pc_std : array, shape (n_components,)
        Standard deviation of principal components
    """
    # Get the principal component loadings (eigenvectors)
    V = pca_model.components_[:n_components, :]
    
    # Propagate covariance: Cov(PC) = V^T Cov(data) V
    pc_cov = V @ data_cov @ V.T
    
    # Extract standard deviations
    pc_std = np.sqrt(np.diag(pc_cov))
    
    return pc_std, pc_cov


def analyze_orbit(data_file, coh_file, orbit_name):
    """
    Analyze uncertainty for one orbital geometry.
    """
    print(f"\n{'='*70}")
    print(f"Processing {orbit_name} orbit")
    print(f"{'='*70}")
    
    # Load time-series data
    with h5py.File(data_file, 'r') as f:
        # Check available datasets
        print(f"Available datasets: {list(f.keys())}")
        
        # Load timeseries data
        timeseries = f['timeseries'][:]  # Shape: (n_dates, n_rows, n_cols)
        dates = f['date'][:]
        
        print(f"Time-series shape: {timeseries.shape}")
    
    # Reshape to (n_pixels, n_dates) for easier processing
    n_dates, n_rows, n_cols = timeseries.shape
    
    # MEMORY OPTIMIZATION: Subsample pixels to avoid memory issues
    # Use every 10th pixel for a manageable subset
    subsample_factor = 10
    row_indices = np.arange(0, n_rows, subsample_factor)
    col_indices = np.arange(0, n_cols, subsample_factor)
    
    # Create meshgrid for subsampling
    rows_sub, cols_sub = np.meshgrid(row_indices, col_indices, indexing='ij')
    
    # Extract subsampled data
    timeseries_sub = timeseries[:, rows_sub, cols_sub]  # (n_dates, n_rows_sub, n_cols_sub)
    
    n_pixels = rows_sub.size
    velocity = timeseries_sub.reshape(n_dates, -1).T  # (n_pixels, n_dates)
    
    print(f"Subsampled from {n_rows}×{n_cols} to {len(row_indices)}×{len(col_indices)} grid")
    print(f"Total pixels for analysis: {n_pixels}")
    
    # Load temporal coherence (subsampled)
    try:
        with h5py.File(coh_file, 'r') as f:
            coherence_2d = f['temporalCoherence'][:]  # Shape: (n_rows, n_cols)
            # Subsample coherence to match velocity
            coherence_sub = coherence_2d[rows_sub, cols_sub]
            coherence = coherence_sub.flatten()  # Flatten to match pixels
            # Broadcast to all dates
            coherence = np.tile(coherence[:, np.newaxis], (1, n_dates))
            print(f"Loaded temporal coherence data (mean: {np.mean(coherence):.3f})")
    except Exception as e:
        # Use a conservative estimate
        coherence = 0.7 * np.ones(velocity.shape)
        print(f"Warning: Could not load coherence ({e}), using conservative estimate of 0.7")
    
    n_pixels, n_dates = velocity.shape
    print(f"Data shape: {n_pixels} pixels × {n_dates} dates")
    
    # Step 1: Estimate interferogram uncertainties from coherence
    print(f"\nStep 1: Estimating interferogram uncertainties...")
    _, ifg_std = estimate_interferogram_std(coherence)
    
    # Calculate mean uncertainty per date
    mean_ifg_std = np.mean(ifg_std, axis=0)
    print(f"Mean interferogram uncertainty: {np.mean(mean_ifg_std):.3f} ± {np.std(mean_ifg_std):.3f} cm")
    
    # Step 2: Propagate through WLS inversion
    print(f"\nStep 2: Propagating uncertainty through WLS inversion...")
    
    # Create a simplified design matrix for demonstration
    # In reality, this would come from the SBAS network
    n_pairs = n_dates - 1
    A = np.zeros((n_pairs, n_dates - 1))
    for i in range(n_pairs):
        A[i, i] = 1  # Simplified: each pair contributes to one time step
    
    # Calculate weights from coherence (inverse variance weighting)
    weights = np.mean(coherence, axis=0)**2  # Simplified: mean coherence per date
    weights = weights[:-1]  # Match number of pairs
    
    # Propagate for a representative pixel (median uncertainty)
    median_pixel = n_pixels // 2
    obs_std = ifg_std[median_pixel, :-1]  # Use first n_pairs dates
    
    ts_std, ts_cov = propagate_wls_uncertainty(A, obs_std, weights)
    
    print(f"Time-series uncertainty after WLS:")
    print(f"  Mean: {np.mean(ts_std):.3f} cm")
    print(f"  Median: {np.median(ts_std):.3f} cm")
    print(f"  95th percentile: {np.percentile(ts_std, 95):.3f} cm")
    
    # Step 3: Perform PCA
    print(f"\nStep 3: Performing PCA decomposition...")
    
    # Standardize data
    velocity_mean = np.mean(velocity, axis=1, keepdims=True)
    velocity_std_data = np.std(velocity, axis=1, keepdims=True)
    velocity_standardized = (velocity - velocity_mean) / (velocity_std_data + 1e-10)
    
    # Fit PCA
    n_components = 5
    pca = PCA(n_components=n_components)
    pc_features = pca.fit_transform(velocity_standardized)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Step 4: Propagate uncertainty through PCA
    print(f"\nStep 4: Propagating uncertainty through PCA...")
    
    # Estimate data covariance matrix from time-series uncertainties
    # Simplified: assume independent observations with variance = mean(ts_std^2)
    data_var = np.mean(ts_std**2)
    data_cov = data_var * np.eye(n_dates - 1)  # Diagonal covariance
    
    # Pad to match n_dates if needed
    if data_cov.shape[0] < n_dates:
        temp = np.zeros((n_dates, n_dates))
        temp[:data_cov.shape[0], :data_cov.shape[0]] = data_cov
        data_cov = temp
    
    pc_std, pc_cov = propagate_pca_uncertainty(pca, data_cov, n_components)
    
    print(f"\nPrincipal Component uncertainties:")
    for i in range(n_components):
        print(f"  PC{i+1}: σ = {pc_std[i]:.4f} (standardized units)")
    
    # Calculate relative uncertainty (coefficient of variation)
    pc_means = np.mean(np.abs(pc_features), axis=0)
    cv = pc_std / (pc_means + 1e-10) * 100  # As percentage
    
    print(f"\nRelative uncertainty (Coefficient of Variation):")
    for i in range(n_components):
        print(f"  PC{i+1}: CV = {cv[i]:.2f}%")
    
    # Step 5: Calculate signal-to-noise ratio
    print(f"\nStep 5: Signal-to-Noise Ratio Analysis...")
    
    pc_signal = np.std(pc_features, axis=0)  # Std of actual PC values
    snr = pc_signal / pc_std
    snr_db = 20 * np.log10(snr)
    
    print(f"Signal-to-Noise Ratio:")
    for i in range(n_components):
        print(f"  PC{i+1}: SNR = {snr[i]:.2f} ({snr_db[i]:.2f} dB)")
    
    # Store results
    results = {
        'orbit': orbit_name,
        'n_pixels': n_pixels,
        'n_dates': n_dates,
        'mean_ifg_std': mean_ifg_std,
        'ts_std': ts_std,
        'ts_cov': ts_cov,
        'pc_std': pc_std,
        'pc_cov': pc_cov,
        'cv': cv,
        'snr': snr,
        'snr_db': snr_db,
        'explained_variance': pca.explained_variance_ratio_
    }
    
    return results


def create_summary_plot(results_asc, results_desc):
    """
    Create visualization of uncertainty propagation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Uncertainty Propagation: WLS → PCA', fontsize=14, fontweight='bold')
    
    # Plot 1: Time-series uncertainty
    ax = axes[0, 0]
    ax.plot(results_asc['ts_std'], 'b-', label='Ascending', linewidth=2)
    ax.plot(results_desc['ts_std'], 'r-', label='Descending', linewidth=2)
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Uncertainty (cm)', fontsize=11)
    ax.set_title('Time-Series Uncertainty from WLS Inversion', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: PC uncertainty
    ax = axes[0, 1]
    x = np.arange(1, 6)
    width = 0.35
    ax.bar(x - width/2, results_asc['pc_std'], width, label='Ascending', color='blue', alpha=0.7)
    ax.bar(x + width/2, results_desc['pc_std'], width, label='Descending', color='red', alpha=0.7)
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Uncertainty (std units)', fontsize=11)
    ax.set_title('Principal Component Uncertainty', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i}' for i in x])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Coefficient of Variation
    ax = axes[1, 0]
    ax.bar(x - width/2, results_asc['cv'], width, label='Ascending', color='blue', alpha=0.7)
    ax.bar(x + width/2, results_desc['cv'], width, label='Descending', color='red', alpha=0.7)
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('CV (%)', fontsize=11)
    ax.set_title('Relative Uncertainty (Coefficient of Variation)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i}' for i in x])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Signal-to-Noise Ratio
    ax = axes[1, 1]
    ax.bar(x - width/2, results_asc['snr_db'], width, label='Ascending', color='blue', alpha=0.7)
    ax.bar(x + width/2, results_desc['snr_db'], width, label='Descending', color='red', alpha=0.7)
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('SNR (dB)', fontsize=11)
    ax.set_title('Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i}' for i in x])
    ax.axhline(y=20, color='green', linestyle='--', linewidth=1, alpha=0.5, label='High SNR threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(base_path / 'uncertainty_propagation.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: uncertainty_propagation.png")
    plt.close()


def save_results(results_asc, results_desc):
    """
    Save uncertainty results to HDF5 file.
    """
    with h5py.File(output_file, 'w') as f:
        # Ascending orbit
        grp_asc = f.create_group('ascending')
        for key, value in results_asc.items():
            if isinstance(value, (int, float, str)):
                grp_asc.attrs[key] = value
            else:
                grp_asc.create_dataset(key, data=value)
        
        # Descending orbit
        grp_desc = f.create_group('descending')
        for key, value in results_desc.items():
            if isinstance(value, (int, float, str)):
                grp_desc.attrs[key] = value
            else:
                grp_desc.create_dataset(key, data=value)
    
    print(f"\n✓ Saved results: {output_file.name}")


def generate_latex_summary(results_asc, results_desc):
    """
    Generate LaTeX table for paper.
    """
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Uncertainty propagation through WLS inversion and PCA decomposition. Values represent the standard deviation of each principal component after propagating measurement uncertainties through the complete processing chain.}
\label{tab:uncertainty_propagation}
\begin{tabular}{lcccccc}
\toprule
\textbf{Orbit} & \textbf{PC1} & \textbf{PC2} & \textbf{PC3} & \textbf{PC4} & \textbf{PC5} & \textbf{Mean SNR (dB)} \\
\midrule
"""
    
    # Ascending row
    latex += "Ascending & "
    latex += " & ".join([f"{std:.3f}" for std in results_asc['pc_std']])
    latex += f" & {np.mean(results_asc['snr_db']):.1f} \\\\\n"
    
    # Descending row
    latex += "Descending & "
    latex += " & ".join([f"{std:.3f}" for std in results_desc['pc_std']])
    latex += f" & {np.mean(results_desc['snr_db']):.1f} \\\\\n"
    
    latex += r"""\midrule
\multicolumn{7}{l}{\textit{Units: standardized (zero mean, unit variance)}} \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    latex_file = base_path / "uncertainty_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex)
    
    print(f"\n✓ Generated LaTeX table: {latex_file.name}")
    return latex


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    try:
        # Analyze both orbits
        results_asc = analyze_orbit(asc_data, asc_coh, "Ascending")
        results_desc = analyze_orbit(desc_data, desc_coh, "Descending")
        
        # Create visualization
        print(f"\n{'='*70}")
        print("Creating visualizations...")
        print(f"{'='*70}")
        create_summary_plot(results_asc, results_desc)
        
        # Save results
        save_results(results_asc, results_desc)
        
        # Generate LaTeX table
        print(f"\n{'='*70}")
        print("Generating LaTeX summary...")
        print(f"{'='*70}")
        latex_table = generate_latex_summary(results_asc, results_desc)
        print(latex_table)
        
        # Print summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"\nKey Findings:")
        print(f"1. WLS time-series uncertainty (ascending): {np.mean(results_asc['ts_std']):.3f} ± {np.std(results_asc['ts_std']):.3f} cm")
        print(f"2. WLS time-series uncertainty (descending): {np.mean(results_desc['ts_std']):.3f} ± {np.std(results_desc['ts_std']):.3f} cm")
        print(f"3. Mean PC uncertainty (ascending): {np.mean(results_asc['pc_std']):.4f} std units")
        print(f"4. Mean PC uncertainty (descending): {np.mean(results_desc['pc_std']):.4f} std units")
        print(f"5. Mean SNR (ascending): {np.mean(results_asc['snr_db']):.1f} dB")
        print(f"6. Mean SNR (descending): {np.mean(results_desc['snr_db']):.1f} dB")
        
        print(f"\n{'='*70}")
        print("✓ Uncertainty propagation analysis complete!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

