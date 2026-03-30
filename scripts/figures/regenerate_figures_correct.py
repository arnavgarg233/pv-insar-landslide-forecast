"""
Regenerate Figures 4 & 5 - CORRECTLY Using Original Data Structure
===================================================================
This version properly loads the spatial dimensions from the original data
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

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# Paths
base_path = REPO_ROOT
mintpy_path = base_path / "Ascending and Descending MintPy data"
asc_ts_file = mintpy_path / "Ascendingdata/timeseries.h5"
desc_ts_file = mintpy_path / "ProcessedDescenindg/timeseriesd.h5"
pca_file = base_path / "RandomForest MovmentLandslide threshodl/pca_results.h5"
output_dir = base_path / "RandomForest MovmentLandslide threshodl"

print("="*70)
print("Regenerating Figures 4 & 5 - CORRECT VERSION")
print("="*70)

# ===========================================================================
# Load Original Timeseries to Get Proper Spatial Dimensions
# ===========================================================================

print("\nLoading spatial dimensions from original InSAR data...")

with h5py.File(asc_ts_file, 'r') as f:
    asc_timeseries = f['timeseries'][:]
    asc_dates_raw = f['date'][:]
    n_dates_asc, n_rows_asc, n_cols_asc = asc_timeseries.shape
    print(f"✓ Ascending: {n_dates_asc} dates × {n_rows_asc} rows × {n_cols_asc} cols")

with h5py.File(desc_ts_file, 'r') as f:
    desc_timeseries = f['timeseries'][:]
    desc_dates_raw = f['date'][:]
    n_dates_desc, n_rows_desc, n_cols_desc = desc_timeseries.shape
    print(f"✓ Descending: {n_dates_desc} dates × {n_rows_desc} rows × {n_cols_desc} cols")

# ===========================================================================
# Load PCA Results
# ===========================================================================

print("\nLoading PCA results...")
with h5py.File(pca_file, 'r') as f:
    # Ascending
    asc_spatial_pcs = f['ascending/spatial_components'][:]  # Shape: (n_components, n_pixels)
    asc_temporal_pcs = f['ascending/scores'][:]  # Shape: (n_dates, n_components)
    asc_variance = f['ascending/variance_ratio'][:]
    asc_dates_pca = f['ascending/dates'][:]
    
    # Descending
    desc_spatial_pcs = f['descending/spatial_components'][:]
    desc_temporal_pcs = f['descending/scores'][:]
    desc_variance = f['descending/variance_ratio'][:]
    desc_dates_pca = f['descending/dates'][:]

print(f"✓ Ascending PCs: {asc_spatial_pcs.shape[0]} components × {asc_spatial_pcs.shape[1]} pixels")
print(f"✓ Descending PCs: {desc_spatial_pcs.shape[0]} components × {desc_spatial_pcs.shape[1]} pixels")

# ===========================================================================
# FIGURE 4: Spatial Patterns of PC1 (Using Correct Dimensions)
# ===========================================================================

print("\n" + "="*70)
print("Creating Figure 4: Spatial Patterns of PC1")
print("="*70)

# Reshape PC1 to proper spatial dimensions
pc1_asc_spatial = asc_spatial_pcs[0, :].reshape(n_rows_asc, n_cols_asc)
pc1_desc_spatial = desc_spatial_pcs[0, :].reshape(n_rows_desc, n_cols_desc)

print(f"PC1 Ascending spatial shape: {pc1_asc_spatial.shape}")
print(f"PC1 Descending spatial shape: {pc1_desc_spatial.shape}")

# Create Figure 4
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Common colorbar limits
vmin = min(np.nanpercentile(pc1_asc_spatial, 2), np.nanpercentile(pc1_desc_spatial, 2))
vmax = max(np.nanpercentile(pc1_asc_spatial, 98), np.nanpercentile(pc1_desc_spatial, 98))

# Plot (a) Ascending PC1
ax = axes[0]
im = ax.imshow(pc1_asc_spatial, cmap='RdBu_r', vmin=vmin, vmax=vmax, 
               aspect='equal', interpolation='bilinear')
ax.set_title('(a) Ascending PC1', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude (pixel)', fontsize=11)
ax.set_ylabel('Latitude (pixel)', fontsize=11)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Amplitude', fontsize=10, rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=9)

# Plot (b) Descending PC1
ax = axes[1]
im = ax.imshow(pc1_desc_spatial, cmap='RdBu_r', vmin=vmin, vmax=vmax,
               aspect='equal', interpolation='bilinear')
ax.set_title('(b) Descending PC1', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude (pixel)', fontsize=11)
ax.set_ylabel('Latitude (pixel)', fontsize=11)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Amplitude', fontsize=10, rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
output_file_4 = output_dir / 'Figure_4_PC1_Spatial_HighRes_CORRECTED.png'
plt.savefig(output_file_4, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {output_file_4.name}")
plt.close()

# ===========================================================================
# FIGURE 5: Temporal Evolution of PC1, PC2, PC3
# ===========================================================================

print("\n" + "="*70)
print("Creating Figure 5: Temporal Evolution of PCs")
print("="*70)

# Convert dates
dates_asc_str = [d.decode('utf-8') if isinstance(d, bytes) else str(d) for d in asc_dates_pca]
dates_desc_str = [d.decode('utf-8') if isinstance(d, bytes) else str(d) for d in desc_dates_pca]

dates_asc = pd.to_datetime(dates_asc_str, format='%Y%m%d')
dates_desc = pd.to_datetime(dates_desc_str, format='%Y%m%d')

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot (a) Ascending
ax = axes[0]
ax.plot(dates_asc, asc_temporal_pcs[:, 0], 'b-', linewidth=1.2, alpha=0.8,
        label=f'PC1 ({asc_variance[0]*100:.1f}%)')
ax.plot(dates_asc, asc_temporal_pcs[:, 1], color='#FF8C00', linewidth=1.2, alpha=0.8,
        label=f'PC2 ({asc_variance[1]*100:.1f}%)')
ax.plot(dates_asc, asc_temporal_pcs[:, 2], 'g-', linewidth=1.2, alpha=0.8,
        label=f'PC3 ({asc_variance[2]*100:.1f}%)')

ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax.set_title('(a) Ascending Orbit', fontsize=13, fontweight='bold')
ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Amplitude', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.2, linestyle=':')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot (b) Descending  
ax = axes[1]
ax.plot(dates_desc, desc_temporal_pcs[:, 0], 'b-', linewidth=1.2, alpha=0.8,
        label=f'PC1 ({desc_variance[0]*100:.1f}%)')
ax.plot(dates_desc, desc_temporal_pcs[:, 1], color='#FF8C00', linewidth=1.2, alpha=0.8,
        label=f'PC2 ({desc_variance[1]*100:.1f}%)')
ax.plot(dates_desc, desc_temporal_pcs[:, 2], 'g-', linewidth=1.2, alpha=0.8,
        label=f'PC3 ({desc_variance[2]*100:.1f}%)')

ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax.set_title('(b) Descending Orbit', fontsize=13, fontweight='bold')
ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Amplitude', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.2, linestyle=':')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
output_file_5 = output_dir / 'Figure_5_PC_Temporal_HighRes_CORRECTED.png'
plt.savefig(output_file_5, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output_file_5.name}")
plt.close()

print("\n" + "="*70)
print("✓ Corrected figures generated!")
print("="*70)
print(f"\n1. {output_file_4.name}")
print(f"2. {output_file_5.name}")
print("\nThese should now match your original figure structure!")
print("="*70 + "\n")

