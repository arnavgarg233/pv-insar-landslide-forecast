"""
Combine Existing High-Quality Figures into Publication Format
==============================================================
Uses your original figures and combines them properly
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
    FIGURE_INPUTS_DIR,
    ASC_H5,
    DESC_H5,
    PCA_H5,
    RAINFALL_DIR,
    MINTPY_STACK_DIR,
)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths
base_path = REPO_ROOT
viz_dir = FIGURE_INPUTS_DIR
output_dir = RF_DATA_DIR

print("="*70)
print("Combining Your Original Figures")
print("="*70)

# ===========================================================================
# FIGURE 4: Spatial Patterns of PC1
# ===========================================================================

print("\nCreating Figure 4: Spatial Patterns of PC1...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
fig.subplots_adjust(top=0.90)  # Make room for title

# Load and display ascending PC1
img_asc = mpimg.imread(viz_dir / "ascending_spatial_PC1.png")
axes[0].imshow(img_asc)
axes[0].axis('off')
axes[0].set_title('(a) Ascending Track', fontsize=13, fontweight='bold', pad=15)
axes[0].text(0.5, -0.05, 'PC1 loadings (standardized amplitude)', 
             transform=axes[0].transAxes, ha='center', fontsize=11, style='italic')

# Load and display descending PC1
img_desc = mpimg.imread(viz_dir / "descending_spatial_PC1.png")
axes[1].imshow(img_desc)
axes[1].axis('off')
axes[1].set_title('(b) Descending Track', fontsize=13, fontweight='bold', pad=15)
axes[1].text(0.5, -0.05, 'PC1 loadings (standardized amplitude)', 
             transform=axes[1].transAxes, ha='center', fontsize=11, style='italic')

fig.suptitle('Spatial Patterns of the First Principal Component (PC1)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
output_file_4 = output_dir / 'Figure_4_PC1_Spatial_FINAL.jpg'
plt.savefig(output_file_4, dpi=300, bbox_inches='tight', facecolor='white', pil_kwargs={'quality': 95})
print(f"✓ Saved: {output_file_4.name}")
plt.close()

# ===========================================================================
# FIGURE 5: Temporal Evolution
# ===========================================================================

print("\nCreating Figure 5: Temporal Evolution of PCs...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
fig.subplots_adjust(top=0.90)  # Make room for title

# Load and display ascending temporal
img_asc_temp = mpimg.imread(viz_dir / "ascending_temporal.png")
axes[0].imshow(img_asc_temp)
axes[0].axis('off')
axes[0].set_title('(a) Ascending Orbit', fontsize=13, fontweight='bold', pad=15)
axes[0].text(0.5, -0.05, 'PC amplitudes (standardized units) vs. Time', 
             transform=axes[0].transAxes, ha='center', fontsize=11, style='italic')

# Load and display descending temporal
img_desc_temp = mpimg.imread(viz_dir / "descending_temporal.png")
axes[1].imshow(img_desc_temp)
axes[1].axis('off')
axes[1].set_title('(b) Descending Orbit', fontsize=13, fontweight='bold', pad=15)
axes[1].text(0.5, -0.05, 'PC amplitudes (standardized units) vs. Time', 
             transform=axes[1].transAxes, ha='center', fontsize=11, style='italic')

fig.suptitle('Temporal Evolution of the First Three Principal Components (PCs)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
output_file_5 = output_dir / 'Figure_5_PC_Temporal_FINAL.jpg'
plt.savefig(output_file_5, dpi=300, bbox_inches='tight', facecolor='white', pil_kwargs={'quality': 95})
print(f"✓ Saved: {output_file_5.name}")
plt.close()

print("\n" + "="*70)
print("✓ Final figures created using your original images!")
print("="*70)
print(f"\nGenerated:")
print(f"  1. {output_file_4.name}")
print(f"  2. {output_file_5.name}")
print("\nThese match your original figures with proper formatting!")
print("="*70 + "\n")

