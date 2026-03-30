"""
Random Forest Feature Importance Analysis
==========================================
Extract and visualize feature importance rankings from the trained RF model
to reveal dominant predictors for landslide forecasting.
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
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Paths
base_path = RF_DATA_DIR
model_dir = base_path / "Final model training scripts"

print("="*70)
print("Random Forest Feature Importance Analysis")
print("="*70)

# Load the trained Random Forest model
model_file = model_dir / "random_forest_model.pkl"

if model_file.exists():
    print(f"\n✓ Loading trained model from: {model_file.name}")
    with open(model_file, 'rb') as f:
        rf_model = pickle.load(f)
    
    # Get feature importances (Mean Decrease in Impurity / Gini Importance)
    importances = rf_model.feature_importances_
    
    # Feature names
    feature_names = [
        'PC1 (Ascending)',
        'PC2 (Ascending)', 
        'PC3 (Ascending)',
        'PC4 (Ascending)',
        'PC5 (Ascending)',
        'PC1 (Descending)',
        'PC2 (Descending)',
        'PC3 (Descending)',
        'PC4 (Descending)',
        'PC5 (Descending)',
        'Daily Rainfall'
    ]
    
    print(f"\n✓ Model has {len(importances)} features")
    
    # Create dataframe for easier manipulation
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Importance (%)': importances * 100
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    
    # Reorder columns
    importance_df = importance_df[['Rank', 'Feature', 'Importance', 'Importance (%)']]
    
    print(f"\n{'='*70}")
    print("Feature Importance Rankings")
    print(f"{'='*70}\n")
    print(importance_df.to_string(index=False, float_format='%.4f'))
    
    # Calculate statistics
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print(f"Top 3 features account for: {importance_df.head(3)['Importance (%)'].sum():.2f}%")
    print(f"PCA features total: {importance_df[importance_df['Feature'] != 'Daily Rainfall']['Importance (%)'].sum():.2f}%")
    print(f"Rainfall importance: {importance_df[importance_df['Feature'] == 'Daily Rainfall']['Importance (%)'].values[0]:.2f}%")
    
    # Aggregate by orbit
    asc_importance = importance_df[importance_df['Feature'].str.contains('Ascending')]['Importance'].sum()
    desc_importance = importance_df[importance_df['Feature'].str.contains('Descending')]['Importance'].sum()
    rainfall_importance = importance_df[importance_df['Feature'] == 'Daily Rainfall']['Importance'].values[0]
    
    print(f"\nBy Category:")
    print(f"  Ascending PCs:  {asc_importance*100:.2f}%")
    print(f"  Descending PCs: {desc_importance*100:.2f}%")
    print(f"  Rainfall:       {rainfall_importance*100:.2f}%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart of all features
    ax = axes[0]
    colors = ['#1f77b4' if 'Ascending' in f else '#ff7f0e' if 'Descending' in f else '#2ca02c' 
              for f in importance_df['Feature']]
    
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['Importance (%)'], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['Feature'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest Feature Importance Rankings', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(importance_df['Importance (%)']):
        ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)
    
    # Plot 2: Grouped by category
    ax = axes[1]
    categories = ['Ascending\nPCs', 'Descending\nPCs', 'Rainfall']
    category_importances = [asc_importance*100, desc_importance*100, rainfall_importance*100]
    category_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(categories, category_importances, color=category_colors, alpha=0.8, width=0.6)
    ax.set_ylabel('Cumulative Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance by Category', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, category_importances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, max(category_importances) * 1.15)
    
    plt.tight_layout()
    plt.savefig(base_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: feature_importance.png")
    plt.close()
    
    # Generate LaTeX table
    latex_table = r"""\begin{table}[htbp]
\centering
\caption{Random Forest feature importance rankings revealing the dominant predictors for 7-day landslide forecasting. Importance values represent the mean decrease in Gini impurity (MDI) normalized to sum to 100\%.}
\label{tab:feature_importance}
\begin{tabular}{clcc}
\toprule
\textbf{Rank} & \textbf{Feature} & \textbf{Importance} & \textbf{Cumulative (\%)} \\
\midrule
"""
    
    cumulative = 0
    for idx, row in importance_df.iterrows():
        cumulative += row['Importance (%)']
        latex_table += f"{int(row['Rank'])} & {row['Feature']} & {row['Importance (%)']:.2f}\\% & {cumulative:.2f}\\% \\\\\n"
    
    latex_table += r"""\midrule
\multicolumn{4}{l}{\textit{Top 3 features account for """ + f"{importance_df.head(3)['Importance (%)'].sum():.1f}" + r"""\% of predictive power}} \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save LaTeX table
    latex_file = base_path / "feature_importance_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\n✓ Generated LaTeX table: {latex_file.name}")
    
    # Generate simplified LaTeX table (Top 5 + Rainfall)
    latex_simple = r"""\begin{table}[htbp]
\centering
\caption{Top-ranked Random Forest feature importance for landslide forecasting. Importance values represent the normalized mean decrease in Gini impurity.}
\label{tab:feature_importance_simple}
\begin{tabular}{clc}
\toprule
\textbf{Rank} & \textbf{Feature} & \textbf{Importance (\%)} \\
\midrule
"""
    
    for idx, row in importance_df.head(6).iterrows():
        latex_simple += f"{int(row['Rank'])} & {row['Feature']} & {row['Importance (%)']:.2f} \\\\\n"
    
    latex_simple += r"""\midrule
\multicolumn{3}{l}{\textbf{Aggregated by Category:}} \\
\midrule
& Ascending PCs (total) & """ + f"{asc_importance*100:.2f}" + r""" \\
& Descending PCs (total) & """ + f"{desc_importance*100:.2f}" + r""" \\
& Daily Rainfall & """ + f"{rainfall_importance*100:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    latex_simple_file = base_path / "feature_importance_table_simple.tex"
    with open(latex_simple_file, 'w') as f:
        f.write(latex_simple)
    
    print(f"✓ Generated simplified LaTeX table: {latex_simple_file.name}")
    
    # Print LaTeX for immediate use
    print(f"\n{'='*70}")
    print("LaTeX Table (Simplified Version)")
    print(f"{'='*70}\n")
    print(latex_simple)
    
    print(f"\n{'='*70}")
    print("✓ Feature importance analysis complete!")
    print(f"{'='*70}\n")
    
else:
    print(f"\n✗ Error: Model file not found at {model_file}")
    print(f"Looking for alternative model files...")
    
    # Search for any .pkl files
    pkl_files = list(base_path.rglob("*.pkl"))
    if pkl_files:
        print(f"\nFound {len(pkl_files)} .pkl files:")
        for f in pkl_files:
            print(f"  - {f}")
    else:
        print("No .pkl files found in the project directory")

