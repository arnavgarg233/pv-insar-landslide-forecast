"""
Extract Feature Importance from Trained Random Forest Model
============================================================
This script trains the model and extracts feature importance rankings.
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
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import os

def load_insar_data(file_path):
    """Load InSAR time series from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        timeseries = f['timeseries'][:]  # Shape: (time_steps, height, width)
        dates = f['date'][:]
        return timeseries, dates

def load_pca_features(pca_file):
    """Load PCA features from the provided HDF5 file."""
    with h5py.File(pca_file, 'r') as f:
        asc_scores = f['ascending/scores'][:]
        desc_scores = f['descending/scores'][:]
    combined_pca_features = np.hstack([asc_scores[:desc_scores.shape[0]], desc_scores])
    return combined_pca_features

def calculate_cumulative_displacement_rolling(asc, desc, window_size):
    """Calculate cumulative displacement over rolling windows."""
    vertical = (asc + desc) / 2
    horizontal = (asc - desc) / 2
    displacement_magnitude = np.sqrt(vertical**2 + horizontal**2)
    time_steps, height, width = displacement_magnitude.shape
    cumulative_rolling = np.zeros_like(displacement_magnitude)

    for t in range(time_steps - window_size + 1):
        cumulative_rolling[t] = displacement_magnitude[t:t + window_size].sum(axis=0)

    return cumulative_rolling

def generate_labels(rolling_displacement, threshold):
    """Generate labels from rolling displacement by applying a threshold."""
    labels = (rolling_displacement > threshold).astype(int)
    labels = labels.reshape(labels.shape[0], -1).sum(axis=1) > 0
    return labels.astype(int)

def load_rainfall_data(rainfall_dir):
    """Load rainfall data from .nc4 files."""
    rainfall_files = [os.path.join(rainfall_dir, f) for f in os.listdir(rainfall_dir) if f.endswith('.nc4')]
    rainfall_data = []

    for file in rainfall_files:
        try:
            ds = xr.open_dataset(file)
            if 'MWprecipitation' in ds:
                daily_precip = ds['MWprecipitation'].mean(dim=['lon', 'lat']).to_dataframe()
                rainfall_data.append(daily_precip)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    rainfall_df = pd.concat(rainfall_data).reset_index()
    rainfall_df["time"] = pd.to_datetime(rainfall_df["time"])
    return rainfall_df

def align_data(rainfall_df, displacement_labels):
    """Align rainfall data with displacement labels."""
    aligned_rainfall = rainfall_df.groupby(rainfall_df["time"].dt.date)["MWprecipitation"].mean().values
    aligned_rainfall = aligned_rainfall[:len(displacement_labels)]
    return aligned_rainfall

def train_and_evaluate_model(X, Y):
    """Train and evaluate a Random Forest model."""
    if len(np.unique(Y)) < 2:
        print("Only one class present in the labels. Skipping configuration.")
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None

    return acc, f1, auc, confusion_matrix(y_test, y_pred), model

# ===========================================================================
# Main Execution
# ===========================================================================

print("="*70)
print("Feature Importance Extraction")
print("="*70)

# File paths (Mac version)
base_path = REPO_ROOT
asc_file = base_path / "RandomForest MovmentLandslide threshodl/timeseries.h5"
desc_file = base_path / "RandomForest MovmentLandslide threshodl/pvtimeseriesd.h5"
pca_file = base_path / "RandomForest MovmentLandslide threshodl/pca_results.h5"
rainfall_dir = base_path / "Average Rainfall data/Rain"

print("\nLoading data...")
asc_data, _ = load_insar_data(asc_file)
desc_data, _ = load_insar_data(desc_file)
if asc_data.shape[0] > desc_data.shape[0]:
    asc_data = asc_data[:desc_data.shape[0]]

pca_features = load_pca_features(pca_file)
rainfall_df = load_rainfall_data(rainfall_dir)

print(f"✓ PCA features shape: {pca_features.shape}")
print(f"✓ Number of features: {pca_features.shape[1]}")

# Parameters
window_size = 7
threshold = 0.35

print("\nGenerating labels...")
rolling_displacement = calculate_cumulative_displacement_rolling(asc_data, desc_data, window_size)
labels = generate_labels(rolling_displacement, threshold)
labels = labels[:pca_features.shape[0]]

print("\nAligning rainfall data...")
aligned_rainfall = align_data(rainfall_df, labels)

print("\nCombining features...")
combined_features = np.hstack([pca_features, aligned_rainfall.reshape(-1, 1)])
print(f"✓ Combined features shape: {combined_features.shape}")

print("\nTraining Random Forest model...")
acc, f1, auc, cm, model = train_and_evaluate_model(combined_features, labels)

if acc is not None and model is not None:
    print("\n" + "="*70)
    print("Model Performance")
    print("="*70)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}" if auc is not None else "AUC Score: N/A")
    
    # ===========================================================================
    # Extract Feature Importance
    # ===========================================================================
    
    print("\n" + "="*70)
    print("Feature Importance Analysis")
    print("="*70)
    
    importances = model.feature_importances_
    
    # Determine feature names based on actual number of features
    n_features = len(importances)
    
    if n_features == 11:  # 10 PCA + 1 rainfall
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
    elif n_features == 7:  # 6 PCA + 1 rainfall
        feature_names = [
            'PC1 (Ascending)',
            'PC2 (Ascending)', 
            'PC3 (Ascending)',
            'PC1 (Descending)',
            'PC2 (Descending)',
            'PC3 (Descending)',
            'Daily Rainfall'
        ]
    else:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Importance (%)': importances * 100
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    importance_df = importance_df[['Rank', 'Feature', 'Importance', 'Importance (%)']]
    
    print(f"\nFeature Importance Rankings:")
    print(importance_df.to_string(index=False, float_format='%.4f'))
    
    # Statistics
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print(f"Top 3 features: {importance_df.head(3)['Importance (%)'].sum():.2f}%")
    
    # Aggregate by category
    asc_importance = importance_df[importance_df['Feature'].str.contains('Ascending')]['Importance'].sum()
    desc_importance = importance_df[importance_df['Feature'].str.contains('Descending')]['Importance'].sum()
    rainfall_importance = importance_df[importance_df['Feature'] == 'Daily Rainfall']['Importance'].values[0]
    
    print(f"\nBy Category:")
    print(f"  Ascending PCs:  {asc_importance*100:.2f}%")
    print(f"  Descending PCs: {desc_importance*100:.2f}%")
    print(f"  Rainfall:       {rainfall_importance*100:.2f}%")
    
    # ===========================================================================
    # Visualization
    # ===========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Individual features
    ax = axes[0]
    colors = ['#1f77b4' if 'Ascending' in f else '#ff7f0e' if 'Descending' in f else '#2ca02c' 
              for f in importance_df['Feature']]
    
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['Importance (%)'], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['Feature'], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest Feature Importance Rankings', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(importance_df['Importance (%)']):
        ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=10)
    
    # Plot 2: Grouped by category
    ax = axes[1]
    categories = ['Ascending\nPCs', 'Descending\nPCs', 'Rainfall']
    category_importances = [asc_importance*100, desc_importance*100, rainfall_importance*100]
    category_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(categories, category_importances, color=category_colors, alpha=0.8, width=0.6)
    ax.set_ylabel('Cumulative Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance by Category', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, category_importances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, max(category_importances) * 1.15)
    
    plt.tight_layout()
    output_dir = RF_DATA_DIR
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: feature_importance.png")
    plt.close()
    
    # ===========================================================================
    # Generate LaTeX Tables
    # ===========================================================================
    
    # Simplified table
    latex_simple = r"""\begin{table}[htbp]
\centering
\caption{Random Forest feature importance rankings for 7-day landslide forecasting. Importance values represent the mean decrease in Gini impurity (MDI) normalized to sum to 100\%.}
\label{tab:feature_importance}
\begin{tabular}{clc}
\toprule
\textbf{Rank} & \textbf{Feature} & \textbf{Importance (\%)} \\
\midrule
"""
    
    for idx, row in importance_df.iterrows():
        latex_simple += f"{int(row['Rank'])} & {row['Feature']} & {row['Importance (%)']:.2f} \\\\\n"
    
    latex_simple += r"""\midrule
\multicolumn{3}{l}{\textbf{Aggregated by Category:}} \\
\midrule
& Ascending PCs (total) & """ + f"{asc_importance*100:.2f}" + r""" \\
& Descending PCs (total) & """ + f"{desc_importance*100:.2f}" + r""" \\
& Rainfall & """ + f"{rainfall_importance*100:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    latex_file = output_dir / "feature_importance_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_simple)
    
    print(f"✓ Generated LaTeX table: {latex_file.name}")
    
    print(f"\n{'='*70}")
    print("LaTeX Table")
    print(f"{'='*70}\n")
    print(latex_simple)
    
    print(f"\n{'='*70}")
    print("✓ Feature importance analysis complete!")
    print(f"{'='*70}\n")

else:
    print("\n✗ Model training failed")

