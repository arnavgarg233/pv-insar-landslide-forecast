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

# regression_model.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

def train_risk_classifier(X, y, use_xgboost=False):
    """
    Train a binary classifier with a 3-way split: train/val/test.
    We return (model, X_test, y_test) so you can do a final hold-out evaluation.

    X: (n_samples, n_features)
    y: (n_samples,) with 0=stable, 1=landslide
    """
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # 1) final test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    # 2) from X_temp (80%) create val set (25% of 80% => 20% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    # => 60% train, 20% val, 20% test

    if use_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("Please install xgboost if use_xgboost=True.")
        model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                              scale_pos_weight=class_weight_dict[1] / class_weight_dict[0])
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                                       class_weight='balanced')

    # train on the train set
    model.fit(X_train, y_train)

    # Evaluate on val set
    y_val_pred = model.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    f1_val = f1_score(y_val, y_val_pred, average='binary')
    auc_val = roc_auc_score(y_val, y_val_pred) if len(np.unique(y_val)) > 1 else None
    cm_val = confusion_matrix(y_val, y_val_pred)

    print("=== Validation Set Metrics ===")
    print(f"Accuracy: {acc_val:.3f}, F1: {f1_val:.3f}, AUC: {auc_val}")
    print("Confusion Matrix (Val):\n", cm_val)
    print("=============================\n")

    # Return the model + final test set
    return model
