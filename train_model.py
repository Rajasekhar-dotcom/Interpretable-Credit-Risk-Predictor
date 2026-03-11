"""
train_model.py
--------------
Handles class-imbalance correction with SMOTE, then trains an XGBoost
classifier using GridSearchCV for hyperparameter optimisation.

WHY SMOTE?
----------
In real-world credit portfolios, defaulters are a small minority (≈ 20-25 %).
A naive model trained on imbalanced data simply learns to predict "no default"
for every client and still achieves 75-80 % accuracy — yet it catches *zero*
actual defaults.  SMOTE (Synthetic Minority Over-sampling Technique) fixes
this by generating new, plausible synthetic minority samples in feature space,
giving the model equal exposure to both classes during training.

IMPORTANT: SMOTE is applied ONLY to the training split.  The test set must
remain untouched so evaluation metrics reflect real-world conditions.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


# ── Default paths ────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "xgboost_model.pkl")


# ── Public API ───────────────────────────────────────────────────────────────

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Oversample the minority class in the training set using SMOTE.

    How SMOTE works
    ---------------
    For each minority sample, SMOTE:
      1. Finds its k nearest minority neighbours.
      2. Draws a random point along the line segment between the sample
         and one of those neighbours.
      3. Adds that interpolated point as a new synthetic sample.

    This is preferable to simple duplication because the model sees
    *varied* minority examples, not identical copies.

    Parameters
    ----------
    X_train      : training features (post-scaling)
    y_train      : training labels
    random_state : seed for reproducibility
    k_neighbors  : number of nearest neighbours used by SMOTE

    Returns
    -------
    X_resampled, y_resampled  – balanced arrays (still DataFrames / Series)
    """
    print("\n[SMOTE] Class distribution BEFORE resampling:")
    print(y_train.value_counts().to_string())

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Restore DataFrame / Series types with original column names
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    print("\n[SMOTE] Class distribution AFTER resampling:")
    print(y_res.value_counts().to_string())
    print(f"[SMOTE] Training set grew from {len(y_train):,} → {len(y_res):,} samples\n")

    return X_res, y_res


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_grid_search: bool = True,
    save_path: str = MODEL_PATH,
    random_state: int = 42,
    n_jobs: int = -1,
) -> XGBClassifier:
    """
    Train an XGBoost classifier, optionally with GridSearchCV tuning.

    XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees
    sequentially, where each new tree corrects the residual errors of the
    previous ones.  It is well-suited for tabular credit data because:
      • It handles mixed feature types naturally.
      • It is robust to outliers via tree-based splits.
      • It natively supports feature importance scores.

    Parameters
    ----------
    X_train        : SMOTE-balanced training features
    y_train        : SMOTE-balanced training labels
    use_grid_search: if True, run GridSearchCV; else use sensible defaults
    save_path      : path to persist the trained model
    random_state   : seed
    n_jobs         : parallel jobs (-1 = all cores)

    Returns
    -------
    best_model : fitted XGBClassifier
    """
    # Base estimator — eval_metric set to avoid a deprecation warning
    base_model = XGBClassifier(
        objective      = "binary:logistic",
        eval_metric    = "auc",
        use_label_encoder = False,
        random_state   = random_state,
        n_jobs         = n_jobs,
    )

    if use_grid_search:
        best_model = _grid_search(base_model, X_train, y_train, random_state, n_jobs)
    else:
        # Sensible defaults — good starting point without tuning overhead
        print("[TRAIN] Fitting XGBoost with default hyperparameters …")
        base_model.set_params(
            learning_rate = 0.05,
            max_depth     = 4,
            n_estimators  = 200,
            subsample     = 0.8,
        )
        base_model.fit(X_train, y_train)
        best_model = base_model

    # ── Persist the trained model ─────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(best_model, save_path)
    print(f"\n[TRAIN] Model saved → {save_path}")

    return best_model


# ── Private helpers ──────────────────────────────────────────────────────────

def _grid_search(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_jobs: int,
) -> XGBClassifier:
    """
    Run an exhaustive GridSearchCV over a curated hyperparameter grid.

    StratifiedKFold is used so each fold preserves the class ratio from
    the (SMOTE-balanced) training data.  ROC-AUC is used as the scoring
    metric because it measures rank-order discrimination — the primary
    goal for credit risk models.

    Grid being searched
    -------------------
    learning_rate : step shrinkage to prevent overfitting
    max_depth     : tree depth — deeper = more complex interactions
    n_estimators  : number of boosting rounds
    subsample     : row-sampling ratio per tree (regularisation)
    """
    param_grid = {
        "learning_rate" : [0.01, 0.05, 0.10],
        "max_depth"     : [3, 4, 6],
        "n_estimators"  : [100, 200, 300],
        "subsample"     : [0.7, 0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator  = estimator,
        param_grid = param_grid,
        scoring    = "roc_auc",
        cv         = cv,
        n_jobs     = n_jobs,
        verbose    = 1,
        refit      = True,   # refit on full training data with best params
    )

    print("[TRAIN] Starting GridSearchCV — this may take several minutes …")
    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n[TRAIN] GridSearchCV finished in {elapsed:.1f}s")
    print(f"[TRAIN] Best parameters : {grid.best_params_}")
    print(f"[TRAIN] Best CV AUC     : {grid.best_score_:.4f}")

    return grid.best_estimator_


def load_model(path: str = MODEL_PATH) -> XGBClassifier:
    """Load a previously saved model from disk."""
    model = joblib.load(path)
    print(f"[TRAIN] Model loaded from {path}")
    return model
