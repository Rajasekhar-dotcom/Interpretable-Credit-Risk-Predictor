"""
evaluate_model.py
-----------------
Computes and visualises all standard classification metrics for the trained
XGBoost credit-risk model.

METRICS EXPLAINED (in the context of credit risk)
--------------------------------------------------
Accuracy   – Overall percentage correct. MISLEADING on imbalanced data.
             A model that labels everyone "no default" gets ~78 % accuracy
             yet catches zero actual defaults.

Precision  – Of all clients we flagged as "will default", what fraction
             actually defaulted?  Low precision → many false alarms, which
             wastes credit analyst time.

Recall     – Of all clients who actually defaulted, what fraction did we
             catch?  Low recall → missed defaults, which causes real losses.
             In credit risk, recall (sensitivity) is usually prioritised.

F1-Score   – Harmonic mean of Precision & Recall. Useful single number
             when both matter and classes are imbalanced.

ROC-AUC    – Area Under the Receiver Operating Characteristic Curve.
             Measures the model's ability to rank a defaulter above a
             non-defaulter regardless of the decision threshold chosen.
             AUC = 1.0 is perfect; AUC = 0.5 is random guessing.
             Industry benchmark for credit scorecards: AUC ≥ 0.75 is good,
             ≥ 0.80 is very good, ≥ 0.85 is excellent.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


# ── Default paths ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ── Public API ───────────────────────────────────────────────────────────────

def evaluate(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = OUTPUT_DIR,
    threshold: float = 0.50,
) -> dict:
    """
    Full evaluation suite: metrics + four diagnostic plots.

    Parameters
    ----------
    model      : trained XGBClassifier
    X_test     : held-out features (scaled)
    y_test     : true labels (0/1)
    output_dir : directory for saved figures
    threshold  : classification threshold (default 0.5)

    Returns
    -------
    metrics : dict  – all scalar metrics for downstream logging
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Predictions ──────────────────────────────────────────────────────
    y_prob  = model.predict_proba(X_test)[:, 1]   # probability of default
    y_pred  = (y_prob >= threshold).astype(int)

    # ── Scalar metrics ────────────────────────────────────────────────────
    metrics = {
        "accuracy"  : accuracy_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred, zero_division=0),
        "recall"    : recall_score(y_test, y_pred, zero_division=0),
        "f1"        : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"   : roc_auc_score(y_test, y_prob),
    }

    # ── Print report ──────────────────────────────────────────────────────
    _print_report(metrics, y_test, y_pred)

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred, output_dir)
    _plot_roc_curve(y_test, y_prob, metrics["roc_auc"], output_dir)
    _plot_feature_importance(model, X_test.columns.tolist(), output_dir)
    _plot_score_distribution(y_test, y_prob, output_dir)

    return metrics


# ── Private helpers ──────────────────────────────────────────────────────────

def _print_report(metrics: dict, y_test, y_pred) -> None:
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1-Score   : {metrics['f1']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print("\n  Full Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Default", "Default"]))
    print("=" * 60 + "\n")


def _plot_confusion_matrix(y_test, y_pred, output_dir: str) -> None:
    """
    Confusion matrix heatmap.

    Cell interpretation (rows = actual, cols = predicted):
      TN (top-left)  – correctly identified non-defaulters
      FP (top-right) – non-defaulters wrongly flagged as risk
      FN (bot-left)  – missed defaulters  ← most costly error in credit
      TP (bot-right) – correctly caught defaulters
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: No Default", "Pred: Default"],
        yticklabels=["True: No Default", "True: Default"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    fig.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[EVAL] Confusion matrix saved → {path}")


def _plot_roc_curve(y_test, y_prob, auc: float, output_dir: str) -> None:
    """
    ROC Curve — plots True Positive Rate vs False Positive Rate.

    Each point on the curve corresponds to a different decision threshold.
    The ideal operating point is the top-left corner (TPR=1, FPR=0).
    The diagonal dashed line represents a random classifier (AUC=0.5).
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5,
            label=f"XGBoost (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC Curve — Credit Default Prediction", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[EVAL] ROC curve saved → {path}")


def _plot_feature_importance(model, feature_names: list, output_dir: str) -> None:
    """
    Bar chart of XGBoost's built-in feature importance scores (gain).

    'Gain' measures the average improvement in the loss function brought
    by a feature across all tree splits — higher gain = more predictive power.
    Note: SHAP-based importance (see interpret_model.py) is more reliable
    because it accounts for feature interactions.
    """
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]   # top-20 features
    top_features = [feature_names[i] for i in idx]
    top_scores   = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]
    bars = ax.barh(top_features[::-1], top_scores[::-1], color=colors[::-1])

    ax.set_xlabel("Importance (Gain)", fontsize=11)
    ax.set_title("Top-20 Feature Importances (XGBoost Gain)",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[EVAL] Feature importance plot saved → {path}")


def _plot_score_distribution(y_test, y_prob, output_dir: str) -> None:
    """
    Distribution of predicted default probabilities, split by actual class.

    A well-calibrated model will show:
      • Non-defaulters (blue) clustered near 0
      • Defaulters      (red)  clustered near 1
    Overlap in the middle reveals where the model is uncertain.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 40)

    ax.hist(y_prob[y_test == 0], bins=bins, alpha=0.55,
            color="#3B82F6", label="No Default (0)", density=True)
    ax.hist(y_prob[y_test == 1], bins=bins, alpha=0.55,
            color="#EF4444", label="Default (1)", density=True)

    ax.axvline(0.5, color="black", linestyle="--", lw=1.5,
               label="Decision threshold (0.50)")
    ax.set_xlabel("Predicted Probability of Default", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Score Distribution by True Class",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "score_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[EVAL] Score distribution plot saved → {path}")
