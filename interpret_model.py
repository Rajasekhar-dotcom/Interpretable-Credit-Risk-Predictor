"""
interpret_model.py
------------------
Generates SHAP (SHapley Additive exPlanations) visualisations for the trained
XGBoost credit risk model.

WHY INTERPRETABILITY MATTERS IN FINANCE
----------------------------------------
Credit decisions affect people's lives — a rejected loan can prevent someone
from buying a home or starting a business.  Regulators (GDPR Article 22, the
EU AI Act, the US Equal Credit Opportunity Act) now require lenders to provide
an "adverse action notice" explaining WHY a client was denied credit.

"Black box" models, however accurate, cannot satisfy this requirement.
SHAP solves this by answering:  "Which features — and by how much — pushed
THIS individual's risk score up or down?"

HOW SHAP WORKS
--------------
SHAP is grounded in cooperative game theory (Shapley values).  Imagine all
features as "players" in a game where the "payout" is the model's prediction.
The Shapley value of a feature is its *average marginal contribution* across
all possible orderings in which features could be introduced to the model.

For a single prediction:
    model_output = base_value + φ₁ + φ₂ + … + φₙ
    where φᵢ is the SHAP value for feature i.

Properties:
  • Efficiency   – SHAP values sum exactly to the model output minus the mean.
  • Consistency  – if a feature's impact increases, its SHAP value never decreases.
  • Null player  – a feature that doesn't change predictions always gets SHAP = 0.

TreeExplainer (used here) computes exact Shapley values for tree-based models
in O(TLD²) time, making it practical even for large XGBoost ensembles.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("[SHAP] shap library not installed.  Run: pip install shap")

from xgboost import XGBClassifier


# ── Default paths ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ── Public API ───────────────────────────────────────────────────────────────

def explain_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
    client_index: int | None = None,
    max_display: int = 20,
    background_samples: int = 100,
    random_state: int = 42,
) -> dict | None:
    """
    Generate all SHAP explanations and save them as PNG files.

    Outputs
    -------
    1. shap_summary_bar.png     – global mean |SHAP| per feature (bar)
    2. shap_summary_beeswarm.png – beeswarm showing direction & magnitude
    3. shap_force_plot.png      – single-client waterfall explanation
    4. shap_dependence_<feature>.png – top feature dependence plot

    Parameters
    ----------
    model             : trained XGBClassifier
    X_test            : test features (scaled DataFrame)
    output_dir        : where to save PNGs
    client_index      : row index of the client to explain locally.
                        If None, a random client is selected.
    max_display       : how many features to show in summary plots
    background_samples: number of background samples for the explainer
    random_state      : seed for random client selection

    Returns
    -------
    dict with keys: shap_values, expected_value, client_idx
    """
    if not SHAP_AVAILABLE:
        print("[SHAP] Skipping — shap not installed.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # ── Build TreeExplainer ───────────────────────────────────────────────
    print("\n[SHAP] Building TreeExplainer …")
    explainer = shap.TreeExplainer(
        model,
        data              = shap.sample(X_test, background_samples,
                                        random_state=random_state),
        feature_perturbation = "interventional",
    )

    # ── Compute SHAP values for the test set ─────────────────────────────
    #   shap_values shape: (n_samples, n_features)
    #   Each value φᵢ for sample j says: "feature i pushed prediction j
    #   by φᵢ log-odds relative to the base rate."
    print("[SHAP] Computing SHAP values for test set (may take ~30 s) …")
    shap_values = explainer.shap_values(X_test)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value)   # binary classifier → scalar

    print(f"[SHAP] Base (expected) value: {expected_value:.4f}")
    print(f"[SHAP] SHAP values shape    : {np.array(shap_values).shape}")

    # ── Select client for local explanation ──────────────────────────────
    rng = np.random.default_rng(random_state)
    if client_index is None:
        client_index = int(rng.integers(0, len(X_test)))
    print(f"[SHAP] Explaining client at index {client_index}")

    # ── Generate plots ────────────────────────────────────────────────────
    _plot_summary_bar(shap_values, X_test, output_dir, max_display)
    _plot_summary_beeswarm(shap_values, X_test, output_dir, max_display)
    _plot_force_plot(shap_values, expected_value, X_test, client_index, output_dir)
    _plot_waterfall(explainer, X_test, client_index, output_dir)
    _plot_dependence(shap_values, X_test, output_dir)

    return {
        "shap_values"    : shap_values,
        "expected_value" : expected_value,
        "client_idx"     : client_index,
    }


# ── Private plot helpers ──────────────────────────────────────────────────────

def _plot_summary_bar(shap_values, X_test, output_dir, max_display):
    """
    GLOBAL INTERPRETABILITY — Bar chart of mean |SHAP| per feature.

    This answers: "Which features does the model rely on most, across all
    clients in the test set?"

    Taller bars = features that swing predictions more on average.
    E.g., if PAY_0 (payment status in September) has the tallest bar, late
    payments in that month are the strongest overall predictor of default.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_test,
        plot_type   = "bar",
        max_display = max_display,
        show        = False,
        plot_size   = None,
    )
    plt.title("Global Feature Importance (Mean |SHAP|)",
              fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_summary_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Summary bar plot saved → {path}")


def _plot_summary_beeswarm(shap_values, X_test, output_dir, max_display):
    """
    GLOBAL INTERPRETABILITY — Beeswarm plot.

    Each dot is one test-set client.
      • X-axis  : SHAP value (positive → pushes toward default)
      • Colour  : actual feature value (red = high, blue = low)
      • Y-axis  : features sorted by mean |SHAP| (most important at top)

    Reading example:
      "PAY_0 — high values (red dots) cluster on the RIGHT, meaning clients
       with recent late payments are strongly pushed toward predicted default."
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        max_display = max_display,
        show        = False,
        plot_size   = None,
    )
    plt.title("SHAP Beeswarm — Feature Impact Direction & Magnitude",
              fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Beeswarm plot saved → {path}")


def _plot_force_plot(shap_values, expected_value, X_test,
                     client_idx, output_dir):
    """
    LOCAL INTERPRETABILITY — Force plot for one client.

    The base value (expected_value) is the model's average output.
    Red arrows push the prediction HIGHER (toward default).
    Blue arrows push the prediction LOWER (away from default).
    The final prediction is where the arrows balance.

    Example reading:
      "This client's prediction is 0.72 (high risk).  The main drivers
       pushing toward default are: PAY_0=2 (+0.18), BILL_AMT1=180,000 (+0.12).
       Partially offset by: LIMIT_BAL=500,000 (−0.09)."
    """
    # Matplotlib-based force plot (saved as PNG — no JS required)
    shap.initjs()
    force = shap.force_plot(
        base_value  = expected_value,
        shap_values = shap_values[client_idx],
        features    = X_test.iloc[client_idx],
        matplotlib  = True,
        show        = False,
    )
    plt.title(
        f"SHAP Force Plot — Client #{client_idx}  "
        f"(predicted default prob ≈ "
        f"{_sigmoid(expected_value + shap_values[client_idx].sum()):.2%})",
        fontsize=11, fontweight="bold",
    )
    path = os.path.join(output_dir, "shap_force_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Force plot saved → {path}")


def _plot_waterfall(explainer, X_test, client_idx, output_dir):
    """
    LOCAL INTERPRETABILITY — Waterfall chart (cleaner alternative to force plot).

    Shows the step-by-step additive decomposition of the prediction.
    Starting from the base value, each bar extends or contracts the score.
    The final bar lands exactly on the model's output for this client.
    """
    try:
        expl_obj = explainer(X_test.iloc[[client_idx]])
        fig, ax  = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(expl_obj[0], max_display=15, show=False)
        plt.title(f"SHAP Waterfall — Client #{client_idx}",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(output_dir, "shap_waterfall.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Waterfall plot saved → {path}")
    except Exception as e:
        print(f"[SHAP] Waterfall plot skipped: {e}")


def _plot_dependence(shap_values, X_test, output_dir):
    """
    GLOBAL INTERPRETABILITY — Dependence plot for the top feature.

    Shows how the SHAP value of the most important feature changes as its
    raw value increases.  The colour encodes the most interacting feature
    (chosen automatically by SHAP).

    Useful for revealing non-linear relationships, e.g.:
      "PAY_0 has near-zero SHAP for values < 1 (on-time payments),
       but SHAP rises sharply for values ≥ 2 (2-month delinquency)."
    """
    # Identify the most important feature by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_feat  = X_test.columns[np.argmax(mean_abs)]

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        top_feat, shap_values, X_test,
        ax   = ax,
        show = False,
    )
    ax.set_title(
        f"SHAP Dependence Plot — '{top_feat}'",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(output_dir, f"shap_dependence_{top_feat}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] Dependence plot saved → {path}")


# ── Utility ──────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Convert log-odds to probability."""
    return 1.0 / (1.0 + np.exp(-x))
