"""
main.py
-------
Entry point for the Interpretable Credit Risk Predictor pipeline.

Run:
    python main.py                          # full pipeline (with GridSearch)
    python main.py --fast                   # skip GridSearch, use defaults
    python main.py --client-idx 42          # explain a specific client
    python main.py --fast --client-idx 42   # both

Pipeline stages
---------------
  1. Preprocess  – load, clean, encode, scale the UCI dataset
  2. SMOTE       – balance the training class distribution
  3. Train       – fit XGBoost (+ optional GridSearchCV)
  4. Evaluate    – metrics, ROC curve, confusion matrix
  5. Interpret   – SHAP summary, force, waterfall, dependence plots
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Ensure project root is on the Python path ────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocessing  import load_and_preprocess
from src.train_model    import apply_smote, train_xgboost
from src.evaluate_model import evaluate
from src.interpret_model import explain_model


# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH  = os.path.join(MODEL_DIR, "xgboost_model.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpretable Credit Risk Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip GridSearchCV; use sensible XGBoost defaults (much faster).",
    )
    parser.add_argument(
        "--client-idx",
        type=int,
        default=None,
        dest="client_idx",
        help="Row index in the test set to use for the SHAP force/waterfall plots.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        dest="test_size",
        help="Fraction of data to hold out for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        dest="random_state",
        help="Global random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    banner = r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║       Interpretable Credit Risk Predictor  v1.0             ║
  ║       XGBoost  +  SMOTE  +  SHAP                            ║
  ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    pipeline_start = time.time()

    # ── STAGE 1: Preprocessing ───────────────────────────────────────────────
    print("\n─── STAGE 1 / 5 : DATA PREPROCESSING ─────────────────────────────")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        test_size    = args.test_size,
        random_state = args.random_state,
        scaler_path  = SCALER_PATH,
    )

    # ── STAGE 2: SMOTE ───────────────────────────────────────────────────────
    print("\n─── STAGE 2 / 5 : SMOTE CLASS-IMBALANCE CORRECTION ───────────────")
    X_train_bal, y_train_bal = apply_smote(
        X_train,
        y_train,
        random_state = args.random_state,
    )

    # ── STAGE 3: Model Training ──────────────────────────────────────────────
    print("\n─── STAGE 3 / 5 : MODEL TRAINING ─────────────────────────────────")
    use_grid = not args.fast
    model = train_xgboost(
        X_train_bal,
        y_train_bal,
        use_grid_search = use_grid,
        save_path       = MODEL_PATH,
        random_state    = args.random_state,
    )

    # ── STAGE 4: Evaluation ──────────────────────────────────────────────────
    print("\n─── STAGE 4 / 5 : MODEL EVALUATION ───────────────────────────────")
    metrics = evaluate(
        model,
        X_test,
        y_test,
        output_dir = OUTPUT_DIR,
    )

    # ── STAGE 5: SHAP Interpretability ──────────────────────────────────────
    print("\n─── STAGE 5 / 5 : SHAP INTERPRETABILITY ──────────────────────────")
    shap_results = explain_model(
        model,
        X_test,
        output_dir   = OUTPUT_DIR,
        client_index = args.client_idx,
        random_state = args.random_state,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    _print_summary(metrics, shap_results, elapsed)


def _print_summary(metrics: dict, shap_results, elapsed: float) -> None:
    print("\n" + "═" * 62)
    print("  PIPELINE COMPLETE")
    print("═" * 62)
    print(f"  Total runtime : {elapsed:.1f} s")
    print(f"\n  Key metrics on held-out test set:")
    print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}  (target ≥ 0.80)")
    print(f"    F1-Score  : {metrics['f1']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}  (defaulter catch-rate)")
    print(f"    Precision : {metrics['precision']:.4f}")
    if shap_results:
        print(f"\n  SHAP analysis completed for client #{shap_results['client_idx']}")
    print(f"\n  Outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("    • confusion_matrix.png")
    print("    • roc_curve.png")
    print("    • feature_importance.png")
    print("    • score_distribution.png")
    print("    • shap_summary_bar.png")
    print("    • shap_summary_beeswarm.png")
    print("    • shap_force_plot.png")
    print("    • shap_waterfall.png")
    print("    • shap_dependence_<feature>.png")
    print(f"\n  Trained model  : {os.path.abspath('models/xgboost_model.pkl')}")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
