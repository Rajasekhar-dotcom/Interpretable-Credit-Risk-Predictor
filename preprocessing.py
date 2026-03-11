"""
preprocessing.py
----------------
Handles all data loading, cleaning, encoding, and scaling steps for the
UCI Default of Credit Card Clients dataset.

WHY PREPROCESSING MATTERS IN CREDIT RISK:
  Raw financial data is messy — it contains invalid category codes, skewed
  distributions, and mixed scales.  A model trained on dirty data will learn
  noise instead of signal, producing unreliable risk scores.  Careful
  preprocessing is therefore the foundation of every trustworthy credit model.
"""

import os
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── Column-name constants ────────────────────────────────────────────────────
TARGET_COL      = "default_payment_next_month"
ID_COL          = "ID"

CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

# Continuous money / age / payment columns that benefit from scaling
NUMERICAL_COLS  = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
    "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6",
]

# ── Dataset URL (UCI via direct Excel link) ──────────────────────────────────
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "credit_default.csv")


# ── Public API ───────────────────────────────────────────────────────────────

def download_dataset(force: bool = False) -> pd.DataFrame:
    """
    Download the UCI dataset if not already cached, then return it as a
    DataFrame.

    Parameters
    ----------
    force : bool
        Re-download even if the local cache already exists.

    Returns
    -------
    pd.DataFrame  – raw, unprocessed data.
    """
    csv_path = os.path.abspath(DATA_PATH)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path) and not force:
        print(f"[INFO] Loading cached dataset from {csv_path}")
        return pd.read_csv(csv_path)

    print("[INFO] Downloading dataset from UCI repository …")
    try:
        response = requests.get(UCI_URL, timeout=60)
        response.raise_for_status()
        xls_path = csv_path.replace(".csv", ".xls")
        with open(xls_path, "wb") as f:
            f.write(response.content)

        # The first row is a secondary header; row 0 is the real header
        df = pd.read_excel(xls_path, header=1, engine="xlrd")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Dataset saved to {csv_path}")
        return df

    except Exception as exc:
        print(f"[WARN] Download failed ({exc}).  Generating synthetic data …")
        return _generate_synthetic_data(csv_path)


def load_and_preprocess(
    test_size: float = 0.20,
    random_state: int = 42,
    scaler_path: str | None = None,
) -> tuple:
    """
    Full preprocessing pipeline:
      1. Load / download data
      2. Clean & validate
      3. Encode categoricals
      4. Train / test split  (stratified)
      5. Scale numericals    (fit on train, transform both)

    Parameters
    ----------
    test_size     : fraction of data held out for evaluation
    random_state  : seed for reproducibility
    scaler_path   : if given, save the fitted scaler here (.pkl)

    Returns
    -------
    X_train, X_test, y_train, y_test  (all pandas DataFrames / Series)
    feature_names : list[str]
    """
    df = download_dataset()

    # ── Step 1: Drop identifier column ──────────────────────────────────────
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # ── Step 2: Rename target for clarity ───────────────────────────────────
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": TARGET_COL})

    print(f"\n[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Target distribution:\n{df[TARGET_COL].value_counts()}\n")

    # ── Step 3: Fix known invalid category codes ─────────────────────────────
    #   EDUCATION: documented values are 1-4.  0, 5, 6 → map to 4 (Other)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    #   MARRIAGE: documented values are 1-3.  0 → map to 3 (Other)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # ── Step 4: Handle missing values ────────────────────────────────────────
    #   Numeric NaN → column median  (robust to outliers)
    #   Categorical NaN → column mode
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if col in CATEGORICAL_COLS:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # ── Step 5: One-Hot Encode categoricals ──────────────────────────────────
    #   drop_first=True avoids multicollinearity (dummy variable trap)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # ── Step 6: Split features / target ──────────────────────────────────────
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    # ── Step 7: Stratified train-test split ──────────────────────────────────
    #   Stratify preserves the original class ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── Step 8: Scale numerical features ─────────────────────────────────────
    #   StandardScaler: z = (x - μ) / σ
    #   FIT only on training data to prevent data leakage.
    scaler = StandardScaler()
    numerical_present = [c for c in NUMERICAL_COLS if c in X_train.columns]

    X_train[numerical_present] = scaler.fit_transform(X_train[numerical_present])
    X_test[numerical_present]  = scaler.transform(X_test[numerical_present])

    # Optionally persist the scaler so it can be reused at inference time
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Scaler saved to {scaler_path}")

    print(f"[INFO] Train size : {X_train.shape[0]} samples")
    print(f"[INFO] Test size  : {X_test.shape[0]} samples")
    print(f"[INFO] Features   : {len(feature_names)}")

    return X_train, X_test, y_train, y_test, feature_names


# ── Private helpers ──────────────────────────────────────────────────────────

def _generate_synthetic_data(save_path: str) -> pd.DataFrame:
    """
    Fallback: generate a synthetic dataset that mirrors the UCI schema.
    Used only when the download fails (e.g., offline CI environments).
    """
    print("[INFO] Building synthetic credit-risk dataset (5,000 rows) …")
    rng = np.random.default_rng(42)
    n   = 5_000

    pay_cols  = [f"PAY_{i}"      for i in [0, 2, 3, 4, 5, 6]]
    bill_cols = [f"BILL_AMT{i}"  for i in range(1, 7)]
    pamt_cols = [f"PAY_AMT{i}"   for i in range(1, 7)]

    df = pd.DataFrame({
        "LIMIT_BAL" : rng.integers(10_000,  800_000, n),
        "SEX"       : rng.integers(1, 3, n),
        "EDUCATION" : rng.integers(1, 5, n),
        "MARRIAGE"  : rng.integers(1, 4, n),
        "AGE"       : rng.integers(20, 75,  n),
        **{c: rng.integers(-2, 9, n)           for c in pay_cols},
        **{c: rng.integers(0,  200_000, n)     for c in bill_cols},
        **{c: rng.integers(0,  100_000, n)     for c in pamt_cols},
        TARGET_COL  : rng.choice([0, 1], n, p=[0.78, 0.22]),
    })

    df.to_csv(save_path, index=False)
    print(f"[INFO] Synthetic data saved to {save_path}")
    return df
