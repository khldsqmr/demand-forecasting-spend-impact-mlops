"""
Final Baseline Demand Model Training
------------------------------------

Purpose:
- Train a single, production-ready baseline forecasting model
- Uses full historical data
- Produces a reusable model artifact
- Safe for downstream prediction & financial impact analysis

Design principles:
- No data leakage
- Explicit feature handling
- Deterministic & reproducible
"""

# ============================================================
# Imports
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# Configuration
# ============================================================

FEATURES_PATH = Path("data/processed/model_features_final.csv")
MODEL_OUTPUT_PATH = Path("models/baseline_model.pkl")

DATE_COL = "DATE"
TARGET_COL = "TOTAL_PRODUCT_DEMAND"
CATEGORICAL_COLS = ["COUNTRY"]

RANDOM_STATE = 42

# ============================================================
# Utility Logging
# ============================================================

def log(msg: str):
    print(f"🔹 {msg}")

# ============================================================
# Main Training Logic
# ============================================================

def main():
    log("Starting FINAL baseline model training")

    # --------------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------------

    log("Loading feature dataset")
    df = pd.read_csv(FEATURES_PATH, parse_dates=[DATE_COL])

    df = df.sort_values(DATE_COL).reset_index(drop=True)

    log(f"Dataset loaded with shape: {df.shape}")
    log(f"Date range: {df[DATE_COL].min()} → {df[DATE_COL].max()}")

    # --------------------------------------------------------
    # Split features & target
    # --------------------------------------------------------

    X = df.drop(columns=[DATE_COL, TARGET_COL])
    y = df[TARGET_COL]

    log(f"Total features: {X.shape[1]}")
    log(f"Target column: {TARGET_COL}")

    # --------------------------------------------------------
    # Identify categorical vs numeric features
    # --------------------------------------------------------

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    log(f"Categorical columns: {cat_cols}")
    log(f"Numeric columns: {len(num_cols)}")

    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    log("Encoding categorical features")

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        # Backward compatibility for older sklearn
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    X_cat = encoder.fit_transform(X[cat_cols])
    X_num = X[num_cols].values

    X_final = np.hstack([X_num, X_cat])

    log(f"Final feature matrix shape: {X_final.shape}")

    # --------------------------------------------------------
    # Train final baseline model
    # --------------------------------------------------------

    log("Training RandomForest baseline model")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_final, y)

    log("Model training completed")

    # --------------------------------------------------------
    # Persist model artifact
    # --------------------------------------------------------

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
        },
        MODEL_OUTPUT_PATH
    )

    log(f"Model artifact saved to: {MODEL_OUTPUT_PATH}")
    log("FINAL baseline model is ready for prediction & impact analysis")

# ============================================================
# CLI Entrypoint
# ============================================================

if __name__ == "__main__":
    main()
