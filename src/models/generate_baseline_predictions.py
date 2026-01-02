"""
Baseline Demand Prediction Generation
-------------------------------------

Purpose:
- Load the FINAL trained baseline model artifact
- Apply identical feature preprocessing used during training
- Generate baseline demand predictions on full history
- Persist predictions for downstream financial impact analysis

Design principles:
- No re-fitting encoders
- Strict schema reuse from model artifact
- Deterministic & reproducible inference
"""

# ============================================================
# Imports
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ============================================================
# Configuration
# ============================================================

FEATURES_PATH = Path("data/processed/model_features_final.csv")
MODEL_PATH = Path("models/baseline_model.pkl")
OUTPUT_PATH = Path("data/processed/baseline_predictions.csv")

DATE_COL = "DATE"
TARGET_COL = "TOTAL_PRODUCT_DEMAND"

# ============================================================
# Utility Logging
# ============================================================

def log(msg: str):
    print(f"🔹 {msg}")

# ============================================================
# Main Prediction Logic
# ============================================================

def main():
    log("Starting baseline prediction generation")

    # --------------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------------

    log("Loading feature dataset")
    df = pd.read_csv(FEATURES_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    log(f"Dataset loaded with shape: {df.shape}")
    log(f"Date range: {df[DATE_COL].min()} → {df[DATE_COL].max()}")

    # --------------------------------------------------------
    # Load trained model artifact
    # --------------------------------------------------------

    log("Loading trained baseline model artifact")

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    encoder = artifact["encoder"]
    num_cols = artifact["numeric_features"]
    cat_cols = artifact["categorical_features"]

    log("Model and encoder loaded successfully")
    log(f"Numeric features: {len(num_cols)}")
    log(f"Categorical features: {cat_cols}")

    # --------------------------------------------------------
    # Prepare feature matrix
    # --------------------------------------------------------

    X = df.drop(columns=[DATE_COL, TARGET_COL], errors="ignore")

    # Safety check
    missing_num = set(num_cols) - set(X.columns)
    missing_cat = set(cat_cols) - set(X.columns)

    if missing_num or missing_cat:
        raise ValueError(
            f"Feature mismatch detected.\n"
            f"Missing numeric: {missing_num}\n"
            f"Missing categorical: {missing_cat}"
        )

    log("Encoding categorical features using trained encoder")

    X_num = X[num_cols].values
    X_cat = encoder.transform(X[cat_cols])

    X_final = np.hstack([X_num, X_cat])

    log(f"Final feature matrix shape: {X_final.shape}")

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    log("Generating baseline demand predictions")

    y_pred = model.predict(X_final)

    # --------------------------------------------------------
    # Assemble output dataframe
    # --------------------------------------------------------

    output_df = pd.DataFrame({
        "DATE": df[DATE_COL],
        "COUNTRY": df["COUNTRY"],
        "ACTUAL_DEMAND": df[TARGET_COL] if TARGET_COL in df.columns else np.nan,
        "BASELINE_PREDICTION": y_pred
    })

    # --------------------------------------------------------
    # Persist predictions
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    log(f"Predictions saved to: {OUTPUT_PATH}")
    log("Baseline prediction generation completed successfully")

# ============================================================
# CLI Entrypoint
# ============================================================

if __name__ == "__main__":
    main()
