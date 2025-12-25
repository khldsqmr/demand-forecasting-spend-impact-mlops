"""
Feature Engineering Module
--------------------------
Transforms the model-ready dataset into a final ML feature matrix.

Design principles:
- Deterministic
- No data leakage
- Explainable features
- Production-ready logging
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path("data/processed/model_training_dataset.csv")
OUTPUT_PATH = Path("data/processed/model_features_final.csv")

DATE_COL = "DATE"
TARGET_COL = "TOTAL_PRODUCT_DEMAND"


# ============================================================
# Utility Functions
# ============================================================

def log(msg: str):
    """Consistent logging format"""
    print(f"🔹 {msg}")


def validate_columns(df: pd.DataFrame, required_cols: list):
    """Ensure required columns exist"""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")
    log("All required columns validated")


# ============================================================
# Feature Engineering Logic
# ============================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-based seasonality features"""
    log("Adding time-based features")

    df["DAY_OF_WEEK"] = df[DATE_COL].dt.dayofweek
    df["WEEK_OF_YEAR"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["MONTH"] = df[DATE_COL].dt.month
    df["YEAR"] = df[DATE_COL].dt.year

    # Cyclical encoding
    df["DOW_SIN"] = np.sin(2 * np.pi * df["DAY_OF_WEEK"] / 7)
    df["DOW_COS"] = np.cos(2 * np.pi * df["DAY_OF_WEEK"] / 7)

    return df


def add_marketing_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Marketing efficiency & saturation signals"""
    log("Adding marketing efficiency features")

    eps = 1e-6  # Avoid division by zero

    df["SPEND_PER_RESPONSE"] = df["TOTAL_SPEND"] / (df["TOTAL_CHANNEL_RESPONSE"] + eps)
    df["RESPONSE_PER_SPEND"] = df["TOTAL_CHANNEL_RESPONSE"] / (df["TOTAL_SPEND"] + eps)

    df["SPEND_VS_BASELINE"] = df["TOTAL_SPEND"] / (df["BASELINE_DEMAND"] + eps)

    return df


def add_macro_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Macro × demand interactions"""
    log("Adding macroeconomic interaction features")

    df["DEMAND_X_ECONOMIC"] = df["BASELINE_DEMAND"] * df["ECONOMIC_INDEX"]
    df["DEMAND_X_INFLATION"] = df["BASELINE_DEMAND"] * df["INFLATION_RATE"]
    df["DEMAND_X_UNEMPLOYMENT"] = df["BASELINE_DEMAND"] * df["UNEMPLOYMENT_RATE"]

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trend & momentum signals"""
    log("Adding trend features")

    df["DEMAND_TREND_7_14"] = df["DEMAND_ROLLING_7"] - df["DEMAND_ROLLING_14"]
    df["SPEND_TREND_7_14"] = df["SPEND_LAG_7"] - df["SPEND_LAG_14"]

    return df


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleanup and safety checks"""
    log("Running final data quality checks")

    # Drop rows with any NaNs
    before = df.shape[0]
    df = df.dropna()
    after = df.shape[0]

    log(f"Dropped {before - after} rows due to NaNs")

    # Sort for time-series safety
    df = df.sort_values([ "COUNTRY", DATE_COL ])

    return df


# ============================================================
# Main Pipeline
# ============================================================

def run_feature_engineering():
    log("Starting feature engineering pipeline")

    # Load data
    df = pd.read_csv(INPUT_PATH, parse_dates=[DATE_COL])
    log(f"Loaded dataset with shape: {df.shape}")

    # Validate schema
    required_columns = [
        DATE_COL,
        "COUNTRY",
        "ECONOMIC_INDEX",
        "INFLATION_RATE",
        "UNEMPLOYMENT_RATE",
        "BASELINE_DEMAND",
        "TOTAL_SPEND",
        "TOTAL_CHANNEL_RESPONSE",
        "TOTAL_PRODUCT_DEMAND",
        "SPEND_LAG_7",
        "SPEND_LAG_14",
        "DEMAND_ROLLING_7",
        "DEMAND_ROLLING_14",
    ]
    validate_columns(df, required_columns)

    # Feature creation
    df = add_time_features(df)
    df = add_marketing_efficiency_features(df)
    df = add_macro_interactions(df)
    df = add_trend_features(df)

    # Final cleanup
    df = clean_and_validate(df)

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    log(f"Feature engineering complete")
    log(f"Final feature matrix saved to: {OUTPUT_PATH}")
    log(f"Final shape: {df.shape}")
    log(f"Target column: {TARGET_COL}")

    return df


# ============================================================
# CLI Execution
# ============================================================

if __name__ == "__main__":
    run_feature_engineering()
