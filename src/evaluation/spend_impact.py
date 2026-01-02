"""
Forecast Spend Impact Analysis
------------------------------

Purpose:
- Translate forecast errors into financial impact
- Quantify cost of over-forecasting vs under-forecasting
- Provide business-readable outputs

Inputs:
- data/processed/baseline_predictions.csv

Outputs:
- Printed financial impact summary
- Optional CSV for downstream reporting
"""

# ============================================================
# Imports
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

PREDICTIONS_PATH = Path("data/processed/baseline_predictions.csv")

# Financial assumptions (example – adjustable)
REVENUE_PER_UNIT = 120.0        # revenue per unit sold
OVER_FORECAST_COST = 30.0       # holding / waste cost per excess unit
UNDER_FORECAST_COST = 80.0      # lost margin per missed unit

# ============================================================
# Utility Logging
# ============================================================

def log(msg: str):
    print(f"💰 {msg}")

# ============================================================
# Core Impact Logic
# ============================================================

def compute_financial_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forecast error and translate into dollar impact
    """

    log("Computing forecast error")

    df["FORECAST_ERROR"] = df["ACTUAL_DEMAND"] - df["BASELINE_PREDICTION"]

    # Positive error → under-forecast (missed demand)
    df["UNDER_FORECAST_UNITS"] = df["FORECAST_ERROR"].clip(lower=0)

    # Negative error → over-forecast (excess supply)
    df["OVER_FORECAST_UNITS"] = (-df["FORECAST_ERROR"]).clip(lower=0)

    log("Applying financial cost assumptions")

    df["UNDER_FORECAST_COST_$"] = (
        df["UNDER_FORECAST_UNITS"] * UNDER_FORECAST_COST
    )

    df["OVER_FORECAST_COST_$"] = (
        df["OVER_FORECAST_UNITS"] * OVER_FORECAST_COST
    )

    df["TOTAL_FORECAST_COST_$"] = (
        df["UNDER_FORECAST_COST_$"] + df["OVER_FORECAST_COST_$"]
    )

    return df

# ============================================================
# Main Execution
# ============================================================

def main():

    log("Starting forecast financial impact analysis")

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    log("Loading baseline predictions")

    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["DATE"])

    log(f"Loaded predictions with shape: {df.shape}")
    log(f"Date range: {df['DATE'].min()} → {df['DATE'].max()}")

    required_cols = [
        "DATE",
        "COUNTRY",
        "ACTUAL_DEMAND",
        "BASELINE_PREDICTION",
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --------------------------------------------------------
    # Compute financial impact
    # --------------------------------------------------------

    df = compute_financial_impact(df)

    # --------------------------------------------------------
    # Aggregate summary
    # --------------------------------------------------------

    log("Aggregating financial impact summary")

    summary = {
        "Total Actual Demand": df["ACTUAL_DEMAND"].sum(),
        "Total Predicted Demand": df["BASELINE_PREDICTION"].sum(),
        "Total Under-Forecast Units": df["UNDER_FORECAST_UNITS"].sum(),
        "Total Over-Forecast Units": df["OVER_FORECAST_UNITS"].sum(),
        "Total Under-Forecast Cost ($)": df["UNDER_FORECAST_COST_$"].sum(),
        "Total Over-Forecast Cost ($)": df["OVER_FORECAST_COST_$"].sum(),
        "Total Forecast Cost ($)": df["TOTAL_FORECAST_COST_$"].sum(),
    }

    print("\n📊 FINANCIAL IMPACT SUMMARY")
    print("--------------------------------------------------")

    for k, v in summary.items():
        print(f"{k:<35}: {v:,.2f}")

    print("\n✅ Forecast financial impact analysis completed")

# ============================================================
# CLI Entrypoint
# ============================================================

if __name__ == "__main__":
    main()
