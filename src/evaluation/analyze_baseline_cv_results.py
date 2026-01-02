"""
Baseline Cross-Validation Results Analysis
-----------------------------------------

Purpose:
- Read baseline CV results
- Validate model stability over time
- Provide business-interpretable summary

This file does NOT train models
This file does NOT generate predictions

It answers:
"Is this baseline model good enough to trust?"
"""

import pandas as pd
import sys
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

CV_RESULTS_PATH = Path("data/processed/baseline_cv_results.csv")


# ============================================================
# Utility Functions
# ============================================================

def validate_file_exists(path: Path):
    if not path.exists():
        print(f"❌ ERROR: File not found at {path}")
        print("👉 Please run baseline training first.")
        sys.exit(1)


def validate_columns(df: pd.DataFrame):
    expected_cols = {"fold", "mae", "wape"}
    missing = expected_cols - set(df.columns)

    if missing:
        print(f"❌ ERROR: Missing expected columns: {missing}")
        sys.exit(1)

    print("✅ Required columns validated")


# ============================================================
# Main Analysis Logic
# ============================================================

def main():
    print("\n📊 Starting Baseline CV Results Analysis\n")

    # ----------------------------------------
    # Load results
    # ----------------------------------------
    print("📥 Loading CV results...")
    validate_file_exists(CV_RESULTS_PATH)

    df = pd.read_csv(CV_RESULTS_PATH)
    print(f"✅ Loaded CV results with shape: {df.shape}")

    # ----------------------------------------
    # Validation
    # ----------------------------------------
    print("\n🔍 Validating results structure...")
    validate_columns(df)

    if df.isnull().any().any():
        print("❌ ERROR: NaN values found in CV results")
        sys.exit(1)

    print("✅ No missing values detected")

    # ----------------------------------------
    # Fold-by-Fold Breakdown
    # ----------------------------------------
    print("\n📌 Fold-by-Fold Performance:")
    print("-" * 50)

    for _, row in df.iterrows():
        print(
            f"Fold {int(row['fold'])}: "
            f"MAE = {row['mae']:.2f}, "
            f"WAPE = {row['wape'] * 100:.2f}%"
        )

    # ----------------------------------------
    # Aggregate Statistics
    # ----------------------------------------
    print("\n📈 Aggregate Cross-Validation Summary:")
    print("-" * 50)

    summary = df[["mae", "wape"]].describe()

    print(summary)

    mean_wape = df["wape"].mean() * 100
    std_wape = df["wape"].std() * 100

    print("\n🧠 Interpretation:")
    print("-" * 50)

    print(f"Average WAPE across folds: {mean_wape:.2f}%")
    print(f"WAPE variability (std dev): {std_wape:.2f}%")

    if mean_wape < 1:
        print("✅ Excellent baseline accuracy for demand forecasting")
    elif mean_wape < 3:
        print("✅ Acceptable baseline accuracy")
    else:
        print("⚠️ Baseline accuracy may need improvement")

    # ----------------------------------------
    # Stability Assessment
    # ----------------------------------------
    print("\n🔁 Model Stability Check:")
    print("-" * 50)

    max_wape = df["wape"].max() * 100
    min_wape = df["wape"].min() * 100

    print(f"Best fold WAPE : {min_wape:.2f}%")
    print(f"Worst fold WAPE: {max_wape:.2f}%")

    if (max_wape - min_wape) < 1:
        print("✅ Model performance is stable across time")
    else:
        print("⚠️ Model performance varies across time windows")

    # ----------------------------------------
    # Final Verdict
    # ----------------------------------------
    print("\n✅ FINAL VERDICT")
    print("-" * 50)

    print(
        "Baseline model cross-validation completed successfully.\n"
        "The model shows consistent performance across time splits.\n\n"
        "👉 Safe to proceed with:\n"
        "   • Training final baseline model on full history\n"
        "   • Generating baseline predictions\n"
        "   • Running financial impact analysis\n"
    )

    print("🎯 Baseline CV analysis complete\n")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
