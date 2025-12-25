"""
Baseline Demand Forecasting Model (TimeSeries-safe)
--------------------------------------------------

Fixes:
- Handles categorical features (COUNTRY)
- Uses OneHotEncoding
- Prevents data leakage
"""

# ============================================================
# Imports
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# Config
# ============================================================

DATA_PATH = Path("data/processed/model_features_final.csv")
OUTPUT_PATH = Path("data/processed/baseline_cv_results.csv")

TARGET_COL = "TOTAL_PRODUCT_DEMAND"
CATEGORICAL_COLS = ["COUNTRY"]

N_SPLITS = 5
RANDOM_STATE = 42

# ============================================================
# Metrics
# ============================================================

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


# ============================================================
# Load data
# ============================================================

print("📥 Loading feature dataset...")

df = pd.read_csv(DATA_PATH, parse_dates=["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

print(f"✅ Dataset loaded: {df.shape}")
print(f"📅 Date range: {df['DATE'].min()} → {df['DATE'].max()}")

# ============================================================
# Split features / target
# ============================================================

X = df.drop(columns=["DATE", TARGET_COL])
y = df[TARGET_COL]

print(f"🧮 Initial features: {X.shape[1]}")
print(f"🎯 Target: {TARGET_COL}")

# ============================================================
# Identify categorical + numeric features
# ============================================================

cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"🔤 Categorical columns: {cat_cols}")
print(f"🔢 Numeric columns: {len(num_cols)}")

# ============================================================
# TimeSeries Cross Validation
# ============================================================

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
results = []

print(f"\n🔁 Running TimeSeriesSplit CV ({N_SPLITS} folds)\n")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

    print(f"================ Fold {fold} ================")

    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"Train range: {df.iloc[train_idx]['DATE'].min()} → {df.iloc[train_idx]['DATE'].max()}")
    print(f"Test  range: {df.iloc[test_idx]['DATE'].min()} → {df.iloc[test_idx]['DATE'].max()}")

    # --------------------------------------------------------
    # Encode categorical variables (fit ONLY on train)
    # --------------------------------------------------------

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        # Fallback for older sklearn versions
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


    X_train_cat = encoder.fit_transform(X_train_raw[cat_cols])
    X_test_cat = encoder.transform(X_test_raw[cat_cols])

    # --------------------------------------------------------
    # Combine numeric + encoded categorical
    # --------------------------------------------------------

    X_train_num = X_train_raw[num_cols].values
    X_test_num = X_test_raw[num_cols].values

    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_test_final = np.hstack([X_test_num, X_test_cat])

    print(f"Final train shape: {X_train_final.shape}")
    print(f"Final test  shape: {X_test_final.shape}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(X_train_final, y_train)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(X_test_final)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    fold_mae = mean_absolute_error(y_test, y_pred)
    fold_wape = wape(y_test.values, y_pred)

    print(f"MAE  : {fold_mae:,.2f}")
    print(f"WAPE : {fold_wape:.2%}")

    results.append({
        "fold": fold,
        "mae": fold_mae,
        "wape": fold_wape
    })

# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\n📊 Cross-validation summary")
print(results_df.describe())

print(f"\n💾 Results saved to: {OUTPUT_PATH}")
print("\n✅ Baseline training completed successfully")
