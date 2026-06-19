"""
preprocess.py — KC House Data Preprocessing Pipeline
=====================================================
Cleans, engineers features, encodes, scales, and splits the KC house dataset
into train/test sets ready for model training.

Usage:
    from preprocess import load_and_preprocess
    X_train, X_test, y_train, y_test, preprocessor_info = load_and_preprocess("kc_house_data.csv")

Or run directly:
    python preprocess.py --data kc_house_data.csv --output processed/
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Columns to drop entirely (identifiers / leakage / low-value)
DROP_COLS = ["id", "date"]

# Target column
TARGET = "price"

# Ordinal features mapped to ordered buckets for interpretability
ORDINAL_FEATURES = {
    "condition": [1, 2, 3, 4, 5],
    "grade":     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "view":      [0, 1, 2, 3, 4],
    "waterfront":[0, 1],
}

# Test split ratio & random seed
TEST_SIZE   = 0.2
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV and do a quick sanity check."""
    df = pd.read_csv(path)
    print(f"[load]  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    assert TARGET in df.columns, f"Target column '{TARGET}' not found."
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop useless columns, fix obvious anomalies, remove hard outliers.
    """
    df = df.copy()

    # --- 3a. Drop identifier columns ---
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # --- 3b. Bedrooms: a row with 33 bedrooms is a known data-entry error ---
    df = df[df["bedrooms"] <= 10]

    # --- 3c. Price must be positive ---
    df = df[df[TARGET] > 0]

    # --- 3d. yr_renovated: 0 means "never renovated", keep as-is ---

    # --- 3e. Remove extreme price outliers (beyond 99.5th percentile) ---
    upper_cap = df[TARGET].quantile(0.995)
    before = len(df)
    df = df[df[TARGET] <= upper_cap]
    print(f"[clean] Removed {before - len(df)} extreme-price outliers  "
          f"(cap: ${upper_cap:,.0f})")

    print(f"[clean] Remaining rows: {len(df):,}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable, domain-meaningful features that improve model
    accuracy and SHAP explainability.
    """
    df = df.copy()

    # --- Age and renovation ---
    sale_year = pd.to_datetime("today").year  # dynamic; use 2015 to reproduce paper results
    df["house_age"]          = sale_year - df["yr_built"]
    df["years_since_reno"]   = sale_year - df["yr_renovated"].replace(0, np.nan)
    df["years_since_reno"]   = df["years_since_reno"].fillna(df["house_age"])
    df["is_renovated"]       = (df["yr_renovated"] > 0).astype(int)

    # --- Size ratios ---
    df["living_to_lot_ratio"]   = df["sqft_living"] / (df["sqft_lot"] + 1)
    df["above_to_living_ratio"] = df["sqft_above"]  / (df["sqft_living"] + 1)
    df["basement_present"]      = (df["sqft_basement"] > 0).astype(int)

    # --- Neighborhood size trend ---
    df["living_vs_neighbors"]   = df["sqft_living"] - df["sqft_living15"]
    df["lot_vs_neighbors"]      = df["sqft_lot"]    - df["sqft_lot15"]

    # --- Composite luxury score (useful for SHAP interactions) ---
    df["luxury_score"] = (
        df["grade"]      * 0.4 +
        df["condition"]  * 0.2 +
        df["view"]       * 0.2 +
        df["waterfront"] * 0.2
    )

    # --- Bathroom-to-bedroom ratio ---
    df["bath_bed_ratio"] = df["bathrooms"] / (df["bedrooms"] + 1)

    # --- Total rooms proxy ---
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"].apply(np.ceil)

    # Drop raw year columns (replaced by engineered ones)
    df.drop(columns=["yr_built", "yr_renovated"], inplace=True)

    print(f"[engineer] Feature count: {df.shape[1] - 1} features  (target excluded)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. TARGET TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def transform_target(y: pd.Series) -> pd.Series:
    """Log1p-transform the target to reduce right skew (matches notebook)."""
    return np.log1p(y)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENCODING & SCALING
# ─────────────────────────────────────────────────────────────────────────────

def encode_and_scale(X_train: pd.DataFrame,
                     X_test:  pd.DataFrame):
    """
    - Ordinal encode the ordered categorical columns.
    - StandardScale all numeric features.

    Returns:
        X_train_proc, X_test_proc, scaler, ordinal_encoders (dict)
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    ordinal_encoders: dict = {}

    # --- 6a. Ordinal encode known ordinal columns ---
    for col in ORDINAL_FEATURES:
        if col not in X_train.columns:
            continue
        categories = [sorted(ORDINAL_FEATURES[col])]
        enc = OrdinalEncoder(categories=categories,
                             handle_unknown="use_encoded_value",
                             unknown_value=-1)
        X_train[[col]] = enc.fit_transform(X_train[[col]])
        X_test[[col]]  = enc.transform(X_test[[col]])
        ordinal_encoders[col] = enc

    # --- 6b. Zipcode: treat as categorical integer (keep raw for tree models,
    #         but scale for SVR) — scaled version added as a separate column ---
    # Tree-based models handle raw integers fine; we add a flag-column instead.
    X_train["zipcode_raw"] = X_train["zipcode"]
    X_test["zipcode_raw"]  = X_test["zipcode"]

    # --- 6c. StandardScale all numeric columns ---
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"[encode] Ordinal-encoded: {list(ordinal_encoders.keys())}")
    print(f"[scale]  Scaled {len(num_cols)} numeric columns via StandardScaler")

    return X_train, X_test, scaler, ordinal_encoders


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(data_path: str, output_dir: str = None):
    """
    Full end-to-end preprocessing pipeline.

    Returns
    -------
    X_train, X_test  : pd.DataFrame (scaled features)
    y_train, y_test  : pd.Series    (log1p-transformed target)
    info             : dict with scaler, encoders, feature names, and raw test target
    """
    # 1. Load
    df = load_raw(data_path)

    # 2. Clean
    df = clean(df)

    # 3. Engineer
    df = engineer_features(df)

    # 4. Split X / y
    y_raw = df[TARGET].copy()
    X     = df.drop(columns=[TARGET])
    y     = transform_target(y_raw)

    # 5. Train / test split  (stratified by price quantile for reproducibility)
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y, y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # 6. Encode + Scale
    X_train, X_test, scaler, ordinal_encoders = encode_and_scale(X_train, X_test)

    print(f"\n[pipeline] Train size : {len(X_train):,}")
    print(f"[pipeline] Test  size : {len(X_test):,}")
    print(f"[pipeline] Features   : {X_train.shape[1]}")

    info = {
        "scaler":           scaler,
        "ordinal_encoders": ordinal_encoders,
        "feature_names":    X_train.columns.tolist(),
        "y_raw_test":       y_raw_test,   # original USD prices for metric display
        "y_raw_train":      y_raw_train,
    }

    # 7. Optionally save to disk
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        X_train.to_parquet(os.path.join(output_dir, "X_train.parquet"), index=False)
        X_test.to_parquet(os.path.join(output_dir,  "X_test.parquet"),  index=False)
        y_train.to_frame("price_log").to_parquet(os.path.join(output_dir, "y_train.parquet"), index=False)
        y_test.to_frame("price_log").to_parquet(os.path.join(output_dir,  "y_test.parquet"),  index=False)
        with open(os.path.join(output_dir, "preprocessor_info.pkl"), "wb") as f:
            pickle.dump(info, f)
        print(f"\n[save] Processed data saved to '{output_dir}/'")

    return X_train, X_test, y_train, y_test, info


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess KC House Data")
    parser.add_argument("--data",   default="kc_house_data.csv",
                        help="Path to kc_house_data.csv")
    parser.add_argument("--output", default="processed",
                        help="Output directory for processed parquet files")
    args = parser.parse_args()

    load_and_preprocess(args.data, output_dir=args.output)
    print("\n✅ Preprocessing complete.")
