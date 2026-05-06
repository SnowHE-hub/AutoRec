from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR      = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR  = BASE_DIR / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLS = ["price", "mileage", "age_at_listing"]
CAT_COLS = ["body_type", "fuel_type", "transmission", "price_tier"]
TFIDF_MAX_FEATURES = 100


def build_item_features(cars_df: pd.DataFrame) -> pd.DataFrame:
    df = cars_df.copy()

    # --- numeric: StandardScaler ---
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[NUM_COLS])
    num_df = pd.DataFrame(
        num_scaled,
        columns=[f"num_{c}" for c in NUM_COLS],
        index=df.index,
    )

    # --- categorical: OneHotEncoder ---
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
    cat_encoded = ohe.fit_transform(df[CAT_COLS])
    cat_cols = [
        f"cat_{feat}_{val}"
        for feat, cats in zip(CAT_COLS, ohe.categories_)
        for val in cats
    ]
    cat_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=df.index)

    # --- text: TF-IDF on make + model + body_type ---
    corpus = (
        df["make"].fillna("") + " "
        + df["model"].fillna("") + " "
        + df["body_type"].fillna("")
    )
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = tfidf.fit_transform(corpus).toarray().astype(np.float32)
    tfidf_cols = [f"tfidf_{t}" for t in tfidf.get_feature_names_out()]
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=df.index)

    # --- concatenate ---
    result = pd.concat(
        [df[["car_id"]].reset_index(drop=True),
         num_df.reset_index(drop=True),
         cat_df.reset_index(drop=True),
         tfidf_df.reset_index(drop=True)],
        axis=1,
    )
    return result


if __name__ == "__main__":
    print("Loading processed data ...")
    cars = pd.read_csv(PROCESSED_DIR / "cars.csv")

    print("Building item features ...")
    itf = build_item_features(cars)

    out = FEATURES_DIR / "item_features.parquet"
    itf.to_parquet(out, index=False)

    print(f"\nSaved  : {out}")
    print(f"Shape  : {itf.shape}")
    print(f"\nColumn groups:")
    print(f"  num   : {[c for c in itf.columns if c.startswith('num_')]}")
    print(f"  cat   : {len([c for c in itf.columns if c.startswith('cat_')])} columns")
    print(f"  tfidf : {len([c for c in itf.columns if c.startswith('tfidf_')])} columns")
    print(f"\nFirst 3 rows (first 10 cols):")
    print(itf.iloc[:3, :10].to_string())
