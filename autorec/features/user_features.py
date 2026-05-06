from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

BASE_DIR      = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR  = BASE_DIR / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DATE = pd.Timestamp("2025-01-01")


def _mode_or_none(s: pd.Series) -> str | None:
    if s.empty:
        return None
    return s.mode().iloc[0]


def _top_brands(s: pd.Series, k: int = 2) -> list[str | None]:
    counts = s.value_counts()
    return [counts.index[i] if i < len(counts) else None for i in range(k)]


def _body_entropy(s: pd.Series) -> float:
    counts = s.value_counts(normalize=True)
    return float(entropy(counts))


def build_user_features(
    interactions_df: pd.DataFrame,
    cars_df: pd.DataFrame,
) -> pd.DataFrame:
    interactions = interactions_df.copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

    # join car attributes onto interactions
    merged = interactions.merge(
        cars_df[["car_id", "price", "body_type", "make", "price_tier"]],
        on="car_id",
        how="left",
    )

    # --- aggregate per user ---
    td = merged[merged["interaction_type"] == "test_drive"]
    pu = merged[merged["interaction_type"] == "purchase"]

    agg = (
        merged.groupby("user_id")
        .agg(
            n_test_drives=("interaction_type", lambda x: (x == "test_drive").sum()),
            n_purchases=("interaction_type", lambda x: (x == "purchase").sum()),
            avg_price_interacted=("price", "mean"),
            preferred_body_type=("body_type", _mode_or_none),
            preferred_price_tier=("price_tier", _mode_or_none),
            last_interaction=("timestamp", "max"),
            diversity_score=("body_type", _body_entropy),
        )
        .reset_index()
    )

    # recency in days from 2025-01-01
    agg["recency_days"] = (
        REFERENCE_DATE - agg["last_interaction"]
    ).dt.days.astype(int)
    agg = agg.drop(columns=["last_interaction"])

    # top-2 brands per user
    brand_top = (
        merged.groupby("user_id")["make"]
        .apply(_top_brands)
        .reset_index()
    )
    brand_top[["brand_top1", "brand_top2"]] = pd.DataFrame(
        brand_top["make"].tolist(), index=brand_top.index
    )
    brand_top = brand_top.drop(columns=["make"])

    result = agg.merge(brand_top, on="user_id", how="left")

    col_order = [
        "user_id",
        "n_test_drives", "n_purchases",
        "avg_price_interacted",
        "preferred_body_type", "preferred_price_tier",
        "brand_top1", "brand_top2",
        "recency_days",
        "diversity_score",
    ]
    return result[col_order]


if __name__ == "__main__":
    print("Loading processed data ...")
    interactions = pd.read_csv(PROCESSED_DIR / "interactions.csv")
    cars         = pd.read_csv(PROCESSED_DIR / "cars.csv")

    print("Building user features ...")
    uf = build_user_features(interactions, cars)

    out = FEATURES_DIR / "user_features.parquet"
    uf.to_parquet(out, index=False)

    print(f"\nSaved  : {out}")
    print(f"Shape  : {uf.shape}")
    print(f"\nFirst 3 rows:")
    print(uf.head(3).to_string())
