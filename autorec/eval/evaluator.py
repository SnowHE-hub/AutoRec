from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from autorec.eval.metrics import (
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    coverage,
    diversity,
)
from autorec.models.popularity import PopularityRecommender
from autorec.models.content_based import ContentBasedRecommender
from autorec.models.matrix_factorization import ALSRecommender

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

BASE_DIR      = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR  = BASE_DIR / "data" / "features"

K = 10


# ---------------------------------------------------------------------------
# Temporal train/test split
# ---------------------------------------------------------------------------

def temporal_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sort each user's interactions by time; last test_ratio fraction → test."""
    interactions = interactions.copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions = interactions.sort_values(["user_id", "timestamp"])

    train_rows, test_rows = [], []
    for _, group in interactions.groupby("user_id", sort=False):
        n = len(group)
        split = max(1, int(n * (1 - test_ratio)))
        train_rows.append(group.iloc[:split])
        test_rows.append(group.iloc[split:])

    return pd.concat(train_rows, ignore_index=True), pd.concat(test_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-model evaluation helper
# ---------------------------------------------------------------------------

def _eval_model(
    model,
    test_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    catalog_size: int,
    model_name: str,
) -> Dict[str, float]:
    """Generate top-K recommendations for every test user and compute metrics."""
    test_users = test_df["user_id"].unique()
    relevant_map: Dict[str, set] = (
        test_df.groupby("user_id")["car_id"].apply(set).to_dict()
    )

    all_recs: List[List[str]] = []
    recalls, ndcgs, hits = [], [], []

    print(f"  Evaluating {model_name} on {len(test_users)} users ...", flush=True)
    for user_id in test_users:
        recs_with_scores = model.recommend(user_id, k=K, exclude_seen=True)
        recs = [car_id for car_id, _ in recs_with_scores]
        relevant = relevant_map.get(user_id, set())

        recalls.append(recall_at_k(recs, relevant, K))
        ndcgs.append(ndcg_at_k(recs, relevant, K))
        hits.append(hit_rate_at_k(recs, relevant, K))
        all_recs.append(recs)

    return {
        "Model":       model_name,
        f"Recall@{K}":  round(float(np.mean(recalls)),  4),
        f"NDCG@{K}":    round(float(np.mean(ndcgs)),    4),
        f"HitRate@{K}": round(float(np.mean(hits)),     4),
        "Coverage":    round(coverage(all_recs, catalog_size), 4),
        "Diversity":   round(diversity(all_recs, item_features_df), 4),
    }


# ---------------------------------------------------------------------------
# Cold-start analysis
# ---------------------------------------------------------------------------

def _cold_start_analysis(
    models: Dict[str, object],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Recall@K grouped by number of training interactions per user."""
    train_counts = train_df.groupby("user_id").size().rename("n_train")
    bins   = [0, 5, 20, int(1e9)]
    labels = ["1-5", "6-20", "20+"]

    relevant_map = test_df.groupby("user_id")["car_id"].apply(set).to_dict()
    test_users = test_df["user_id"].unique()

    rows = []
    for user_id in test_users:
        n = int(train_counts.get(user_id, 0))
        group = labels[np.searchsorted(bins[1:], n, side="left")]
        relevant = relevant_map.get(user_id, set())
        row = {"user_id": user_id, "group": group}
        for name, model in models.items():
            recs = [c for c, _ in model.recommend(user_id, k=K, exclude_seen=True)]
            row[name] = recall_at_k(recs, relevant, K)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = df.groupby("group")[[m for m in models]].mean().round(4)
    summary.index = pd.CategoricalIndex(summary.index, categories=labels, ordered=True)
    return summary.sort_index()


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_all_models() -> pd.DataFrame:
    print("\n=== AutoRec Offline Evaluation ===\n")

    # load data
    interactions  = pd.read_csv(PROCESSED_DIR / "interactions.csv")
    cars          = pd.read_csv(PROCESSED_DIR / "cars.csv")
    item_features = pd.read_parquet(FEATURES_DIR / "item_features.parquet")
    catalog_size  = len(cars)

    # temporal split
    train_df, test_df = temporal_split(interactions, test_ratio=0.2)
    print(f"Train interactions : {len(train_df):,}")
    print(f"Test  interactions : {len(test_df):,}")
    print(f"Test  users        : {test_df['user_id'].nunique():,}")
    print(f"Catalogue size     : {catalog_size:,}\n")

    # ---------- fit models ----------
    print("Fitting models ...")

    t0 = time.time()
    pop = PopularityRecommender().fit(train_df)
    print(f"  Popularity   fitted in {time.time()-t0:.1f}s")

    t0 = time.time()
    cb = ContentBasedRecommender().fit(train_df, item_features=item_features)
    cb.attach_car_meta(cars)
    print(f"  ContentBased fitted in {time.time()-t0:.1f}s")

    t0 = time.time()
    als = ALSRecommender().fit(train_df)
    print(f"  ALS          fitted in {time.time()-t0:.1f}s")

    models = {"Popularity": pop, "ContentBased": cb, "ALS": als}

    # ---------- metric evaluation ----------
    print("\nComputing metrics ...\n")
    results = []
    for name, model in models.items():
        row = _eval_model(model, test_df, item_features, catalog_size, name)
        results.append(row)

    results_df = pd.DataFrame(results).set_index("Model")

    print("\n" + "=" * 70)
    print("Overall Evaluation Results")
    print("=" * 70)
    print(results_df.to_string())

    # ---------- cold-start ----------
    print("\n\nCold-start Analysis (Recall@10 by training interaction count)")
    print("=" * 70)
    cold_df = _cold_start_analysis(models, train_df, test_df)
    print(cold_df.to_string())

    # ---------- save ----------
    out_path = PROCESSED_DIR / "eval_results.csv"
    results_df.reset_index().to_csv(out_path, index=False)
    cold_path = PROCESSED_DIR / "eval_coldstart.csv"
    cold_df.reset_index().to_csv(cold_path, index=False)

    print(f"\nSaved → {out_path}")
    print(f"Saved → {cold_path}")

    return results_df


if __name__ == "__main__":
    evaluate_all_models()
