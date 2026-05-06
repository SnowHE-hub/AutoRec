from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from autorec.models.base import BaseRecommender
from autorec.models.popularity import PopularityRecommender

CONFIDENCE = {"purchase": 5.0, "test_drive": 1.0}
ALS_FACTORS     = 64
ALS_ITERATIONS  = 20
ALS_REGULARIZE  = 0.01


class ALSRecommender(BaseRecommender):
    """Implicit-feedback ALS via the `implicit` library."""

    def fit(
        self,
        interactions_df: pd.DataFrame,
        user_features: pd.DataFrame | None = None,
        item_features: pd.DataFrame | None = None,
    ) -> "ALSRecommender":
        self._interactions_df = interactions_df.copy()

        # index mappings
        self._user_ids = interactions_df["user_id"].unique()
        self._car_ids  = interactions_df["car_id"].unique()
        self._user2idx: dict[str, int] = {u: i for i, u in enumerate(self._user_ids)}
        self._car2idx:  dict[str, int] = {c: i for i, c in enumerate(self._car_ids)}
        self._idx2car:  dict[int, str] = {i: c for c, i in self._car2idx.items()}

        # build sparse user-item confidence matrix (CSR, shape: users × items)
        df = interactions_df.copy()
        df["conf"] = df["interaction_type"].map(CONFIDENCE).fillna(0)
        row = df["user_id"].map(self._user2idx).values
        col = df["car_id"].map(self._car2idx).values
        # sum confidences for duplicate (user, item) pairs
        mat = sp.csr_matrix(
            (df["conf"].values, (row, col)),
            shape=(len(self._user_ids), len(self._car_ids)),
            dtype=np.float32,
        )

        # implicit.ALS treats rows as users: fit with (n_users × n_items)
        self._user_item_mat = mat  # (n_users, n_items) — CSR

        self._als = AlternatingLeastSquares(
            factors=ALS_FACTORS,
            iterations=ALS_ITERATIONS,
            regularization=ALS_REGULARIZE,
            use_gpu=False,
        )
        self._als.fit(self._user_item_mat)

        # cold-start fallback
        self._fallback = PopularityRecommender().fit(interactions_df)

        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float]]:
        if user_id not in self._user2idx:
            return self._fallback.recommend(user_id, k=k, exclude_seen=False)

        u_idx = self._user2idx[user_id]
        filter_items: list[int] = []
        if exclude_seen:
            seen = self._seen_items(user_id)
            filter_items = [self._car2idx[c] for c in seen if c in self._car2idx]

        item_ids, scores = self._als.recommend(
            u_idx,
            self._user_item_mat[u_idx],
            N=k,
            filter_already_liked_items=exclude_seen,
            filter_items=filter_items if filter_items else None,
        )
        return [(self._idx2car[int(i)], float(s)) for i, s in zip(item_ids, scores)]

    def explain(self, user_id: str, car_id: str) -> str:
        if user_id not in self._user2idx:
            return self._fallback.explain(user_id, car_id)
        return "基于协同过滤，与你相似的用户也喜欢这款车"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
    interactions = pd.read_csv(PROCESSED / "interactions.csv")

    print("Fitting ALS ...")
    model = ALSRecommender().fit(interactions)

    user = interactions["user_id"].value_counts().index[0]
    recs = model.recommend(user, k=5)
    print(f"\nALSRecommender — top-5 for {user}:")
    for car_id, score in recs:
        print(f"  {car_id}  score={score:.4f}  | {model.explain(user, car_id)}")

    cold_recs = model.recommend("NEW_USER_999", k=3)
    print(f"\nCold-start fallback top-3: {cold_recs}")
