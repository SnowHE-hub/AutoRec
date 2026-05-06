from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from autorec.models.base import BaseRecommender

FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
WEIGHTS = {"purchase": 5.0, "test_drive": 1.0}


class ContentBasedRecommender(BaseRecommender):
    """Cosine-similarity content-based recommender using item_features.parquet."""

    def fit(
        self,
        interactions_df: pd.DataFrame,
        user_features: pd.DataFrame | None = None,
        item_features: pd.DataFrame | None = None,
    ) -> "ContentBasedRecommender":
        self._interactions_df = interactions_df.copy()

        # load item feature matrix
        if item_features is not None:
            feat_df = item_features.copy()
        else:
            feat_df = pd.read_parquet(FEATURES_DIR / "item_features.parquet")

        self._car_ids: np.ndarray = feat_df["car_id"].values
        self._feat_matrix: np.ndarray = feat_df.drop(columns=["car_id"]).values.astype(np.float32)
        self._car_index: dict[str, int] = {cid: i for i, cid in enumerate(self._car_ids)}

        # stash original car attributes for explain()
        # we need body_type and make — look them up from interactions merge if possible
        # but we keep a lookup from feature index only if the caller passes it
        self._body_type_lookup: dict[str, str] = {}
        self._make_lookup: dict[str, str] = {}

        return self

    def attach_car_meta(self, cars_df: pd.DataFrame) -> "ContentBasedRecommender":
        """Optional: provide make/body_type for richer explain() output."""
        self._body_type_lookup = cars_df.set_index("car_id")["body_type"].to_dict()
        self._make_lookup       = cars_df.set_index("car_id")["make"].to_dict()
        return self

    def _user_profile(self, user_id: str) -> np.ndarray | None:
        """Weighted mean of item feature vectors for items the user interacted with."""
        df = self._interactions_df
        user_rows = df[df["user_id"] == user_id].copy()
        if user_rows.empty:
            return None

        user_rows["weight"] = user_rows["interaction_type"].map(WEIGHTS).fillna(0)
        vectors, weights = [], []
        for _, row in user_rows.iterrows():
            idx = self._car_index.get(row["car_id"])
            if idx is not None:
                vectors.append(self._feat_matrix[idx])
                weights.append(row["weight"])

        if not vectors:
            return None

        weights_arr = np.array(weights, dtype=np.float32)
        profile = np.average(np.stack(vectors), axis=0, weights=weights_arr)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float]]:
        profile = self._user_profile(user_id)
        if profile is None:
            return []

        sims = cosine_similarity(profile.reshape(1, -1), self._feat_matrix)[0]

        if exclude_seen:
            seen = self._seen_items(user_id)
            for cid in seen:
                idx = self._car_index.get(cid)
                if idx is not None:
                    sims[idx] = -1.0

        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        return [(self._car_ids[i], float(sims[i])) for i in top_indices]

    def explain(self, user_id: str, car_id: str) -> str:
        profile = self._user_profile(user_id)
        if profile is None:
            return "暂无足够交互数据生成解释"

        idx = self._car_index.get(car_id)
        if idx is None:
            return "车辆特征数据缺失"

        sim = float(cosine_similarity(
            profile.reshape(1, -1),
            self._feat_matrix[idx].reshape(1, -1),
        )[0][0])

        body = self._body_type_lookup.get(car_id, "")
        make = self._make_lookup.get(car_id, "")
        desc = f"{body} {make}".strip() or car_id
        return f"与你偏好的 {desc} 车型相似度 {sim * 100:.1f}%"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
    interactions = pd.read_csv(PROCESSED / "interactions.csv")
    cars         = pd.read_csv(PROCESSED / "cars.csv")

    model = ContentBasedRecommender().fit(interactions)
    model.attach_car_meta(cars)

    user = interactions["user_id"].value_counts().index[5]
    recs = model.recommend(user, k=5)
    print(f"ContentBasedRecommender — top-5 for {user}:")
    for car_id, score in recs:
        print(f"  {car_id}  sim={score:.4f}  | {model.explain(user, car_id)}")

    cold_recs = model.recommend("NEW_USER_999", k=3)
    print(f"\nCold-start (unknown user): {cold_recs}")
