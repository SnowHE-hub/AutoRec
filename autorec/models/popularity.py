from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from autorec.models.base import BaseRecommender

WEIGHTS = {"purchase": 5, "test_drive": 1}


class PopularityRecommender(BaseRecommender):
    """Global popularity baseline: score = 5 * purchases + test_drives."""

    def fit(
        self,
        interactions_df: pd.DataFrame,
        user_features: pd.DataFrame | None = None,
        item_features: pd.DataFrame | None = None,
    ) -> "PopularityRecommender":
        self._interactions_df = interactions_df.copy()

        scored = interactions_df.copy()
        scored["weight"] = scored["interaction_type"].map(WEIGHTS).fillna(0)

        self._scores: pd.Series = (
            scored.groupby("car_id")["weight"].sum().sort_values(ascending=False)
        )

        # per-car counts for explain()
        counts = (
            interactions_df.groupby(["car_id", "interaction_type"])
            .size()
            .unstack(fill_value=0)
        )
        self._purchases   = counts.get("purchase",   pd.Series(dtype=int))
        self._test_drives = counts.get("test_drive", pd.Series(dtype=int))

        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float]]:
        scores = self._scores
        if exclude_seen:
            seen = self._seen_items(user_id)
            scores = scores[~scores.index.isin(seen)]
        top = scores.head(k)
        return list(zip(top.index.tolist(), top.values.tolist()))

    def explain(self, user_id: str, car_id: str) -> str:
        total     = int(self._scores.get(car_id, 0))
        purchases = int(self._purchases.get(car_id, 0))
        return f"该车累计被交互 {total} 次，其中购买 {purchases} 次"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
    interactions = pd.read_csv(PROCESSED / "interactions.csv")

    model = PopularityRecommender().fit(interactions)

    user = interactions["user_id"].iloc[0]
    recs = model.recommend(user, k=5)
    print(f"PopularityRecommender — top-5 for {user}:")
    for car_id, score in recs:
        print(f"  {car_id}  score={score:.1f}  | {model.explain(user, car_id)}")

    cold_recs = model.recommend("NEW_USER_999", k=3)
    print(f"\nCold-start (unknown user) top-3: {cold_recs}")
