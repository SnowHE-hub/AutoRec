from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd


class BaseRecommender(ABC):
    """Common interface for all AutoRec recommenders."""

    @abstractmethod
    def fit(
        self,
        interactions_df: pd.DataFrame,
        user_features: pd.DataFrame | None = None,
        item_features: pd.DataFrame | None = None,
    ) -> "BaseRecommender":
        """Train the model on interaction data.

        Parameters
        ----------
        interactions_df:
            Columns: user_id, car_id, interaction_type, timestamp
        user_features:
            Optional pre-computed user feature matrix (user_id as first column).
        item_features:
            Optional pre-computed item feature matrix (car_id as first column).
        """

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float]]:
        """Return top-k (car_id, score) pairs for *user_id*.

        Parameters
        ----------
        user_id:
            Target user.  Unknown users should be handled gracefully (cold start).
        k:
            Number of results to return.
        exclude_seen:
            When True, do not include cars the user has already interacted with.
        """

    @abstractmethod
    def explain(self, user_id: str, car_id: str) -> str:
        """Return a human-readable explanation for recommending *car_id* to *user_id*."""

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------
    def _seen_items(self, user_id: str) -> set[str]:
        """Return the set of car_ids the user has already interacted with."""
        if not hasattr(self, "_interactions_df"):
            return set()
        df: pd.DataFrame = self._interactions_df
        return set(df.loc[df["user_id"] == user_id, "car_id"])
