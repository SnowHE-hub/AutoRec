"""Ranking and catalogue metrics for offline recommender evaluation."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd


def precision_at_k(recommended: List[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant.

    P@k = |{top-k} ∩ relevant| / k

    Parameters
    ----------
    recommended:  Ranked list of item IDs (highest score first).
    relevant:     Set of ground-truth item IDs for this user.
    k:            Cut-off rank.
    """
    if k == 0 or not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: List[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k recommendations.

    R@k = |{top-k} ∩ relevant| / |relevant|

    Parameters
    ----------
    recommended:  Ranked list of item IDs (highest score first).
    relevant:     Set of ground-truth item IDs for this user.
    k:            Cut-off rank.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: List[str], relevant: set[str], k: int) -> float:
    """1 if at least one relevant item appears in top-k, else 0.

    HR@k = 1  if |{top-k} ∩ relevant| > 0, else 0

    Parameters
    ----------
    recommended:  Ranked list of item IDs (highest score first).
    relevant:     Set of ground-truth item IDs for this user.
    k:            Cut-off rank.
    """
    if not relevant:
        return 0.0
    return float(any(item in relevant for item in recommended[:k]))


def ndcg_at_k(recommended: List[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank k (binary relevance).

    DCG@k  = Σ_{i=1}^{k} rel_i / log2(i+1)
    IDCG@k = Σ_{i=1}^{min(k,|relevant|)} 1 / log2(i+1)
    NDCG@k = DCG@k / IDCG@k

    Parameters
    ----------
    recommended:  Ranked list of item IDs (highest score first).
    relevant:     Set of ground-truth item IDs for this user.
    k:            Cut-off rank.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommendations: List[List[str]], catalog_size: int) -> float:
    """Proportion of the catalogue that appears in at least one recommendation list.

    Coverage = |⋃ recommended_lists| / catalog_size

    Parameters
    ----------
    all_recommendations:  List of per-user recommendation lists.
    catalog_size:         Total number of distinct items in the system.
    """
    if catalog_size == 0:
        return 0.0
    unique_recommended = {item for recs in all_recommendations for item in recs}
    return len(unique_recommended) / catalog_size


def diversity(
    all_recommendations: List[List[str]],
    item_features_df: pd.DataFrame,
) -> float:
    """Mean intra-list diversity (1 - mean pairwise cosine similarity) averaged over users.

    For each user's recommendation list L:
        ILD(L) = 1 - (1 / |L|(|L|-1)) * Σ_{i≠j} cos_sim(v_i, v_j)
    Diversity = mean ILD over all users.

    A score of 1 means every pair of recommended items is orthogonal;
    0 means all items in every list are identical.

    Parameters
    ----------
    all_recommendations:  List of per-user recommendation lists.
    item_features_df:     DataFrame with car_id as first column, feature columns following.
    """
    if item_features_df.empty or not all_recommendations:
        return 0.0

    feat_cols = [c for c in item_features_df.columns if c != "car_id"]
    feat_matrix = item_features_df[feat_cols].values.astype(np.float32)
    # row-normalise for cosine similarity
    norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feat_norm = feat_matrix / norms
    id_to_idx = {cid: i for i, cid in enumerate(item_features_df["car_id"])}

    ild_scores = []
    for recs in all_recommendations:
        indices = [id_to_idx[r] for r in recs if r in id_to_idx]
        if len(indices) < 2:
            ild_scores.append(0.0)
            continue
        vecs = feat_norm[indices]                         # (L, D)
        sim_matrix = vecs @ vecs.T                        # (L, L)
        # upper-triangle without diagonal
        n = len(indices)
        off_diag_sum = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
        ild_scores.append(float(1.0 - off_diag_sum))

    return float(np.mean(ild_scores)) if ild_scores else 0.0
