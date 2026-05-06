from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from autorec.models.popularity import PopularityRecommender
from autorec.models.content_based import ContentBasedRecommender
from autorec.models.matrix_factorization import ALSRecommender

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("autorec.api")

BASE_DIR      = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR  = BASE_DIR / "data" / "features"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RecommendFilters(BaseModel):
    price_max:  Optional[float] = None
    price_min:  Optional[float] = None
    body_type:  Optional[str]   = None
    fuel_type:  Optional[str]   = None

class RecommendRequest(BaseModel):
    user_id: str
    model:   str  = Field("als", pattern="^(popularity|content_based|als)$")
    k:       int  = Field(10, ge=1, le=50)
    filters: Optional[RecommendFilters] = None

class RecommendationItem(BaseModel):
    car_id:      str
    score:       float
    explanation: str

class RecommendResponse(BaseModel):
    user_id:         str
    model:           str
    recommendations: List[RecommendationItem]
    latency_ms:      float

class HealthResponse(BaseModel):
    status:        str
    models_loaded: List[str]


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    cars_df:         pd.DataFrame
    user_features_df: pd.DataFrame
    item_features_df: pd.DataFrame
    eval_results_df: Optional[pd.DataFrame]
    models:          Dict[str, Any]

state = AppState()


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

app = FastAPI(title="AutoRec Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    user_id = request.query_params.get("user_id", "-")
    log.info(
        "%-6s %-35s  user=%-10s  %.1f ms",
        request.method,
        request.url.path,
        user_id,
        latency_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    log.info("Loading data ...")
    state.cars_df          = pd.read_csv(PROCESSED_DIR / "cars.csv")
    interactions           = pd.read_csv(PROCESSED_DIR / "interactions.csv")
    state.item_features_df = pd.read_parquet(FEATURES_DIR / "item_features.parquet")

    uf_path = FEATURES_DIR / "user_features.parquet"
    state.user_features_df = (
        pd.read_parquet(uf_path) if uf_path.exists() else pd.DataFrame()
    )

    er_path = PROCESSED_DIR / "eval_results.csv"
    state.eval_results_df = (
        pd.read_csv(er_path) if er_path.exists() else None
    )

    log.info("Fitting models ...")
    pop = PopularityRecommender().fit(interactions)

    cb = ContentBasedRecommender().fit(
        interactions, item_features=state.item_features_df
    )
    cb.attach_car_meta(state.cars_df)

    als = ALSRecommender().fit(interactions)

    state.models = {
        "popularity":    pop,
        "content_based": cb,
        "als":           als,
    }
    log.info("All models ready: %s", list(state.models.keys()))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded=list(state.models.keys()),
    )


@app.get("/cars/{car_id}")
def get_car(car_id: str):
    row = state.cars_df[state.cars_df["car_id"] == car_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"car_id '{car_id}' not found")
    return row.iloc[0].to_dict()


@app.get("/users/{user_id}/profile")
def get_user_profile(user_id: str):
    if state.user_features_df.empty:
        raise HTTPException(status_code=503, detail="User features not available")
    row = state.user_features_df[state.user_features_df["user_id"] == user_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"user_id '{user_id}' not found")
    return row.iloc[0].to_dict()


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request):
    t0 = time.perf_counter()

    model = state.models[req.model]

    # apply optional filters to restrict candidate pool
    car_ids: Optional[set[str]] = None
    if req.filters:
        filtered = state.cars_df.copy()
        if req.filters.price_max is not None:
            filtered = filtered[filtered["price"] <= req.filters.price_max]
        if req.filters.price_min is not None:
            filtered = filtered[filtered["price"] >= req.filters.price_min]
        if req.filters.body_type is not None:
            filtered = filtered[
                filtered["body_type"].str.lower() == req.filters.body_type.lower()
            ]
        if req.filters.fuel_type is not None:
            filtered = filtered[
                filtered["fuel_type"].str.lower() == req.filters.fuel_type.lower()
            ]
        car_ids = set(filtered["car_id"].tolist())

    # ask for more candidates than needed so filtering doesn't leave us short
    fetch_k = req.k * 5 if car_ids is not None else req.k
    raw_recs = model.recommend(req.user_id, k=fetch_k, exclude_seen=True)

    # post-filter then truncate to req.k
    if car_ids is not None:
        raw_recs = [(cid, s) for cid, s in raw_recs if cid in car_ids][: req.k]
    else:
        raw_recs = raw_recs[: req.k]

    recommendations = [
        RecommendationItem(
            car_id=cid,
            score=round(score, 6),
            explanation=model.explain(req.user_id, cid),
        )
        for cid, score in raw_recs
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "recommend  user=%-10s  model=%-14s  k=%d  results=%d  %.1f ms",
        req.user_id, req.model, req.k, len(recommendations), latency_ms,
    )

    return RecommendResponse(
        user_id=req.user_id,
        model=req.model,
        recommendations=recommendations,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/metrics")
def get_metrics():
    if state.eval_results_df is None:
        raise HTTPException(
            status_code=503,
            detail="Evaluation results not found. Run autorec/eval/evaluator.py first.",
        )
    return state.eval_results_df.to_dict(orient="records")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
