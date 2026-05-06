"""AutoRec — Automotive Recommendation System  (Streamlit UI)"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
API_BASE      = "http://localhost:8000"

sys.path.insert(0, str(ROOT))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoRec — Automotive Recommendation System",
    page_icon="🚗",
    layout="wide",
)

# ── cached data loaders ───────────────────────────────────────────────────────

@st.cache_data
def load_cars() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cars.csv")

@st.cache_data
def load_interactions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "interactions.csv")

@st.cache_data
def load_user_features() -> pd.DataFrame:
    p = FEATURES_DIR / "user_features.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_item_features() -> pd.DataFrame:
    p = FEATURES_DIR / "item_features.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_eval_results() -> Optional[pd.DataFrame]:
    p = PROCESSED_DIR / "eval_results.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_coldstart() -> Optional[pd.DataFrame]:
    p = PROCESSED_DIR / "eval_coldstart.csv"
    return pd.read_csv(p) if p.exists() else None

# ── API helpers with fallback ─────────────────────────────────────────────────

def _api_alive() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False

@st.cache_resource
def _get_fallback_models():
    """Load models directly (fallback when API is down)."""
    from autorec.models.popularity import PopularityRecommender
    from autorec.models.content_based import ContentBasedRecommender
    from autorec.models.matrix_factorization import ALSRecommender

    interactions  = load_interactions()
    item_features = load_item_features()
    cars          = load_cars()

    pop = PopularityRecommender().fit(interactions)
    cb  = ContentBasedRecommender().fit(interactions, item_features=item_features)
    cb.attach_car_meta(cars)
    als = ALSRecommender().fit(interactions)
    return {"popularity": pop, "content_based": cb, "als": als}

def recommend(user_id: str, model: str, k: int) -> List[Dict[str, Any]]:
    """Call API; fall back to local models if API is down."""
    if _api_alive():
        try:
            resp = requests.post(
                f"{API_BASE}/recommend",
                json={"user_id": user_id, "model": model, "k": k},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("recommendations", [])
        except Exception:
            pass

    # fallback
    models = _get_fallback_models()
    m = models[model]
    raw = m.recommend(user_id, k=k, exclude_seen=True)
    return [
        {"car_id": cid, "score": round(score, 6), "explanation": m.explain(user_id, cid)}
        for cid, score in raw
    ]

# ── colour palette ─────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Popularity":   "#636EFA",
    "ContentBased": "#EF553B",
    "ALS":          "#00CC96",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🚗 AutoRec")
    st.caption("Automotive Recommendation System")
    st.divider()

    page = st.radio(
        "Navigation",
        ["User Recommendations", "Model Comparison", "Data Overview"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Data Summary")
    try:
        cars_df   = load_cars()
        iact_df   = load_interactions()
        n_users   = iact_df["user_id"].nunique()
        n_cars    = len(cars_df)
        n_iact    = len(iact_df)
        col1, col2 = st.columns(2)
        col1.metric("Users",   f"{n_users:,}")
        col2.metric("Cars",    f"{n_cars:,}")
        st.metric("Interactions", f"{n_iact:,}")
        api_ok = _api_alive()
        st.caption(f"API: {'🟢 online' if api_ok else '🔴 offline (local fallback)'}")
    except Exception as e:
        st.warning(f"Could not load summary: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — USER RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

if page == "User Recommendations":
    st.title("🎯 User Recommendations")

    # ── controls ──────────────────────────────────────────────────────────────
    left, right = st.columns([1, 3])

    with left:
        st.subheader("Settings")
        user_num = st.slider("User ID number", 1, 5000, 1)
        user_id  = f"U{user_num:05d}"
        st.caption(f"Selected: **{user_id}**")
        k_choice = st.select_slider("Top-k", options=[5, 10, 20], value=10)
        compare  = st.toggle("Compare all 3 models", value=False)
        if not compare:
            model_choice = st.selectbox(
                "Model",
                ["popularity", "content_based", "als"],
                format_func=lambda x: x.replace("_", " ").title(),
            )

    # ── user profile ──────────────────────────────────────────────────────────
    with right:
        st.subheader("User Profile")
        uf = load_user_features()
        if not uf.empty:
            row = uf[uf["user_id"] == user_id]
            if not row.empty:
                r = row.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Test drives",  int(r.get("n_test_drives", 0)))
                c2.metric("Purchases",    int(r.get("n_purchases", 0)))
                c3.metric("Avg price",    f"${r.get('avg_price_interacted', 0):,.0f}")
                c4.metric("Recency (d)",  int(r.get("recency_days", 0)))
                c1.metric("Top brand",    str(r.get("brand_top1", "—")))
                c2.metric("2nd brand",    str(r.get("brand_top2", "—")))
                c3.metric("Pref. body",   str(r.get("preferred_body_type", "—")))
                c4.metric("Pref. tier",   str(r.get("preferred_price_tier", "—")))
            else:
                st.info(f"No feature data for {user_id}")
        else:
            st.warning("user_features.parquet not found — run user_features.py first")

        st.divider()

        # ── recommendations ───────────────────────────────────────────────────
        cars_df = load_cars()
        cars_lookup = cars_df.set_index("car_id").to_dict("index")

        models_to_show = (
            ["popularity", "content_based", "als"]
            if compare
            else [model_choice]
        )
        cols = st.columns(len(models_to_show))

        for col, model_name in zip(cols, models_to_show):
            with col:
                st.subheader(model_name.replace("_", " ").title())
                with st.spinner("Loading..."):
                    recs = recommend(user_id, model_name, k_choice)

                if not recs:
                    st.info("No recommendations returned.")
                else:
                    for rec in recs:
                        cid   = rec["car_id"]
                        score = rec["score"]
                        expl  = rec.get("explanation", "")
                        meta  = cars_lookup.get(cid, {})
                        with st.container(border=True):
                            st.markdown(
                                f"**{cid}** &nbsp; `{meta.get('make','').title()} "
                                f"{meta.get('year','')}`"
                            )
                            st.caption(
                                f"{meta.get('body_type','—')} · "
                                f"{meta.get('fuel_type','—')} · "
                                f"${meta.get('price', 0):,}"
                            )
                            tier = meta.get("price_tier", "—")
                            st.progress(
                                min(float(score), 1.0),
                                text=f"score {score:.4f} · {tier}",
                            )
                            st.caption(f"💡 {expl}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Model Comparison":
    st.title("📊 Model Comparison")

    eval_df = load_eval_results()
    if eval_df is None:
        st.error("eval_results.csv not found. Run `autorec/eval/evaluator.py` first.")
        st.stop()

    # ── overall table ─────────────────────────────────────────────────────────
    st.subheader("Overall Evaluation Results")

    def _highlight_max(s: pd.Series) -> list[str]:
        is_max = s == s.max()
        return ["background-color: #d4edda; font-weight:bold" if v else "" for v in is_max]

    metric_cols = [c for c in eval_df.columns if c != "Model"]
    styled = (
        eval_df.set_index("Model")[metric_cols]
        .style
        .apply(_highlight_max, axis=0)
        .format("{:.4f}")
    )
    st.dataframe(styled, use_container_width=True)

    # ── bar charts ────────────────────────────────────────────────────────────
    st.subheader("Metric Comparison")
    bar_metrics = ["Recall@10", "NDCG@10", "HitRate@10", "Coverage"]
    bar_metrics = [m for m in bar_metrics if m in eval_df.columns]

    bcols = st.columns(len(bar_metrics))
    for col, metric in zip(bcols, bar_metrics):
        fig = px.bar(
            eval_df,
            x="Model",
            y=metric,
            color="Model",
            color_discrete_map=MODEL_COLORS,
            text_auto=".4f",
            title=metric,
            height=320,
        )
        fig.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        fig.update_traces(textposition="outside")
        col.plotly_chart(fig, use_container_width=True)

    # ── diversity separate (different scale) ─────────────────────────────────
    if "Diversity" in eval_df.columns:
        fig_div = px.bar(
            eval_df, x="Model", y="Diversity",
            color="Model", color_discrete_map=MODEL_COLORS,
            text_auto=".4f", title="Diversity (intra-list)", height=320,
        )
        fig_div.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        fig_div.update_traces(textposition="outside")
        st.plotly_chart(fig_div, use_container_width=True)

    # ── cold-start heatmap ────────────────────────────────────────────────────
    st.subheader("Cold-start Analysis — Recall@10 by User Activity")
    cold_df = load_coldstart()
    if cold_df is not None:
        cold_df = cold_df.set_index("group") if "group" in cold_df.columns else cold_df
        model_cols = [c for c in cold_df.columns if c != cold_df.index.name]
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=cold_df[model_cols].values,
                x=model_cols,
                y=cold_df.index.tolist(),
                colorscale="Blues",
                text=cold_df[model_cols].round(4).values,
                texttemplate="%{text}",
                textfont={"size": 14},
                showscale=True,
                colorbar=dict(title="Recall@10"),
            )
        )
        fig_heat.update_layout(
            xaxis_title="Model",
            yaxis_title="Training interactions",
            height=320,
            margin=dict(t=20, b=40, l=80, r=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("eval_coldstart.csv not found. Run evaluator.py to generate it.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Data Overview":
    st.title("🔍 Data Overview")

    cars_df = load_cars()
    iact_df = load_interactions()

    row1_l, row1_r = st.columns(2)

    # ── price distribution ────────────────────────────────────────────────────
    with row1_l:
        st.subheader("Car Price Distribution")
        fig_price = px.histogram(
            cars_df, x="price", nbins=60,
            color_discrete_sequence=["#636EFA"],
            labels={"price": "Price (USD)"},
            height=350,
        )
        fig_price.update_layout(margin=dict(t=10, b=40, l=40, r=10))
        st.plotly_chart(fig_price, use_container_width=True)

    # ── top-10 brands by interaction count ───────────────────────────────────
    with row1_r:
        st.subheader("Top 10 Brands by Interactions")
        brand_counts = (
            iact_df.merge(cars_df[["car_id", "make"]], on="car_id", how="left")
            .groupby("make")
            .size()
            .nlargest(10)
            .reset_index(name="count")
            .sort_values("count")
        )
        fig_brand = px.bar(
            brand_counts, x="count", y="make",
            orientation="h",
            color="count",
            color_continuous_scale="Blues",
            labels={"make": "Brand", "count": "Interactions"},
            height=350,
        )
        fig_brand.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=40, l=10, r=10),
        )
        st.plotly_chart(fig_brand, use_container_width=True)

    row2_l, row2_r = st.columns(2)

    # ── user activity distribution ────────────────────────────────────────────
    with row2_l:
        st.subheader("User Activity Distribution")
        user_act = (
            iact_df.groupby("user_id")
            .size()
            .reset_index(name="n_interactions")
        )
        bins   = [0, 5, 10, 15, 20, 30, 50, 200]
        labels = ["1-5", "6-10", "11-15", "16-20", "21-30", "31-50", "50+"]
        user_act["bucket"] = pd.cut(
            user_act["n_interactions"], bins=bins, labels=labels, right=True
        )
        bucket_counts = (
            user_act.groupby("bucket", observed=True)
            .size()
            .reset_index(name="users")
        )
        fig_act = px.bar(
            bucket_counts, x="bucket", y="users",
            color_discrete_sequence=["#EF553B"],
            labels={"bucket": "Interactions per user", "users": "Users"},
            height=350,
        )
        fig_act.update_layout(margin=dict(t=10, b=40, l=40, r=10))
        st.plotly_chart(fig_act, use_container_width=True)

    # ── purchase conversion by price_tier ─────────────────────────────────────
    with row2_r:
        st.subheader("Purchase Conversion Rate by Price Tier")
        merged = iact_df.merge(cars_df[["car_id", "price_tier"]], on="car_id", how="left")
        conv = (
            merged.groupby(["price_tier", "interaction_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        if "test_drive" in conv.columns and "purchase" in conv.columns:
            conv["conversion_rate"] = (
                conv["purchase"] / conv["test_drive"].replace(0, float("nan"))
            )
            tier_order = ["budget", "mid", "premium", "luxury"]
            conv["price_tier"] = pd.Categorical(
                conv["price_tier"], categories=tier_order, ordered=True
            )
            conv = conv.sort_values("price_tier")
            fig_conv = px.bar(
                conv, x="price_tier", y="conversion_rate",
                color="price_tier",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"price_tier": "Price Tier", "conversion_rate": "Conversion Rate"},
                text_auto=".2%",
                height=350,
            )
            fig_conv.update_layout(
                showlegend=False,
                yaxis_tickformat=".1%",
                margin=dict(t=10, b=40, l=50, r=10),
            )
            fig_conv.update_traces(textposition="outside")
            st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("Not enough interaction types to compute conversion rate.")
