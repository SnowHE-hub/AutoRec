from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, Engine, text

from autorec.db.schema import Base, DimUser, DimCar, DimDate, FactInteraction

# ---------------------------------------------------------------------------
# Switch: set USE_SQLITE=True to run without a real PostgreSQL instance.
# ---------------------------------------------------------------------------
USE_SQLITE: bool = True

BASE_DIR   = Path(__file__).resolve().parents[2]
PROCESSED  = BASE_DIR / "data" / "processed"
SQLITE_PATH = BASE_DIR / "data" / "autorec.db"
CHUNKSIZE  = 1_000


def _get_engine() -> Engine:
    if USE_SQLITE:
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(f"sqlite:///{SQLITE_PATH}", echo=False)
    # PostgreSQL — reads .env via connection.py
    from autorec.db.connection import get_engine
    return get_engine()


def _recreate_table(engine: Engine, table) -> None:
    """Drop then (re)create a single ORM-mapped table."""
    table.__table__.drop(engine, checkfirst=True)
    table.__table__.create(engine, checkfirst=True)


def _build_dim_date(start: str = "2022-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame({"full_date": dates})
    df["date_id"]     = df["full_date"].dt.strftime("%Y%m%d").astype(int)
    df["year"]        = df["full_date"].dt.year.astype("int16")
    df["quarter"]     = df["full_date"].dt.quarter.astype("int16")
    df["month"]       = df["full_date"].dt.month.astype("int16")
    df["day_of_week"] = df["full_date"].dt.weekday.astype("int16")  # 0=Monday
    df["is_weekend"]  = df["day_of_week"] >= 5
    df["full_date"]   = df["full_date"].dt.date
    return df[["date_id", "full_date", "year", "quarter", "month", "day_of_week", "is_weekend"]]


def _add_date_id(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df["date_id"] = df[ts_col].dt.strftime("%Y%m%d").astype(int)
    return df


def load_all() -> None:
    engine = _get_engine()

    # --- 1. dim_user ---
    print("\n[1/4] Loading dim_user ...")
    users = pd.read_csv(PROCESSED / "users.csv", parse_dates=["registration_date"])
    users["registration_date"] = users["registration_date"].dt.date
    _recreate_table(engine, DimUser)
    users.to_sql("dim_user", engine, if_exists="append", index=False, chunksize=CHUNKSIZE)
    print(f"      Written: {len(users):,} rows")

    # --- 2. dim_car ---
    print("\n[2/4] Loading dim_car ...")
    cars = pd.read_csv(PROCESSED / "cars.csv")
    _recreate_table(engine, DimCar)
    cars.to_sql("dim_car", engine, if_exists="append", index=False, chunksize=CHUNKSIZE)
    print(f"      Written: {len(cars):,} rows")

    # --- 3. dim_date (auto-generated) ---
    print("\n[3/4] Loading dim_date ...")
    dates = _build_dim_date()
    _recreate_table(engine, DimDate)
    dates.to_sql("dim_date", engine, if_exists="append", index=False, chunksize=CHUNKSIZE)
    print(f"      Written: {len(dates):,} rows")

    # --- 4. fact_interactions ---
    print("\n[4/4] Loading fact_interactions ...")
    interactions = pd.read_csv(PROCESSED / "interactions.csv")
    interactions = _add_date_id(interactions, ts_col="timestamp")
    interactions = interactions.rename(columns={"timestamp": "interaction_timestamp"})
    # keep only columns that match the ORM (drop any extras)
    keep = ["interaction_id", "user_id", "car_id", "date_id",
            "interaction_type", "interaction_timestamp"]
    interactions = interactions[keep]

    _recreate_table(engine, FactInteraction)
    interactions.to_sql(
        "fact_interactions", engine,
        if_exists="append", index=False, chunksize=CHUNKSIZE,
    )
    print(f"      Written: {len(interactions):,} rows")

    # --- Summary ---
    print("\n=== Load complete ===")
    backend = f"SQLite ({SQLITE_PATH})" if USE_SQLITE else "PostgreSQL"
    print(f"Backend : {backend}")

    with engine.connect() as conn:
        for tbl in ("dim_user", "dim_car", "dim_date", "fact_interactions"):
            n = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            print(f"  {tbl:<22}: {n:>7,} rows")


if __name__ == "__main__":
    load_all()
