from __future__ import annotations

import pandas as pd

# (column, min, max) — only checked when column is present in the dataframe
NUMERIC_RULES: dict[str, tuple[str, dict[str, tuple[float, float]]]] = {
    "users": {
        "age": (18, 75),
    },
    "cars": {
        "price": (1_000, 500_000),
        "mileage": (0, 500_000),
        "age_at_listing": (0, 50),
    },
    "interactions": {},
}

PRIMARY_KEYS = {
    "users": "user_id",
    "cars": "car_id",
    "interactions": "interaction_id",
}

NULL_THRESHOLD = 0.10  # 10 %


def _check_null_rates(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results = []
    for col in df.columns:
        rate = df[col].isna().mean()
        passed = rate < NULL_THRESHOLD
        results.append((
            f"null_rate[{col}]",
            passed,
            f"{rate:.2%} {'< 10% OK' if passed else '>= 10% FAIL'}",
        ))
    return results


def _check_pk_uniqueness(df: pd.DataFrame, name: str) -> tuple[str, bool, str]:
    pk = PRIMARY_KEYS.get(name)
    if pk is None or pk not in df.columns:
        return ("pk_uniqueness", True, "no PK defined - skipped")
    n_dupes = df[pk].duplicated().sum()
    passed = n_dupes == 0
    return (
        f"pk_uniqueness[{pk}]",
        passed,
        f"{n_dupes} duplicate(s) {'OK' if passed else 'FAIL'}",
    )


def _check_numeric_ranges(df: pd.DataFrame, name: str) -> list[tuple[str, bool, str]]:
    results = []
    rules = NUMERIC_RULES.get(name, {})
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        n_out = (~df[col].between(lo, hi)).sum()
        passed = n_out == 0
        results.append((
            f"range[{col}]",
            passed,
            f"{n_out} value(s) outside [{lo}, {hi}] {'OK' if passed else 'FAIL'}",
        ))
    return results


def run_checks(df: pd.DataFrame, name: str) -> bool:
    """Run all quality checks on *df* labelled *name*. Returns True if all pass."""
    checks: list[tuple[str, bool, str]] = []
    checks += _check_null_rates(df)
    checks.append(_check_pk_uniqueness(df, name))
    checks += _check_numeric_ranges(df, name)

    passed_all = all(c[1] for c in checks)
    status_line = "PASS" if passed_all else "FAIL"

    width = 60
    print(f"\n{'='*width}")
    print(f"  Quality report: {name.upper()}  [{status_line}]  ({len(df):,} rows)")
    print(f"{'='*width}")
    for check_name, passed, detail in checks:
        icon = "  PASS" if passed else "  FAIL"
        print(f"{icon}  {check_name:<35} {detail}")
    print(f"{'='*width}")
    return passed_all


# ---------------------------------------------------------------------------
# Full ETL pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from autorec.etl.transform import clean_cars, clean_users, clean_interactions

    BASE = Path(__file__).resolve().parents[2]
    RAW = BASE / "data" / "raw"
    PROCESSED = BASE / "data" / "processed"
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("\n>>> Loading raw data...")
    users_raw = pd.read_csv(RAW / "users.csv")
    cars_raw = pd.read_csv(RAW / "cars.csv")
    interactions_raw = pd.read_csv(RAW / "interactions.csv")
    print(f"  users       : {len(users_raw):,} rows")
    print(f"  cars        : {len(cars_raw):,} rows")
    print(f"  interactions: {len(interactions_raw):,} rows")

    print("\n>>> Cleaning...")
    users = clean_users(users_raw)
    cars = clean_cars(cars_raw)
    interactions = clean_interactions(
        interactions_raw,
        valid_user_ids=set(users["user_id"]),
        valid_car_ids=set(cars["car_id"]),
    )

    print("\n>>> Running quality checks...")
    all_passed = True
    all_passed &= run_checks(users, "users")
    all_passed &= run_checks(cars, "cars")
    all_passed &= run_checks(interactions, "interactions")

    print("\n>>> Saving processed data...")
    users.to_csv(PROCESSED / "users.csv", index=False)
    cars.to_csv(PROCESSED / "cars.csv", index=False)
    interactions.to_csv(PROCESSED / "interactions.csv", index=False)
    print(f"  users        → {PROCESSED / 'users.csv'} ({len(users):,} rows)")
    print(f"  cars         → {PROCESSED / 'cars.csv'} ({len(cars):,} rows)")
    print(f"  interactions → {PROCESSED / 'interactions.csv'} ({len(interactions):,} rows)")

    print(f"\n>>> ETL complete. Overall quality: {'PASS' if all_passed else 'FAIL'}")
