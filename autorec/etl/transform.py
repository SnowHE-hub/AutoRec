import pandas as pd


def clean_cars(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["price"].between(1000, 500_000)].copy()
    removed = before - len(df)
    if removed:
        print(f"  [cars] dropped {removed} rows with price out of [1000, 500000]")

    df["make"] = df["make"].str.strip().str.lower()
    df["model"] = df["model"].str.strip().str.lower()
    df["age_at_listing"] = 2024 - df["year"]
    return df.reset_index(drop=True)


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna().copy()
    dropped_null = before - len(df)
    if dropped_null:
        print(f"  [users] dropped {dropped_null} rows with null values")

    before = len(df)
    df = df[df["age"].between(18, 75)]
    dropped_age = before - len(df)
    if dropped_age:
        print(f"  [users] dropped {dropped_age} rows with age out of [18, 75]")

    return df.reset_index(drop=True)


def clean_interactions(
    df: pd.DataFrame,
    valid_user_ids: set,
    valid_car_ids: set,
) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "car_id", "timestamp"]).copy()
    dropped_dup = before - len(df)
    if dropped_dup:
        print(f"  [interactions] dropped {dropped_dup} duplicate rows")

    before = len(df)
    df = df[df["user_id"].isin(valid_user_ids) & df["car_id"].isin(valid_car_ids)]
    dropped_fk = before - len(df)
    if dropped_fk:
        print(f"  [interactions] dropped {dropped_fk} rows with invalid foreign keys")

    return df.reset_index(drop=True)
