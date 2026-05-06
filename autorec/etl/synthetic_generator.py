import numpy as np
import pandas as pd
from faker import Faker
from pathlib import Path
import random

fake = Faker("zh_CN")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

MAKES = ["Toyota", "Honda", "BMW", "Mercedes", "Ford", "Tesla", "Audi", "Hyundai", "Nissan", "Volkswagen"]
BODY_TYPES = ["Sedan", "SUV", "Hatchback", "Coupe", "Pickup", "Minivan"]
FUEL_TYPES = ["Gasoline", "Diesel", "Electric", "Hybrid"]
TRANSMISSIONS = ["Automatic", "Manual", "CVT"]

# income → price_tier affinity weights (budget, mid, premium, luxury)
AFFINITY = {
    "low":  [0.55, 0.35, 0.08, 0.02],
    "mid":  [0.20, 0.45, 0.28, 0.07],
    "high": [0.05, 0.15, 0.45, 0.35],
}

# monthly seasonality multipliers (index 0 = Jan)
SEASON_WEIGHTS = [0.8, 0.8, 1.3, 0.9, 0.9, 0.9, 0.9, 0.9, 1.3, 1.0, 1.0, 1.0]


def generate_users(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    ages_raw = rng.normal(35, 12, n * 3)
    ages = ages_raw[(ages_raw >= 18) & (ages_raw <= 75)][:n].astype(int)

    income_brackets = rng.choice(
        ["low", "mid", "high"], size=n, p=[0.30, 0.50, 0.20]
    )
    reg_dates = pd.to_datetime(
        rng.integers(
            pd.Timestamp("2018-01-01").value,
            pd.Timestamp("2023-12-31").value,
            n,
        )
    ).normalize()

    return pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(1, n + 1)],
            "age": ages,
            "gender": rng.choice(["M", "F"], size=n, p=[0.52, 0.48]),
            "income_bracket": income_brackets,
            "city": [fake.city() for _ in range(n)],
            "registration_date": reg_dates,
        }
    )


def generate_cars(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    makes = rng.choice(MAKES, size=n)
    years = rng.integers(2010, 2025, size=n)

    # price shaped by make prestige
    prestige = {
        "Toyota": (18000, 8000), "Honda": (17000, 7000), "Hyundai": (16000, 6000),
        "Nissan": (17000, 7000), "Ford": (22000, 10000), "Volkswagen": (21000, 8000),
        "Audi": (45000, 18000), "BMW": (50000, 20000), "Mercedes": (55000, 22000),
        "Tesla": (52000, 20000),
    }
    prices = np.array(
        [max(5000, int(rng.normal(*prestige[m]))) for m in makes], dtype=int
    )

    q = np.percentile(prices, [25, 50, 75])
    price_tier = np.where(
        prices <= q[0], "budget",
        np.where(prices <= q[1], "mid",
                 np.where(prices <= q[2], "premium", "luxury")),
    )

    models = [f"{m}-{fake.bothify(text='??##').upper()}" for m in makes]

    return pd.DataFrame(
        {
            "car_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "make": makes,
            "model": models,
            "year": years,
            "price": prices,
            "mileage": rng.integers(0, 150000, size=n),
            "body_type": rng.choice(BODY_TYPES, size=n),
            "fuel_type": rng.choice(FUEL_TYPES, size=n, p=[0.55, 0.20, 0.15, 0.10]),
            "transmission": rng.choice(TRANSMISSIONS, size=n, p=[0.65, 0.20, 0.15]),
            "price_tier": price_tier,
        }
    )


def _sample_timestamp(rng: np.random.Generator) -> pd.Timestamp:
    """Sample a timestamp in [2022-01-01, 2024-12-31] with seasonal bias."""
    month = rng.choice(range(1, 13), p=np.array(SEASON_WEIGHTS) / sum(SEASON_WEIGHTS))
    year = int(rng.choice([2022, 2023, 2024]))
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = rng.integers(1, max_day + 1)
    hour = rng.integers(8, 21)
    minute = rng.integers(0, 60)
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)


def generate_interactions(
    users: pd.DataFrame,
    cars: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tier_order = ["budget", "mid", "premium", "luxury"]
    cars_by_tier = {t: cars[cars["price_tier"] == t]["car_id"].tolist() for t in tier_order}

    records = []
    interaction_id = 1

    for _, user in users.iterrows():
        uid = user["user_id"]
        income = user["income_bracket"]
        weights = AFFINITY[income]

        n_test_drives = int(rng.integers(3, 16))
        n_purchases = int(rng.integers(0, 3))

        # sample tiers for this user's test drives
        tiers_td = rng.choice(tier_order, size=n_test_drives, p=weights)
        td_cars = []
        for tier in tiers_td:
            pool = cars_by_tier[tier]
            if not pool:
                continue
            td_cars.append(rng.choice(pool))

        td_timestamps = sorted(
            [_sample_timestamp(rng) for _ in td_cars]
        )

        for car_id, ts in zip(td_cars, td_timestamps):
            records.append(
                {
                    "interaction_id": f"I{interaction_id:07d}",
                    "user_id": uid,
                    "car_id": car_id,
                    "interaction_type": "test_drive",
                    "timestamp": ts,
                }
            )
            interaction_id += 1

        # purchases must be among test-driven cars, after their test-drive ts
        if td_cars and n_purchases > 0:
            n_purchases = min(n_purchases, len(td_cars))
            purchase_indices = rng.choice(len(td_cars), size=n_purchases, replace=False)
            for idx in purchase_indices:
                car_id = td_cars[idx]
                td_ts = td_timestamps[idx]
                delta_days = int(rng.integers(1, 60))
                purchase_ts = td_ts + pd.Timedelta(days=delta_days)
                if purchase_ts.year > 2024:
                    purchase_ts = pd.Timestamp("2024-12-31 18:00")
                records.append(
                    {
                        "interaction_id": f"I{interaction_id:07d}",
                        "user_id": uid,
                        "car_id": car_id,
                        "interaction_type": "purchase",
                        "timestamp": purchase_ts,
                    }
                )
                interaction_id += 1

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    # re-assign sequential IDs after sort
    df["interaction_id"] = [f"I{i:07d}" for i in range(1, len(df) + 1)]
    return df


if __name__ == "__main__":
    print("=== Generating users ===")
    users = generate_users(5000)
    users_path = RAW_DIR / "users.csv"
    users.to_csv(users_path, index=False)
    print(f"Saved {len(users)} rows → {users_path}")
    print(users.head(3).to_string(), "\n")

    print("=== Generating cars ===")
    cars = generate_cars(3000)
    cars_path = RAW_DIR / "cars.csv"
    cars.to_csv(cars_path, index=False)
    print(f"Saved {len(cars)} rows → {cars_path}")
    print(cars.head(3).to_string(), "\n")

    print("=== Generating interactions ===")
    interactions = generate_interactions(users, cars)
    interactions_path = RAW_DIR / "interactions.csv"
    interactions.to_csv(interactions_path, index=False)
    print(f"Saved {len(interactions)} rows → {interactions_path}")
    print(interactions.head(3).to_string(), "\n")

    print("=== Summary ===")
    print(f"Users      : {len(users):>6,}")
    print(f"Cars       : {len(cars):>6,}")
    print(f"Interactions: {len(interactions):>6,}")
    td = (interactions["interaction_type"] == "test_drive").sum()
    pu = (interactions["interaction_type"] == "purchase").sum()
    print(f"  test_drive: {td:,}  |  purchase: {pu:,}")
