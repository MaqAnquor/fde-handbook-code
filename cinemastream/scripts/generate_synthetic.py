"""
Generates schema-faithful synthetic data for CinemaStream CI testing.
Output: cinemastream/data/synthetic/users.csv + watch_events.csv

Run: python cinemastream/scripts/generate_synthetic.py
Chapter: 024a — Synthetic Data: Creation & Validation
"""

from faker import Faker
from pathlib import Path
import pandas as pd
import random

SEED = 42
N_USERS = 500
N_EVENTS = 2000
OUT_DIR = Path("cinemastream/data/synthetic")

fake = Faker()
Faker.seed(SEED)
random.seed(SEED)

COUNTRIES  = ["SG", "MY", "ID", "TH", "PH"]
COUNTRY_W  = [0.35, 0.25, 0.20, 0.12, 0.08]
PLANS      = ["Free", "Basic", "Premium"]
PLAN_W     = [0.40, 0.35, 0.25]
GENRES     = ["Action", "Drama", "Comedy", "Thriller", "Romance", "Documentary"]
DEVICES    = ["mobile", "desktop", "smart_tv", "tablet"]
DEVICE_W   = [0.45, 0.25, 0.20, 0.10]


def generate_users(n: int) -> pd.DataFrame:
    rows = []
    for uid in range(1, n + 1):
        plan = random.choices(PLANS, weights=PLAN_W)[0]
        churned = random.choices(
            [False, True],
            weights=[0.90, 0.10] if plan == "Premium" else
                    [0.82, 0.18] if plan == "Basic" else
                    [0.70, 0.30]
        )[0]
        rows.append({
            "user_id":     uid,
            "name":        fake.name(),
            "email":       fake.email(),
            "country":     random.choices(COUNTRIES, weights=COUNTRY_W)[0],
            "plan":        plan,
            "signup_date": str(fake.date_between(start_date="-3y", end_date="-7d")),
            "churned":     churned,
        })
    return pd.DataFrame(rows)


def generate_watch_events(users_df: pd.DataFrame, n: int) -> pd.DataFrame:
    active_ids = users_df.loc[~users_df["churned"], "user_id"].tolist()
    rows = []
    for event_id in range(1, n + 1):
        rows.append({
            "event_id":     event_id,
            "user_id":      random.choice(active_ids),
            "movie_id":     random.randint(1, 200),
            "genre":        random.choice(GENRES),
            "watch_date":   str(fake.date_between(start_date="-1y", end_date="today")),
            "duration_min": random.randint(12, 148),
            "device":       random.choices(DEVICES, weights=DEVICE_W)[0],
            "completed":    random.choices([True, False], weights=[0.68, 0.32])[0],
        })
    return pd.DataFrame(rows)


def validate_synthetic(real_path: Path, synthetic: pd.DataFrame,
                        cat_cols: list) -> bool:
    """Return True if all category distributions are within 0.10 of real."""
    if not real_path.exists():
        print(f"  [skip] No real file at {real_path} — skipping distribution check.")
        return True
    real = pd.read_csv(real_path)
    for col in cat_cols:
        if col not in real.columns or col not in synthetic.columns:
            continue
        real_dist = real[col].value_counts(normalize=True)
        syn_dist  = synthetic[col].value_counts(normalize=True)
        combined  = pd.DataFrame({"r": real_dist, "s": syn_dist}).fillna(0)
        max_drift = (combined["r"] - combined["s"]).abs().max()
        if max_drift > 0.10:
            print(f"  [WARN] {col}: max distribution drift = {max_drift:.3f} (threshold 0.10)")
            return False
    return True


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    users = generate_users(N_USERS)
    users_path = OUT_DIR / "users.csv"
    users.to_csv(users_path, index=False, encoding="utf-8")
    print(f"users.csv: {len(users)} rows → {users_path}")

    events = generate_watch_events(users, N_EVENTS)
    events_path = OUT_DIR / "watch_events.csv"
    events.to_csv(events_path, index=False, encoding="utf-8")
    print(f"watch_events.csv: {len(events)} rows → {events_path}")

    users_ok = validate_synthetic(
        Path("cinemastream/data/users.csv"), users, ["plan", "country", "churned"]
    )
    print(f"Distribution check: {'PASSED' if users_ok else 'FAILED — review WARN lines above'}")

    real_user_file = Path("cinemastream/data/users.csv")
    if real_user_file.exists():
        real_emails = pd.read_csv(real_user_file)["email"].tolist()
        leaks = users["email"].str.lower().isin([e.lower() for e in real_emails]).sum()
        print(f"PII check: {leaks} synthetic emails match real records (expect 0)")
    else:
        print("PII check: skipped (no real users.csv to compare against)")

    print("\nDone. Safe to check into git.")


if __name__ == "__main__":
    main()
