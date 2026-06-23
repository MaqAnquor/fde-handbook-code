#!/usr/bin/env python3
"""
generate_data.py — THE single source of truth for CinemaStream's sample data.

This is the ONLY thing that defines the dataset. It is fully seeded, so running it
produces byte-identical CSVs every time. Chapters and notebooks READ these files;
they never write them. The committed CSVs in cinemastream/data/ are exactly this
script's output (a contract test enforces that).

Canonical facts (kept consistent with the book's documented numbers):
  - users.csv            100 rows, plan split 56 Free / 28 Basic / 16 Premium,
                         24 churned (24%), canonical rows 1-3 (Ravi/Siti/Minh)
  - movies.csv           300 rows (movie_id 101-400), multilingual, canonical 1-3
                         (Monsoon Heart / Hujan di Singapura / Office Hari Ini)
  - watch_events.csv     381 rows, canonical rows 1-3, FK-valid
  - ratings.csv          500 rows, FK-valid
  - subscriptions.csv    100 rows (one per user)
  - support_tickets.csv  150 rows, FK-valid
  - plan prices (SGD):   Free 0.00 / Basic 8.90 / Premium 12.90

Run:  python cinemastream/scripts/generate_data.py
"""
import csv
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Work both as a script (python cinemastream/scripts/generate_data.py) and as an
# import (from cinemastream.scripts.generate_data import generate).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cinemastream.scripts.make_movies_data import make_movies

SEED = 42

DATA = Path(__file__).resolve().parents[1] / "data"

# ── Canonical counts ──────────────────────────────────────────────────────────
N_USERS, N_EVENTS, N_RATINGS, N_TICKETS = 100, 381, 500, 150
N_CHURNED = 24                                  # 24% churn rate
PLAN_COUNTS = {"Free": 56, "Basic": 28, "Premium": 16}
MOVIE_IDS = list(range(101, 401))              # matches movies.csv (300 titles)

# ── Reference tables ──────────────────────────────────────────────────────────
COUNTRIES = ["SG", "MY", "ID", "PH", "TH", "VN", "IN"]
COUNTRY_W = [0.08, 0.20, 0.29, 0.09, 0.11, 0.16, 0.07]
LANG = {"SG": "en", "MY": "ms", "ID": "id", "PH": "tl", "TH": "th", "VN": "vi", "IN": "hi"}
CURRENCY = {"SG": "SGD", "MY": "MYR", "ID": "IDR", "PH": "PHP", "TH": "THB", "VN": "VND", "IN": "INR"}
FX = {"SGD": 1.0, "MYR": 3.5, "IDR": 12000.0, "PHP": 43.0, "THB": 27.0, "VND": 19000.0, "INR": 62.0}
PLAN_SGD = {"Free": 0.0, "Basic": 8.90, "Premium": 12.90}   # book canon
DEVICES = ["Mobile", "TV", "Web", "Tablet"]
DEVICE_W = [0.5, 0.2, 0.2, 0.1]

CANON_USERS = [
    (1, "Ravi Kumar", "ravi@example.com", "IN", "hi", "Premium", "2024-01-15", "FALSE"),
    (2, "Siti Rahman", "siti@example.com", "MY", "ms", "Basic", "2024-02-03", "FALSE"),
    (3, "Nguyen Van Minh", "minh@example.com", "VN", "vi", "Free", "2024-03-22", "TRUE"),
]
CANON_EVENTS = [
    [1, 1, 101, "2024-04-01 14:45:00", 132, "TRUE", "TV", "IN"],
    [2, 1, 102, "2024-04-02 17:00:00", 45, "FALSE", "Mobile", "IN"],
    [3, 2, 103, "2024-04-03 11:30:00", 96, "TRUE", "TV", "MY"],
]
FIRST = ["Aarav", "Mei", "Wei", "Putri", "Ananya", "Budi", "Somchai", "Linh", "Arjun",
         "Nurul", "Chai", "Dewi", "Kumar", "Hana", "Tuan", "Aishah", "Rahul", "Thanh",
         "Min", "Aung", "Farah", "Iqbal", "Lan", "Surya", "Mai"]
LAST = ["Tan", "Lim", "Wong", "Kumar", "Sharma", "Nguyen", "Pham", "Santos", "Reyes",
        "Putra", "Sari", "Rahman", "Iyer", "Menon", "Tran", "Le", "Cruz", "Bautista",
        "Wati", "Chen", "Goh", "Ng", "Hassan", "Devi", "Rai"]
CATEGORIES = ["Billing", "Playback", "Account", "Content", "Other"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
STATUSES = ["Open", "InProgress", "Resolved", "Closed"]
REVIEW_SNIPPETS = [
    "Loved it.", "Bagus sekali.", "Could be better.", "Tidak terlalu menarik.",
    "Great regional pick.", "Bohut acchi movie.", "Slow start, strong finish.",
    "Hay qua.", "Worth the watch.", "Mediocre plot.", "Pacing issues.", "Beautiful.",
]
TICKET_DESC = {
    "Billing": "Charged twice this month, please check.",
    "Playback": "Video keeps buffering on mobile.",
    "Account": "Cannot reset my password.",
    "Content": "When will regional-language titles be added?",
    "Other": "General feedback about the app.",
}
BASE = date(2024, 1, 1)


def _rdate(rng, start, end):
    return (start + timedelta(days=rng.randint(0, (end - start).days))).isoformat()


def _rts(rng, start, end):
    dt = datetime(start.year, start.month, start.day) + timedelta(
        seconds=rng.randint(0, int((end - start).total_seconds())))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _write(name, header, rows):
    with open(DATA / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name}: {len(rows)} rows")


def generate() -> None:
    """Generate all six canonical CSVs into cinemastream/data/. Deterministic."""
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # ---- movies (from the seeded movie generator) ----------------------------
    _write("movies.csv",
           ["movie_id", "title", "original_lang", "genre", "release_year", "runtime_min", "description"],
           make_movies())

    # ---- users : exact plan split + exact churn count ------------------------
    users = [list(u) for u in CANON_USERS]
    # canonical rows already consume: 1 Premium, 1 Basic, 1 Free
    remaining = (["Free"] * (PLAN_COUNTS["Free"] - 1)
                 + ["Basic"] * (PLAN_COUNTS["Basic"] - 1)
                 + ["Premium"] * (PLAN_COUNTS["Premium"] - 1))
    rng.shuffle(remaining)
    for offset, uid in enumerate(range(4, N_USERS + 1)):
        country = rng.choices(COUNTRIES, COUNTRY_W)[0]
        users.append([uid, f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                      f"user{uid}@example.com", country, LANG[country],
                      remaining[offset], _rdate(rng, BASE, date(2025, 6, 1)), "FALSE"])
    # churn: user 3 is canonically churned; pick the rest to reach N_CHURNED
    churn_ids = {3}
    pool = [u[0] for u in users if u[0] != 3]
    rng.shuffle(pool)
    churn_ids.update(pool[:N_CHURNED - 1])
    for u in users:
        u[7] = "TRUE" if u[0] in churn_ids else "FALSE"
    _write("users.csv",
           ["user_id", "name", "email", "country", "language_pref", "plan", "signup_date", "churned"],
           users)
    USER = {u[0]: u for u in users}

    # ---- watch_events --------------------------------------------------------
    events = [list(e) for e in CANON_EVENTS]
    for eid in range(len(events) + 1, N_EVENTS + 1):
        uid = rng.randint(1, N_USERS)
        minutes = rng.randint(2, 165)
        completed = "TRUE" if minutes >= 90 and rng.random() < 0.7 else "FALSE"
        events.append([eid, uid, rng.choice(MOVIE_IDS),
                       _rts(rng, date(2024, 4, 1), date(2025, 6, 30)), minutes,
                       completed, rng.choices(DEVICES, DEVICE_W)[0], USER[uid][3]])
    _write("watch_events.csv",
           ["event_id", "user_id", "movie_id", "watch_started", "watch_minutes", "completed", "device", "country"],
           events)

    # ---- ratings -------------------------------------------------------------
    ratings = [[rid, rng.randint(1, N_USERS), rng.choice(MOVIE_IDS), rng.randint(1, 5),
                rng.choice(REVIEW_SNIPPETS), _rdate(rng, date(2024, 4, 1), date(2025, 6, 30))]
               for rid in range(1, N_RATINGS + 1)]
    _write("ratings.csv",
           ["rating_id", "user_id", "movie_id", "stars", "review_text", "rating_date"], ratings)

    # ---- subscriptions (one per user) ----------------------------------------
    subs = []
    for uid in range(1, N_USERS + 1):
        u = USER[uid]
        cur = CURRENCY[u[3]]
        amount_sgd = PLAN_SGD[u[5]]
        end = "" if u[7] == "FALSE" else _rdate(rng, date(2024, 6, 1), date(2025, 6, 30))
        subs.append([uid, uid, u[5], u[6], end, round(amount_sgd * FX[cur], 2), cur, amount_sgd])
    _write("subscriptions.csv",
           ["subscription_id", "user_id", "plan", "start_date", "end_date", "amount_local", "currency", "amount_sgd"],
           subs)

    # ---- support_tickets -----------------------------------------------------
    tickets = []
    for tid in range(1, N_TICKETS + 1):
        cat = rng.choice(CATEGORIES)
        tickets.append([tid, rng.randint(1, N_USERS),
                        _rts(rng, date(2024, 5, 1), date(2025, 6, 30)), cat,
                        rng.choices(PRIORITIES, [0.4, 0.35, 0.2, 0.05])[0],
                        rng.choices(STATUSES, [0.2, 0.2, 0.3, 0.3])[0], TICKET_DESC[cat]])
    _write("support_tickets.csv",
           ["ticket_id", "user_id", "created_at", "category", "priority", "status", "description"],
           tickets)


if __name__ == "__main__":
    print("Generating CinemaStream canonical data (seed 42)...")
    generate()
    print("Done. SYNTHETIC, reproducible, FK-valid (users 1-100, movies 101-400).")
