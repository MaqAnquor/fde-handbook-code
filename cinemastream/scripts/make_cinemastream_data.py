#!/usr/bin/env python3
"""
make_cinemastream_data.py - generate the canonical CinemaStream sample datasets.

Creates users / watch_events / ratings / subscriptions / support_tickets, all
consistent with bible_core.md schema, the canonical first-3 rows, and the existing
movies.csv (movie_id 101-400). SYNTHETIC + flagged + reproducible (seed 42).
FK-valid by construction (user_id in 1..N_USERS, movie_id in 101..400).

Run:  python cinemastream/scripts/make_cinemastream_data.py
"""
import csv
import random
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(parents=True, exist_ok=True)

N_USERS, N_EVENTS, N_RATINGS, N_TICKETS = 200, 2000, 500, 150
MOVIE_IDS = list(range(101, 401))  # matches movies.csv

COUNTRIES = ["SG", "MY", "ID", "PH", "TH", "VN", "IN"]
COUNTRY_W = [0.14, 0.16, 0.18, 0.14, 0.10, 0.10, 0.18]
LANG = {"SG": "en", "MY": "ms", "ID": "id", "PH": "tl", "TH": "th", "VN": "vi", "IN": "hi"}
CURRENCY = {"SG": "SGD", "MY": "MYR", "ID": "IDR", "PH": "PHP", "TH": "THB", "VN": "VND", "IN": "INR"}
FX = {"SGD": 1.0, "MYR": 3.5, "IDR": 12000.0, "PHP": 43.0, "THB": 27.0, "VND": 19000.0, "INR": 62.0}
PLANS = ["Free", "Basic", "Premium"]
PLAN_W = [0.45, 0.35, 0.20]
PLAN_SGD = {"Free": 0.0, "Basic": 12.90, "Premium": 19.90}  # matches Ch070 canon
DEVICES = ["Mobile", "TV", "Web", "Tablet"]
DEVICE_W = [0.5, 0.2, 0.2, 0.1]
CHURN_P = {"SG": 0.03, "MY": 0.05, "ID": 0.05, "PH": 0.065, "TH": 0.05, "VN": 0.082, "IN": 0.04}

CANON_USERS = [
    (1, "Ravi Kumar", "ravi@example.com", "IN", "hi", "Premium", "2024-01-15", "FALSE"),
    (2, "Siti Rahman", "siti@example.com", "MY", "ms", "Basic", "2024-02-03", "FALSE"),
    (3, "Nguyen Van Minh", "minh@example.com", "VN", "vi", "Free", "2024-03-22", "TRUE"),
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


def rdate(start, end):
    return (start + timedelta(days=random.randint(0, (end - start).days))).isoformat()


def rts(start, end):
    dt = datetime(start.year, start.month, start.day) + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def write(name, header, rows):
    with open(DATA / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name}: {len(rows)} rows")


# ---- users -------------------------------------------------------------------
users = [list(u) for u in CANON_USERS]
used_emails = {u[2] for u in CANON_USERS}
for uid in range(4, N_USERS + 1):
    country = random.choices(COUNTRIES, COUNTRY_W)[0]
    name = f"{random.choice(FIRST)} {random.choice(LAST)}"
    email = f"user{uid}@example.com"
    plan = random.choices(PLANS, PLAN_W)[0]
    signup = rdate(BASE, date(2025, 6, 1))
    churned = "TRUE" if random.random() < CHURN_P[country] else "FALSE"
    users.append([uid, name, email, country, LANG[country], plan, signup, churned])
write("users.csv",
      ["user_id", "name", "email", "country", "language_pref", "plan", "signup_date", "churned"],
      users)
USER = {u[0]: u for u in users}

# ---- watch_events ------------------------------------------------------------
CANON_EVENTS = [
    [1, 1, 101, "2024-04-01 14:45:00", 132, "TRUE", "TV", "IN"],
    [2, 1, 102, "2024-04-02 17:00:00", 45, "FALSE", "Mobile", "IN"],
    [3, 2, 103, "2024-04-03 11:30:00", 96, "TRUE", "TV", "MY"],
]
events = [list(e) for e in CANON_EVENTS]
for eid in range(4, N_EVENTS + 1):
    uid = random.randint(1, N_USERS)
    country = USER[uid][3]
    minutes = random.randint(2, 165)
    completed = "TRUE" if minutes >= 90 and random.random() < 0.7 else "FALSE"
    events.append([eid, uid, random.choice(MOVIE_IDS),
                   rts(date(2024, 4, 1), date(2025, 6, 30)), minutes, completed,
                   random.choices(DEVICES, DEVICE_W)[0], country])
write("watch_events.csv",
      ["event_id", "user_id", "movie_id", "watch_started", "watch_minutes", "completed", "device", "country"],
      events)

# ---- ratings -----------------------------------------------------------------
ratings = []
for rid in range(1, N_RATINGS + 1):
    ratings.append([rid, random.randint(1, N_USERS), random.choice(MOVIE_IDS),
                    random.randint(1, 5), random.choice(REVIEW_SNIPPETS),
                    rdate(date(2024, 4, 1), date(2025, 6, 30))])
write("ratings.csv",
      ["rating_id", "user_id", "movie_id", "stars", "review_text", "rating_date"], ratings)

# ---- subscriptions -----------------------------------------------------------
subs = []
for uid in range(1, N_USERS + 1):
    u = USER[uid]
    plan, country = u[5], u[3]
    cur = CURRENCY[country]
    amount_sgd = PLAN_SGD[plan]
    amount_local = round(amount_sgd * FX[cur], 2)
    end = "" if u[7] == "FALSE" else rdate(date(2024, 6, 1), date(2025, 6, 30))
    subs.append([uid, uid, plan, u[6], end, amount_local, cur, amount_sgd])
write("subscriptions.csv",
      ["subscription_id", "user_id", "plan", "start_date", "end_date", "amount_local", "currency", "amount_sgd"],
      subs)

# ---- support_tickets ---------------------------------------------------------
tickets = []
for tid in range(1, N_TICKETS + 1):
    cat = random.choice(CATEGORIES)
    tickets.append([tid, random.randint(1, N_USERS),
                    rts(date(2024, 5, 1), date(2025, 6, 30)), cat,
                    random.choices(PRIORITIES, [0.4, 0.35, 0.2, 0.05])[0],
                    random.choices(STATUSES, [0.2, 0.2, 0.3, 0.3])[0],
                    TICKET_DESC[cat]])
write("support_tickets.csv",
      ["ticket_id", "user_id", "created_at", "category", "priority", "status", "description"],
      tickets)

print("Done. SYNTHETIC sample data (seed 42), FK-valid against movies.csv (101-400).")
