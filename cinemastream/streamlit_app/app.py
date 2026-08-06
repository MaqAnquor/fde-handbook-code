"""
CinemaStream "Ask Anything" dashboard — Chapter 60 MVP.

Run with:
    streamlit run cinemastream/streamlit_app/app.py

This first version is a read-only analytics dashboard over synthetic
CinemaStream data (same schema as the star schema from Ch048-051).
Chapter 61 will add the LLM-powered "Ask Anything" query box and a
naive RAG layer on top of this data. Chapter 62 adds deployment config.

Data is generated in-memory with a fixed random seed so every reader
sees the same numbers. In a real deployment, `load_data()` would read
from `cinemastream/data/*.csv` (Ch024) or the warehouse (Ch048-051).
"""

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants (mirrors bible_core.md canonical values)
# ---------------------------------------------------------------------------
COUNTRIES = ["SG", "MY", "ID", "PH", "TH", "VN", "IN"]
GENRES = ["Action", "Drama", "Comedy", "Documentary", "Thriller", "Romance"]
LANG_BY_COUNTRY = {"SG": "en", "MY": "ms", "ID": "id", "PH": "tl", "TH": "th", "VN": "vi", "IN": "hi"}
PLAN_PRICE_SGD = {"Free": 0.0, "Basic": 8.90, "Premium": 12.90}
DEVICES = ["Mobile", "TV", "Web", "Tablet"]


# ---------------------------------------------------------------------------
# Data generation (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def make_users_df(seed: int = 42) -> pd.DataFrame:
    """Build a 100-row users_df: 56 Free, 28 Basic, 16 Premium (bible split)."""
    rng = np.random.default_rng(seed)
    plans = ["Free"] * 56 + ["Basic"] * 28 + ["Premium"] * 16
    plans.remove("Premium")  # Ravi (user_id=1) takes one Premium slot
    plans.remove("Basic")    # Siti (user_id=2) takes one Basic slot
    plans.remove("Free")     # Minh (user_id=3) takes one Free slot
    rng.shuffle(plans)

    rows = [
        (1, "Ravi Kumar", "ravi@example.com", "IN", "hi", "Premium", "2024-01-15", False),
        (2, "Siti Rahman", "siti@example.com", "MY", "ms", "Basic", "2024-02-03", False),
        (3, "Nguyen Van Minh", "minh@example.com", "VN", "vi", "Free", "2024-03-22", True),
    ]
    for idx, plan in enumerate(plans, start=4):
        country = COUNTRIES[(idx * 3) % len(COUNTRIES)]
        signup = pd.Timestamp("2023-06-01") + pd.Timedelta(days=int(rng.integers(0, 500)))
        churned = bool(rng.random() < 0.18)
        rows.append((idx, f"User{idx:03d}", f"user{idx:03d}@example.com", country,
                      LANG_BY_COUNTRY[country], plan, signup.strftime("%Y-%m-%d"), churned))

    return pd.DataFrame(rows, columns=["user_id", "name", "email", "country", "language_pref",
                                         "plan", "signup_date", "churned"])


@st.cache_data
def make_movies_df(n_movies: int = 30) -> pd.DataFrame:
    """Build a 30-row movies_df, seeded with the bible's first 3 canonical titles."""
    rows = [
        (101, "Monsoon Heart", "hi", "Drama", 2024, 132,
         "A monsoon love story set in coastal Kerala."),
        (102, "Hujan di Singapura", "ms", "Thriller", 2023, 118,
         "A Singapore cybercrime unit races a deadline."),
        (103, "Office Hari Ini", "id", "Comedy", 2024, 96,
         "A Jakarta ad agency adjusts to permanent remote work."),
    ]
    titles_pool = [
        "Last Train to Hanoi", "Manila Nights", "Bangkok Static", "Quiet Tides",
        "The Durian Detective", "Songs from Sabah", "Curry & Code", "Mekong Drift",
        "Neon Temple", "Paper Boats", "The Last Recipe", "Kerala Skies",
        "Borneo Signal", "Cafe Aroy", "The Understudy", "Highway 19",
        "Glasshouse", "Echoes of Penang", "Midnight Ferry", "The Negotiator",
        "Salt and Monsoon", "Small Island Diary", "Sea of Memory", "Lagu Senja",
        "Cermin Retak", "The Long Layover", "Pulau", "Static Lines",
    ]
    for i, title in enumerate(titles_pool[: n_movies - 3], start=104):
        genre = GENRES[i % len(GENRES)]
        lang = list(LANG_BY_COUNTRY.values())[i % len(LANG_BY_COUNTRY)]
        year = 2018 + (i % 7)
        runtime = 85 + (i * 3) % 60
        rows.append((i, title, lang, genre, year, runtime, f"{title} -- a CinemaStream original."))

    return pd.DataFrame(rows, columns=["movie_id", "title", "original_lang", "genre",
                                         "release_year", "runtime_min", "description"])


@st.cache_data
def make_watch_events_df(users_df: pd.DataFrame, movies_df: pd.DataFrame,
                          n_events: int = 381, seed: int = 99) -> pd.DataFrame:
    """Build a 381-row events_df, seeded with the bible's first 3 canonical rows."""
    rng = np.random.default_rng(seed)
    user_ids = users_df["user_id"].to_numpy()
    movie_ids = movies_df["movie_id"].to_numpy()
    runtime_map = movies_df.set_index("movie_id")["runtime_min"]
    country_map = users_df.set_index("user_id")["country"]

    rows = [
        (1, 1, 101, "2024-04-01 14:45:00", 132, True, "TV", "IN"),
        (2, 1, 102, "2024-04-02 17:00:00", 45, False, "Mobile", "IN"),
        (3, 2, 103, "2024-04-03 11:30:00", 96, True, "TV", "MY"),
    ]
    base = pd.Timestamp("2024-04-01")
    for event_id in range(4, n_events + 1):
        uid = int(rng.choice(user_ids))
        mid = int(rng.choice(movie_ids))
        runtime = int(runtime_map[mid])
        ts = base + pd.Timedelta(days=int(rng.integers(0, 75)),
                                  hours=int(rng.integers(0, 24)),
                                  minutes=int(rng.integers(0, 60)))
        completed = bool(rng.random() < 0.55)
        watch_minutes = runtime if completed else int(rng.integers(5, max(6, runtime - 5)))
        device = DEVICES[int(rng.integers(0, len(DEVICES)))]
        country = country_map[uid]
        rows.append((event_id, uid, mid, ts.strftime("%Y-%m-%d %H:%M:%S"),
                      watch_minutes, completed, device, country))

    return pd.DataFrame(rows, columns=["event_id", "user_id", "movie_id", "watch_started",
                                         "watch_minutes", "completed", "device", "country"])


@st.cache_data
def load_data():
    users_df = make_users_df()
    movies_df = make_movies_df()
    events_df = make_watch_events_df(users_df, movies_df)
    return users_df, movies_df, events_df


# ---------------------------------------------------------------------------
# Page config + header
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CinemaStream Analytics", page_icon=":bar_chart:", layout="wide")

st.title("CinemaStream Analytics Dashboard")
st.caption("Internal tool -- prototype built with Wei Lin, Module 8 (Streamlit + Prompt Engineering)")

users_df, movies_df, events_df = load_data()

# ---------------------------------------------------------------------------
# Sidebar filters (st.session_state-backed)
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

if "selected_countries" not in st.session_state:
    st.session_state.selected_countries = COUNTRIES.copy()

selected_countries = st.sidebar.multiselect(
    "Country",
    options=COUNTRIES,
    default=st.session_state.selected_countries,
    key="selected_countries",
)

selected_plans = st.sidebar.multiselect(
    "Plan",
    options=["Free", "Basic", "Premium"],
    default=["Free", "Basic", "Premium"],
)

if st.sidebar.button("Reset filters"):
    st.session_state.selected_countries = COUNTRIES.copy()
    st.rerun()

# Guard: if the user deselects everything, fall back to "all" rather than
# showing an empty dashboard (a Wei Lin UX note from this chapter).
active_countries = selected_countries or COUNTRIES
active_plans = selected_plans or ["Free", "Basic", "Premium"]

filtered_users = users_df[
    users_df["country"].isin(active_countries) & users_df["plan"].isin(active_plans)
]
filtered_user_ids = set(filtered_users["user_id"])
filtered_events = events_df[events_df["user_id"].isin(filtered_user_ids)]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_users = len(filtered_users)
churn_rate = filtered_users["churned"].mean() * 100 if total_users else 0.0
mrr = (
    filtered_users.loc[~filtered_users["churned"], "plan"].map(PLAN_PRICE_SGD).sum()
)
total_watch_minutes = filtered_events["watch_minutes"].sum()

col1.metric("Users (filtered)", f"{total_users:,}")
col2.metric("Churn rate", f"{churn_rate:.1f}%")
col3.metric("MRR (S$, filtered)", f"S${mrr:,.2f}")
col4.metric("Total watch minutes", f"{total_watch_minutes:,}")

st.divider()

# ---------------------------------------------------------------------------
# Charts: top genres + plan mix
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Watch minutes by genre")
    genre_minutes = (
        filtered_events.merge(movies_df[["movie_id", "genre"]], on="movie_id")
        .groupby("genre")["watch_minutes"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(genre_minutes)

with right:
    st.subheader("Plan mix (filtered users)")
    plan_counts = filtered_users["plan"].value_counts()
    st.bar_chart(plan_counts)

st.divider()

# ---------------------------------------------------------------------------
# Drill-down table
# ---------------------------------------------------------------------------
st.subheader("Recent watch events (filtered)")

show_completed_only = st.checkbox("Completed sessions only", value=False)

# filtered_events already carries `country`; only pull name + plan from users_df
# (merging a second `country` would create country_x/country_y and drop plain `country`)
table = filtered_events.merge(
    movies_df[["movie_id", "title", "genre"]], on="movie_id"
).merge(
    users_df[["user_id", "name", "plan"]], on="user_id"
)

if show_completed_only:
    table = table[table["completed"]]

table = table.sort_values("watch_started", ascending=False).head(20)

st.dataframe(
    table[["watch_started", "name", "country", "plan", "title", "genre",
           "watch_minutes", "completed", "device"]],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Data: synthetic, generated in-memory (seeded) -- mirrors the star schema "
    "from Ch048-051. Chapter 61 connects this dashboard to the warehouse and "
    "adds an LLM-powered 'Ask Anything' box."
)
