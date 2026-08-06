# cinemastream/scripts/generate_profile_report.py
"""
Run YData-Profiling on the real CinemaStream users export (joined with
watch_events.csv and support_tickets.csv into the same feature set the churn
model trains on) and print the alerts the report actually surfaced.

Writes the HTML report to reports/cinemastream_profile.html (gitignored —
regenerate by running this script).

Usage:
    python scripts/generate_profile_report.py
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from ydata_profiling import ProfileReport

DATA_DIR       = Path(__file__).parent.parent / "data"
REPORTS_DIR    = Path(__file__).parent.parent / "reports"
REFERENCE_DATE = pd.Timestamp("2026-06-01")


def build_feature_table() -> pd.DataFrame:
    """Same feature set the churn model trains on, built from the real CSV exports."""
    users   = pd.read_csv(DATA_DIR / "users.csv")
    watch   = pd.read_csv(DATA_DIR / "watch_events.csv")
    tickets = pd.read_csv(DATA_DIR / "support_tickets.csv")

    users["signup_date"]   = pd.to_datetime(users["signup_date"])
    users["tenure_months"] = ((REFERENCE_DATE - users["signup_date"]).dt.days / 30.44).round(1)

    watch_avg     = watch.groupby("user_id")["watch_minutes"].mean().rename("watch_minutes_avg")
    ticket_counts = tickets.groupby("user_id").size().rename("support_tickets")

    df = (
        users
        .merge(watch_avg, on="user_id", how="left")
        .merge(ticket_counts, on="user_id", how="left")
    )
    df["support_tickets"] = df["support_tickets"].fillna(0)
    return df[["user_id", "email", "plan", "country", "tenure_months",
               "watch_minutes_avg", "support_tickets", "churned"]]


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_feature_table()

    profile = ProfileReport(
        df,
        title="CinemaStream Users — Churn Feature Profile",
        sensitive=True,
        correlations={"pearson": {"calculate": True}, "cramers": {"calculate": True}},
    )

    out_path = REPORTS_DIR / "cinemastream_profile.html"
    profile.to_file(str(out_path))

    desc = profile.get_description()
    print(f"Dataset: {desc.table['n']} rows x {desc.table['n_var']} columns")
    print(f"Missing cells: {desc.table['p_cells_missing']:.1%}")
    print(f"Duplicate rows: {desc.table.get('n_duplicates', 0)}")
    print()
    print(f"Alerts ({len(desc.alerts)} total):")
    for alert in desc.alerts:
        print(f"  {alert}")
    print(f"\nHTML report: {out_path}")
