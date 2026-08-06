# cinemastream/scripts/generate_sweetviz_report.py
"""
Run a sweetviz comparison between the earliest CinemaStream signup cohort and the
most recent signup cohort, using the same churn-model feature set built from the
real users.csv, watch_events.csv, and support_tickets.csv exports.

Writes the HTML report to reports/cinemastream_cohort_compare.html (gitignored —
regenerate by running this script).

Usage:
    python scripts/generate_sweetviz_report.py
"""

from pathlib import Path

import pandas as pd
import sweetviz as sv

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
    df["churned"] = df["churned"].astype(int)
    return df.sort_values("signup_date").reset_index(drop=True)


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_feature_table()

    # Earliest-signup half = the cohort the churn model trained on.
    # Most recent-signup half = this month's new users.
    midpoint = len(df) // 2
    earliest = df.iloc[:midpoint]
    recent   = df.iloc[midpoint:]

    cols = ["plan", "country", "tenure_months", "watch_minutes_avg",
            "support_tickets", "churned"]

    report = sv.compare(
        [earliest[cols], "Earliest signups"],
        [recent[cols], "Most recent signups"],
        target_feat="churned",
        pairwise_analysis="off",
    )
    out_path = REPORTS_DIR / "cinemastream_cohort_compare.html"
    report.show_html(str(out_path), open_browser=False)

    print(f"Earliest cohort: {len(earliest)} users, Most recent cohort: {len(recent)} users")
    print()
    for col in ["tenure_months", "watch_minutes_avg", "support_tickets"]:
        m1 = earliest[col].mean()
        m2 = recent[col].mean()
        delta = (m2 - m1) / m1 * 100
        flag = "DRIFT " if abs(delta) > 15 else "       "
        print(f"{flag}{col:20s}: earliest={m1:.1f}  recent={m2:.1f}  delta={delta:+.0f}%")

    print()
    print(f"Churn rate, earliest cohort: {earliest['churned'].mean():.1%}")
    print(f"Churn rate, recent cohort:   {recent['churned'].mean():.1%}")
    print(f"\nHTML report: {out_path}")
