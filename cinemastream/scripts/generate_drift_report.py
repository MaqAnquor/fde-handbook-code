# cinemastream/scripts/generate_drift_report.py
"""
Run an Evidently drift report comparing the earliest CinemaStream signup cohort
(reference / "training") against the most recent signup cohort (current /
"production") using the real users.csv, watch_events.csv, and support_tickets.csv data.

Writes the HTML drift report to reports/churn_drift_report.html (gitignored —
regenerate by running this script).

Usage:
    python scripts/generate_drift_report.py
"""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, MissingValueCount, ValueDrift

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
    return df.sort_values("signup_date").reset_index(drop=True)


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_feature_table()

    # Reference = earliest-signup half (the cohort the churn model trained on).
    # Current   = most recent-signup half (this month's production traffic).
    midpoint  = len(df) // 2
    reference = df.iloc[:midpoint]
    current   = df.iloc[midpoint:]

    report = Report(
        metrics=[
            ValueDrift(column="tenure_months"),
            ValueDrift(column="watch_minutes_avg"),
            ValueDrift(column="support_tickets"),
            DriftedColumnsCount(),
            MissingValueCount(column="watch_minutes_avg"),
        ],
        include_tests=True,
    )
    snapshot = report.run(reference_data=reference, current_data=current)

    out_path = REPORTS_DIR / "churn_drift_report.html"
    snapshot.save_html(str(out_path))

    d = snapshot.dict()
    print(f"CinemaStream Churn Feature Drift — earliest {len(reference)} signups "
          f"vs. most recent {len(current)} signups")
    print("=" * 70)
    for m in d["metrics"]:
        val  = m["value"]
        name = m["metric_name"].split("(")[0]
        if name == "ValueDrift":
            col      = m["metric_name"].split("column=")[1].split(",")[0]
            flag     = "DRIFT" if val < 0.05 else "  OK "
            ref_mean = float(reference[col].mean())
            cur_mean = float(current[col].mean())
            print(f"  {flag}  {col:20s}  p={val:.4f}  ref_mean={ref_mean:.1f}  cur_mean={cur_mean:.1f}")
        elif name == "DriftedColumnsCount":
            print(f"\n  Dataset drift summary: {val['count']:.0f}/3 columns drifted ({val['share']:.0%})")
        elif name == "MissingValueCount":
            print(f"  Null check watch_minutes_avg: {val['share']:.1%} missing")

    n_fail  = sum(1 for t in d["tests"] if t["status"].value == "FAIL")
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n  Gate: {verdict}  (failed_tests={n_fail}/{len(d['tests'])})")
    print(f"  HTML report: {out_path}")
