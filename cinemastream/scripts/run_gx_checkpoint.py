# cinemastream/scripts/run_gx_checkpoint.py
"""
Run a Great Expectations ingestion gate against the real CinemaStream users export.

Builds the same feature set the churn model trains on (plan, country, tenure_months,
watch_minutes_avg, support_tickets, churned) from data/users.csv, data/watch_events.csv,
and data/support_tickets.csv, then validates it against the CinemaStream ingestion suite.

Writes a human-readable JSON summary of the run to reports/gx_checkpoint_result.json
(gitignored — regenerate by running this script).

Usage:
    python scripts/run_gx_checkpoint.py
"""

import json
from pathlib import Path

import great_expectations as gx
import pandas as pd

DATA_DIR    = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REFERENCE_DATE = pd.Timestamp("2026-06-01")

CINEMASTREAM_INGESTION_SUITE = [
    {"type": "expect_column_values_to_not_be_null", "args": ["user_id"]},
    {"type": "expect_column_values_to_be_unique",     "args": ["user_id"]},
    {"type": "expect_column_values_to_not_be_null",  "args": ["plan"]},
    {"type": "expect_column_values_to_be_in_set",    "args": ["plan"],
     "kwargs": {"value_set": ["Free", "Basic", "Premium"]}},
    {"type": "expect_column_values_to_be_in_set",    "args": ["country"],
     "kwargs": {"value_set": ["SG", "MY", "VN", "PH", "ID", "TH", "IN"]}},
    {"type": "expect_column_values_to_be_between",   "args": ["tenure_months"],
     "kwargs": {"min_value": 0, "max_value": 60}},
    {"type": "expect_column_values_to_not_be_null",  "args": ["watch_minutes_avg"]},
    {"type": "expect_column_values_to_be_between",   "args": ["watch_minutes_avg"],
     "kwargs": {"min_value": 0, "max_value": 300}},
    {"type": "expect_column_values_to_be_between",   "args": ["support_tickets"],
     "kwargs": {"min_value": 0, "max_value": 50}},
]


def build_feature_batch() -> pd.DataFrame:
    """Rebuild the churn-model feature table from the real CSV exports."""
    users   = pd.read_csv(DATA_DIR / "users.csv")
    watch   = pd.read_csv(DATA_DIR / "watch_events.csv")
    tickets = pd.read_csv(DATA_DIR / "support_tickets.csv")

    users["signup_date"]  = pd.to_datetime(users["signup_date"])
    users["tenure_months"] = ((REFERENCE_DATE - users["signup_date"]).dt.days / 30.44).round(1)

    watch_avg = watch.groupby("user_id")["watch_minutes"].mean().rename("watch_minutes_avg")
    ticket_counts = tickets.groupby("user_id").size().rename("support_tickets")

    df = (
        users
        .merge(watch_avg, on="user_id", how="left")
        .merge(ticket_counts, on="user_id", how="left")
    )
    df["watch_minutes_avg"] = df["watch_minutes_avg"]  # NaN preserved deliberately — real gap
    df["support_tickets"]   = df["support_tickets"].fillna(0)
    return df[["user_id", "plan", "country", "tenure_months",
               "watch_minutes_avg", "support_tickets", "churned"]]


def run_checkpoint(df: pd.DataFrame) -> dict:
    context = gx.get_context()
    ds = context.sources.add_pandas("cs_users_ingest")
    da = ds.add_dataframe_asset("users_batch")
    batch_req = da.build_batch_request(dataframe=df)
    validator = context.get_validator(batch_request=batch_req)

    for exp in CINEMASTREAM_INGESTION_SUITE:
        method = getattr(validator, exp["type"])
        method(*exp.get("args", []), **exp.get("kwargs", {}))

    results = validator.validate()

    failures = []
    for r in results.results:
        if not r.success:
            failures.append({
                "column": r.expectation_config.kwargs.get("column", "table"),
                "expectation": r.expectation_config.expectation_type,
                "unexpected_percent": r.result.get("unexpected_percent", 0.0),
                "unexpected_count": r.result.get("unexpected_count", 0),
            })

    return {
        "success": results.success,
        "rows_validated": len(df),
        "evaluated_expectations": results.statistics["evaluated_expectations"],
        "successful_expectations": results.statistics["successful_expectations"],
        "unsuccessful_expectations": results.statistics["unsuccessful_expectations"],
        "failures": failures,
    }


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_feature_batch()
    summary = run_checkpoint(df)

    out_path = REPORTS_DIR / "gx_checkpoint_result.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))

    print(f"Ingestion gate: {'PASS' if summary['success'] else 'FAIL'}")
    print(f"  {summary['successful_expectations']}/{summary['evaluated_expectations']} expectations passed")
    for f in summary["failures"]:
        print(f"  FAIL  {f['column']:20s}  {f['expectation']}  "
              f"({f['unexpected_percent']:.1f}% bad, {f['unexpected_count']} rows)")
    print(f"\nSummary written to {out_path}")
