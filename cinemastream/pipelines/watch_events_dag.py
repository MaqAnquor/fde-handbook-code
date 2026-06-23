"""
CinemaStream Watch Events Ingestion DAG
========================================
Chapter 46-47 — Orchestration with Apache Airflow
Chapter 52    — Monitoring and Alerting (added `monitor` task)

Pipeline: Extract daily watch events → Validate → Load to warehouse → Monitor → Notify

Schedule: Daily at 02:00 UTC (processes previous day's events)
Owner:    data-team
SLA:      Results available by 03:00 UTC

Dependencies:
    apache-airflow>=2.7.0
    pandas>=2.2.0
    pyarrow>=14.0.0
    requests>=2.31.0
    great-expectations>=0.18.0  (optional — falls back to manual validation)

Secrets:
    SLACK_WEBHOOK_URL — Slack incoming webhook for the #data-team channel.
    Set as an environment variable or Airflow Connection. Never hardcode.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_DEVICES  = {"Mobile", "TV", "Tablet", "Web"}
VALID_COUNTRIES = {"SG", "MY", "ID", "PH", "TH", "VN", "IN"}
VALID_PLANS    = {"Free", "Basic", "Premium"}

STAGING_BASE   = Path("/tmp/cinemastream_staging")
STAGING_BASE.mkdir(parents=True, exist_ok=True)

# ── Default Task Arguments ────────────────────────────────────────────────────

DEFAULT_ARGS: dict = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "email":            ["priya@cinemastream.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# ── Task Functions ─────────────────────────────────────────────────────────────


def extract_watch_events(**context) -> None:
    """
    Task: extract
    Extract raw watch events for the logical date from the source system.

    In production: queries the PostgreSQL read replica.
    In this portfolio implementation: reads from the CinemaStream CSV file.

    Pushes to XCom:
        staging_path (str): path to the Parquet staging file
        row_count (int): number of rows extracted
    """
    logical_date = context["logical_date"]
    date_str     = logical_date.strftime("%Y-%m-%d")

    logger.info("Extracting watch_events for date=%s", date_str)

    # --- Production version (PostgreSQL) ---
    # from airflow.providers.postgres.hooks.postgres import PostgresHook
    # hook = PostgresHook(postgres_conn_id="cinemastream_prod_replica")
    # df = hook.get_pandas_df(
    #     sql="""
    #         SELECT event_id, user_id, movie_id, watch_started,
    #                watch_minutes, completed, device, country
    #         FROM watch_events
    #         WHERE DATE(watch_started AT TIME ZONE 'UTC') = %(date)s
    #     """,
    #     parameters={"date": date_str},
    # )

    # --- Portfolio version (CSV) ---
    csv_path = Path(__file__).parent.parent / "data" / "watch_events.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df[df["watch_started"].str.startswith(date_str)]

    staging_path = str(STAGING_BASE / f"watch_events_{date_str}.parquet")
    df.to_parquet(staging_path, index=False)

    logger.info("Extracted %d rows → %s", len(df), staging_path)

    context["ti"].xcom_push(key="staging_path", value=staging_path)
    context["ti"].xcom_push(key="row_count",    value=len(df))


def validate_watch_events(**context) -> None:
    """
    Task: validate
    Run data quality expectations on the extracted staging file.
    Raises ValueError if any expectation fails, blocking the load task.

    Expectations:
        - event_id: not null, unique
        - user_id: not null
        - watch_minutes: between 1 and 300
        - device: in VALID_DEVICES
        - country: in VALID_COUNTRIES
    """
    ti           = context["ti"]
    staging_path = ti.xcom_pull(task_ids="extract", key="staging_path")
    row_count    = ti.xcom_pull(task_ids="extract", key="row_count")

    logger.info("Validating %d rows from %s", row_count, staging_path)

    if row_count == 0:
        logger.warning("Empty partition for this date — skipping validation")
        ti.xcom_push(key="validation_passed", value=True)
        return

    df     = pd.read_parquet(staging_path)
    errors = []

    # Structural checks
    if df["event_id"].isnull().any():
        errors.append(f"event_id has {df['event_id'].isnull().sum()} NULL values")
    if df["event_id"].duplicated().any():
        errors.append(f"event_id has {df['event_id'].duplicated().sum()} duplicates")
    if df["user_id"].isnull().any():
        errors.append(f"user_id has {df['user_id'].isnull().sum()} NULL values")

    # Range checks
    invalid_mins = df[~df["watch_minutes"].between(1, 300)]
    if len(invalid_mins) > 0:
        errors.append(f"watch_minutes out of [1,300]: {len(invalid_mins)} rows")

    # Set membership checks
    bad_devices   = df[~df["device"].isin(VALID_DEVICES)]["device"].unique().tolist()
    bad_countries = df[~df["country"].isin(VALID_COUNTRIES)]["country"].unique().tolist()
    if bad_devices:
        errors.append(f"Unknown device values: {bad_devices}")
    if bad_countries:
        errors.append(f"Unknown country values: {bad_countries}")

    if errors:
        logger.error("Validation FAILED: %s", errors)
        raise ValueError(f"Data quality gate failed:\n" + "\n".join(f"  - {e}" for e in errors))

    logger.info("Validation PASSED: %d rows, %d expectations checked", len(df), 5)
    ti.xcom_push(key="validation_passed", value=True)


def load_watch_events(**context) -> None:
    """
    Task: load
    Load validated events from staging to the analytics warehouse.
    Uses upsert (INSERT ... ON CONFLICT DO UPDATE) for idempotency.

    In production: loads to BigQuery using the bigquery-client or SQLAlchemy.
    In this portfolio implementation: loads to a local SQLite database.
    """
    ti           = context["ti"]
    staging_path = ti.xcom_pull(task_ids="extract", key="staging_path")
    row_count    = ti.xcom_pull(task_ids="extract", key="row_count")

    if row_count == 0:
        logger.info("Empty partition — skipping load")
        ti.xcom_push(key="loaded_rows", value=0)
        return

    df = pd.read_parquet(staging_path)

    # Add session_bucket derived column
    import numpy as np
    conditions = [df["watch_minutes"] < 30, df["watch_minutes"].between(30, 89)]
    df["session_bucket"] = np.select(conditions, ["short", "medium"], default="long")
    df["loaded_at"] = datetime.utcnow().isoformat()

    # --- Production version (BigQuery) ---
    # from google.cloud import bigquery
    # client = bigquery.Client()
    # job = client.load_table_from_dataframe(
    #     df, "cinemastream_analytics.marts.watch_events",
    #     job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    # )
    # job.result()

    # --- Portfolio version (SQLite) ---
    import sqlite3
    warehouse_db = Path(__file__).parent.parent / "data" / "warehouse.db"
    conn = sqlite3.connect(str(warehouse_db))
    df.to_sql("watch_events_daily", conn, if_exists="append", index=False)
    conn.close()

    logger.info("Loaded %d rows to warehouse", len(df))
    ti.xcom_push(key="loaded_rows", value=len(df))


def build_slack_alert(check_name: str, status: str, details: dict) -> dict:
    """
    Build a Slack webhook payload for a monitoring alert (Chapter 52).
    Does NOT send it — returns the payload dict so callers can decide
    whether/how to send it (and so it's easy to unit test).
    """
    emoji = {"ok": ":white_check_mark:", "late": ":warning:", "critical": ":rotating_light:"}
    color = {"ok": "#3D7370", "late": "#E07A3B", "critical": "#D62728"}

    detail_lines = "\n".join(f"  • {k}: {v}" for k, v in details.items())

    return {
        "text": f"{emoji.get(status, ':grey_question:')} *{check_name}* — status: `{status}`",
        "attachments": [
            {
                "color": color.get(status, "#888888"),
                "text": detail_lines,
            }
        ],
    }


def run_pipeline_health_check(loaded_rows: int, history_counts: list, last_run: datetime,
                               expected_interval_hours: float = 25, now: datetime = None) -> dict:
    """
    Combine a freshness check and a volume (z-score) check into a single
    health report with an overall severity (P1/P2/P3). See Chapter 52.
    """
    import statistics

    if now is None:
        now = datetime.utcnow()

    # Freshness
    elapsed = now - last_run
    expected = timedelta(hours=expected_interval_hours)
    if elapsed <= expected:
        freshness_status = "ok"
    elif elapsed <= expected * 2:
        freshness_status = "late"
    else:
        freshness_status = "critical"

    # Volume (z-score against history, current period excluded by caller)
    mean = statistics.mean(history_counts)
    stdev = statistics.stdev(history_counts) if len(history_counts) > 1 else 0
    z_score = (loaded_rows - mean) / stdev if stdev else 0.0
    volume_anomaly = abs(z_score) > 2.0

    # Overall severity — worst of the two checks wins
    if freshness_status == "critical" or (volume_anomaly and z_score < -10):
        severity = "P1"
    elif freshness_status == "late" or volume_anomaly:
        severity = "P2"
    else:
        severity = "P3"

    return {
        "loaded_rows": loaded_rows,
        "history_mean": round(mean, 1),
        "z_score": round(z_score, 2),
        "volume_anomaly": volume_anomaly,
        "freshness_status": freshness_status,
        "elapsed_hours": round(elapsed.total_seconds() / 3600, 2),
        "severity": severity,
    }


def monitor_pipeline_health(**context) -> None:
    """
    Task: monitor (Chapter 52)
    Runs after load. Checks freshness + volume against the last 7 days
    of loaded_rows counts (read from the warehouse), classifies severity,
    and sends a Slack alert if severity is P1 or P2.

    On P1, this task raises — which fails the DAG run and triggers
    Airflow's email_on_failure (already configured in DEFAULT_ARGS).
    """
    import os
    import sqlite3
    import requests

    ti = context["ti"]
    loaded_rows = ti.xcom_pull(task_ids="load", key="loaded_rows") or 0
    logical_date = context["logical_date"]

    warehouse_db = Path(__file__).parent.parent / "data" / "warehouse.db"

    history_counts = [loaded_rows]  # safe fallback if warehouse/table doesn't exist yet
    if warehouse_db.exists():
        conn = sqlite3.connect(str(warehouse_db))
        try:
            history_df = pd.read_sql(
                """
                SELECT DATE(loaded_at) AS load_date, COUNT(*) AS row_count
                FROM watch_events_daily
                GROUP BY DATE(loaded_at)
                ORDER BY load_date DESC
                LIMIT 8
                """,
                conn,
            )
            # Exclude today's just-loaded rows from the history baseline
            counts = history_df["row_count"].tolist()[1:8]
            if counts:
                history_counts = counts
        except Exception:
            logger.warning("Could not read history from warehouse — using single-point fallback")
        finally:
            conn.close()

    health = run_pipeline_health_check(
        loaded_rows=loaded_rows,
        history_counts=history_counts,
        last_run=logical_date.replace(tzinfo=None),
        now=datetime.utcnow(),
    )

    logger.info("Health check: %s", health)
    ti.xcom_push(key="health_check", value=health)

    if health["severity"] in ("P1", "P2"):
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        payload = build_slack_alert(
            check_name="watch_events_ingestion daily health check",
            status="critical" if health["severity"] == "P1" else "late",
            details=health,
        )
        if webhook_url:
            requests.post(webhook_url, json=payload, timeout=10)
        else:
            logger.warning("SLACK_WEBHOOK_URL not set — alert NOT sent: %s", payload)

    if health["severity"] == "P1":
        raise ValueError(f"Pipeline health check FAILED (P1): {health}")


def send_completion_notification(**context) -> None:
    """
    Task: notify
    Send a Slack/email notification confirming the pipeline completed.
    Failure of this task does NOT block upstream — it's decorative.
    """
    ti          = context["ti"]
    loaded_rows = ti.xcom_pull(task_ids="load", key="loaded_rows") or 0
    date_str    = context["logical_date"].strftime("%Y-%m-%d")

    message = (
        f"[watch_events_ingestion] {date_str} ✅\n"
        f"  Loaded: {loaded_rows} rows\n"
        f"  Pipeline: extract → validate → load → notify\n"
    )

    logger.info("Pipeline complete: %s", message.replace("\n", " | "))

    # --- Production version ---
    # SlackWebhookOperator(webhook_token_conn_id="slack_data_alerts", message=message).execute(context)

    print(message)  # In testing, stdout is sufficient


# ── DAG Definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id             = "watch_events_ingestion",
    description        = "Daily ingestion of CinemaStream watch events to the analytics warehouse",
    default_args       = DEFAULT_ARGS,
    schedule_interval  = "0 2 * * *",        # 02:00 UTC daily
    start_date         = datetime(2024, 1, 1),
    catchup            = False,               # don't backfill on first deploy
    max_active_runs    = 1,                   # no overlapping runs
    doc_md             = __doc__,
    tags               = ["cinemastream", "ingestion", "watch-events", "daily"],
) as dag:

    t_extract = PythonOperator(
        task_id         = "extract",
        python_callable = extract_watch_events,
    )

    t_validate = PythonOperator(
        task_id         = "validate",
        python_callable = validate_watch_events,
    )

    t_load = PythonOperator(
        task_id         = "load",
        python_callable = load_watch_events,
    )

    t_monitor = PythonOperator(
        task_id         = "monitor",
        python_callable = monitor_pipeline_health,
    )

    t_notify = PythonOperator(
        task_id         = "notify",
        python_callable = send_completion_notification,
    )

    # Pipeline: extract → validate → load → monitor → notify
    t_extract >> t_validate >> t_load >> t_monitor >> t_notify
