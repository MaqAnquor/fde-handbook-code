"""
Prefect flow: sync subscription events from billing API to warehouse.
Schedule: every 30 minutes via system cron  →  python subscription_sync_flow.py
Chapter: 047a — Prefect

pip install prefect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from prefect import flow, task
from prefect.logging import get_run_logger


@dataclass
class SubscriptionEvent:
    event_id: str
    user_id: int
    event_type: str          # UPGRADE | DOWNGRADE | CANCEL | REACTIVATE
    old_plan: Optional[str]
    new_plan: Optional[str]
    occurred_at: datetime


@dataclass
class SyncResult:
    run_date: str
    extracted: int = 0
    loaded: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# --- Simulated data store (in production: DuckDB / BigQuery / Postgres) ---
_loaded_event_ids: set[str] = set()


def _simulate_billing_api(limit: int = 10) -> list[SubscriptionEvent]:
    """Return synthetic subscription events (replaces real API call)."""
    now = datetime.now(timezone.utc)
    return [
        SubscriptionEvent("evt_001", 12, "UPGRADE",     "Basic",   "Premium", now),
        SubscriptionEvent("evt_002", 47, "CANCEL",      "Basic",   None,      now),
        SubscriptionEvent("evt_003", 91, "REACTIVATE",  None,      "Free",    now),
        SubscriptionEvent("evt_004", 33, "DOWNGRADE",   "Premium", "Basic",   now),
        SubscriptionEvent("evt_005", 78, "UPGRADE",     "Free",    "Basic",   now),
    ][:limit]


# --- Tasks ---

@task(name="extract-subscription-events", retries=2, retry_delay_seconds=30)
def extract_events(since_minutes: int = 30) -> list[SubscriptionEvent]:
    logger = get_run_logger()
    events = _simulate_billing_api()
    logger.info(f"Extracted {len(events)} events from billing API.")
    return events


@task(name="validate-events")
def validate_events(events: list[SubscriptionEvent]) -> list[SubscriptionEvent]:
    logger = get_run_logger()
    valid = []
    for e in events:
        if not e.event_id or not e.user_id:
            logger.warning(f"Skipping malformed event: {e}")
            continue
        if e.event_type not in {"UPGRADE", "DOWNGRADE", "CANCEL", "REACTIVATE"}:
            logger.warning(f"Unknown event_type '{e.event_type}' for {e.event_id}")
            continue
        valid.append(e)
    logger.info(f"Validated {len(valid)}/{len(events)} events.")
    return valid


@task(name="load-events", retries=1, retry_delay_seconds=10)
def load_events(events: list[SubscriptionEvent]) -> SyncResult:
    logger = get_run_logger()
    result = SyncResult(run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    result.extracted = len(events)

    for e in events:
        if e.event_id in _loaded_event_ids:
            logger.debug(f"  {e.event_id}: already loaded — skip.")
            result.skipped += 1
            continue
        _loaded_event_ids.add(e.event_id)
        logger.info(f"  {e.event_id}: {e.event_type} user {e.user_id} "
                    f"{e.old_plan or '?'} → {e.new_plan or 'churned'}")
        result.loaded += 1

    return result


# --- Flow ---

@flow(name="subscription-sync", log_prints=True)
def subscription_sync_flow(since_minutes: int = 30) -> SyncResult:
    """Idempotent sync of subscription events from billing API to warehouse."""
    raw     = extract_events(since_minutes)
    valid   = validate_events(raw)
    result  = load_events(valid)

    print(f"Sync complete: {result.loaded} loaded, "
          f"{result.skipped} skipped, {len(result.errors)} errors.")
    return result


if __name__ == "__main__":
    r = subscription_sync_flow()
    print(f"Success: {r.success}")

    # Re-run proves idempotency
    print("\n--- Re-run (idempotency check) ---")
    r2 = subscription_sync_flow()
    print(f"Second run loaded: {r2.loaded} (expect 0)")
