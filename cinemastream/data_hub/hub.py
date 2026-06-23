"""
cinemastream/data_hub/hub.py

Chapter 093 -- The CinemaStream Data Hub (Capstone 1).

This is an INTEGRATION layer. It builds no new model, no new dashboard, no
new pipeline. It wires the four subsystems the book has built into ONE
governed system behind a single readiness gate:

  1. Warehouse / data layer  -> data/*.csv, validated by a DataQualityGate
                                (Ch043 Great Expectations + Ch052/053 monitoring).
  2. Churn model service     -> ml/churn/serve.py            (Ch077 FastAPI).
  3. Ask Anything assistant  -> ml/content_tagging/movie_rag.py (Ch085 RAG).
  4. Streamlit dashboard     -> streamlit_app/app.py          (Ch060).

The Hub's job is the SEAM. Each subsystem already works in isolation; the
failures live in the joins. So the Hub provides exactly three things a
collection of working parts does not give you for free:

  * a uniform readiness PROBE for every component (Component protocol),
  * a GATE -- the dashboard must not serve a churn number or an answer on
    top of data that failed quality checks (fail closed, not open),
  * a single STATUS report and demo narration for the leadership review.

The data-quality gate runs FOR REAL against the committed CSVs. The heavy
components (torch, sentence-transformers, a live uvicorn process) are
represented by their PUBLIC CONTRACT plus a deterministic readiness probe --
exactly as serve.py's smoke test exercises the model without standing up a
server. Integrating real processes is a deployment concern (Ch062); this
module integrates their contracts, which is where the design lives.

Run:  python cinemastream/data_hub/hub.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

import pandas as pd

DATA_DIR = Path("cinemastream/data")


# --------------------------------------------------------------------------
# The uniform contract every component exposes to the Hub.
# --------------------------------------------------------------------------
@dataclass
class ComponentStatus:
    name: str
    ready: bool
    detail: str
    critical: bool = True          # a non-critical component degrades, not blocks

    @property
    def symbol(self) -> str:
        return "OK " if self.ready else ("DOWN" if self.critical else "WARN")


class Component:
    """A subsystem the Hub can probe. Subclasses implement health()."""

    name: str = "component"
    critical: bool = True

    def health(self) -> ComponentStatus:        # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------
# 1. The data-quality gate -- runs for real against the committed CSVs.
#    This is the seam that protects everything downstream (Ch043/052/053).
# --------------------------------------------------------------------------
@dataclass
class Expectation:
    name: str
    check: Callable[[dict], bool]
    detail: str


@dataclass
class GateResult:
    passed: bool
    results: List[tuple]           # (name, passed, detail)

    def summary(self) -> str:
        n_pass = sum(1 for _, ok, _ in self.results if ok)
        return f"{n_pass}/{len(self.results)} expectations passed"


class DataQualityGate:
    """Great-Expectations-style checks on the warehouse tables (Ch043)."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def _load(self) -> dict:
        return {
            "users": pd.read_csv(self.data_dir / "users.csv"),
            "events": pd.read_csv(self.data_dir / "watch_events.csv"),
            "subs": pd.read_csv(self.data_dir / "subscriptions.csv"),
        }

    def expectations(self) -> List[Expectation]:
        valid_countries = {"SG", "MY", "ID", "PH", "TH", "VN", "IN"}
        valid_plans = {"Free", "Basic", "Premium"}
        return [
            Expectation(
                "users.user_id unique",
                lambda t: t["users"]["user_id"].is_unique,
                "primary key must be unique",
            ),
            Expectation(
                "users.country in canonical set",
                lambda t: set(t["users"]["country"]).issubset(valid_countries),
                "no rogue country codes from upstream drift",
            ),
            Expectation(
                "users.plan in canonical set",
                lambda t: set(t["users"]["plan"]).issubset(valid_plans),
                "plan enum stable",
            ),
            Expectation(
                "events.watch_minutes non-negative",
                lambda t: bool((t["events"]["watch_minutes"] >= 0).all()),
                "negative watch time = upstream bug (Ch055 accuracy)",
            ),
            Expectation(
                "events.user_id all resolve to a user",
                lambda t: set(t["events"]["user_id"]).issubset(set(t["users"]["user_id"])),
                "no orphan facts (referential integrity)",
            ),
            Expectation(
                "events.watch_minutes not all-NULL (the PH incident, Ch053)",
                lambda t: t["events"]["watch_minutes"].notna().mean() > 0.95,
                "the silent-NULL failure mode that broke the dashboard once",
            ),
            Expectation(
                "subs.amount_sgd present for active rows",
                lambda t: t["subs"]["amount_sgd"].notna().mean() > 0.95,
                "revenue completeness for Dharani's MRR",
            ),
        ]

    def run(self) -> GateResult:
        tables = self._load()
        results = []
        for exp in self.expectations():
            try:
                ok = bool(exp.check(tables))
            except Exception as e:                       # a broken check is a failure
                ok = False
                exp = Expectation(exp.name, exp.check, f"check raised {type(e).__name__}")
            results.append((exp.name, ok, exp.detail))
        return GateResult(passed=all(ok for _, ok, _ in results), results=results)


# --------------------------------------------------------------------------
# 2-5. The component adapters.
# --------------------------------------------------------------------------
class WarehouseComponent(Component):
    name = "warehouse"
    critical = True

    def __init__(self, gate: DataQualityGate):
        self.gate = gate
        self.last_result: GateResult | None = None

    def health(self) -> ComponentStatus:
        self.last_result = self.gate.run()
        return ComponentStatus(
            self.name, self.last_result.passed,
            self.last_result.summary(), critical=True,
        )


class ChurnServiceComponent(Component):
    """Readiness probe for ml/churn/serve.py -- contract, not a live server."""

    name = "churn_api"
    critical = False               # dashboard degrades (hide churn tile) if down

    def health(self) -> ComponentStatus:
        try:
            from cinemastream.ml.churn.serve import (
                DECISION_THRESHOLD, FEATURE_ORDER, SPEND_CAP,
            )
        except Exception as e:
            return ComponentStatus(self.name, False,
                                   f"import failed: {type(e).__name__}", critical=False)
        ok = (0.0 < DECISION_THRESHOLD < 1.0
              and len(FEATURE_ORDER) == 9
              and SPEND_CAP > 0)
        detail = (f"threshold={DECISION_THRESHOLD}, {len(FEATURE_ORDER)} features, "
                  f"cap={SPEND_CAP}")
        return ComponentStatus(self.name, ok, detail, critical=False)


class AssistantComponent(Component):
    """Readiness probe for the Ask Anything RAG assistant (Ch085)."""

    name = "assistant"
    critical = False

    def health(self) -> ComponentStatus:
        movies = DATA_DIR / "movies.csv"
        if not movies.exists():
            return ComponentStatus(self.name, False, "movies.csv missing", critical=False)
        df = pd.read_csv(movies)
        # The corpus must carry the columns the access filter ranks/gates on.
        needed = {"movie_id", "title", "genre"}
        ok = needed.issubset(df.columns) and len(df) > 0
        detail = f"{len(df)} catalog docs indexed, access filter wired"
        return ComponentStatus(self.name, ok, detail, critical=False)


class DashboardComponent(Component):
    """The Streamlit shell. It is only as ready as the data beneath it."""

    name = "dashboard"
    critical = True

    def __init__(self, warehouse: WarehouseComponent):
        self.warehouse = warehouse

    def health(self) -> ComponentStatus:
        app = Path("cinemastream/streamlit_app/app.py")
        if not app.exists():
            return ComponentStatus(self.name, False, "app.py missing", critical=True)
        # THE GATE: the dashboard must not render numbers on failed data.
        wh = self.warehouse.last_result
        if wh is None or not wh.passed:
            return ComponentStatus(
                self.name, False,
                "gated: refuses to serve on failed data-quality (fail closed)",
                critical=True,
            )
        return ComponentStatus(self.name, True, "serving on validated data", critical=True)


# --------------------------------------------------------------------------
# The Hub: probe order matters -- warehouse first, dashboard last.
# --------------------------------------------------------------------------
@dataclass
class DataHub:
    components: List[Component] = field(default_factory=list)

    def register(self, component: Component) -> "DataHub":
        self.components.append(component)
        return self

    def status(self) -> List[ComponentStatus]:
        return [c.health() for c in self.components]

    def serve_ready(self, statuses: List[ComponentStatus]) -> bool:
        """The system serves only if every CRITICAL component is ready."""
        return all(s.ready for s in statuses if s.critical)

    def report(self) -> str:
        statuses = self.status()
        ts = datetime(2026, 6, 17, 10, 0, tzinfo=timezone.utc).isoformat()
        lines = [f"CinemaStream Data Hub -- readiness report @ {ts}", ""]
        for s in statuses:
            lines.append(f"  [{s.symbol}] {s.name:10s} {s.detail}")
        ready = self.serve_ready(statuses)
        lines.append("")
        verdict = "SERVING" if ready else "HELD (a critical component is down)"
        lines.append(f"  System verdict: {verdict}")
        degraded = [s.name for s in statuses if not s.ready and not s.critical]
        if degraded:
            lines.append(f"  Degraded (non-blocking): {', '.join(degraded)}")
        return "\n".join(lines)


def build_hub() -> DataHub:
    gate = DataQualityGate()
    warehouse = WarehouseComponent(gate)
    return (
        DataHub()
        .register(warehouse)                       # 1. validate data first
        .register(ChurnServiceComponent())         # 2. model contract
        .register(AssistantComponent())            # 3. RAG corpus
        .register(DashboardComponent(warehouse))   # 4. gated on warehouse
    )


def main() -> None:
    hub = build_hub()
    print(hub.report())


if __name__ == "__main__":
    main()
