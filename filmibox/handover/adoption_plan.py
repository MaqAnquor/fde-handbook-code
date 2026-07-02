"""
FilmiBox Engagement — Adoption Plan & Handover
Chapter 092: Adoption Strategy

Dataclasses and utilities for:
- Training plan design (skill transfer, not feature walkthrough)
- Adoption metrics tracking (usage, resolution rate, ticket deflection)
- Handover document structure (scope boundaries, health checks, checklist)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingScenario:
    scenario_name: str
    role: str
    setup: str
    steps: List[str]
    common_failure: str
    success_signal: str


@dataclass
class TrainingPlan:
    tool_name: str
    audience: str
    duration_minutes: int
    scenarios: List[TrainingScenario]
    self_practice_task: str
    owner_contact: str


@dataclass
class AdoptionMetric:
    metric_name: str
    definition: str
    baseline: float
    target: float
    target_date: str
    current: float = 0.0


@dataclass
class AdoptionPlan:
    tool_name: str
    launch_date: str
    metrics: List[AdoptionMetric]
    review_cadence: str


@dataclass
class HandoverChecklistItem:
    item: str
    verified_by: str
    done: bool = False


@dataclass
class HandoverDocument:
    system_name: str
    engagement_end_date: str
    internal_owner: str
    escalation_contact: str
    what_it_does: str
    what_it_does_not_do: str
    how_to_check_health: str
    how_to_restart: str
    known_limitations: str
    checklist: List[HandoverChecklistItem] = field(default_factory=list)


def check_adoption_status(plan: AdoptionPlan) -> None:
    print(f"Adoption status: {plan.tool_name}")
    print(f"Review cadence: {plan.review_cadence}")
    print()
    for m in plan.metrics:
        if m.target == m.baseline:
            pct = 100.0
        else:
            pct = min(100.0, (m.current - m.baseline) / (m.target - m.baseline) * 100)
        status = "✓ ON TRACK" if pct >= 70 else "⚠ AT RISK" if pct >= 40 else "✗ STALLED"
        print(f"{status}  {m.metric_name}: {m.current} (target {m.target} by {m.target_date})")


def run_handover_checklist(doc: HandoverDocument) -> None:
    done = sum(1 for i in doc.checklist if i.done)
    total = len(doc.checklist)
    print(f"Handover: {doc.system_name}")
    print(f"Owner: {doc.internal_owner} | Escalation: {doc.escalation_contact}")
    print(f"Checklist: {done}/{total}")
    print()
    for item in doc.checklist:
        symbol = "✓" if item.done else "✗"
        print(f"  [{symbol}] {item.item}")


def build_billing_adoption() -> AdoptionPlan:
    """The billing-tool adoption plan (week-one pilot data)."""
    return AdoptionPlan(
        tool_name="Billing Lookup Tool — FilmiBox",
        launch_date="2026-07-01",
        metrics=[
            AdoptionMetric(
                "Weekly active users (% of support team)",
                "Distinct support agents who ran ≥1 billing query this week / 5 total agents",
                baseline=0.0, target=0.80, target_date="2026-07-28",
                current=0.60,
            ),
            AdoptionMetric(
                "In-tool resolution rate",
                "Billing queries resolved in-tool without escalation / total billing queries",
                baseline=0.0, target=0.70, target_date="2026-07-28",
                current=0.786,
            ),
            AdoptionMetric(
                "Manual billing ticket rate",
                "Billing tickets opened per week / total billing contacts received",
                baseline=1.0, target=0.30, target_date="2026-07-28",
                current=0.50,
            ),
        ],
        review_cadence="Weekly for first 30 days; Dev reviews and sends Anjali a one-line update",
    )


def main():
    billing_adoption = build_billing_adoption()

    print("=== ADOPTION STATUS ===")
    check_adoption_status(billing_adoption)

    filmibox_handover = HandoverDocument(
        system_name="FilmiBox Internal Assistant (billing + wiki + complaint themes)",
        engagement_end_date="2026-07-07",
        internal_owner="Dev (dev@filmibox.io)",
        escalation_contact="FDE (reader@cinemastream.com) — available for 90 days post-handover",
        what_it_does=(
            "Answers billing and subscription questions using the subscriber database (MCP tool). "
            "Answers new-hire policy and process questions from the internal wiki (RAG, bilingual). "
            "Surfaces top complaint themes from Zendesk tickets indexed overnight (RAG, last 7 days)."
        ),
        what_it_does_not_do=(
            "Does NOT answer questions about content licensing, contracts, or partner agreements. "
            "Does NOT answer HR or payroll questions. "
            "Does NOT give answers about individual employee accounts or salary. "
            "Does NOT have access to the founder MRR dashboard (Phase 2, not yet built). "
            "If a query falls outside billing/wiki/complaint-themes scope, the system will say so."
        ),
        how_to_check_health=(
            "Open the assistant URL and type 'health check'. It returns index freshness timestamps. "
            "Zendesk index must show last-updated within 25 hours (refresh runs 02:00 IST nightly). "
            "Wiki index must show last-updated within 49 hours (refresh runs 03:00 IST every other night). "
            "If either index is stale: check the #data-pipeline-alerts Slack channel first."
        ),
        how_to_restart=(
            "Zendesk refresh: ssh filmibox-data 'cd /opt/assistant && python refresh_tickets.py'. "
            "Wiki refresh: same host, python refresh_wiki.py. "
            "Both scripts log to /var/log/assistant/."
        ),
        known_limitations=(
            "Billing tool accuracy: 78.6% resolution rate in pilot (14 test queries). "
            "Subscriber names with special characters may not match — use email. "
            "Wiki in Hindi: works for standard queries; legal/technical language may miss relevant docs. "
            "Complaint themes: Zendesk only — does not pull from email or phone records. "
            "System is not designed for subscriber-facing use — internal team only."
        ),
        checklist=[
            HandoverChecklistItem("Dev can log in and run a billing query", "Dev live demo", done=True),
            HandoverChecklistItem("Dev knows how to check index freshness", "Dev live demo", done=True),
            HandoverChecklistItem("Dev can trigger a manual wiki refresh", "Dev live demo", done=True),
            HandoverChecklistItem("Dev can add a document to the wiki knowledge base", "Dev live demo", done=True),
            HandoverChecklistItem("Monitoring alert goes to Dev's Slack, not FDE", "test alert verified", done=True),
            HandoverChecklistItem("Handover document read and signed by Dev", "written sign-off", done=True),
            HandoverChecklistItem("Anjali briefed on Phase 2 PoV and timeline", "Anjali confirmation", done=True),
            HandoverChecklistItem("FDE escalation path tested (Dev messaged, FDE replied)", "test message", done=True),
        ],
    )

    print()
    print("=== HANDOVER CHECKLIST ===")
    run_handover_checklist(filmibox_handover)
    print()
    print("What this system does:")
    print(filmibox_handover.what_it_does)
    print()
    print("What it does NOT do:")
    print(filmibox_handover.what_it_does_not_do)
    print()
    print("Known limitations:")
    print(filmibox_handover.known_limitations)


if __name__ == "__main__":
    main()
