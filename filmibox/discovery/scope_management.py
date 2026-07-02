"""
filmibox/discovery/scope_management.py

Chapter 089 -- change requests, saying no, and protecting the MVP.

Chapter 088 committed a scope: billing lookup as a Must, wiki and
complaint themes as Shoulds, catalog count as a Could, founder MRR
explicitly Won't. Week three, two change requests arrive in one
thirty-minute check-in. This module is the scope tracker that assesses
them: a three-bucket change-request classifier (YES_TRADE / NO_PHASE /
DEFER_ASSESS), a scope basket that makes trades visible instead of
silent, an MVP checker that holds the acceptance-criteria line, and the
response language that protects the relationship while protecting the
deadline.

The point: adding to a committed scope without removing something is an
overdraft. Invisible trade-offs don't disappear -- they surprise you in
week five.

Run:
    python filmibox/discovery/scope_management.py
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ── The three-bucket change request response ─────────────────────────────────

CHANGE_BUCKETS = {
    "YES_TRADE":    "In scope or genuine must-have: accept, but remove something of equal cost",
    "NO_PHASE":     "Valuable but not this phase: decline with a named future home",
    "DEFER_ASSESS": "Unclear value or effort: defer until scoped; don't absorb unknowns",
}


def classify_change(description, is_in_scope_extension, effort_weeks, phase_capacity_remaining):
    """
    Returns a bucket + one-sentence rationale.
    This is a decision aid, not an algorithm — the output informs judgement.
    """
    if effort_weeks <= 0:
        return ("DEFER_ASSESS", "Effort unknown — do not absorb until scoped.")
    if effort_weeks > phase_capacity_remaining:
        return ("NO_PHASE", f"Effort {effort_weeks}w exceeds remaining capacity {phase_capacity_remaining:.1f}w.")
    if is_in_scope_extension:
        return ("YES_TRADE", "Fits existing story scope; displaces an equal-cost Could item.")
    return ("NO_PHASE", "New scope — valid but not this phase; add to next-phase backlog.")


# ── The scope trade: additions must displace equal cost ──────────────────────

@dataclass
class ScopeItem:
    name: str
    moscow: str     # Must / Should / Could / Won't
    effort_weeks: float
    status: str = "committed"  # committed / candidate_for_removal


@dataclass
class ScopeBasket:
    capacity_weeks: float
    items: List[ScopeItem] = field(default_factory=list)

    def total_committed(self):
        return sum(i.effort_weeks for i in self.items if i.status == "committed")

    def remaining(self):
        return self.capacity_weeks - self.total_committed()

    def can_add(self, effort_weeks):
        return self.remaining() >= effort_weeks

    def add_requires_trade(self, new_item, displaced_item=None):
        """Returns the trade decision."""
        if self.can_add(new_item.effort_weeks):
            return ("YES_DIRECT", f"Capacity has room ({self.remaining():.1f}w remaining).")
        if displaced_item:
            displaced_item.status = "candidate_for_removal"
            return ("YES_TRADE", f"Trade: add {new_item.name!r} ({new_item.effort_weeks}w), "
                                 f"remove {displaced_item.name!r} ({displaced_item.effort_weeks}w).")
        return ("NO_ROOM", f"No room without a trade. Remaining: {self.remaining():.1f}w.")


# ── MVP discipline: the acceptance criteria line ─────────────────────────────

def check_mvp_scope(story_name, acceptance_criteria_met, proposed_additions):
    """
    Determines whether proposed additions are in-MVP or above-MVP.
    acceptance_criteria_met: dict of criterion -> bool
    proposed_additions: list of feature description strings
    """
    unmet = [k for k, v in acceptance_criteria_met.items() if not v]
    if unmet:
        return {
            "mvp_status": "INCOMPLETE",
            "unmet_criteria": unmet,
            "verdict": "Finish the MVP before considering additions.",
            "additions": "ALL deferred until MVP criteria are met.",
        }
    return {
        "mvp_status": "COMPLETE",
        "unmet_criteria": [],
        "verdict": f"{story_name} MVP is done. Additions are new scope decisions.",
        "additions": {a: "New scope — assess via change request process" for a in proposed_additions},
    }


# ── Saying no: language that protects the relationship ───────────────────────

def draft_response(change_request, bucket, future_home=None, trade_offer=None):
    templates = {
        "YES_TRADE": (
            f"Let's add '{change_request}' — it fits the current arc. To make room, "
            f"we'd need to move {trade_offer!r} to Could or drop it this phase. "
            f"Does that trade work for you?"
        ),
        "NO_PHASE": (
            f"'{change_request}' is worth building — I don't want to lose it. "
            f"This phase it would displace work we've both committed to, so I'd put it "
            f"in '{future_home}' as the first candidate for next engagement. "
            f"Is there a story in the current scope you'd trade for it today?"
        ),
        "DEFER_ASSESS": (
            f"'{change_request}' sounds valuable — I need one working session to scope "
            f"the effort before I can say yes or no. Can I come back to you by end of week "
            f"with a size and a trade-off?"
        ),
    }
    return templates[bucket]


# ── The week-three FilmiBox scenario: two CRs against the live scope ─────────

@dataclass
class ChangeRequest:
    description: str
    source: str  # who asked
    effort_estimate_weeks: float
    is_extension_of_story: Optional[str] = None  # None if new scope


@dataclass
class ActiveScope:
    capacity_weeks: float
    phase_weeks_remaining: float
    items: list

    def assess(self, cr: ChangeRequest):
        if cr.effort_estimate_weeks <= 0:
            return ("DEFER_ASSESS", "Scope unknown. Need sizing before deciding.")
        if cr.effort_estimate_weeks > self.phase_weeks_remaining:
            return ("NO_PHASE",
                    f"Effort {cr.effort_estimate_weeks}w exceeds {self.phase_weeks_remaining}w remaining.")
        if cr.is_extension_of_story:
            return ("YES_TRADE",
                    f"Extension of {cr.is_extension_of_story}. Absorbs {cr.effort_estimate_weeks}w "
                    f"from remaining capacity ({self.phase_weeks_remaining}w).")
        return ("NO_PHASE", "New scope. Valid, but would displace committed work.")


def main():
    print("FilmiBox scope tracker -- week three, two change requests\n")

    scope = ActiveScope(
        capacity_weeks=6.0,
        phase_weeks_remaining=2.2,  # 3.8w spent, 2.2w left
        items=["Story 1 (done)", "Story 4 wiki (Should)", "Story 2 themes (Should)", "Story 3 catalog (Could)"],
    )

    watch_history_cr = ChangeRequest(
        description="Watch history tab alongside billing charges",
        source="Support team lead",
        effort_estimate_weeks=0.5,
        is_extension_of_story="Story 1: Support billing lookup",
    )

    mrr_dashboard_cr = ChangeRequest(
        description="Founder churn-adjusted MRR dashboard",
        source="Co-founder (Dev)",
        effort_estimate_weeks=2.0,
        is_extension_of_story=None,
    )

    print("CR 1:", scope.assess(watch_history_cr))
    print("CR 2:", scope.assess(mrr_dashboard_cr))

    # Final updated scope after both CRs assessed
    SCOPE_V2 = {
        "Must":  ["1. Support billing lookup + watch history (extended, 1.5w)"],
        "Should":["4. New-hire refund wiki (1.5w)", "2. Support complaint themes (1.5w)"],
        "Could": [],  # catalog count dropped to fund CR 1
        "Won't": [
            "5. Founder churn-adj MRR (this phase) — Phase 2, founders-only separate system",
            "3. Content catalog count (this phase) — traded for watch-history extension",
        ],
    }

    for bucket, items in SCOPE_V2.items():
        print(f"\n{bucket}:")
        for i in items:
            print(f"  • {i}")


if __name__ == "__main__":
    main()
