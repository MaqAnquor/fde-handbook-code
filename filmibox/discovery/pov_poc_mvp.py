"""PoV-PoC-MVP lifecycle framework for the FilmiBox engagement (Ch 090).

A document that runs: the lifecycle stages, the PoV articulation check,
the PoC gate review, and the week-4 FilmiBox scenario (Story 1 pilot
verdict + Phase 2 founder-MRR PoV proposal).

Fourth file in filmibox/discovery/, alongside scope_management.py (Ch 089).
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# The lifecycle: each stage retires one risk, in order of cheapness.
# ---------------------------------------------------------------------------

LIFECYCLE = [
    ("PoV", "Proof of Value", "Value risk",
     "Is this worth building at all?",
     "A written hypothesis with measurable success criteria"),
    ("PoC", "Proof of Concept", "Feasibility risk",
     "Can we build it with this data and stack?",
     "Working code, time-boxed, deliberately ugly"),
    ("MVP", "Minimum Viable Product", "Adoption risk",
     "Will real users actually use it?",
     "The smallest version that satisfies the acceptance criteria"),
]

# What upgrades when a PoC graduates to MVP. The feature is the same;
# everything around it changes.
POC_TO_MVP = {
    "hardcoded credentials":      "real auth + secrets management",
    "happy path only":            "error handling for the failures users will actually hit",
    "manually run by the FDE":    "deployed where users reach it themselves",
    "prints to the console":      "logging someone can read at 2am",
    "works on the demo extract":  "works on the live, messy, current data",
    "no one is on the hook":      "a named owner and a runbook entry",
}


@dataclass
class PoV:
    """A Proof of Value: a falsifiable bet on business value.

    Every field is load-bearing. A PoV missing any of these is an
    opinion, not a hypothesis.
    """

    hypothesis: str        # what we believe will happen if this exists
    metric: str            # the ONE number that proves or disproves it
    baseline: str          # where that number is today (measured, not guessed)
    target: str            # where it must reach for the bet to pay off
    time_box_weeks: float  # how long we give ourselves to find out

    def is_testable(self):
        """Checks the PoV can actually be won or lost."""
        problems = []
        if not self.metric.strip():
            problems.append("no metric - 'better' is not measurable")
        if not self.baseline.strip():
            problems.append("no baseline - you cannot show improvement "
                            "without a starting point")
        if not self.target.strip():
            problems.append("no target - when do you declare success?")
        if self.time_box_weeks <= 0 or self.time_box_weeks > 4:
            problems.append("time box missing or too long - "
                            "a PoV is weeks, not months")
        return (len(problems) == 0, problems)


def review_poc(criteria_results, pivot_insight=None):
    """The PoC gate: PROCEED, PIVOT, or KILL.

    criteria_results: dict of success criterion -> bool (met or not),
                      written BEFORE the PoC started, not after.
    pivot_insight:    if the PoC missed but revealed a better direction,
                      name it here - that converts a KILL into a PIVOT.
    """
    met = sum(1 for v in criteria_results.values() if v)
    total = len(criteria_results)
    if met == total:
        return ("PROCEED", f"All {total} criteria met -> fund the MVP.")
    if pivot_insight:
        return ("PIVOT", f"{met}/{total} criteria met, but the PoC "
                         f"revealed: {pivot_insight}")
    return ("KILL", f"Only {met}/{total} criteria met and no new direction "
                    "-> stop. Two weeks spent learning this is cheap. "
                    "Three months would not be.")


# ---------------------------------------------------------------------------
# Week-4 FilmiBox scenario: Story 1 gate review + Phase 2 PoV proposal.
# ---------------------------------------------------------------------------

def run_filmibox_week4():
    print("=" * 70)
    print("FilmiBox week 4 - PoV-PoC-MVP lifecycle review")
    print("=" * 70)

    print("\nThe lifecycle:")
    for code, name, risk, question, deliverable in LIFECYCLE:
        print(f"  {code} - {name}")
        print(f"     Retires:     {risk}")
        print(f"     Question:    {question}")
        print(f"     Deliverable: {deliverable}")

    # Story 1's PoV - written in discovery (Ch 086-087), formalized here.
    story1_pov = PoV(
        hypothesis="If support agents can look up a subscriber's charges "
                   "themselves, billing questions stop escalating to "
                   "engineering",
        metric="% of billing queries resolved in-tool without engineering "
               "escalation",
        baseline="0% - every billing question becomes an engineering ticket "
                 "(~4h round-trip)",
        target=">=70% resolved in-tool during a one-week pilot with 5 agents",
        time_box_weeks=2.0,
    )
    ok, _ = story1_pov.is_testable()
    print(f"\nStory 1 PoV testable: {ok}")

    # Week-4 pilot: 5 agents, one week, 14 billing queries.
    queries_total = 14
    resolved_in_tool = 11  # 2 needed an actual refund, 1 was bank-side
    pilot_results = {
        ">=70% of billing queries resolved in-tool without escalation":
            resolved_in_tool / queries_total >= 0.70,
        "median time-to-answer under 5 minutes "
        "(baseline: ~4h escalation round-trip)":
            True,  # pilot median was ~3 minutes
        "answers come from the live read replica, not a static extract":
            True,
    }
    print(f"Pilot: {resolved_in_tool}/{queries_total} resolved in-tool "
          f"({resolved_in_tool / queries_total:.1%})")
    print(f"Gate: {review_poc(pilot_results)}")

    # Phase 2, first item (per the Ch 089 deferral): founders-only MRR view.
    phase2_pov = PoV(
        hypothesis="A founders-only churn-adjusted MRR view replaces the "
                   "monthly spreadsheet assembly and surfaces churn-driven "
                   "revenue dips weeks earlier",
        metric="time from 'Anjali wants the number' to having it, and the "
               "lag between a churn-driven dip and a founder seeing it",
        baseline="~4 hours of manual assembly, once a month - a dip can sit "
                 "invisible for up to 30 days",
        target="on-demand in under a minute, refreshed daily - dips visible "
               "within 24 hours",
        time_box_weeks=2.0,
    )
    ok, _ = phase2_pov.is_testable()
    print(f"\nPhase 2 PoV testable: {ok}")
    print(f"  The bet:  {phase2_pov.hypothesis}")
    print(f"  Metric:   {phase2_pov.metric}")
    print(f"  Baseline: {phase2_pov.baseline}")
    print(f"  Target:   {phase2_pov.target}")
    print(f"  Time box: {phase2_pov.time_box_weeks} weeks")

    print("\nPoC -> MVP upgrade floor:")
    for poc_state, mvp_state in POC_TO_MVP.items():
        print(f"  {poc_state:<30} ->  {mvp_state}")


if __name__ == "__main__":
    run_filmibox_week4()
