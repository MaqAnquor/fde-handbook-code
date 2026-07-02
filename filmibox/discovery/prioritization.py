"""
filmibox/discovery/prioritization.py

Chapter 088 -- ordering the backlog so the six weeks build the highest-
leverage stories first.

Chapter 087 produced four INVEST-clean FilmiBox stories. You can't build
all four well in six weeks, and Anjali picked Story 1 (billing lookup) by
gut. This module replaces the gut with two standard frameworks:

  RICE   = (Reach x Impact x Confidence) / Effort  -- a single score that
           ranks stories by value-per-unit-effort.
  MoSCoW = Must / Should / Could / Won't(-this-phase) -- buckets that turn
           the ranked list into a committed scope.

The point: prioritization is a defensible decision, not a hunch. RICE
confirms Anjali's instinct on #1 AND surfaces something her gut missed
(the all-staff wiki out-reaches a single-team feature).

Run:
    python filmibox/discovery/prioritization.py
"""

from dataclasses import dataclass


@dataclass
class Story:
    name: str
    reach: int          # people (or uses) per week the story touches
    impact: float       # 3=massive, 2=high, 1=medium, 0.5=low, 0.25=minimal
    confidence: float   # 1.0=high, 0.8=medium, 0.5=low (how sure are the inputs)
    effort: float       # person-weeks to build
    moscow: str         # Must / Should / Could / Won't

    def rice(self):
        return (self.reach * self.impact * self.confidence) / self.effort


# The Chapter 087 backlog, scored. (Story 5 -- founder MRR -- is access-scoped
# to founders and deferred to a later phase, so it's Won't-this-phase.)
BACKLOG = [
    Story("1. Support billing lookup",  reach=40, impact=2, confidence=0.9, effort=1.0, moscow="Must"),
    Story("2. Support complaint themes", reach=20, impact=2, confidence=0.7, effort=1.5, moscow="Should"),
    Story("3. Content catalog count",    reach=10, impact=1, confidence=0.9, effort=0.5, moscow="Could"),
    Story("4. New-hire refund wiki",     reach=60, impact=1, confidence=0.8, effort=1.5, moscow="Should"),
    Story("5. Founder churn-adj MRR",    reach=2,  impact=3, confidence=0.8, effort=2.0, moscow="Won't (this phase)"),
]


def main():
    ranked = sorted(BACKLOG, key=lambda s: s.rice(), reverse=True)
    print("FilmiBox backlog -- ranked by RICE\n")
    print(f"{'story':30s} {'R':>4} {'I':>4} {'C':>5} {'E':>5} {'RICE':>7}  MoSCoW")
    print("-" * 78)
    for s in ranked:
        print(f"{s.name:30s} {s.reach:>4} {s.impact:>4.1f} {s.confidence:>5.2f} "
              f"{s.effort:>5.1f} {s.rice():>7.1f}  {s.moscow}")

    must = [s for s in BACKLOG if s.moscow == "Must"]
    should = [s for s in BACKLOG if s.moscow == "Should"]
    print(f"\nBuild order (Must first, then Should by RICE):")
    plan = sorted(must, key=lambda s: s.rice(), reverse=True) + \
        sorted(should, key=lambda s: s.rice(), reverse=True)
    for i, s in enumerate(plan, 1):
        print(f"  {i}. {s.name}  (RICE {s.rice():.1f}, {s.moscow})")
    print(f"\nGut-check: Anjali picked Story 1 by instinct -- RICE confirms it #1 "
          f"({ranked[0].rice():.0f}). But Story 4 (all-staff wiki) out-ranks the\n"
          f"support-themes story on REACH ({ranked[1].name.strip()}) -- the surprise "
          f"gut missed.")


if __name__ == "__main__":
    main()
