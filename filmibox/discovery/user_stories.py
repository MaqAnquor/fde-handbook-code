"""
filmibox/discovery/user_stories.py

Chapter 087 -- turning discovery needs into buildable, testable units.

Chapter 086 produced FilmiBox's data-source map and five laddered "real
needs." A need is not yet buildable -- it has no owner-role, no clear
benefit, and no way to know when it's DONE. This module converts each
need into a USER STORY ("As a <role>, I want <capability>, so that
<benefit>") with GIVEN-WHEN-THEN acceptance criteria, and checks each
story against the INVEST quality bar. The output is the backlog the
engagement is built and demoed against.

The point: a need you can't test is a need you can't finish. Acceptance
criteria are the definition of done, agreed before any code is written.

Run:
    python filmibox/discovery/user_stories.py
"""

from dataclasses import dataclass, field


@dataclass
class AcceptanceCriterion:
    """Given-When-Then: the concrete, testable definition of done."""
    given: str
    when: str
    then: str

    def __str__(self):
        return f"GIVEN {self.given}\n      WHEN {self.when}\n      THEN {self.then}"


@dataclass
class UserStory:
    role: str
    want: str
    so_that: str                       # the benefit -- the 'why', not optional
    access: str                        # who this story is scoped to
    criteria: list = field(default_factory=list)

    def story(self):
        return f"As a {self.role}, I want to {self.want}, so that {self.so_that}."

    def invest(self):
        """INVEST: a quick quality gate. Returns the checks that PASS.
        (Independent/Negotiable are reviewed in conversation; the four
        below are checkable from the story object itself.)"""
        checks = {
            "Valuable": bool(self.so_that),                  # has a stated benefit
            "Estimable": len(self.want) < 120,               # small enough to scope
            "Small": " and " not in self.want.lower(),       # one capability, not many
            "Testable": len(self.criteria) >= 1,             # has acceptance criteria
        }
        return checks


# FilmiBox stories, derived from Chapter 086's five real needs.
STORIES = [
    UserStory(
        role="support agent",
        want="look up a specific subscriber's recent charges by email",
        so_that="I can answer 'why was I charged twice' without escalating to engineering",
        access="support + eng",
        criteria=[AcceptanceCriterion(
            given="a subscriber email that exists",
            when="the agent asks for that user's charges in the last 60 days",
            then="the assistant returns the dated charge list from the live database")],
    ),
    UserStory(
        role="support lead",
        want="see this week's top recurring complaint themes",
        so_that="I can flag an emerging bug before it reaches more users",
        access="support + ops",
        criteria=[AcceptanceCriterion(
            given="the last 7 days of support tickets are indexed",
            when="the lead asks for the top complaint themes",
            then="the assistant returns 3-5 themes ranked by ticket count")],
    ),
    UserStory(
        role="content manager",
        want="ask how many titles we hold in a given genre",
        so_that="I can avoid licensing duplicates",
        access="all staff",
        criteria=[AcceptanceCriterion(
            given="the content catalog is reachable",
            when="the manager asks 'how many Tamil thrillers from the last 2 years?'",
            then="the assistant returns the count and titles from the live catalog")],
    ),
    UserStory(
        role="new hire",
        want="ask how to perform a common procedure like issuing a refund",
        so_that="I don't interrupt senior staff for things the wiki already answers",
        access="all staff",
        criteria=[AcceptanceCriterion(
            given="the bilingual (Hindi/English) wiki is indexed",
            when="the new hire asks 'how do I issue a refund?' in either language",
            then="the assistant answers from the wiki and cites the source page")],
    ),
    # A DELIBERATELY BAD story -- too big, no benefit, untestable -- for contrast.
    UserStory(
        role="founder",
        want="get full visibility into everything happening across the company "
             "and all our metrics and all the data in one place",
        so_that="",
        access="founders",
        criteria=[],
    ),
]


def main():
    print("FilmiBox backlog -- user stories vs the INVEST bar\n")
    for i, s in enumerate(STORIES, 1):
        checks = s.invest()
        passed = sum(checks.values())
        flag = "OK" if passed == len(checks) else f"NEEDS WORK ({passed}/{len(checks)})"
        print(f"Story {i} [{s.access}]  -- INVEST: {flag}")
        print(f"  {s.story()}")
        if passed < len(checks):
            failed = [k for k, v in checks.items() if not v]
            print(f"  FAILS: {', '.join(failed)}")
        for c in s.criteria:
            print(f"  - {c}")
        print()


if __name__ == "__main__":
    main()
