"""
filmibox/demo/demo_story.py

Chapter 091 -- demo storytelling: three acts, a hero, and a fallback.

Week five of the FilmiBox engagement is demo day: twenty-two staff, a
projector, and one chance to earn belief in the tools built over five
weeks. This module is the demo builder -- a three-act structure checker
(Before -> Change -> After), a hero mapper that keeps the user (not the
tool) at the centre, the prep checklist, a Before-state drafting helper,
and the full FilmiBox demo-day runsheet with real pilot numbers.

The point: a demo is not an inventory of everything you were asked to
build. It is the story of the things that changed. The Before state does
the caring-generation work; the change act only proves the After is real.

Run:
    python filmibox/demo/demo_story.py
"""

from dataclasses import dataclass
from typing import List


# ── The three-act structure ──────────────────────────────────────────────────

@dataclass
class DemoAct:
    act: str          # "before" | "change" | "after"
    headline: str     # one-sentence summary
    script: str       # what you say (or show)
    duration_sec: int


@dataclass
class DemoStory:
    story_name: str
    hero: str             # the person whose day changes
    pain: str             # what was painful before
    acts: List[DemoAct]

    def total_duration_min(self):
        return sum(a.duration_sec for a in self.acts) / 60

    def check_structure(self):
        act_names = [a.act for a in self.acts]
        required = {"before", "change", "after"}
        missing = required - set(act_names)
        if missing:
            return f"INCOMPLETE — missing acts: {missing}"
        before = next(a for a in self.acts if a.act == "before")
        change = next(a for a in self.acts if a.act == "change")
        if change.duration_sec > before.duration_sec * 2:
            return "WARNING — change act is longer than twice the before act. Demo may feel like a tour."
        return "OK"


# ── The hero mapping: the user is the hero, the tool is not ─────────────────

def map_hero(story_name, tool_description, user_role, before_state, after_state):
    print(f"Story:   {story_name}")
    print(f"Tool:    {tool_description}  ← this is NOT the hero")
    print(f"Hero:    {user_role}")
    print(f"Before:  {before_state}")
    print(f"After:   {after_state}")
    print()


# ── The live demo risk — and the fallback ────────────────────────────────────

DEMO_PREP_CHECKLIST = [
    ("LIVE_DATA",      "Verify the data source is reachable from the demo device"),
    ("CREDENTIALS",    "Confirm API keys and DB credentials have not rotated since last run"),
    ("FALLBACK",       "Record a screen-capture fallback video on the morning of the demo"),
    ("TIMER",          "Rehearse the change act — it should feel unhurried in ≤90s"),
    ("SILENT_WINDOW",  "Leave the demo interface open before the meeting — no live typing of URLs"),
    ("QUESTION_PREP",  "List the 3 questions most likely to come, with one-sentence answers"),
    ("CLOSE",          "End with 'What questions do you have?' not 'So that's the demo.'"),
]


def print_checklist():
    print("Demo prep checklist:")
    for i, (label, action) in enumerate(DEMO_PREP_CHECKLIST, 1):
        print(f"  {i}. [{label:16s}] {action}")


# ── Drafting the Before state: specific scenario, concrete cost, named emotion ─

def draft_before_state(user_role, specific_scenario, time_cost, emotional_cost):
    """
    Before state should be specific (a named scenario),
    have a concrete cost (time or money),
    and name the emotional reality (frustration, embarrassment, uncertainty).
    """
    return (
        f"Picture {user_role}. "
        f"{specific_scenario}. "
        f"That {time_cost}. "
        f"{emotional_cost}."
    )


# ── The FilmiBox demo-day runsheet ───────────────────────────────────────────

@dataclass
class Act:
    kind: str; headline: str; script: str; seconds: int


@dataclass
class FilmiBoxDemo:
    story: str; hero: str; acts: List[Act]
    pilot_result: str = ""

    def duration_min(self):
        return round(sum(a.seconds for a in self.acts) / 60, 1)

    def print_runsheet(self):
        print(f"\n{'='*60}")
        print(f"Story: {self.story}")
        print(f"Hero:  {self.hero}")
        if self.pilot_result:
            print(f"Data:  {self.pilot_result}")
        for act in self.acts:
            bar = "▶" if act.kind == "change" else " "
            print(f"  {bar} [{act.kind.upper():6s} {act.seconds:3d}s] {act.headline}")
        print(f"  Total: {self.duration_min()} min")


BILLING = FilmiBoxDemo(
    story="1. Support billing lookup + watch history",
    hero="Support agent on the phone with an upset subscriber",
    pilot_result="11 of 14 queries resolved in-tool (78.6%) vs ≥70% target — PROCEED",
    acts=[
        Act("before", "A subscriber waits 4 hours for a yes/no",
            "Picture your support agent on a call. A subscriber says 'I was charged twice.' "
            "They need a yes or a no. Before today, the agent had to open an engineering ticket, "
            "wait — sometimes hours, sometimes until tomorrow — and call back. The subscriber is "
            "still on hold. Let me show you what it looks like now.",
            90),
        Act("change", "Live: email → charges + watch history in 30 seconds",
            "[TYPE EMAIL. SHOW CHARGES. SHOW LAST 5 WATCH EVENTS.] "
            "Thirty seconds. No ticket. No paging engineering. The subscriber is still on the line.",
            60),
        Act("after", "Same question resolved before the call ends",
            "That subscriber's question — 'was I charged twice?' — answered before they hung up. "
            "In the pilot this week, 11 of 14 billing queries resolved this way. "
            "The other 3 were refund edge cases that are genuinely outside this tool's scope — "
            "they escalated appropriately. The ones that escalated now do so with a reason, "
            "not just 'I don't know yet.'",
            60),
    ]
)

WIKI = FilmiBoxDemo(
    story="4. New-hire refund wiki (bilingual)",
    hero="New hire on first solo shift",
    acts=[
        Act("before", "A new hire's first solo call ends with an interruption",
            "Picture your newest hire on their first solo day. A subscriber asks about your "
            "refund policy for a cancelled subscription. They look in the shared drive — "
            "three folders, six documents, nothing clear. They interrupt a senior colleague. "
            "Fifteen minutes of someone else's morning. The new hire feels like a burden "
            "before they've finished their first shift.",
            75),
        Act("change", "Live: 'how do I issue a refund?' in Hindi → answer in 8 seconds",
            "[TYPE 'refund kaise karein' IN HINDI. SHOW ANSWER WITH SOURCE LINK.]",
            45),
        Act("after", "New hire self-serves, senior staff uninterrupted",
            "Hindi or English. The answer comes with a source link so they can read the full "
            "policy if they need to. Your senior staff's morning is their own. "
            "And the new hire learns where the answer lives — next time they look there first.",
            50),
    ]
)

THEMES = FilmiBoxDemo(
    story="2. Support complaint themes — early bug signal",
    hero="Support lead skimming 400 tickets a week",
    acts=[
        Act("before", "A bug hides in 400 tickets for a week",
            "Your support lead reads roughly 400 tickets a week to find patterns. "
            "Last month a subtitle bug was reported by 30 subscribers over 4 days before anyone "
            "connected the dots — by then there were 90 open tickets. "
            "That is the problem: patterns are in the data before a human can see them.",
            70),
        Act("change", "Live: top 5 complaint themes from last 7 days",
            "[RUN THEME QUERY. SHOW TOP 5 THEMES WITH TICKET COUNTS.]",
            50),
        Act("after", "Patterns visible in hours, not days",
            "That query runs in seconds. If a bug starts at 10am, by the afternoon stand-up "
            "it is a named theme, not a noise spike. Your support lead still reads the tickets "
            "that need a human — but they read *toward* something, not hoping to spot it.",
            55),
    ]
)


def run_demo_day():
    demos = [BILLING, WIKI, THEMES]
    print("FilmiBox Demo Day — Week 5 Runsheet")
    print(f"Stories: {len(demos)}  |  "
          f"Total: {sum(d.duration_min() for d in demos):.1f} min + Q&A")

    for d in demos:
        d.print_runsheet()

    print("\n" + "="*60)
    print("Prep checklist (verify before the call):")
    checks = [
        "DB live: subscriber_db reachable from demo machine",
        "Wiki RAG index fresh (last ingested <24h ago)",
        "Zendesk tickets indexed through yesterday",
        "Screen-capture fallback recorded this morning",
        "Billing demo rehearsed — change act under 90s",
        "Three likely questions prepared: Phase 2, Hindi languages, ticket security",
    ]
    for i, c in enumerate(checks, 1):
        print(f"  {i}. {c}")


if __name__ == "__main__":
    run_demo_day()
