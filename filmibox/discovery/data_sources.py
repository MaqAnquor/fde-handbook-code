"""
filmibox/discovery/data_sources.py

Chapter 086 -- the deliverable from a discovery interview.

The reader is loaned to FilmiBox (a Mumbai streaming startup) for a
six-week FDE engagement. Founder Anjali Rao's stated ask: "I want my team
to be able to ask Claude questions about our data." Discovery's job is to
turn that one vague sentence into a MAP: every data source, what kind it
is, who owns it, who may see it, how often it changes -- because that map
decides what gets built (structured data -> MCP tool; unstructured data
-> RAG) and is the single artifact that prevents an FDE from building the
wrong thing.

This module encodes the discovery output as structured data, so the map
is queryable, not a slide. It is a DOCUMENT that happens to run.

Run:
    python filmibox/discovery/data_sources.py
"""

from dataclasses import dataclass, field
from enum import Enum


class Kind(Enum):
    STRUCTURED = "structured"      # rows/columns, an API -> MCP tool
    UNSTRUCTURED = "unstructured"  # free text, PDFs, wiki -> RAG pipeline


@dataclass
class DataSource:
    """One row of the discovery map. Every field is a question discovery
    must answer -- a blank here is a risk you haven't surfaced yet."""
    name: str
    kind: Kind
    owner: str                 # the human accountable (people leave; teams own)
    access: str                # who may see it -- the access-control boundary
    update_freq: str           # how often it changes -> freshness strategy
    real_need: str             # the actual job, not the stated ask

    def build_path(self) -> str:
        """Structured -> MCP tool (Claude calls it live); unstructured -> RAG
        (embed + retrieve chunks). This routing is the whole point of
        categorising sources during discovery."""
        return "MCP tool" if self.kind is Kind.STRUCTURED else "RAG pipeline"


# The FilmiBox discovery map -- the output of the Chapter 086 interview.
FILMIBOX_SOURCES = [
    DataSource(
        "Subscriber database (Postgres)", Kind.STRUCTURED,
        owner="Backend team", access="PII -- restricted to support + eng",
        update_freq="real-time",
        real_need="Support agents answer 'why was this user charged twice?' "
                  "without filing a ticket to engineering."),
    DataSource(
        "Content catalog (titles API)", Kind.STRUCTURED,
        owner="Content team", access="internal (all staff)",
        update_freq="daily",
        real_need="Content team checks 'do we have any Tamil thrillers from "
                  "the last 2 years?' before licensing more."),
    DataSource(
        "Support tickets (Zendesk export)", Kind.UNSTRUCTURED,
        owner="Support lead", access="PII -- support + ops only",
        update_freq="hourly",
        real_need="Spot recurring complaints early instead of reading 400 "
                  "tickets a week by hand (the Forever-Manual Report)."),
    DataSource(
        "Company wiki + runbooks (Hindi/English)", Kind.UNSTRUCTURED,
        owner="Ops", access="internal (all staff)",
        update_freq="weekly",
        real_need="New hires self-serve 'how do I issue a refund?' instead "
                  "of interrupting senior staff -- multilingual content."),
    DataSource(
        "Revenue reports (PDF/spreadsheet)", Kind.UNSTRUCTURED,
        owner="Anjali (CFO duties)", access="CONFIDENTIAL -- founders only",
        update_freq="monthly",
        real_need="Anjali asks 'what's our churn-adjusted MRR this month?' -- "
                  "but this is NOT for the all-staff assistant."),
]


def summarize(sources):
    print(f"FilmiBox discovery map: {len(sources)} data sources\n")
    print(f"{'source':40s} {'build path':12s} {'access'}")
    print("-" * 92)
    for s in sources:
        print(f"{s.name:40s} {s.build_path():12s} {s.access}")

    mcp = [s for s in sources if s.kind is Kind.STRUCTURED]
    rag = [s for s in sources if s.kind is Kind.UNSTRUCTURED]
    print(f"\n-> {len(mcp)} structured sources become MCP tools, "
          f"{len(rag)} unstructured become RAG pipelines.")

    # The access-control finding discovery must surface BEFORE building.
    confidential = [s for s in sources if "CONFIDENTIAL" in s.access]
    print(f"\nACCESS-CONTROL FLAG: {len(confidential)} source(s) are "
          f"founder-confidential and must NOT enter the all-staff assistant:")
    for s in confidential:
        print(f"   - {s.name} (owner: {s.owner})")


def main():
    summarize(FILMIBOX_SOURCES)


if __name__ == "__main__":
    main()
