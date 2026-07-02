"""
filmibox/migration/migrate.py

Chapter 093b -- Capstone 3: Autonomous Legacy Migration.

The job an FDE actually gets paid for: a client hands you an old database full
of unstructured, half-structured, inconsistent records and says "make our AI
answer questions about this." You cannot RAG a mess. The work is the pipeline
that turns the mess into something serveable:

    INGEST  ->  CLEAN  ->  STRUCTURE  ->  VECTORIZE  ->  SERVE (local API)

Each stage has a real-world failure mode this module handles explicitly:

  * INGEST     -- legacy rows: mixed casing, whitespace junk, duplicate emails,
                  three different date formats, free-text `notes` blobs.
  * CLEAN      -- normalize whitespace/casing; parse the date formats to ISO.
  * STRUCTURE  -- extract plan/amount/currency from the free-text notes with
                  regex; a record that cannot be parsed is QUARANTINED for
                  human review, never silently dropped (the cardinal rule).
  * DEDUPE     -- collapse duplicate emails, keeping the MOST COMPLETE row.
  * VECTORIZE  -- chunk + index the leftover free text for semantic serving
                  (lexical stand-in here; pgvector + Ch085 encoder in prod).
  * SERVE      -- a real local FastAPI app, exercised via TestClient (Ch077),
                  exposing /customer/{email}, /search, and /migration/report.

The migration is IDEMPOTENT (re-running on the same source yields the same
result) and AUDITED (it emits a MigrationReport you can defend to the client).

Run:  python filmibox/migration/migrate.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# The legacy source -- a dump from FilmiBox's old CRM. Deliberately messy.
# (Synthetic, flagged: invented legacy records for the migration demo.)
# --------------------------------------------------------------------------
LEGACY_ROWS: List[dict] = [
    {"id": 1, "name": "  ravi  KUMAR ", "email": "Ravi@Example.COM ",
     "notes": "Plan: Premium | paid 1499 INR | signed 15/01/2024. Heavy watcher."},
    {"id": 2, "name": "Siti Rahman", "email": "siti@example.com",
     "notes": "plan basic - 299 inr - joined 2024-02-03. asks about refunds often"},
    {"id": 3, "name": "Nguyen Van Minh", "email": "minh@example.com",
     "notes": "Free tier. Cancelled. last seen Jan 15 2024"},
    {"id": 4, "name": "RAVI KUMAR", "email": "ravi@example.com",
     "notes": "Plan: Premium | 1499 INR | 15/01/2024 | prefers Hindi originals"},
    {"id": 5, "name": "Aarti Desai", "email": "aarti@example.com",
     "notes": "legacy import error ###  unreadable blob  ???"},
    {"id": 6, "name": "Lim Wei", "email": "",      # missing email -> quarantine
     "notes": "Plan: Basic | 299 INR | 2024-03-10"},
]


# --------------------------------------------------------------------------
# CLEAN
# --------------------------------------------------------------------------
def clean_text(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def clean_name(s: str) -> str:
    return clean_text(s).title()


def clean_email(s: str) -> str:
    return clean_text(s).lower()


DATE_PATTERNS = [
    ("%d/%m/%Y", re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")),
    ("%Y-%m-%d", re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")),
    ("%b %d %Y", re.compile(r"\b([A-Z][a-z]{2} \d{1,2} \d{4})\b")),
]


def parse_date(notes: str) -> Optional[str]:
    """Find a date in any of three legacy formats; normalize to ISO."""
    for fmt, pat in DATE_PATTERNS:
        m = pat.search(notes)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt).date().isoformat()
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------
# STRUCTURE -- extract fields from the free-text notes.
# --------------------------------------------------------------------------
PLAN_RE = re.compile(r"\b(Premium|Basic|Free)\b", re.IGNORECASE)
AMOUNT_RE = re.compile(r"\b(\d{2,5})\s*(INR|inr)\b")


@dataclass
class CleanRecord:
    email: str
    name: str
    plan: Optional[str]
    amount_inr: Optional[int]
    signed_date: Optional[str]
    notes: str
    completeness: int = 0       # how many structured fields we recovered


@dataclass
class Quarantined:
    raw: dict
    reason: str


def structure_row(raw: dict) -> Tuple[Optional[CleanRecord], Optional[Quarantined]]:
    email = clean_email(raw.get("email", ""))
    if not email or "@" not in email:
        return None, Quarantined(raw, "missing/invalid email (no natural key)")

    notes = clean_text(raw.get("notes"))
    plan_m = PLAN_RE.search(notes)
    amt_m = AMOUNT_RE.search(notes)
    plan = plan_m.group(1).title() if plan_m else None
    amount = int(amt_m.group(1)) if amt_m else None
    signed = parse_date(notes)

    rec = CleanRecord(
        email=email, name=clean_name(raw.get("name", "")),
        plan=plan, amount_inr=amount, signed_date=signed, notes=notes,
    )
    rec.completeness = sum(x is not None for x in (plan, amount, signed))
    if rec.completeness == 0:
        return None, Quarantined(raw, "no structured fields recoverable from notes")
    return rec, None


# --------------------------------------------------------------------------
# DEDUPE -- collapse duplicate emails, keep the MOST COMPLETE row.
# --------------------------------------------------------------------------
def dedupe(records: List[CleanRecord]) -> Tuple[List[CleanRecord], int]:
    best: Dict[str, CleanRecord] = {}
    removed = 0
    for r in records:
        prev = best.get(r.email)
        if prev is None:
            best[r.email] = r
        else:
            removed += 1
            if r.completeness > prev.completeness:
                best[r.email] = r          # keep the more complete duplicate
    return sorted(best.values(), key=lambda r: r.email), removed


# --------------------------------------------------------------------------
# VECTORIZE -- index the free text for serving (lexical stand-in).
# --------------------------------------------------------------------------
def _tokens(s: str) -> set:
    return {w.strip(".,?!|#").lower() for w in s.split() if len(w) > 2}


@dataclass
class VectorIndex:
    docs: List[CleanRecord] = field(default_factory=list)

    def search(self, query: str, k: int = 3) -> List[Tuple[int, CleanRecord]]:
        q = _tokens(query)
        scored = [(len(q & _tokens(d.notes)), d) for d in self.docs]
        scored = [(s, d) for s, d in scored if s > 0]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


# --------------------------------------------------------------------------
# The pipeline + audit report.
# --------------------------------------------------------------------------
@dataclass
class MigrationReport:
    ingested: int = 0
    structured: int = 0
    quarantined: int = 0
    duplicates_removed: int = 0
    final_records: int = 0
    indexed: int = 0
    quarantine_reasons: List[str] = field(default_factory=list)

    def lines(self) -> List[str]:
        return [
            "Migration report",
            f"  ingested ............ {self.ingested}",
            f"  structured .......... {self.structured}",
            f"  quarantined ......... {self.quarantined}",
            f"  duplicates removed .. {self.duplicates_removed}",
            f"  final records ....... {self.final_records}",
            f"  indexed for search .. {self.indexed}",
            "  quarantine reasons:",
            *[f"    - {r}" for r in self.quarantine_reasons],
        ]


@dataclass
class MigrationResult:
    records: List[CleanRecord]
    index: VectorIndex
    quarantine: List[Quarantined]
    report: MigrationReport


def migrate(rows: List[dict] = None) -> MigrationResult:
    rows = LEGACY_ROWS if rows is None else rows
    report = MigrationReport(ingested=len(rows))

    structured, quarantine = [], []
    for raw in rows:
        rec, q = structure_row(raw)
        if rec is not None:
            structured.append(rec)
        else:
            quarantine.append(q)
            report.quarantine_reasons.append(f"id={raw.get('id')}: {q.reason}")
    report.structured = len(structured)
    report.quarantined = len(quarantine)

    deduped, removed = dedupe(structured)
    report.duplicates_removed = removed
    report.final_records = len(deduped)

    index = VectorIndex(docs=deduped)
    report.indexed = len(index.docs)

    return MigrationResult(deduped, index, quarantine, report)


# --------------------------------------------------------------------------
# SERVE -- a real local FastAPI app over the migrated data (Ch077).
# --------------------------------------------------------------------------
def build_api(result: MigrationResult):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="FilmiBox Migrated CRM")
    by_email = {r.email: r for r in result.records}

    @app.get("/customer/{email}")
    def get_customer(email: str):
        rec = by_email.get(email.lower())
        if rec is None:
            raise HTTPException(status_code=404, detail="not found")
        return {"email": rec.email, "name": rec.name, "plan": rec.plan,
                "amount_inr": rec.amount_inr, "signed_date": rec.signed_date}

    @app.get("/search")
    def search(q: str):
        hits = result.index.search(q)
        return {"query": q, "hits": [{"email": d.email, "score": s,
                                      "notes": d.notes} for s, d in hits]}

    @app.get("/migration/report")
    def get_report():
        r = result.report
        return {"ingested": r.ingested, "structured": r.structured,
                "quarantined": r.quarantined, "final_records": r.final_records,
                "indexed": r.indexed}

    return app


def main() -> None:
    result = migrate()
    print("\n".join(result.report.lines()))

    print("\n=== Served via local API (FastAPI TestClient) ===")
    from fastapi.testclient import TestClient
    client = TestClient(build_api(result))

    r1 = client.get("/customer/ravi@example.com")
    print(f"GET /customer/ravi@example.com -> {r1.status_code} {r1.json()}")

    r2 = client.get("/customer/ghost@example.com")
    print(f"GET /customer/ghost@example.com -> {r2.status_code} (not found)")

    r3 = client.get("/search", params={"q": "refunds basic"})
    hits = r3.json()["hits"]
    print(f"GET /search?q=refunds basic -> {len(hits)} hit(s): "
          f"{[h['email'] for h in hits]}")

    r4 = client.get("/migration/report")
    print(f"GET /migration/report -> {r4.json()}")


if __name__ == "__main__":
    main()
