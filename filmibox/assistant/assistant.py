"""
filmibox/assistant/assistant.py

Chapter 093a -- Capstone 2: the Production AI Assistant.

This is the client-ready version of "Ask Anything", built for FilmiBox during
the Part 9 engagement. It is the architecture an FDE actually ships, and it
combines the two retrieval modes the book separated in Ch066/085c:

  * MCP / TOOL layer   -> STRUCTURED, live data (the subscriber DB).
                          A billing question calls a tool, at query time.
  * RAG layer          -> UNSTRUCTURED, static knowledge (wiki, tickets).
                          A policy question retrieves chunks, with an access
                          filter (Ch085) and injection sanitising (Ch085k).

Around that core sits the HARNESS that turns a demo into a product:

  * Access control      -> role decides which sources a request may touch.
  * Routing             -> tool vs RAG vs REFUSE (out-of-scope -> say so).
  * Prompt versioning   -> every answer records the prompt version that made it.
  * Observability       -> a Trace of Spans per request (Ch085i): latency,
                           route, cost; aggregated to p50/p95 + cost totals.
  * Cost control        -> tokens estimated and priced per request (Ch056/085i).
  * Evals               -> a golden set scored on routing + refusal (Ch085j).

The LLM call is SIMULATED deterministically (as Ch061/064 did): the assistant's
job is the 80% around the model -- retrieval, tools, access, traces, evals --
not the model weights. Retrieval is dependency-free lexical overlap; in
production the dense leg is the Ch085 multilingual encoder over pgvector
(the Ch085b decision). Latencies are deterministic (hash-derived) so the
observability numbers reproduce exactly.

Run:  python filmibox/assistant/assistant.py
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

# --------------------------------------------------------------------------
# Config -- everything the assistant is allowed to decide lives here.
# --------------------------------------------------------------------------
ACTIVE_PROMPT_VERSION = "v2"
PROMPT_REGISTRY: Dict[str, str] = {
    "v1": "Answer the question using the context. If unsure, guess.",
    # v2 added the refusal path + grounding instruction after eval (Ch085j).
    "v2": "Answer ONLY from the provided context or tool result. "
          "If the answer is not present, say you don't know and name the team to ask.",
}

# Claude Sonnet 4.6 pricing (Ch085i canonical): $3/MTok in, $15/MTok out, USD->SGD 1.36
USD_PER_MTOK_IN, USD_PER_MTOK_OUT, USD_TO_SGD = 3.00, 15.00, 1.36
REFUSAL = ("I don't have that in my approved sources. "
           "Please ask the {team} team directly.")


# --------------------------------------------------------------------------
# Access control -- role -> the sources a request may touch (Ch085/085k).
# Note: there is NO revenue source. The Ch086 discovery flagged founder MRR
# reports as out-of-scope for the all-staff assistant; it was never ingested.
# --------------------------------------------------------------------------
class Source(str, Enum):
    BILLING = "billing_tool"     # MCP tool over subscriber DB (structured)
    WIKI = "wiki_rag"            # RAG over internal wiki (unstructured)
    TICKETS = "tickets_rag"      # RAG over Zendesk tickets (unstructured)


ROLE_ACCESS: Dict[str, set] = {
    "support_agent": {Source.BILLING, Source.WIKI, Source.TICKETS},
    "new_hire": {Source.WIKI},
    "ops_lead": {Source.WIKI, Source.TICKETS},
}


# --------------------------------------------------------------------------
# Observability -- Trace/Span (Ch085i). Deterministic latency for reproducible
# percentiles: a base per span-type plus a hash-derived jitter from the query.
# --------------------------------------------------------------------------
SPAN_BASE_MS = {"access": 2, "route": 12, "tool": 30, "retrieve": 45,
                "generate": 360, "guardrail": 8}


def _jitter(query: str, span: str) -> int:
    h = hashlib.sha256(f"{span}:{query}".encode()).hexdigest()
    return int(h[:3], 16) % 40           # 0..39 ms, deterministic per (span, query)


@dataclass
class Span:
    name: str
    duration_ms: int


@dataclass
class Trace:
    query_hash: str
    role: str
    spans: List[Span] = field(default_factory=list)
    route: str = "?"
    prompt_version: str = ACTIVE_PROMPT_VERSION
    cost_sgd: float = 0.0

    def add(self, name: str, query: str) -> None:
        self.spans.append(Span(name, SPAN_BASE_MS[name] + _jitter(query, name)))

    @property
    def latency_ms(self) -> int:
        return sum(s.duration_ms for s in self.spans)


# --------------------------------------------------------------------------
# MCP / tool layer -- structured, live data. One bounded-domain tool.
# --------------------------------------------------------------------------
SUBSCRIBER_DB = {
    "ravi@example.com":  {"plan": "Premium", "last_charge": "1499 INR on 2026-06-01",
                          "status": "active", "double_charge": False},
    "siti@example.com":  {"plan": "Basic", "last_charge": "299 INR on 2026-06-03",
                          "status": "active", "double_charge": True},
    "minh@example.com":  {"plan": "Free", "last_charge": "none",
                          "status": "cancelled", "double_charge": False},
}


def billing_lookup(email: str) -> Optional[dict]:
    """MCP tool: query the subscriber DB by email. Bounded, validated input."""
    return SUBSCRIBER_DB.get(email.strip().lower())


# --------------------------------------------------------------------------
# RAG layer -- unstructured, static knowledge with access metadata (Ch085c).
# --------------------------------------------------------------------------
@dataclass
class Doc:
    source: Source
    title: str
    text: str


KNOWLEDGE_BASE: List[Doc] = [
    Doc(Source.WIKI, "Refund policy",
        "FilmiBox refunds are issued within 14 days of a billing complaint if the "
        "subscriber did not stream more than two titles that cycle."),
    Doc(Source.WIKI, "New-hire onboarding",
        "New engineers request laptop and VPN access on day one via the IT portal; "
        "manager approves within 24 hours."),
    Doc(Source.TICKETS, "Top complaint theme (last 7 days)",
        "Buffering on mobile during evening peak is the most common complaint this "
        "week, concentrated in Mumbai and Pune."),
]


def _tokens(s: str) -> set:
    return {w.strip(".,?!").lower() for w in s.split() if len(w) > 2}


def retrieve(query: str, allowed: set, k: int = 1) -> List[tuple]:
    """Lexical overlap retrieval, FILTERED by the caller's allowed sources."""
    q = _tokens(query)
    scored = []
    for doc in KNOWLEDGE_BASE:
        if doc.source not in allowed:
            continue                      # access filter BEFORE ranking (Ch085)
        overlap = len(q & _tokens(doc.text))
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: -x[0])
    return [(s, d) for s, d in scored[:k]]


# --------------------------------------------------------------------------
# Router -- tool vs RAG vs REFUSE.
# --------------------------------------------------------------------------
BILLING_WORDS = {"charge", "charged", "billing", "refund", "double", "payment", "plan"}
TEAM_FOR_OUT_OF_SCOPE = "partnerships"


def route(query: str) -> str:
    q = _tokens(query)
    if q & BILLING_WORDS and any(c in query for c in "@"):
        return "BILLING_TOOL"
    if q & {"refund", "policy", "onboarding", "vpn", "laptop", "complaint",
            "buffering", "theme"}:
        return "RAG"
    return "REFUSE"


# --------------------------------------------------------------------------
# Output guardrail -- scrub obvious PII the assistant must never echo (Ch085k).
# --------------------------------------------------------------------------
def guardrail(text: str) -> str:
    import re
    return re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[email]", text)


# --------------------------------------------------------------------------
# The simulated LLM -- deterministic. Grounds in the tool result / context.
# --------------------------------------------------------------------------
def generate(route_kind: str, payload, query: str) -> str:
    if route_kind == "BILLING_TOOL":
        if payload is None:
            return "No subscriber found for that email."
        flag = " A duplicate charge is present." if payload["double_charge"] else ""
        return (f"Account is {payload['status']} on the {payload['plan']} plan; "
                f"last charge {payload['last_charge']}.{flag}")
    if route_kind == "RAG":
        if not payload:
            return REFUSAL.format(team="support")
        return payload[0][1].text          # quote the top retrieved chunk
    return REFUSAL.format(team=TEAM_FOR_OUT_OF_SCOPE)


# --------------------------------------------------------------------------
# The one instrumented request path.
# --------------------------------------------------------------------------
@dataclass
class Response:
    answer: str
    trace: Trace
    blocked: bool = False


def _email_in(query: str) -> Optional[str]:
    for tok in query.split():
        if "@" in tok:
            return tok.strip(".,?!")
    return None


def ask(role: str, query: str) -> Response:
    qhash = hashlib.sha256(query.encode()).hexdigest()[:16]
    trace = Trace(query_hash=qhash, role=role)
    allowed = ROLE_ACCESS.get(role, set())

    trace.add("access", query)
    trace.add("route", query)
    kind = route(query)

    payload = None
    if kind == "BILLING_TOOL":
        if Source.BILLING not in allowed:                 # access denied
            trace.route = "REFUSE(access)"
            answer = ("The billing tool is not available for your role. "
                      "Please ask a support agent or your manager.")
            trace.add("guardrail", query)
            _price(trace, query, answer)
            return Response(answer, trace, blocked=True)
        trace.add("tool", query)
        payload = billing_lookup(_email_in(query) or "")
        trace.route = "BILLING_TOOL"
    elif kind == "RAG":
        trace.add("retrieve", query)
        payload = retrieve(query, allowed)
        trace.route = "WIKI_RAG" if (payload and payload[0][1].source == Source.WIKI) \
            else ("TICKETS_RAG" if payload else "RAG_EMPTY")
    else:
        trace.route = "REFUSE(scope)"

    trace.add("generate", query)
    answer = guardrail(generate(kind, payload, query))
    trace.add("guardrail", query)
    _price(trace, query, answer)
    return Response(answer, trace)


def _price(trace: Trace, query: str, answer: str) -> None:
    sys_tokens = len(PROMPT_REGISTRY[trace.prompt_version].split()) * 2
    in_tok = (len(query.split()) + sys_tokens) * 1.3
    out_tok = len(answer.split()) * 1.3
    usd = in_tok / 1e6 * USD_PER_MTOK_IN + out_tok / 1e6 * USD_PER_MTOK_OUT
    trace.cost_sgd = round(usd * USD_TO_SGD, 6)


# --------------------------------------------------------------------------
# Observability roll-up + a golden-set eval (Ch085j).
# --------------------------------------------------------------------------
def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


GOLDEN = [
    ("support_agent", "double charge on siti@example.com?", "BILLING_TOOL"),
    ("support_agent", "what is the refund policy?", "WIKI_RAG"),
    ("ops_lead", "what is the top complaint theme this week?", "TICKETS_RAG"),
    ("support_agent", "what are our partner contract terms?", "REFUSE(scope)"),
    ("new_hire", "how do I get VPN access?", "WIKI_RAG"),
    ("new_hire", "double charge on ravi@example.com?", "REFUSE(access)"),
]


def run_evals() -> tuple:
    passed, traces = 0, []
    for role, q, expected_route in GOLDEN:
        r = ask(role, q)
        traces.append(r.trace)
        if r.trace.route == expected_route:
            passed += 1
    return passed, len(GOLDEN), traces


def main() -> None:
    print("=== Live requests ===")
    demos = [
        ("support_agent", "double charge on siti@example.com?"),
        ("new_hire", "how do I get VPN access?"),
        ("support_agent", "what are our partner contract terms?"),
        ("new_hire", "double charge on ravi@example.com?"),
    ]
    for role, q in demos:
        r = ask(role, q)
        print(f"[{role}] {q}")
        print(f"   route={r.trace.route} | prompt={r.trace.prompt_version} | "
              f"latency={r.trace.latency_ms}ms | cost=S${r.trace.cost_sgd}")
        print(f"   -> {r.answer}")

    print("\n=== Eval harness (golden set) ===")
    passed, total, traces = run_evals()
    print(f"Routing/refusal accuracy: {passed}/{total}")

    print("\n=== Observability roll-up (eval traffic) ===")
    lats = [t.latency_ms for t in traces]
    print(f"requests={len(traces)} | p50={percentile(lats,50)}ms | "
          f"p95={percentile(lats,95)}ms | total_cost=S${round(sum(t.cost_sgd for t in traces),5)}")
    routes = {}
    for t in traces:
        routes[t.route] = routes.get(t.route, 0) + 1
    print("route mix: " + ", ".join(f"{k}={v}" for k, v in sorted(routes.items())))


if __name__ == "__main__":
    main()
