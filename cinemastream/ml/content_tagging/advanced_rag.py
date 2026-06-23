"""
cinemastream/ml/content_tagging/advanced_rag.py

Chapter 085a -- Advanced RAG patterns, layered on Ch085's movie_rag.py.
Naive RAG (Ch085) = embed -> vector-search -> stuff -> answer, with two honest
failures: the RECALL TRAP (semantic search misses an exact title like "Hujan")
and no recovery when retrieval is weak. This module adds the production patterns
that fix those:

  1. Query rewriting   -- expand a vague query so it uses words the corpus has.
  2. HyDE              -- retrieve with a *hypothetical answer document*, not
                          the bare query.
  3. Reranking         -- a cheap first stage proposes; a stronger reranker
                          (query-aspect coverage) reorders the shortlist.
  4. Hybrid + keyword  -- a BM25 leg catches exact tokens the dense leg blurs
                          away (the recall-trap fix).
  5. Self-/Corrective  -- retrieve -> GRADE -> if weak, rewrite & retry, else
     RAG                 refuse. The loop that stops confident wrong answers.

NOTE ON THE EMBEDDER: Ch085's movie_rag.py uses the real Ch084 multilingual
sentence-transformer for its dense leg. This chapter's runnable code uses a
dependency-free lexical substrate (TF-IDF cosine for the "dense/semantic" leg,
BM25 for the keyword leg) so every output reproduces with no GPU or model
download. The PATTERNS are embedder-agnostic -- in production the dense leg is
movie_rag.py's `retrieve()`; everything here sits on top unchanged.

Run:  python cinemastream/ml/content_tagging/advanced_rag.py
"""
import sys
sys.path.insert(0, ".")

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

DATA_PATH = "cinemastream/data/movies.csv"

STOPWORDS = {"a", "an", "the", "and", "of", "in", "to", "for", "on", "about",
             "with", "that", "this", "is", "it", "as", "their", "who", "where",
             "while", "before", "during"}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


# --------------------------------------------------------------------------- #
# Dense (semantic) leg -- TF-IDF cosine stand-in for the multilingual encoder
# --------------------------------------------------------------------------- #
class TfidfIndex:
    def __init__(self, docs):
        self.docs_tokens = [tokenize(d) for d in docs]
        df = Counter()
        for toks in self.docs_tokens:
            for t in set(toks):
                df[t] += 1
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.vocab = {t: i for i, t in enumerate(sorted(self.idf))}
        self.matrix = np.stack([self._vec(toks) for toks in self.docs_tokens])

    def _vec(self, toks):
        v = np.zeros(len(self.vocab))
        for t, c in Counter(toks).items():
            if t in self.vocab:
                v[self.vocab[t]] = c * self.idf[t]
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def scores(self, query):
        return self.matrix @ self._vec(tokenize(query))


# --------------------------------------------------------------------------- #
# Keyword leg -- BM25 (catches exact tokens; also offers query-aspect coverage)
# --------------------------------------------------------------------------- #
class BM25Index:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs_tokens = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = float(np.mean([len(t) for t in self.docs_tokens]))
        df = Counter()
        for toks in self.docs_tokens:
            for t in set(toks):
                df[t] += 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.docs_tokens))
        q = tokenize(query)
        for i, toks in enumerate(self.docs_tokens):
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in q:
                if t in tf:
                    s += self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (
                        tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out

    def coverage(self, query, i):
        """Reranker signal: idf summed over the DISTINCT query terms a doc
        matches -- rewards covering more query aspects, not repeating one."""
        present = set(tokenize(query)) & set(self.docs_tokens[i])
        return sum(self.idf.get(t, 0.0) for t in present)


def rank(scores, k=None):
    order = list(np.argsort(-scores))
    return order if k is None else order[:k]


def rrf(rankings, k=60, top=5):
    """Reciprocal Rank Fusion: merge ranked id-lists into one candidate pool."""
    fused = Counter()
    for r in rankings:
        for pos, doc_id in enumerate(r):
            fused[doc_id] += 1.0 / (k + pos + 1)
    return [doc_id for doc_id, _ in fused.most_common(top)]


# Pattern 1: query rewriting (deterministic expansion of vague terms)
REWRITE_MAP = {"scary": "thriller suspense", "hacker": "hacker cybercrime",
               "funny": "comedy", "sad": "tragic drama"}


def rewrite_query(q):
    out = q.lower()
    for k, v in REWRITE_MAP.items():
        out = re.sub(rf"\b{k}\b", v, out)
    return out


# Pattern 2: HyDE -- a hypothetical answer document (an LLM writes this in
# production; here a theme lexicon stands in for that generation step).
HYDE_LEXICON = {"rainy": "monsoon rain love story tender couple",
                "hacker": "cybercrime security breach unit",
                "courtroom": "trial verdict justice inheritance"}


def hyde(query):
    extra = " ".join(HYDE_LEXICON[t] for t in tokenize(query) if t in HYDE_LEXICON)
    return f"A film about {query}. {extra}".strip()


# Pattern 3: reranking -- reorder a shortlist by a stronger scorer (coverage)
def rerank(query, cand_ids, bm25):
    return sorted(cand_ids, key=lambda i: -bm25.coverage(query, i))


# Pattern 5: Self-/Corrective RAG -- retrieve -> grade -> correct OR refuse
def relevance(query, doc_text):
    q = set(tokenize(query))
    d = set(tokenize(doc_text))
    return len(q & d) / (len(q) + 1e-9)


def self_rag(query, dense, bm25, doc_texts, threshold=0.30, k=3):
    log = []

    def retrieve_and_grade(qq):
        pool = rrf([rank(dense.scores(qq), 10), rank(bm25.scores(qq), 10)], top=k)
        best = max((relevance(qq, doc_texts[i]) for i in pool), default=0.0)
        return pool, best

    pool, grade = retrieve_and_grade(query)
    log.append(("retrieve", round(grade, 3), query))
    if grade >= threshold:
        return pool, log, "answer"
    rq = rewrite_query(query)                       # CORRECTIVE: rewrite & retry
    pool2, grade2 = retrieve_and_grade(rq)
    log.append(("correct(rewrite)", round(grade2, 3), rq))
    if grade2 >= threshold:
        return pool2, log, "answer-after-correction"
    log.append(("refuse", round(grade2, 3), "no document cleared the relevance bar"))
    return [], log, "refuse"


# --------------------------------------------------------------------------- #
# Section 2 -- a small, controlled CinemaStream doc set (crisp mechanics)
# --------------------------------------------------------------------------- #
DEMO = [
    ("Hujan di Singapura", "A Singapore cybercrime unit races to stop a hacker before a tense deadline."),
    ("Monsoon Heart", "A tender love story unfolds during the Kerala monsoon rain."),
    ("Office Hari Ini", "A Jakarta advertising agency adjusts to permanent remote work, a workplace comedy."),
    ("Hack Attack Hijinks", "A slapstick comedy where a hacker outwits a hacker who outwits a hacker, thriller jokes."),
    ("The Silent Verdict", "A courtroom thriller about a bitterly contested family inheritance."),
]
DEMO_TITLES = [t for t, _ in DEMO]
DEMO_DESCS = [d for _, d in DEMO]


def _show(ids, titles, label, scores=None):
    print(f"  {label}")
    for i in ids:
        sc = f"[{scores[i]:.3f}] " if scores is not None else ""
        print(f"    {sc}{titles[i]}")


def section2():
    dense = TfidfIndex(DEMO_DESCS)
    bm25 = BM25Index([f"{t} {d}" for t, d in zip(DEMO_TITLES, DEMO_DESCS)])
    print("== SECTION 2 (controlled 5-doc set) ==\n")

    print("1. Query rewriting (vague words -> words the corpus actually uses):")
    q = "a scary picture"
    _show(rank(dense.scores(q), 2), DEMO_TITLES, f"bare {q!r} (no corpus terms -> noise):", dense.scores(q))
    rq = rewrite_query(q)
    _show(rank(dense.scores(rq), 2), DEMO_TITLES, f"rewritten {rq!r}:", dense.scores(rq))
    print()

    print("2. HyDE (retrieve with a hypothetical answer document):")
    q = "a film for a rainy evening"
    _show(rank(dense.scores(q), 2), DEMO_TITLES, f"bare {q!r} (no corpus terms -> noise):", dense.scores(q))
    h = hyde(q)
    _show(rank(dense.scores(h), 2), DEMO_TITLES, f"HyDE doc {h!r}:", dense.scores(h))
    print()

    print("3. Reranking (cheap BM25 recall -> coverage rerank):")
    q = "cybercrime hacker thriller"
    stage1 = rank(bm25.scores(q), 4)
    _show(stage1, DEMO_TITLES, "stage-1 BM25 (term frequency favours the keyword-stuffed decoy):", bm25.scores(q))
    reranked = rerank(q, stage1, bm25)
    _show(reranked, DEMO_TITLES, "after coverage rerank (rewards covering distinct query aspects):")
    print()


# --------------------------------------------------------------------------- #
# Section 3 -- the full 300-movie catalog (realistic application)
# --------------------------------------------------------------------------- #
def section3():
    df = pd.read_csv(DATA_PATH)
    titles, descs = list(df["title"]), list(df["description"])
    dense = TfidfIndex(descs)
    bm25 = BM25Index([f"{t} {d}" for t, d in zip(titles, descs)])
    print(f"== SECTION 3 (full catalog, {len(titles)} movies) ==\n")

    print("A. Recall-trap fix -- query 'Hujan' (exact title token):")
    q = "Hujan"
    _show(rank(dense.scores(q), 3), titles, "dense/semantic leg (over descriptions) -- MISSES the title:", dense.scores(q))
    _show(rank(bm25.scores(q), 3), titles, "keyword/BM25 leg (title+desc) -- CATCHES it at #1:", bm25.scores(q))
    print(f"    -> dense found 'Hujan di Singapura'? "
          f"{'Hujan di Singapura' in [titles[i] for i in rank(dense.scores(q), 3)]}; "
          f"keyword found it? {'Hujan di Singapura' == titles[rank(bm25.scores(q), 1)[0]]}\n")

    print("B. Self-/Corrective RAG -- grade, then correct or refuse:")
    doc_texts = [f"{t} {d}" for t, d in zip(titles, descs)]
    for q in ["a Singapore cybercrime thriller",
              "a scary movie",
              "a Korean zombie outbreak on a bullet train"]:
        pool, log, decision = self_rag(q, dense, bm25, doc_texts)
        print(f"  query: {q!r}")
        for step, grade, detail in log:
            print(f"    {step:22s} grade={grade:<5} :: {detail}")
        print(f"    decision: {decision}"
              + (f" -> top: {titles[pool[0]]}" if pool else " (no answer returned)") + "\n")


def main():
    section2()
    section3()


if __name__ == "__main__":
    main()
