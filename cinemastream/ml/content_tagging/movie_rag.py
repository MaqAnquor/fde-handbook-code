"""
cinemastream/ml/content_tagging/movie_rag.py

Chapter 085 -- a Retrieval-Augmented Generation (RAG) pipeline over the
CinemaStream movie catalog, closing Part 8 and the "Ask Anything" arc.

Naive RAG, end to end:
    user query -> embed query -> vector-search top-K descriptions
                -> stuff into a grounded prompt -> (LLM answers)

This module builds the RETRIEVAL half (the part an FDE actually owns;
the LLM call is mocked as prompt construction, exactly as Ch061 did). It
uses the Chapter 084 MULTILINGUAL embeddings, so a query in Bahasa or
Hindi retrieves the right English movie -- cross-lingual search the
Chapter 083 bag-of-words could never do. It also demonstrates two things
every real RAG system needs and tutorials skip:

  1. ACCESS CONTROL -- the corpus mixes public (released) titles with
     internal (unreleased) ones. Retrieval MUST filter by the user's
     access level AT QUERY TIME. Skipping the filter = an access-control
     bypass that leaks unreleased titles to the public.
  2. The naive-RAG RECALL TRAP -- semantic search can miss an exact
     keyword (a title like "Hujan"), motivating hybrid RAG (Ch066).

Run (requires `pip install sentence-transformers`):
    python cinemastream/ml/content_tagging/movie_rag.py
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA_PATH = "cinemastream/data/movies.csv"

# Unreleased / staff-only titles -- NOT in the public catalog. The whole
# point of access control: a public user must never retrieve these.
INTERNAL_DOCS = [
    {"title": "Project Nightfall", "genre": "Thriller", "access": "internal",
     "description": "An unreleased CinemaStream original: a Singapore "
                    "detective hunts a serial hacker. Slated for 2027."},
    {"title": "Monsoon Heart 2", "genre": "Romance", "access": "internal",
     "description": "Unannounced sequel: the Kerala lovers reunite years "
                    "later. Confidential, not yet greenlit."},
]


def build_corpus():
    """Public catalog (movies.csv) + internal unreleased titles, each row
    tagged with an access level used for filtering at query time."""
    df = pd.read_csv(DATA_PATH)
    public = [{"title": r.title, "genre": r.genre, "access": "public",
               "description": r.description} for r in df.itertuples()]
    return public + INTERNAL_DOCS


def build_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return lambda texts: model.encode(list(texts), show_progress_bar=False)


def retrieve(query, corpus, doc_embeddings, encode, k=3, access=None):
    """Embed the query, cosine-rank the corpus, return top-K.
    If `access` is given, filter to docs the user is allowed to see
    BEFORE ranking -- the access-control gate. If None, NO filter (the
    bug)."""
    q = encode([query])[0]
    q = q / (np.linalg.norm(q) + 1e-9)
    docs = np.array([d / (np.linalg.norm(d) + 1e-9) for d in doc_embeddings])
    sims = docs @ q
    order = np.argsort(-sims)
    out = []
    for i in order:
        if access is not None and corpus[i]["access"] not in access:
            continue                     # access-control filter at query time
        out.append((corpus[i], float(sims[i])))
        if len(out) == k:
            break
    return out


def build_prompt(query, retrieved):
    """Naive RAG prompt: stuff retrieved descriptions in as grounding."""
    context = "\n".join(f"- {d['title']} ({d['genre']}): {d['description']}"
                        for d, _ in retrieved)
    return (f"Use ONLY the context to answer.\nContext:\n{context}\n\n"
            f"Question: {query}\nAnswer:")


def show(label, results):
    print(f"  {label}")
    for d, s in results:
        print(f"    [{s:.3f}] {d['title']} ({d['genre']}, {d['access']})")


def main():
    corpus = build_corpus()
    encode = build_embedder()
    doc_embeddings = encode([d["description"] for d in corpus])
    print(f"Indexed {len(corpus)} documents "
          f"({sum(d['access']=='public' for d in corpus)} public, "
          f"{sum(d['access']=='internal' for d in corpus)} internal)\n")

    # 1. Naive RAG: a semantic query retrieves the right released movie.
    print("1. Semantic retrieval (English query):")
    q1 = "a tense thriller about a cybercrime unit chasing hackers in Singapore"
    show(q1, retrieve(q1, corpus, doc_embeddings, encode, access={"public"}))
    print()

    # 2. Multilingual retrieval: a Malay query finds an English movie.
    print("2. Cross-lingual retrieval (Malay query -> English catalog):")
    q2 = "kisah cinta yang romantis pada musim hujan"   # "a romantic love story in the rainy season"
    show(q2, retrieve(q2, corpus, doc_embeddings, encode, access={"public"}))
    print()

    # 3. Access control: an "upcoming" query. Public user must NOT see
    #    internal titles; an unfiltered query leaks them.
    print("3. Access control on an 'upcoming thriller' query:")
    q3 = "an upcoming unreleased thriller about a detective and a hacker"
    show("PUBLIC user (filtered):",
         retrieve(q3, corpus, doc_embeddings, encode, access={"public"}))
    show("NO filter (the bug -- leaks internal titles):",
         retrieve(q3, corpus, doc_embeddings, encode, access=None))
    print()

    # 4. The recall trap: an exact-title query semantic search can fumble.
    print("4. Recall trap (exact title 'Hujan' -- semantic != keyword):")
    q4 = "Hujan"
    res = retrieve(q4, corpus, doc_embeddings, encode, k=3, access={"public"})
    show(q4, res)
    found = any("Hujan" in d["title"] for d, _ in res)
    print(f"    -> 'Hujan di Singapura' in top-3? {found}  "
          f"(if not: the recall trap -> motivates hybrid RAG, Ch066)")
    print()

    # The grounded prompt naive RAG would send to the LLM:
    print("5. The grounded prompt (naive RAG output for query 1):")
    print(build_prompt(q1, retrieve(q1, corpus, doc_embeddings, encode,
                                    k=2, access={"public"})))


if __name__ == "__main__":
    main()
