"""
cinemastream/ml/content_tagging/multilingual_genre.py

Chapter 084 -- closing Chapter 083's multilingual gap with transformer
sentence embeddings.

Chapter 083's TF-IDF genre tagger (text_genre.py) scored 0.887 on English
descriptions but only 0.278 on the non-English ones -- it learned an
English vocabulary and was blind to Malay/Hindi/etc. This module replaces
the TF-IDF features with MULTILINGUAL TRANSFORMER EMBEDDINGS: each
description is encoded into a dense vector by a model pretrained on many
languages at once, so cross-language synonyms (English "killer", Malay
"pembunuh") map to NEARBY vectors. The same logistic-regression head then
works across languages, and the multilingual gap collapses.

Two things are demonstrated:
  1. A genre classifier on multilingual embeddings -- English accuracy
     stays strong AND non-English accuracy jumps from 0.278 to ~0.72.
  2. Cross-lingual similarity: why it works (synonyms across languages are
     closer than unrelated phrases in the same language).

Run (requires `pip install sentence-transformers`; falls back to mean-pooled
mBERT from `transformers` if unavailable):
    python cinemastream/ml/content_tagging/multilingual_genre.py
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from cinemastream.scripts.make_movies_data import NON_ENGLISH_TEMPLATES

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DATA_PATH = "cinemastream/data/movies.csv"
NON_ENGLISH = {text for (_lang, _genre, text) in NON_ENGLISH_TEMPLATES}


def build_embedder():
    """Prefer the multilingual sentence-transformer; fall back to
    mean-pooled multilingual BERT. Returns (encode_fn, model_name)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        def encode(texts):
            return model.encode(list(texts), batch_size=32,
                                show_progress_bar=False)
        return encode, "paraphrase-multilingual-MiniLM-L12-v2"
    except Exception:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        bert = AutoModel.from_pretrained("bert-base-multilingual-cased").eval()

        def encode(texts):  # masked mean-pool of the last hidden state
            out = []
            texts = list(texts)
            with torch.no_grad():
                for i in range(0, len(texts), 16):
                    enc = tok(texts[i:i + 16], padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
                    hs = bert(**enc).last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    out.append(((hs * mask).sum(1) /
                                mask.sum(1).clamp(min=1e-9)).numpy())
            return np.vstack(out)
        return encode, "mBERT (mean-pooled)"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    df = pd.read_csv(DATA_PATH)
    df["is_english"] = ~df["description"].isin(NON_ENGLISH)
    encode, model_name = build_embedder()
    print(f"Embedding model: {model_name}")
    print(f"{len(df)} movies | English {df['is_english'].sum()} | "
          f"non-English {(~df['is_english']).sum()}\n")

    # --- Genre classifier on multilingual embeddings ---
    eng = df[df["is_english"]]
    ne = df[~df["is_english"]]
    X_eng = encode(eng["description"].values)
    Xtr, Xte, ytr, yte = train_test_split(
        X_eng, eng["genre"].values, test_size=0.25,
        random_state=RANDOM_SEED, stratify=eng["genre"].values)
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced",
                             random_state=RANDOM_SEED).fit(Xtr, ytr)

    eng_acc = accuracy_score(yte, clf.predict(Xte))
    eng_f1 = f1_score(yte, clf.predict(Xte), average="macro")
    ne_acc = accuracy_score(ne["genre"].values, clf.predict(encode(ne["description"].values)))

    print(f"{'':24s} {'English':>9s} {'non-English':>12s} {'gap':>7s}")
    print(f"{'Ch083 TF-IDF baseline':24s} {0.887:>9.3f} {0.278:>12.3f} {0.610:>7.3f}")
    print(f"{'multilingual embeddings':24s} {eng_acc:>9.3f} {ne_acc:>12.3f} "
          f"{eng_acc - ne_acc:>7.3f}")
    print(f"(English macro-F1 {eng_f1:.3f})\n")

    # --- Why it works: cross-lingual similarity ---
    phrases = {
        "EN thriller": "a killer and a detective",
        "MS thriller": "seorang pembunuh dan seorang detektif",
        "EN wedding ": "a wedding and a celebration",
    }
    e = dict(zip(phrases, encode(list(phrases.values()))))
    print("Cross-lingual similarity (cosine):")
    print(f"  EN-thriller vs MS-thriller (synonyms across languages): "
          f"{cosine(e['EN thriller'], e['MS thriller']):.3f}")
    print(f"  EN-thriller vs EN-wedding  (unrelated, same language):  "
          f"{cosine(e['EN thriller'], e['EN wedding ']):.3f}")


if __name__ == "__main__":
    main()
