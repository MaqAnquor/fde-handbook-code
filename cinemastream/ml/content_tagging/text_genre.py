"""
cinemastream/ml/content_tagging/text_genre.py

Genre tagging from a movie DESCRIPTION using TF-IDF + LogisticRegression.

This is the workhorse text classifier for CinemaStream's content-tagging
pipeline (NLP chapters 083-085). It teaches two things at once:

  1. A bag-of-words / TF-IDF model can predict GENRE from an English
     description with high accuracy when the descriptions carry genuine
     genre signal (here, macro-F1 >= 0.85 on a held-out English test set).

  2. The SAME model does NOTABLY WORSE on the non-English descriptions in
     the catalog -- even though those rows are correctly labeled. The model
     simply never learned the Malay/Indonesian/Hindi/Tamil/Vietnamese/
     Tagalog vocabulary, so it has no signal to go on. That honest gap is
     the multilingual-NLP teaching point; we measure it, we don't hide it.

Run:
    python cinemastream/ml/content_tagging/text_genre.py
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
DATA_PATH = Path("cinemastream/data/movies.csv")

# The exact non-English description strings the generator emits. We split
# English vs non-English rows by membership in this set -- a deterministic,
# leak-free rule (we control the generator, so this stays in sync).
try:
    # Preferred: import the templates straight from the generator.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from cinemastream.scripts.make_movies_data import NON_ENGLISH_TEMPLATES
    NON_ENGLISH_DESCRIPTIONS = {text for (_lang, _genre, text) in NON_ENGLISH_TEMPLATES}
except Exception:
    # Fallback: empty set means "detect by original_lang != en AND non-ascii-ish".
    NON_ENGLISH_DESCRIPTIONS = set()


def is_english_description(row):
    """A row is non-English iff its description is one the generator wrote
    in a regional language. Everything else (incl. the canonical English-text
    rows 101-103) counts as English."""
    if NON_ENGLISH_DESCRIPTIONS:
        return row["description"] not in NON_ENGLISH_DESCRIPTIONS
    # Fallback heuristic if the import failed.
    return row["original_lang"] == "en"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["is_english"] = df.apply(is_english_description, axis=1)
    return df


def build_model():
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        max_iter=2000,
        C=10.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return vec, clf


def main():
    df = load_data()
    english = df[df["is_english"]].copy()
    non_english = df[~df["is_english"]].copy()

    print("=" * 60)
    print("CinemaStream genre tagger -- TF-IDF + LogisticRegression")
    print("=" * 60)
    print(f"Total movies                : {len(df)}")
    print(f"English-description rows     : {len(english)}")
    print(f"Non-English-description rows : {len(non_english)}")
    print()

    # --- Train / test on ENGLISH rows ---
    X = english["description"].values
    y = english["genre"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    vec, clf = build_model()
    X_train_v = vec.fit_transform(X_train)
    clf.fit(X_train_v, y_train)

    X_test_v = vec.transform(X_test)
    y_pred = clf.predict(X_test_v)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("-" * 60)
    print("ENGLISH held-out test set (test_size=0.25, stratified)")
    print("-" * 60)
    print(f"Test accuracy : {acc:.3f}")
    print(f"Macro-F1      : {macro_f1:.3f}   (target >= 0.85)")
    print()
    print("Per-genre report (support = test rows per genre):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # --- Evaluate SAME model on the NON-ENGLISH rows ---
    print("-" * 60)
    print("NON-ENGLISH descriptions (separate held-out set)")
    print("-" * 60)
    if len(non_english):
        Xn = non_english["description"].values
        yn = non_english["genre"].values
        Xn_v = vec.transform(Xn)
        yn_pred = clf.predict(Xn_v)
        n_acc = accuracy_score(yn, yn_pred)
        n_macro_f1 = f1_score(yn, yn_pred, average="macro")
        print(f"Rows          : {len(non_english)}")
        print(f"Accuracy      : {n_acc:.3f}   (expected: much lower)")
        print(f"Macro-F1      : {n_macro_f1:.3f}")
        gap = acc - n_acc
        print(f"Multilingual gap (English acc - non-English acc): {gap:.3f}")
    else:
        print("No non-English rows found.")

    print("=" * 60)
    status = "PASS" if macro_f1 >= 0.85 else "BELOW TARGET"
    print(f"English macro-F1 target (>=0.85): {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
