"""
cinemastream/scripts/make_movies_data.py

Builds cinemastream/data/movies.csv -- CinemaStream's movie catalog with
multilingual descriptions, used by the NLP chapters (083-085) to teach a
content-tagging model that reads a description and predicts its genre.

Schema (bible-canonical):
    movie_id, title, original_lang, genre, release_year, runtime_min, description

The first three rows are the bible's canonical sample movies, verbatim.
The rest are generated from genre-specific vocabulary and regional
settings, so the descriptions carry a learnable genre signal while
reflecting CinemaStream's SEA/India, multi-language reality.

A meaningful slice of descriptions are written in the title's original
language (not English), on purpose -- the multilingual messiness an FDE
must handle. These non-English descriptions are GENRE-CONSISTENT: each
non-English template is tagged with the genre it belongs to, so the rows
are correctly labeled. An English-vocabulary TF-IDF model still struggles
on them because it shares no vocabulary -- that gap is the teaching point.

Run:
    python cinemastream/scripts/make_movies_data.py
"""

import csv
from pathlib import Path

import numpy as np

RANDOM_SEED = 42
OUT_PATH = Path("cinemastream/data/movies.csv")

GENRES = ["Action", "Drama", "Comedy", "Documentary", "Thriller", "Romance"]
LANGS = ["en", "ms", "id", "tl", "th", "vi", "hi", "ta"]

# Regional settings give the descriptions CinemaStream's geography.
SETTINGS = ["Singapore", "Jakarta", "Manila", "Bangkok", "Kuala Lumpur",
            "Mumbai", "Hanoi", "coastal Kerala", "a Jakarta high-rise",
            "the streets of Chennai", "a Bangkok night market",
            "a Manila call center", "the Hanoi old quarter"]

# Genre-specific vocabulary pools -- the signal a text model learns.
GENRE_WORDS = {
    "Action": ["explosion", "chase", "gunfight", "mission", "escape", "soldier",
               "heist", "ambush", "rescue", "showdown", "fugitive", "elite commando unit",
               "high-speed pursuit", "weapon", "betrayal", "combat", "firefight",
               "raid", "sabotage", "extraction", "warlord", "mercenary"],
    "Drama": ["family", "grief", "sacrifice", "estranged brothers", "memory", "illness",
              "reconciliation", "quiet dignity", "loss", "inheritance", "generations",
              "hardship", "redemption", "tender", "burden", "forgiveness",
              "regret", "duty", "mourning", "fractured marriage", "longing", "atonement"],
    "Comedy": ["mishap", "hilarious", "awkward", "wedding", "mix-up", "nosy neighbors",
               "scheme", "blunder", "chaotic", "misunderstanding", "office",
               "prank", "ridiculous", "rivalry", "disastrous", "uproarious",
               "farce", "bumbling", "slapstick", "absurd", "zany", "embarrassing"],
    "Documentary": ["explores", "examines", "archival", "interviews", "footage",
                    "investigates", "real", "decades", "ecosystem", "portrait",
                    "uncovers", "chronicle", "firsthand", "vanishing",
                    "testimony", "landscape", "fieldwork", "researchers",
                    "endangered", "migration", "ethnographic", "observational"],
    "Thriller": ["killer", "conspiracy", "detective", "suspense", "vanishes",
                 "stalker", "deadline", "cover-up", "witness", "manhunt",
                 "double-cross", "surveillance", "ransom", "paranoia", "clue",
                 "informant", "assassin", "interrogation", "hostage", "wiretap",
                 "fugitive lead", "cat-and-mouse"],
    "Romance": ["love", "longing", "reunite", "heartbreak", "wedding", "passion",
                "love letters", "first meeting", "yearning", "tender", "courtship",
                "fate", "rekindle", "embrace", "devotion", "promise",
                "infatuation", "slow dance", "stolen glances", "marriage proposal",
                "heartache", "soulmate"],
}

# Genre-NEUTRAL connective phrases. Any genre can draw these, so a slice of
# each description is genuinely ambiguous. This is what keeps the English
# classifier HONEST -- it lands around 0.90-0.93 macro-F1, not a suspicious
# 1.0 you'd only get from perfectly disjoint vocabulary. The pedagogical
# point is "real signal, real errors", not a rigged toy.
SHARED_WORDS = [
    "a stranger arrives", "an old secret surfaces", "the city at night",
    "a phone rings at midnight", "a long-buried past", "time running out",
    "a knock at the door", "rain over the harbor", "a final confrontation",
    "an unexpected visitor", "nothing is what it seems", "a fateful choice",
]

# Longer templates that pack 5 words each -> clear but not trivial TF-IDF signal.
TEMPLATES = [
    "Set in {setting}, a story of {w1} and {w2} unfolds as {w3} forces a "
    "reckoning, while {w4} and {w5} pull everyone toward the edge.",
    "In {setting}, {w1} collides with {w2} in a tale where {w3} gives way to "
    "{w4}, and {w5} decides who is left standing.",
    "A {w1}-driven journey through {setting}, where {w2} and {w3} test every "
    "bond as {w4} closes in and {w5} cannot be undone.",
    "When {w1} comes to {setting}, {w2} and {w3} are never far behind, and "
    "soon {w4} and {w5} consume the lives caught in between.",
    "{setting} becomes the stage for {w1}, {w2}, and {w3}, until {w4} turns "
    "everything upside down and {w5} changes the ending.",
    "Across {setting}, {w1} and {w2} simmer beneath the surface while {w3}, "
    "{w4}, and {w5} drive the story to its breaking point.",
]

# Non-English descriptions, tagged PER GENRE so they are correctly labeled.
# (Romanized SEA/Indian languages -- ascii, but disjoint vocab from English.)
# Each row picks one matching its genre, so labels are clean; the model
# struggles only because it never saw this vocabulary, not because of noise.
NON_ENGLISH_BY_GENRE = {
    "Action": [
        ("ms", "Satu misi rahsia, letupan dan kejar-mengejar di lebuh raya "
               "Kuala Lumpur ketika seorang askar memburu pengkhianat."),
        ("id", "Sebuah perampokan bersenjata berubah menjadi baku tembak dan "
               "pengejaran berkecepatan tinggi di jalanan Jakarta."),
        ("hi", "Ek sainik, ek dhamaka aur Mumbai ki sadkon par khatarnak "
               "peechha jab gaddaar ko pakadna hi mission ban jaata hai."),
    ],
    "Drama": [
        ("ms", "Sebuah kisah tentang keluarga, pengorbanan dan kenangan yang "
               "perlahan memulihkan dua beradik yang terpisah di Kuala Lumpur."),
        ("hi", "Bichhde hue parivaar, bimari aur kurbani ki ek shaant kahani "
               "jahan maafi aur pachhtawa dheere dheere ghar laut aate hain."),
        ("ta", "Pirintha kudumbam, thiyaagam matrum nினைவுகள் parri oru amaithiyana "
               "kathai, Chennai theruvil manippu thedi alaiyum manidhargal."),
    ],
    "Comedy": [
        ("id", "Cerita lucu tentang salah faham dan kekacauan ketika sebuah "
               "kantor Jakarta merancang pesta pernikahan yang berantakan."),
        ("tl", "Isang nakakatawang gulo ng mga kapitbahay, kasalan, at sunod-"
               "sunod na kalokohan sa isang gusali sa Maynila."),
        ("ms", "Satu komedi kekok tentang jiran yang sibuk dan rancangan majlis "
               "kahwin yang bertukar menjadi bencana yang kelakar di Singapura."),
    ],
    "Documentary": [
        ("vi", "Mot bo phim tai lieu kham pha he sinh thai dang bien mat va "
               "phong van nhung nguoi dan chai cuoi cung o vinh Ha Long."),
        ("id", "Film dokumenter yang menyelidiki migrasi dan ekosistem yang "
               "terancam lewat wawancara dan rekaman arsip selama puluhan tahun."),
    ],
    "Thriller": [
        ("vi", "Mot cau chuyen hoi hop ve mot tham tu lan theo ke giet nguoi va "
               "vu mat tich bi an giua mang luoi giam sat o Ha Noi."),
        ("ms", "Satu siasatan menegangkan apabila seorang detektif memburu "
               "pembunuh dan saksi yang hilang sebelum tarikh akhir tiba."),
        ("hi", "Ek jasoos, ek saazish aur gumshuda gawah ki talash, jab har "
               "suraag Chennai ki galiyon mein khatre ki taraf le jaata hai."),
    ],
    "Romance": [
        ("hi", "Pyaar, intezaar aur taqdeer ki ek kahani jahan do dil Mumbai ki "
               "baarish mein milte, bichhadte aur phir se ek hote hain."),
        ("ms", "Kisah cinta, kerinduan dan janji yang menyatukan semula dua "
               "kekasih di bawah hujan di Kuala Lumpur."),
        ("ta", "Kaadhal, yengudhal matrum vaakkurudhi parri oru kathai, mazhaiyil "
               "மீண்டும் சந்திக்கும் iru ullangal Chennai-yil."),
    ],
}

# Flat list for sampling, with the genre each belongs to.
NON_ENGLISH_TEMPLATES = [
    (lang, genre, text)
    for genre, items in NON_ENGLISH_BY_GENRE.items()
    for (lang, text) in items
]

# The bible's canonical first three movies -- verbatim, do not change.
CANONICAL = [
    (101, "Monsoon Heart", "hi", "Drama", 2024, 132,
     "A monsoon love story set in coastal Kerala."),
    (102, "Hujan di Singapura", "ms", "Thriller", 2023, 118,
     "A Singapore cybercrime unit races a deadline."),
    (103, "Office Hari Ini", "id", "Comedy", 2024, 96,
     "A Jakarta ad agency adjusts to permanent remote work."),
]

TITLE_WORDS = ["Last", "Silent", "Monsoon", "Midnight", "Broken", "Golden",
               "Hidden", "Lost", "Burning", "Quiet", "Crimson", "Northern",
               "Echo", "Paper", "Iron", "Velvet", "Distant", "Riverside"]
TITLE_NOUNS = ["Promise", "Harbor", "Signal", "Garden", "Tide", "Market",
               "Letter", "Station", "Mirror", "Season", "Crossing", "Vow",
               "Shadow", "Lantern", "Current", "Verdict", "Ledger", "Bridge"]

# Target number of non-English rows (genre-consistent, correctly labeled).
N_NON_ENGLISH = 18
WORDS_PER_DESC = 5
# Of the 5 words in an English description, up to this many slots may be
# replaced by genre-NEUTRAL phrases (each with SHARED_PROB chance), injecting
# realistic ambiguity so the classifier scores honestly (~0.90), not 1.0.
N_SHARED_SLOTS = 3
SHARED_PROB = 0.8


def make_movies(n=300, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    rows = list(CANONICAL)
    movie_id = 104
    n_generated = n - len(CANONICAL)

    # Assign genres in a balanced round-robin (shuffled) so counts are even.
    genre_assignments = (GENRES * ((n_generated // len(GENRES)) + 1))[:n_generated]
    genre_assignments = list(rng.permutation(genre_assignments))

    # Decide which generated rows get a non-English (but genre-correct) desc.
    non_eng_positions = set(
        rng.choice(n_generated, size=min(N_NON_ENGLISH, n_generated), replace=False).tolist()
    )

    for i in range(n_generated):
        genre = genre_assignments[i]
        setting = SETTINGS[rng.integers(len(SETTINGS))]
        words = rng.choice(GENRE_WORDS[genre], size=WORDS_PER_DESC, replace=False)

        if i in non_eng_positions:
            # Pick a non-English template that MATCHES this row's genre.
            candidates = NON_ENGLISH_BY_GENRE[genre]
            lang, text = candidates[rng.integers(len(candidates))]
            desc = text
        else:
            lang = "en"
            words = list(words)
            # Replace some middle slots with genre-neutral phrases for honest
            # ambiguity (so the model makes a few real, defensible mistakes).
            # Slot 0 (w1) stays a clean genre noun -- one template renders it as
            # "A {w1}-driven journey", which only reads well with a single word.
            for k in range(1, 1 + N_SHARED_SLOTS):
                if rng.random() < SHARED_PROB:
                    words[k] = SHARED_WORDS[rng.integers(len(SHARED_WORDS))]
            tmpl = TEMPLATES[rng.integers(len(TEMPLATES))]
            desc = tmpl.format(setting=setting, w1=words[0], w2=words[1],
                               w3=words[2], w4=words[3], w5=words[4])

        title = f"The {TITLE_WORDS[rng.integers(len(TITLE_WORDS))]} " \
                f"{TITLE_NOUNS[rng.integers(len(TITLE_NOUNS))]}"
        year = int(rng.integers(2019, 2027))
        runtime = int(rng.integers(82, 168))
        rows.append((movie_id, title, lang, genre, year, runtime, desc))
        movie_id += 1
    return rows


def main():
    rows = make_movies()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["movie_id", "title", "original_lang", "genre",
                    "release_year", "runtime_min", "description"])
        w.writerows(rows)
    print(f"wrote {len(rows)} movies to {OUT_PATH}")
    from collections import Counter
    counts = Counter(r[3] for r in rows)
    n_non_eng = sum(1 for r in rows if r[2] != "en" and r[0] >= 104)
    # canonical 101/103 are also non-English-language but English-text;
    # report rows whose description text is non-English.
    print("genre counts:", dict(sorted(counts.items())))
    print(f"non-English-language generated rows: {n_non_eng}")


if __name__ == "__main__":
    main()
