"""
CinemaStream catalogue and analytics-session helpers.

Introduced in Chapter 11b (Protocols & Context Managers). Demonstrates the three
container/resource protocols on real CinemaStream shapes:

  - watch_session(...)  : a context manager (guaranteed cleanup) via @contextmanager
  - Catalogue           : an indexable + iterable view over the movie catalogue
                          (__getitem__, __len__, __iter__)

Chapter 24 (Reading & Writing Data) swaps the hardcoded sample rows for movies.csv.

Usage:
    from cinemastream.scripts.catalogue import Catalogue, watch_session

    cat = Catalogue(movies)
    first = cat[0]
    top_two = cat[:2]
    for movie in cat:
        ...

    with watch_session("Ravi Kumar") as session:
        session["events_processed"] += 1
"""

from contextlib import contextmanager


@contextmanager
def watch_session(user_name: str):
    """Open an analytics session and guarantee it closes, even on error.

    Everything before ``yield`` is setup (__enter__); the ``finally`` block is
    teardown (__exit__) and runs whether the body completes or raises.
    """
    print(f"[session] opened for {user_name}")
    session = {"user": user_name, "events_processed": 0}
    try:
        yield session
    finally:
        print(f"[session] closed for {user_name} "
              f"({session['events_processed']} events processed)")


class Catalogue:
    """An indexable, iterable view over the CinemaStream movie catalogue.

    Implements the container protocols so callers use ordinary Python:
        cat[2]        -> the third movie        (__getitem__ with int)
        cat[:2]       -> first two as a list    (__getitem__ with slice)
        len(cat)      -> number of movies       (__len__)
        for m in cat  -> iterate the movies     (__iter__, re-iterable)
    """

    def __init__(self, movies):
        self._movies = list(movies)

    def __getitem__(self, index):
        return self._movies[index]

    def __len__(self) -> int:
        return len(self._movies)

    def __iter__(self):
        return iter(self._movies)

    def __repr__(self) -> str:
        return f"Catalogue({len(self._movies)} movies)"
