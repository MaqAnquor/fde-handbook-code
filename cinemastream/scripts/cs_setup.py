"""
cs_setup.py — Cross-platform / Colab-compatible data path setup.

Usage in any chapter or notebook:

    from cinemastream.scripts.cs_setup import DATA_DIR, setup_colab
    setup_colab()   # no-op outside Colab; clones data on Colab
    users = pd.read_csv(DATA_DIR / "users.csv")

Or just:

    from cinemastream.scripts.cs_setup import DATA_DIR

DATA_DIR is a pathlib.Path pointing to the directory that contains
users.csv, movies.csv, watch_events.csv, ratings.csv,
subscriptions.csv, support_tickets.csv.
"""

from pathlib import Path
import os

GITHUB_REPO = "https://github.com/MaqAnquor/fde-handbook-code.git"
COLAB_CLONE_DIR = "/content/fde-handbook-code"

# ── Colab detection ────────────────────────────────────────────────────────────
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def setup_colab(repo: str = GITHUB_REPO, clone_dir: str = COLAB_CLONE_DIR) -> None:
    """Clone the repo and set the working directory when running on Colab.

    No-op outside Colab. Safe to call multiple times (skips clone if already done).
    """
    if not IN_COLAB:
        return
    import subprocess

    target = Path(clone_dir)
    if not target.exists():
        print(f"[cs_setup] Cloning {repo} into {clone_dir} …")
        subprocess.run(["git", "clone", "--depth", "1", repo, clone_dir], check=True)
        print("[cs_setup] Clone complete.")
    else:
        print(f"[cs_setup] Repo already at {clone_dir} — skipping clone.")

    os.chdir(clone_dir)
    print(f"[cs_setup] Working directory set to {clone_dir}")


def _resolve_data_dir() -> Path:
    """Return the path to cinemastream/data/, searching common locations."""
    candidates = [
        Path("cinemastream/data"),                      # run from project root (local)
        Path(COLAB_CLONE_DIR) / "cinemastream" / "data",  # Colab after Cell 1 clone
        Path(__file__).parent.parent / "data",          # installed as package
    ]
    for p in candidates:
        if p.exists() and (p / "users.csv").exists():
            return p.resolve()
    # Return the relative path as a fallback — will raise a clear FileNotFoundError
    # at the point of use rather than at import time.
    return Path("cinemastream/data")


DATA_DIR: Path = _resolve_data_dir()


# ── Windows compatibility note ─────────────────────────────────────────────────
# All file paths in this book use pathlib.Path objects, which normalise
# separators automatically on Windows (Path("a/b") → Path("a\\b") on Windows).
# All open() calls include encoding="utf-8" so they behave identically on
# Windows (default cp1252) and Linux/macOS.
#
# On Windows, replace "python3" with "python" in terminal commands if your
# installation doesn't provide the python3 alias.
