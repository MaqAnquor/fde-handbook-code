# cinemastream/tests/conftest.py
# Shared pytest fixtures for the CinemaStream test suite.
# pytest loads this automatically from any test in this directory — no imports needed.

import sys
from pathlib import Path
import pytest

# Add cinemastream/scripts to sys.path so tests can import modules there.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from cs_client import CinemaStreamClient


@pytest.fixture
def client():
    """A configured CinemaStreamClient backed by the in-memory canonical sample rows."""
    return CinemaStreamClient(
        base_url="https://api.cinemastream.sg/v1",
        token="tok-test",
    )
