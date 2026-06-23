# cinemastream/tests/test_cs_client.py
# Unit tests for cs_client.py (introduced in Chapter 11a).
# Run with: pytest cinemastream/tests/test_cs_client.py -v

import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from cs_client import CinemaStreamClient, total_completed_minutes


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return CinemaStreamClient(
        base_url="https://api.cinemastream.sg/v1",
        token="tok-test",
    )


# ── get_user() ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("user_id, expected_name, expected_plan", [
    (1, "Ravi Kumar",      "Premium"),
    (2, "Siti Rahman",     "Basic"),
    (3, "Nguyen Van Minh", "Free"),
])
def test_get_user_returns_correct_record(client, user_id, expected_name, expected_plan):
    user = client.get_user(user_id)
    assert user["name"] == expected_name
    assert user["plan"] == expected_plan
    assert "churned" in user


def test_get_user_raises_for_missing_id(client):
    with pytest.raises(KeyError):
        client.get_user(999)


def test_get_user_returns_canonical_country(client):
    assert client.get_user(1)["country"] == "IN"   # Ravi is in India
    assert client.get_user(2)["country"] == "MY"   # Siti is in Malaysia
    assert client.get_user(3)["country"] == "VN"   # Minh is in Vietnam


# ── get_watch_events() ─────────────────────────────────────────────────────

def test_get_watch_events_returns_list(client):
    events = client.get_watch_events(user_id=1)
    assert isinstance(events, list)
    assert len(events) == 2    # Ravi has 2 events in the canonical sample


def test_get_watch_events_only_for_that_user(client):
    events = client.get_watch_events(user_id=2)
    assert all(e["user_id"] == 2 for e in events)


def test_get_watch_events_returns_empty_for_no_events(client):
    # user_id=3 (Minh) has no watch events in the canonical sample
    events = client.get_watch_events(user_id=3)
    assert events == []


# ── total_completed_minutes() ──────────────────────────────────────────────

def test_total_completed_minutes_ravi(client):
    # Ravi: event_id=1 (132 min, completed=True), event_id=2 (45 min, completed=False)
    # Only completed events count → 132
    mins = total_completed_minutes(client, user_id=1)
    assert mins == 132


def test_total_completed_minutes_siti(client):
    # Siti: event_id=3 (96 min, completed=True) → 96
    mins = total_completed_minutes(client, user_id=2)
    assert mins == 96


def test_total_completed_minutes_no_events(client):
    # Minh has no events → 0
    mins = total_completed_minutes(client, user_id=3)
    assert mins == 0


def test_total_completed_minutes_uses_stream(client):
    # Verify that total_completed_minutes calls stream_watch_events, not the list version.
    fake_events = [
        {"watch_minutes": 50, "completed": True},
        {"watch_minutes": 30, "completed": False},
        {"watch_minutes": 20, "completed": True},
    ]
    with patch.object(client, "stream_watch_events", return_value=iter(fake_events)):
        mins = total_completed_minutes(client, user_id=1)
    assert mins == 70    # 50 + 20 (the two completed ones)


# ── headers property ───────────────────────────────────────────────────────

def test_headers_include_authorization(client):
    headers = client.headers
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("Bearer ")


def test_headers_do_not_expose_token_in_repr(client):
    rep = repr(client)
    assert "tok-test" not in rep   # token should NOT appear
    assert "***" in rep            # masked placeholder should appear


# ── Preview: mocking an HTTP call ─────────────────────────────────────────

def test_future_http_get_user_mock_preview(client):
    """
    When Chapter 19 adds real HTTP, get_user() will call requests.get().
    This test shows the pattern for mocking that — CI never needs network access.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "user_id": 1, "name": "Ravi Kumar", "plan": "Premium",
        "country": "IN", "churned": False,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        # Current implementation uses dict lookup (not requests.get yet),
        # so this test validates the existing contract while the pattern waits for Ch 019.
        user = client.get_user(1)
        assert user["name"] == "Ravi Kumar"
