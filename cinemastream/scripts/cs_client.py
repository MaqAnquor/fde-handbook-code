"""
CinemaStream internal analytics API client.

Introduced in Chapter 11a (Object-Oriented & Functional Python).
Chapter 19 (APIs) extends this with real HTTP calls via the requests library.

Usage:
    from cinemastream.scripts.cs_client import CinemaStreamClient

    cs = CinemaStreamClient(
        base_url="https://api.cinemastream.sg/v1",
        token="your-token-here",
    )
    user = cs.get_user(1)
    for event in cs.stream_watch_events(user_id=1):
        process(event)
"""

import functools
import time
from typing import Iterator


def retry_on_error(max_attempts: int = 3, delay_seconds: float = 0.5):
    """
    Decorator factory: retry the wrapped function up to max_attempts times.

    Prints a message on each failed attempt (except the final one, which
    lets the exception propagate so the caller can handle it).

    Usage:
        @retry_on_error(max_attempts=3, delay_seconds=1.0)
        def fetch_user(user_id: int) -> dict: ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        raise
                    print(
                        f"[retry] {func.__name__} attempt {attempt} failed: {exc}. "
                        f"Retrying in {delay_seconds}s..."
                    )
                    time.sleep(delay_seconds)
        return wrapper
    return decorator


class BaseAPIClient:
    """
    Connection configuration shared by all CinemaStream API clients.

    Stores base_url and auth token. Exposes auth headers via @property
    (never as a plain dict attribute) so the token cannot leak into logs.
    """

    def __init__(self, base_url: str, token: str):
        self._base_url = base_url
        self._token = token

    @property
    def headers(self) -> dict:
        """Return auth headers. Computed fresh on every access."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base_url='{self._base_url}', token='***')"


class CinemaStreamClient(BaseAPIClient):
    """
    Client for the CinemaStream internal analytics API.

    Chapter 11a: uses hardcoded canonical sample rows (bible rows 1-3).
    Chapter 19:  replace _USERS/_EVENTS with real requests.get() calls.

    Inherits from BaseAPIClient: base_url, token, headers property, __repr__.
    """

    # Bible-canonical sample rows — kept in sync with bible_core.md
    _USERS: dict = {
        1: {"user_id": 1, "name": "Ravi Kumar",      "plan": "Premium", "country": "IN", "churned": False},
        2: {"user_id": 2, "name": "Siti Rahman",     "plan": "Basic",   "country": "MY", "churned": False},
        3: {"user_id": 3, "name": "Nguyen Van Minh", "plan": "Free",    "country": "VN", "churned": True},
    }
    _EVENTS: list = [
        {"event_id": 1, "user_id": 1, "movie_id": 101, "watch_minutes": 132, "completed": True},
        {"event_id": 2, "user_id": 1, "movie_id": 102, "watch_minutes":  45, "completed": False},
        {"event_id": 3, "user_id": 2, "movie_id": 103, "watch_minutes":  96, "completed": True},
    ]

    @retry_on_error(max_attempts=3, delay_seconds=0.5)
    def get_user(self, user_id: int) -> dict:
        """
        Return a single user record by ID.

        Retries up to 3 times on transient network errors.
        Chapter 19 replaces the dict lookup with:
            response = requests.get(
                f"{self._base_url}/users/{user_id}", headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        """
        if user_id not in self._USERS:
            raise KeyError(f"User {user_id} not found")
        return self._USERS[user_id]

    def get_watch_events(self, user_id: int) -> list:
        """
        Return all watch events for a user as a list.

        Warning: loads everything into memory. Prefer stream_watch_events()
        when processing large event histories.
        """
        return [e for e in self._EVENTS if e["user_id"] == user_id]

    def stream_watch_events(self, user_id: int) -> Iterator[dict]:
        """
        Yield watch events for a user one at a time — generator pattern.

        Memory use is O(1) regardless of the number of events. The caller
        can start processing event N before event N+1 is fetched from the API.
        In production this would use cursor-based pagination to request one
        page at a time, yielding each event as it arrives.
        """
        for event in self.get_watch_events(user_id):
            yield event


def total_completed_minutes(client: CinemaStreamClient, user_id: int) -> int:
    """Return total watch_minutes for all completed events for a given user."""
    return sum(
        event["watch_minutes"]
        for event in client.stream_watch_events(user_id)
        if event["completed"]
    )


if __name__ == "__main__":
    cs = CinemaStreamClient(
        base_url="https://api.cinemastream.sg/v1",
        token="tok-analytics-read",
    )

    print(repr(cs))
    print()

    print("User records:")
    for uid in [1, 2, 3]:
        user = cs.get_user(uid)
        mins = total_completed_minutes(cs, uid)
        status = "churned" if user["churned"] else "active"
        print(f"  {user['name']:20s} ({status:7s}) — completed: {mins} min")
