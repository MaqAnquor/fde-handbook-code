"""
Redis-based session cache for CinemaStream's "continue watching" feature.
Stores and retrieves the user's resume position per movie.

Run: python cinemastream/scripts/session_cache.py
Requires: pip install fakeredis (or a real Redis instance with redis-py)
Chapter: 041a — NoSQL Databases
"""

from dataclasses import dataclass
from typing import Optional
import fakeredis        # swap for: import redis; r = redis.Redis(host="localhost")


# In production: r = redis.Redis(host="redis.cinemastream.internal", port=6379, decode_responses=True)
r = fakeredis.FakeRedis(decode_responses=True)

SESSION_TTL_SECONDS = 60 * 60 * 24 * 7   # 7 days — resume persists across devices


@dataclass
class ResumeState:
    user_id: int
    movie_id: int
    resume_sec: int
    duration_sec: int
    device: str


def _session_key(user_id: int, movie_id: int) -> str:
    return f"resume:{user_id}:{movie_id}"


def save_resume(state: ResumeState) -> None:
    """Write resume position. Called on pause/stop events."""
    key = _session_key(state.user_id, state.movie_id)
    r.hset(key, mapping={
        "resume_sec":   str(state.resume_sec),
        "duration_sec": str(state.duration_sec),
        "device":       state.device,
    })
    r.expire(key, SESSION_TTL_SECONDS)


def get_resume(user_id: int, movie_id: int) -> Optional[ResumeState]:
    """Read resume position. Called on play events."""
    key = _session_key(user_id, movie_id)
    data = r.hgetall(key)
    if not data:
        return None
    return ResumeState(
        user_id=user_id,
        movie_id=movie_id,
        resume_sec=int(data["resume_sec"]),
        duration_sec=int(data["duration_sec"]),
        device=data["device"],
    )


def clear_resume(user_id: int, movie_id: int) -> None:
    """Delete resume when user finishes or manually restarts."""
    r.delete(_session_key(user_id, movie_id))


def main() -> None:
    # Priya pauses "Hujan di Singapura" at 23 minutes on her phone
    save_resume(ResumeState(
        user_id=1, movie_id=101, resume_sec=1380,
        duration_sec=5640, device="mobile"
    ))
    print("Saved resume for user 1, movie 101.")

    # She opens the app on her smart TV — picks up where she left off
    state = get_resume(user_id=1, movie_id=101)
    if state:
        pct = state.resume_sec / state.duration_sec * 100
        print(f"Resuming movie {state.movie_id} at {state.resume_sec}s "
              f"({pct:.0f}%) on {state.device}.")
    else:
        print("No resume position found — starting from beginning.")

    # She finishes the movie — clear the resume
    clear_resume(user_id=1, movie_id=101)
    print(f"Movie finished. Resume cleared. "
          f"Key exists: {r.exists(_session_key(1, 101))}")

    # User 2 has no saved position
    state2 = get_resume(user_id=2, movie_id=101)
    print(f"User 2 resume: {state2}")


if __name__ == "__main__":
    main()
