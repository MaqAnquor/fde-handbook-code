"""
Reusable logging factory for all CinemaStream scripts.
Introduced in Chapter 013a (Production Logging).

Usage:
    from cinemastream.scripts.logging_setup import get_logger, new_correlation_id, setup_logging

    setup_logging("INFO")  # call once in main()
    logger = get_logger(__name__)

    with new_correlation_id("ingest"):
        logger.info("Batch started", rows=500)
"""

import json
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from loguru import logger as _loguru_logger

# Stores the current correlation ID for this thread/asyncio task.
# Default "no-id" ensures logs outside a batch context are still valid JSON.
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="no-id")


def _make_serializer():
    """Return a loguru format function that produces one JSON line per record."""
    def serialize(record) -> str:
        payload = {
            "ts":     record["time"].isoformat(),
            "level":  record["level"].name,
            "event":  record["message"],
            "module": record["name"],
            "cid":    _correlation_id.get(),
        }
        payload.update(record["extra"])
        record["extra"]["_json"] = json.dumps(payload)
        return "{extra[_json]}\n"
    return serialize


def setup_logging(level: str = "INFO") -> None:
    """
    Configure loguru to emit one JSON line per log record.
    Call once at the top of your script's main() function.
    """
    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stderr,
        level=level,
        format=_make_serializer(),
        colorize=False,
    )


def get_logger(name: str):
    """
    Return a loguru logger bound to `name`.
    Conventional usage: get_logger(__name__) at module level.
    """
    return _loguru_logger.bind(logger_name=name)


@contextmanager
def new_correlation_id(prefix: str = ""):
    """
    Context manager that sets a fresh UUID correlation ID for the duration of a block.

    Example:
        with new_correlation_id("ingest") as cid:
            logger.info("Batch started")   # every line in here carries the same cid
    """
    cid = f"{prefix}-{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:8]
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)
