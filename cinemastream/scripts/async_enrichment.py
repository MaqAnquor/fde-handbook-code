"""
async_enrichment.py — Async ContentHub catalog enricher.
Created: Chapter 019b (Concurrency & Asyncio)

Usage:
    python cinemastream/scripts/async_enrichment.py

In production, set CONTENTHUB_API_KEY environment variable.
The ContentHubMockTransport is used when no live key is available.
"""

import asyncio
import json
import os
import re
import time
import httpx


# ---------------------------------------------------------------------------
# Mock transport (development / CI only)
# ---------------------------------------------------------------------------
class ContentHubMockTransport(httpx.AsyncBaseTransport):
    """Simulates the ContentHub REST API locally for testing."""
    # Canonical from session_log 019a: 101=8.2, 102=7.9, 103=404 (not found)
    # Invented for Ch019b: 104=9.1, 105=7.5, 106=8.8
    RATINGS: dict[int, float] = {101: 8.2, 102: 7.9, 104: 9.1, 105: 7.5, 106: 8.8}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        match = re.search(r"/movies/(\d+)", str(request.url))
        if not match:
            return httpx.Response(400, content=b'{"error": "bad request"}')
        mid = int(match.group(1))
        if mid in self.RATINGS:
            body = json.dumps({"movie_id": mid, "ext_rating": self.RATINGS[mid]}).encode()
            return httpx.Response(200, content=body)
        return httpx.Response(404, content=b'{"error": "not found"}')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONTENTHUB_BASE  = "https://api.contenthub.io/v2"
MAX_CONCURRENT   = 10    # caps concurrent in-flight requests (rate-limit safety)
DEFAULT_TIMEOUT  = 10.0  # seconds


# ---------------------------------------------------------------------------
# Core async functions
# ---------------------------------------------------------------------------
async def _fetch_one(
    client: httpx.AsyncClient,
    movie_id: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Fetch ContentHub rating for a single movie_id."""
    async with semaphore:
        try:
            resp = await client.get(f"{CONTENTHUB_BASE}/movies/{movie_id}")
            if resp.status_code == 200:
                data = resp.json()
                return {"movie_id": movie_id, "ext_rating": data["ext_rating"], "status": "ok"}
            if resp.status_code == 404:
                return {"movie_id": movie_id, "ext_rating": None, "status": "not_found"}
            return {"movie_id": movie_id, "ext_rating": None, "status": f"http_{resp.status_code}"}
        except httpx.RequestError:
            return {"movie_id": movie_id, "ext_rating": None, "status": "request_error"}


async def enrich_catalog(
    movie_ids: list[int],
    api_key: str | None = None,
    use_mock: bool = True,
) -> list[dict]:
    """
    Fetch ContentHub ratings for all movie_ids concurrently.

    Args:
        movie_ids:  List of integer movie IDs to enrich.
        api_key:    ContentHub API key. Reads CONTENTHUB_API_KEY env var if None.
        use_mock:   Use ContentHubMockTransport instead of live network.

    Returns:
        List of dicts in the same order as movie_ids, each with keys:
        movie_id, ext_rating (float or None), status ('ok'/'not_found'/'http_NNN').
    """
    if api_key is None:
        api_key = os.environ.get("CONTENTHUB_API_KEY", "")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    transport = ContentHubMockTransport() if use_mock else None

    async with httpx.AsyncClient(
        transport=transport,
        headers={"X-API-Key": api_key},
        timeout=DEFAULT_TIMEOUT,
    ) as client:
        tasks = [_fetch_one(client, mid, semaphore) for mid in movie_ids]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    catalog_ids = list(range(101, 131))   # 30-movie catalog (IDs 101-130)

    print(f"Enriching {len(catalog_ids)} movies via ContentHub...")
    start = time.perf_counter()
    results = asyncio.run(enrich_catalog(catalog_ids, use_mock=True))
    elapsed = time.perf_counter() - start

    found     = [r for r in results if r["status"] == "ok"]
    not_found = [r for r in results if r["status"] == "not_found"]

    print(f"Done in {elapsed:.3f}s")
    print(f"  Enriched  : {len(found)}")
    print(f"  Not found : {len(not_found)}")
    for r in found:
        print(f"    movie {r['movie_id']:>3}: ext_rating={r['ext_rating']:.1f}")


if __name__ == "__main__":
    main()
