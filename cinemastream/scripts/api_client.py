# cinemastream/scripts/api_client.py
"""
Production-grade base API client for CinemaStream's external API integrations.

Provides BaseAPIClient with session pooling, transport-layer retry (urllib3),
application-layer rate-limit handling, and both offset and cursor pagination.
Extend this class for each external service.

Usage:
    from cinemastream.scripts.api_client import ContentHubClient

    client = ContentHubClient()                # reads CONTENTHUB_API_KEY from env
    movie  = client.get_movie(101)
    for m in client.search_movies("thriller"):
        print(m["title"])
    enriched = client.enrich_catalog([101, 102, 103])
"""

import os
import time
import logging
from typing import Any, Dict, Generator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Raised when an API call fails unrecoverably after all retries are exhausted."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class BaseAPIClient:
    """
    Reusable HTTP client for external API integrations.

    What it handles automatically:
    - Connection pooling (Session reuses TCP connections)
    - Transport-layer retry on network errors and 5xx (urllib3 HTTPAdapter)
    - Application-layer retry on 429 with Retry-After header support
    - Offset-based pagination (page=1, 2, 3...)
    - Cursor-based pagination (next_cursor pointer)

    What subclasses provide:
    - BASE_URL = "https://api.service.io/v2"
    - _auth_headers() returning {"X-API-Key": self.api_key} or equivalent
    """

    BASE_URL = ""
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 1.0      # urllib3: waits 2s, 4s between transport retries
    APP_BACKOFF_BASE = 2.0    # application layer: base seconds for 429 backoff
    TIMEOUT = 10              # seconds until a single request times out

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.environ.get("API_KEY", "")
        if base_url:
            self.BASE_URL = base_url
        self.session = self._build_session()

    def _auth_headers(self) -> Dict[str, str]:
        """Override in subclasses to inject service-specific auth headers."""
        return {}

    def _build_session(self) -> requests.Session:
        """Build a Session with pooling, base headers, and transport retry."""
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "CinemaStream-DataPipeline/1.0",
            **self._auth_headers(),
        })
        retry = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _wait_for_rate_limit(self, response: requests.Response, attempt: int) -> None:
        """Sleep before the next retry after a 429. Respects Retry-After header."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                wait = float(retry_after)
            except ValueError:
                wait = self.APP_BACKOFF_BASE * (2 ** attempt)
        else:
            wait = self.APP_BACKOFF_BASE * (2 ** attempt)
        logger.warning(
            "Rate limited (429) on attempt %d. Waiting %.1fs.", attempt + 1, wait
        )
        time.sleep(wait)

    def get(self, path: str, params: Dict = None) -> Any:
        """GET a JSON endpoint with application-layer 429 retry and backoff."""
        url = f"{self.BASE_URL}{path}"
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.TIMEOUT)
                if resp.status_code == 429:
                    if attempt >= self.MAX_RETRIES:
                        raise APIError(
                            f"Rate limit exceeded after {self.MAX_RETRIES} retries: {url}",
                            status_code=429,
                        )
                    self._wait_for_rate_limit(resp, attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout as exc:
                if attempt >= self.MAX_RETRIES:
                    raise APIError(f"Timed out after {self.TIMEOUT}s: {url}") from exc
                time.sleep(self.APP_BACKOFF_BASE * (2 ** attempt))
            except requests.exceptions.ConnectionError as exc:
                if attempt >= self.MAX_RETRIES:
                    raise APIError(f"Connection failed: {url}") from exc
                time.sleep(self.APP_BACKOFF_BASE * (2 ** attempt))
        raise APIError(f"All {self.MAX_RETRIES + 1} attempts failed: {url}")

    def post(self, path: str, json: Dict = None) -> Any:
        """POST JSON — used for GraphQL and write operations."""
        url = f"{self.BASE_URL}{path}"
        resp = self.session.post(url, json=json, timeout=self.TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def paginate_offset(
        self,
        path: str,
        params: Dict = None,
        page_size: int = 100,
        page_param: str = "page",
        data_key: str = "results",
    ) -> Generator[Dict, None, None]:
        """
        Yield all items from an offset-paginated endpoint.
        Stops when a response returns fewer items than page_size (last page).
        """
        params = dict(params or {})
        params["per_page"] = page_size
        page = 1
        while True:
            params[page_param] = page
            data = self.get(path, params=params)
            items = data.get(data_key, []) if isinstance(data, dict) else []
            if not items:
                break
            yield from items
            if len(items) < page_size:
                break
            page += 1

    def paginate_cursor(
        self,
        path: str,
        params: Dict = None,
        data_key: str = "results",
        cursor_key: str = "next_cursor",
    ) -> Generator[Dict, None, None]:
        """
        Yield all items from a cursor-paginated endpoint.
        Stops when next_cursor is null/missing.
        """
        params = dict(params or {})
        while True:
            data = self.get(path, params=params)
            items = data.get(data_key, []) if isinstance(data, dict) else []
            yield from items
            cursor = data.get(cursor_key)
            if not cursor:
                break
            params["cursor"] = cursor


class ContentHubClient(BaseAPIClient):
    """
    Client for CinemaStream's ContentHub partner API (external content metadata).

    ContentHub provides external ratings, content tags, and similar-movie
    recommendations for our catalog. Uses X-API-Key header auth.
    Rate limit: 60 requests/minute.

    Set environment variable: CONTENTHUB_API_KEY
    """

    BASE_URL = "https://api.contenthub.io/v2"

    def __init__(self, api_key: str = None):
        key = api_key or os.environ.get("CONTENTHUB_API_KEY", "")
        super().__init__(api_key=key)

    def _auth_headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key}

    def get_movie(self, movie_id: int) -> Dict:
        """Fetch enriched metadata for a single CinemaStream movie ID."""
        return self.get(f"/movies/{movie_id}")

    def search_movies(
        self, query: str, genre: str = None, page_size: int = 50
    ) -> Generator[Dict, None, None]:
        """Search and yield all matching movies — pagination handled transparently."""
        params = {"q": query}
        if genre:
            params["genre"] = genre
        yield from self.paginate_offset(
            "/movies/search", params=params, page_size=page_size
        )

    def enrich_catalog(self, movie_ids: list) -> list:
        """
        Fetch enriched metadata for a list of CinemaStream movie IDs.
        Skips individual APIError failures rather than crashing the whole batch.
        """
        enriched = []
        for movie_id in movie_ids:
            try:
                metadata = self.get_movie(movie_id)
                enriched.append({**metadata, "cs_movie_id": movie_id})
            except APIError as err:
                logger.warning("Skipping movie_id=%d — %s", movie_id, err)
        return enriched
