"""
tools/search.py

Web search tool (DuckDuckGo via ddgs).

Adds, on top of the original bare try/except:
  - a minimum delay between calls (self-imposed rate limiting, since ddgs
    has no official API key/quota to check against)
  - retry with exponential backoff on failure (covers transient rate-limit
    / network errors from the search backend)
  - a simple in-process call counter, logged, as a lightweight stand-in for
    quota tracking
"""

import time
from datetime import datetime
from typing import Any, Dict, List

from ddgs import DDGS

from config import SEARCH_RATE_LIMIT_SEC, SEARCH_MAX_RETRIES, SEARCH_BACKOFF_BASE_SEC
from logging_config import get_logger

logger = get_logger(__name__)

_last_call_at: float = 0.0
_call_count: int = 0  # simple quota/usage counter for this process


def _throttle() -> None:
    """Enforce a minimum delay between successive search calls."""
    global _last_call_at
    elapsed = time.monotonic() - _last_call_at
    wait = SEARCH_RATE_LIMIT_SEC - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using DDGS, with rate limiting and retry/backoff.
    Always returns a list (possibly empty) - never raises, so callers can
    continue the pipeline even if a query ultimately fails.
    """
    global _call_count
    results: List[Dict[str, Any]] = []

    for attempt in range(1, SEARCH_MAX_RETRIES + 1):
        _throttle()
        _call_count += 1
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)

                for item in search_results:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("href", ""),
                            "snippet": item.get("body", ""),
                            "retrieved_at": datetime.now().isoformat(),
                        }
                    )
            logger.info(
                "Search OK for '%s' (attempt %d/%d, call #%d, %d results)",
                query, attempt, SEARCH_MAX_RETRIES, _call_count, len(results),
            )
            return results

        except Exception as e:
            logger.warning(
                "Search failed for '%s' (attempt %d/%d, call #%d): %s",
                query, attempt, SEARCH_MAX_RETRIES, _call_count, e,
            )
            if attempt < SEARCH_MAX_RETRIES:
                backoff = SEARCH_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                logger.info("Backing off %.1fs before retrying '%s'", backoff, query)
                time.sleep(backoff)

    logger.error("Search exhausted %d retries for '%s', giving up (returning [])", SEARCH_MAX_RETRIES, query)
    return results
