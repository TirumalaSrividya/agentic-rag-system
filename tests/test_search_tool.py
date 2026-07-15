"""
tests/test_search_tool.py

Covers tools/search.py:
- Retries on failure and eventually returns [] rather than raising
- Succeeds on a later attempt after earlier ones fail
- Rate limiting enforces a minimum delay between calls
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from unittest.mock import patch, MagicMock

import pytest

import tools.search as search_tool


def test_search_web_returns_empty_list_after_exhausting_retries():
    with patch("tools.search.DDGS", side_effect=RuntimeError("boom")):
        with patch("tools.search.time.sleep"):  # skip real backoff delay in test
            results = search_tool.search_web("anything", max_results=5)
    assert results == []


def test_search_web_succeeds_after_transient_failure():
    call_count = {"n": 0}

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=5):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("transient failure")
            return [{"title": "T", "href": "https://example.com/x", "body": "snippet"}]

    with patch("tools.search.DDGS", FakeDDGS):
        with patch("tools.search.time.sleep"):
            results = search_tool.search_web("query", max_results=5)

    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/x"
    assert call_count["n"] == 2


def test_throttle_enforces_minimum_delay():
    with patch("tools.search.SEARCH_RATE_LIMIT_SEC", 0.05):
        search_tool._last_call_at = 0.0
        start = time.monotonic()
        search_tool._throttle()
        search_tool._throttle()
        elapsed = time.monotonic() - start
    assert elapsed >= 0.04  # allow small scheduling slack


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
