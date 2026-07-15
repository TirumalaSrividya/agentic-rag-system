"""
tests/test_scraper.py

Covers tools/scraper.py:
- Published-date extraction from common meta tag patterns
- Falls back to None when no date metadata exists
- Returns None (not raise) on short content / network errors
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

import pytest

from tools.scraper import scrape_article, _extract_published_date
from bs4 import BeautifulSoup


def _fake_response(html: str, status_ok: bool = True):
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock() if status_ok else MagicMock(side_effect=RuntimeError("HTTP error"))
    return resp


def test_extract_published_date_from_article_meta():
    html = '<html><head><meta property="article:published_time" content="2026-07-01T10:00:00Z"></head></html>'
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_published_date(soup) == "2026-07-01T10:00:00"


def test_extract_published_date_from_time_tag():
    html = '<html><body><time datetime="2026-06-15T08:30:00Z">June 15</time></body></html>'
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_published_date(soup) == "2026-06-15T08:30:00"


def test_extract_published_date_returns_none_when_absent():
    html = "<html><body><p>no date here</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_published_date(soup) is None


def test_scrape_article_returns_none_on_short_content():
    html = "<html><body><p>too short</p></body></html>"
    with patch("tools.scraper.requests.get", return_value=_fake_response(html)):
        result = scrape_article("https://example.com/short")
    assert result is None


def test_scrape_article_returns_none_on_request_failure():
    with patch("tools.scraper.requests.get", side_effect=ConnectionError("network down")):
        result = scrape_article("https://example.com/unreachable")
    assert result is None


def test_scrape_article_success_includes_published_date():
    paragraph = "<p>" + ("word " * 100) + "</p>"
    html = f'<html><head><meta property="article:published_time" content="2026-07-01T00:00:00Z"></head><body>{paragraph}</body></html>'
    with patch("tools.scraper.requests.get", return_value=_fake_response(html)):
        result = scrape_article("https://example.com/ok")
    assert result is not None
    assert result["published_date"] == "2026-07-01T00:00:00"
    assert len(result["content"]) >= 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
