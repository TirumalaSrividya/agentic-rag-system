"""
tools/scraper.py

Scrapes article text (and, where available, its published date) from a
webpage. The published date is what ingestion.py uses to actually enforce
RECENCY_DAYS - the search-result "retrieved_at" timestamp is just when
*we* fetched it, not when the article was published, so it can't be used
for a recency filter.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup

from logging_config import get_logger

logger = get_logger(__name__)

_MIN_CONTENT_CHARS = 200

_DATE_META_CANDIDATES = [
    ("meta", {"property": "article:published_time"}),
    ("meta", {"name": "date"}),
    ("meta", {"name": "pubdate"}),
    ("meta", {"name": "publish-date"}),
    ("meta", {"itemprop": "datePublished"}),
    ("meta", {"property": "og:updated_time"}),
]


def _extract_published_date(soup: BeautifulSoup) -> Optional[str]:
    """Best-effort published-date extraction. Returns an ISO string or None."""
    for tag_name, attrs in _DATE_META_CANDIDATES:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get("content"):
            raw = tag["content"][:19].replace("Z", "")
            try:
                return datetime.fromisoformat(raw).isoformat()
            except ValueError:
                continue

    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        raw = time_tag["datetime"][:19].replace("Z", "")
        try:
            return datetime.fromisoformat(raw).isoformat()
        except ValueError:
            pass

    return None


def scrape_article(url: str) -> Optional[Dict[str, Any]]:
    """
    Scrape article text (and published_date if detectable) from a webpage.

    Returns:
        {"url": "...", "content": "...", "published_date": "<iso string or None>"}

    If scraping fails (network error, too-short content, parse error):
        returns None
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)

        if len(text) < _MIN_CONTENT_CHARS:
            return None

        return {
            "url": url,
            "content": text,
            "published_date": _extract_published_date(soup),
        }

    except Exception as e:
        logger.warning("Scraping failed for %s: %s", url, e)
        return None
