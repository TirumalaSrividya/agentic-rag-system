"""
tests/test_ingestion_filters.py

Covers ingestion.py:
- Recency filter drops stale articles, keeps articles with no detectable date
- Relevance filter drops low-overlap content, keeps on-topic content
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

from ingestion import _is_recent_enough, _is_relevant_enough, _keyword_set


def test_recent_article_passes():
    recent_date = (datetime.now() - timedelta(days=1)).isoformat()
    assert _is_recent_enough(recent_date, recency_days=14) is True


def test_stale_article_is_filtered():
    stale_date = (datetime.now() - timedelta(days=60)).isoformat()
    assert _is_recent_enough(stale_date, recency_days=14) is False


def test_no_date_is_kept_not_penalized():
    assert _is_recent_enough(None, recency_days=14) is True


def test_malformed_date_is_kept():
    assert _is_recent_enough("not-a-date", recency_days=14) is True


def test_relevant_content_passes():
    topic_keywords = _keyword_set("battery technology")
    content = "New battery technology breakthroughs announced this week in energy storage."
    assert _is_relevant_enough(content, topic_keywords, min_overlap=0.08) is True


def test_irrelevant_content_is_filtered():
    topic_keywords = _keyword_set("battery technology")
    content = "A recipe for chocolate chip cookies with step by step instructions."
    assert _is_relevant_enough(content, topic_keywords, min_overlap=0.5) is False


def test_empty_topic_keywords_never_filters():
    assert _is_relevant_enough("anything at all", set(), min_overlap=0.5) is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
