"""
tests/test_search_agent.py

Covers:
- LLM-based sub-query generation with a mocked Ollama response
- Fallback to generic (topic-agnostic) templates when the LLM call fails
- The generic fallback templates contain no hardcoded domain vocabulary
- URL deduplication across sub-queries
- Ranking by topic relevance
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import patch, MagicMock

import pytest

from agents.search_agent import SearchAgent, _GENERIC_TEMPLATES


def test_fallback_templates_are_topic_agnostic():
    """None of the fallback templates should hardcode a specific domain
    (e.g. medical terms like 'hospitals', 'clinical trials', 'diagnosis')."""
    banned_terms = {"hospital", "clinical", "diagnosis", "treatment", "medicine", "healthcare"}
    rendered = " ".join(_GENERIC_TEMPLATES).lower()
    for term in banned_terms:
        assert term not in rendered, f"Fallback template hardcodes domain term '{term}'"


def test_generate_queries_uses_llm_when_available():
    agent = SearchAgent(topic="advancements in battery technology")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "message": {
            "content": json.dumps([
                "battery technology breakthroughs",
                "solid-state battery research 2026",
                "battery energy density improvements",
            ])
        }
    }
    with patch("agents.search_agent.requests.post", return_value=mock_response):
        queries = agent.generate_queries()

    assert len(queries) == 3
    assert "battery technology breakthroughs" in queries


def test_generate_queries_falls_back_on_llm_failure():
    agent = SearchAgent(topic="advancements in battery technology")
    with patch("agents.search_agent.requests.post", side_effect=ConnectionError("no ollama")):
        queries = agent.generate_queries()

    assert len(queries) > 0
    assert all("battery technology" in q for q in queries)
    # must not have picked up medical-domain vocabulary for a non-medical topic
    assert not any("hospital" in q.lower() for q in queries)


def test_search_dedups_urls_across_queries():
    agent = SearchAgent(topic="test topic")

    def fake_search_web(query, max_results=5):
        # every query "finds" the same URL plus one unique one
        return [
            {"title": "Shared", "url": "https://example.com/shared", "snippet": "test topic info"},
            {"title": f"Unique for {query}", "url": f"https://example.com/{query}", "snippet": "test topic"},
        ]

    with patch.object(agent, "generate_queries", return_value=["q1", "q2", "q3"]):
        with patch("agents.search_agent.search_web", side_effect=fake_search_web):
            results = agent.search()

    urls = [r["url"] for r in results]
    assert len(urls) == len(set(urls)), "Duplicate URLs were not deduplicated"
    assert "https://example.com/shared" in urls
    # shared URL should only appear once despite being returned by all 3 queries
    assert urls.count("https://example.com/shared") == 1


def test_search_ranks_by_relevance():
    agent = SearchAgent(topic="battery technology")

    def fake_search_web(query, max_results=5):
        return [
            {"title": "Irrelevant story", "url": "https://example.com/irrelevant", "snippet": "cooking recipes"},
            {"title": "Battery technology news", "url": "https://example.com/relevant", "snippet": "battery technology breakthrough"},
        ]

    with patch.object(agent, "generate_queries", return_value=["q1"]):
        with patch("agents.search_agent.search_web", side_effect=fake_search_web):
            results = agent.search()

    assert results[0]["url"] == "https://example.com/relevant"
    assert results[0]["rank"] == 1


def test_search_respects_max_sources_per_run():
    agent = SearchAgent(topic="test topic")

    def fake_search_web(query, max_results=5):
        return [{"title": f"T{i}", "url": f"https://example.com/{query}/{i}", "snippet": ""} for i in range(5)]

    with patch.object(agent, "generate_queries", return_value=["q1", "q2", "q3"]):
        with patch("agents.search_agent.search_web", side_effect=fake_search_web):
            with patch("agents.search_agent.MAX_SOURCES_PER_RUN", 4):
                results = agent.search()

    assert len(results) <= 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
