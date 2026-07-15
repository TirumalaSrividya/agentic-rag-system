"""
agents/search_agent.py

SearchTool Agent

Responsibilities:
1. Generate 5-10 sub-queries by decomposing TOPIC (topic-agnostic - no
   hardcoded domain vocabulary; uses the LLM, with a generic fallback)
2. Search the web for each sub-query
3. Deduplicate URLs across queries
4. Rank results (by cross-query relevance + search-result position)
5. Return a ranked list of URLs with titles and snippet previews

Note: recency filtering happens later in ingestion.py, after scraping,
because publish dates generally aren't reliably available from search
result snippets alone - see tools/scraper.py's date extraction.
"""

import json
import re
from typing import Any, Dict, List

import requests

from config import (
    TOPIC,
    MAX_SOURCES_PER_RUN,
    NUM_SUBQUERIES,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_REQUEST_TIMEOUT,
)
from tools.search import search_web
from logging_config import get_logger

logger = get_logger(__name__)

# Generic (non-domain-specific) fallback templates, used only if the LLM
# call to decompose the topic fails. These make no assumption about the
# topic's subject area, so the system stays topic-agnostic even in the
# fallback path.
_GENERIC_TEMPLATES = [
    "{topic}",
    "{topic} latest news",
    "{topic} recent developments",
    "{topic} 2026",
    "{topic} breakthrough",
    "{topic} research",
    "{topic} announcement",
    "{topic} industry update",
]


def _keyword_set(text: str) -> set:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


class SearchAgent:
    def __init__(self, topic: str = None):
        self.topic: str = topic or TOPIC
        self.topic_keywords: set = _keyword_set(self.topic)

    # ---------- Step 1: topic-agnostic sub-query generation ----------
    def generate_queries(self) -> List[str]:
        """
        Decompose self.topic into NUM_SUBQUERIES search queries via the LLM.
        No hardcoded domain vocabulary here - whatever subject TOPIC names,
        this must produce sensible queries for it. Falls back to generic
        (still topic-agnostic) templates if the LLM call fails.
        """
        prompt = (
            f'Break the topic "{self.topic}" into {NUM_SUBQUERIES} diverse, specific web '
            "search queries that together would surface the latest news, research, and "
            "developments on this topic. The queries must stay strictly relevant to the "
            "topic's actual subject matter - do not assume any particular industry or "
            "domain beyond what the topic itself states. "
            "Return ONLY a JSON array of strings, nothing else."
        )
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=OLLAMA_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            raw = response.json().get("message", {}).get("content", "").strip()
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            queries = json.loads(raw)
            queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
            if queries:
                logger.info("Generated %d sub-queries via LLM for topic '%s'", len(queries), self.topic)
                return queries[:NUM_SUBQUERIES]
        except Exception as e:
            logger.warning("LLM sub-query generation failed (%s), falling back to generic templates", e)

        return [t.format(topic=self.topic) for t in _GENERIC_TEMPLATES[:NUM_SUBQUERIES]]

    # ---------- Step 2 + 3: search, dedup, rank ----------
    def search(self) -> List[Dict[str, Any]]:
        queries = self.generate_queries()
        seen_urls = set()
        candidates: List[Dict[str, Any]] = []

        for query_rank, query in enumerate(queries):
            results = search_web(query, max_results=5)

            for position_in_query, article in enumerate(results):
                url = article["url"]
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                article["source_query"] = query
                article["query_rank"] = query_rank            # earlier sub-queries are more central to topic
                article["position_in_query"] = position_in_query  # search engine's own relevance order
                article["relevance_score"] = self._relevance_score(article)
                candidates.append(article)

        ranked = self._rank(candidates)[:MAX_SOURCES_PER_RUN]
        for i, item in enumerate(ranked, start=1):
            item["rank"] = i

        logger.info(
            "SearchAgent: %d queries -> %d unique URLs -> %d kept after ranking/truncation",
            len(queries), len(candidates), len(ranked),
        )
        return ranked

    def _relevance_score(self, article: Dict[str, Any]) -> float:
        """Keyword overlap between the topic and the article's title+snippet."""
        text_keywords = _keyword_set(f"{article.get('title', '')} {article.get('snippet', '')}")
        if not self.topic_keywords:
            return 0.0
        return len(self.topic_keywords & text_keywords) / len(self.topic_keywords)

    @staticmethod
    def _rank(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank by topic relevance first, then by how early the sub-query was
        (earlier queries are broader/more central), then by the search
        engine's own result position for that query.
        """
        return sorted(
            candidates,
            key=lambda a: (-a["relevance_score"], a["query_rank"], a["position_in_query"]),
        )
