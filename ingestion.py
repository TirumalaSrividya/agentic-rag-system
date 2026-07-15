"""
ingestion.py

Daily Ingestion Pipeline: SearchAgent -> scrape_article -> VectorDBAgent

- Generates 5-10 sub-queries from TOPIC, collects up to MAX_SOURCES_PER_RUN URLs.
- Scrapes each URL; failures are skipped, pipeline continues.
- Enforces MAX_INGESTION_TOKEN_BUDGET: once exceeded, stop scraping NEW articles
  but still index whatever was already scraped.
- Chunks/embeds/dedups/upserts via VectorDBAgent.
- Writes a Daily Ingestion Report JSON to REPORT_FOLDER.
- All steps logged with timestamps.
"""

import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Set

from config import (
    MAX_INGESTION_TOKEN_BUDGET,
    REPORT_FOLDER,
    TOPIC,
    RECENCY_DAYS,
    MIN_RELEVANCE_KEYWORD_OVERLAP,
)
from agents.search_agent import SearchAgent
from agents.vector_db_agent import VectorDBAgent
from tools.scraper import scrape_article
from logging_config import get_logger

logger = get_logger(__name__)

os.makedirs(REPORT_FOLDER, exist_ok=True)


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _keyword_set(text: str) -> Set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


def _is_recent_enough(published_date: Optional[str], recency_days: int) -> bool:
    """
    Recency filter (gap #2). If no publish date could be detected, we keep
    the article rather than drop it - many legitimate sources lack clean
    date metadata, and dropping them would over-filter good content.
    """
    if not published_date:
        return True
    try:
        pub_dt = datetime.fromisoformat(published_date)
    except ValueError:
        return True
    return pub_dt >= datetime.now() - timedelta(days=recency_days)


def _is_relevant_enough(content: str, topic_keywords: Set[str], min_overlap: float) -> bool:
    """
    Post-scrape relevance filter (gap #5). Cheap keyword-overlap check so
    that whatever gets scraped isn't blindly indexed regardless of whether
    it actually relates to TOPIC.
    """
    if not topic_keywords:
        return True
    content_keywords = _keyword_set(content)
    overlap = len(topic_keywords & content_keywords) / len(topic_keywords)
    return overlap >= min_overlap


def run_pipeline() -> dict:
    started_at = datetime.now()
    logger.info("=== Starting daily ingestion pipeline for topic: '%s' ===", TOPIC)

    search_agent = SearchAgent()
    vector_agent = VectorDBAgent()
    topic_keywords = _keyword_set(TOPIC)

    # Step 1: Search
    search_results = search_agent.search()
    logger.info("SearchAgent returned %d candidate URLs.", len(search_results))

    # Step 2: Scrape with token-budget enforcement and per-URL failure handling
    scraped_documents = []
    failed_urls = []
    filtered_stale_urls = []
    filtered_irrelevant_urls = []
    budget_used = 0
    budget_exhausted = False

    for item in search_results:
        url = item["url"]

        if budget_used >= MAX_INGESTION_TOKEN_BUDGET:
            budget_exhausted = True
            logger.warning("Token budget (%d) exhausted. Stopping new scrapes; indexing what we have.", MAX_INGESTION_TOKEN_BUDGET)
            break

        logger.info("Scraping: %s", url)
        scraped = scrape_article(url)

        if scraped is None:
            logger.warning("Skipping failed/empty scrape: %s", url)
            failed_urls.append(url)
            continue

        if not _is_recent_enough(scraped.get("published_date"), RECENCY_DAYS):
            logger.info("Skipping stale article (published %s): %s", scraped.get("published_date"), url)
            filtered_stale_urls.append(url)
            continue

        if not _is_relevant_enough(scraped["content"], topic_keywords, MIN_RELEVANCE_KEYWORD_OVERLAP):
            logger.info("Skipping low-relevance article: %s", url)
            filtered_irrelevant_urls.append(url)
            continue

        doc = {
            "url": url,
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "content": scraped["content"],
            "retrieved_at": item.get("retrieved_at", datetime.now().isoformat()),
            "published_date": scraped.get("published_date"),
        }

        budget_used += _approx_tokens(doc["content"])
        scraped_documents.append(doc)

    logger.info(
        "Scraping complete: %d succeeded, %d failed, %d stale, %d irrelevant, ~%d tokens used.",
        len(scraped_documents), len(failed_urls), len(filtered_stale_urls),
        len(filtered_irrelevant_urls), budget_used,
    )

    # Step 3: Index
    indexing_report = vector_agent.index_documents(scraped_documents)

    finished_at = datetime.now()

    report = {
        "topic": TOPIC,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "urls_found": len(search_results),
        "urls_scraped_ok": len(scraped_documents),
        "urls_failed": failed_urls,
        "urls_filtered_stale": filtered_stale_urls,
        "urls_filtered_irrelevant": filtered_irrelevant_urls,
        "recency_days": RECENCY_DAYS,
        "min_relevance_keyword_overlap": MIN_RELEVANCE_KEYWORD_OVERLAP,
        "token_budget": MAX_INGESTION_TOKEN_BUDGET,
        "token_budget_used_approx": budget_used,
        "token_budget_exhausted": budget_exhausted,
        "indexing": indexing_report,
    }

    report_path = os.path.join(
        REPORT_FOLDER, f"ingestion_report_{started_at.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=== Ingestion pipeline complete. Report saved to %s ===", report_path)
    return report


if __name__ == "__main__":
    run_pipeline()
