"""
tests/test_vector_db_agent.py

Covers:
- Word-based chunking respects chunk_size / overlap
- Documents with too-short content are skipped
- Indexing report totals (chunks_seen / added / deduped / purged) are consistent
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock

import pytest

from agents.vector_db_agent import VectorDBAgent


def test_chunk_text_respects_chunk_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = VectorDBAgent.chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 100

    # overlap: last 20 words of chunk[0] should equal first 20 words of chunk[1]
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    assert first_chunk_words[-20:] == second_chunk_words[:20]


def test_chunk_text_empty_input():
    assert VectorDBAgent.chunk_text("", chunk_size=100, overlap=20) == []
    assert VectorDBAgent.chunk_text("   ", chunk_size=100, overlap=20) == []


def test_index_documents_skips_too_short_content():
    fake_store = MagicMock()
    fake_store.upsert_chunks.return_value = 1
    fake_store.purge_expired.return_value = 0
    fake_store.count.return_value = 1

    agent = VectorDBAgent(store=fake_store)
    documents = [
        {"url": "https://example.com/short", "title": "Short", "content": "too short", "retrieved_at": "2026-07-14T00:00:00"},
        {"url": "https://example.com/ok", "title": "OK", "content": "word " * 200, "retrieved_at": "2026-07-14T00:00:00"},
    ]

    report = agent.index_documents(documents)

    assert report["documents_processed"] == 1
    assert report["documents_skipped_empty_or_short"] == 1
    fake_store.upsert_chunks.assert_called_once()


def test_index_documents_report_math_is_consistent():
    fake_store = MagicMock()
    # simulate: every doc produces chunks, but only half get through dedup
    fake_store.upsert_chunks.side_effect = lambda chunks, metas: len(chunks) // 2
    fake_store.purge_expired.return_value = 2
    fake_store.count.return_value = 42

    agent = VectorDBAgent(store=fake_store)
    documents = [
        {"url": "https://example.com/1", "title": "A", "content": "word " * 300, "retrieved_at": "2026-07-14T00:00:00"},
    ]

    report = agent.index_documents(documents)

    assert report["chunks_seen"] == report["chunks_added"] + report["chunks_deduped"]
    assert report["chunks_purged_retention"] == 2
    assert report["total_chunks_in_db"] == 42


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
