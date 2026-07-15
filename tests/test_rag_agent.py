"""
tests/test_rag_agent.py

Covers agents/rag_agent.py:
- No relevant chunks -> explicit "no info" answer, no LLM call, no hallucination
- Cross-encoder rerank is tried first; falls back to lexical rerank if the
  model can't be loaded
- Relevant chunks -> sources returned match reranked chunks
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

import pytest

import agents.rag_agent as rag_module
from agents.rag_agent import RAGAgent, rerank


def _make_result(content, score, url="https://example.com/a"):
    return {"content": content, "score": score, "metadata": {"url": url, "title": "T", "retrieved_at": ""}}


def test_no_relevant_chunks_returns_explicit_no_info_and_skips_llm():
    fake_store = MagicMock()
    fake_store.query.return_value = []  # nothing retrieved

    with patch("agents.rag_agent.ChatMemory") as MockMemory:
        mock_memory = MagicMock()
        mock_memory.turns = []
        MockMemory.return_value = mock_memory

        with patch("agents.rag_agent._call_ollama") as mock_llm:
            agent = RAGAgent(store=fake_store)
            result = agent.chat("session1", "What happened with X?")

    mock_llm.assert_not_called()
    assert result["sources"] == []
    assert "don't have any relevant information" in result["answer"].lower()


def test_low_score_chunks_are_filtered_out_before_generation():
    fake_store = MagicMock()
    fake_store.query.return_value = [_make_result("irrelevant chunk", score=0.01)]

    with patch("agents.rag_agent.ChatMemory") as MockMemory:
        mock_memory = MagicMock()
        mock_memory.turns = []
        MockMemory.return_value = mock_memory

        with patch("agents.rag_agent._call_ollama") as mock_llm:
            with patch("agents.rag_agent.rerank", side_effect=lambda q, r, n: r):
                agent = RAGAgent(store=fake_store)
                result = agent.chat("session1", "question")

    mock_llm.assert_not_called()
    assert result["sources"] == []


def test_relevant_chunks_generate_grounded_answer_with_sources():
    fake_store = MagicMock()
    fake_store.query.return_value = [_make_result("on-topic chunk", score=0.9)]

    with patch("agents.rag_agent.ChatMemory") as MockMemory:
        mock_memory = MagicMock()
        mock_memory.turns = []
        mock_memory.get_context_messages.return_value = []
        MockMemory.return_value = mock_memory

        with patch("agents.rag_agent._call_ollama", return_value="Grounded answer [1]"):
            with patch("agents.rag_agent.rerank", side_effect=lambda q, r, n: r):
                agent = RAGAgent(store=fake_store)
                result = agent.chat("session1", "question")

    assert result["answer"] == "Grounded answer [1]"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["url"] == "https://example.com/a"
    mock_memory.add_turn.assert_any_call("user", "question")
    mock_memory.add_turn.assert_any_call("assistant", "Grounded answer [1]")


def test_rerank_falls_back_to_lexical_when_cross_encoder_unavailable():
    results = [_make_result("some content about topic", score=0.5)]
    with patch("agents.rag_agent._cross_encoder_rerank", return_value=None):
        with patch("agents.rag_agent.USE_CROSS_ENCODER_RERANK", True):
            out = rerank("topic", results, top_n=1)
    assert len(out) == 1
    assert "rerank_score" in out[0]


def test_rerank_uses_cross_encoder_when_available():
    results = [_make_result("some content", score=0.5)]
    fake_cross_result = [{"content": "some content", "score": 0.5, "metadata": {}, "rerank_score": 9.9}]
    with patch("agents.rag_agent._cross_encoder_rerank", return_value=fake_cross_result) as mock_ce:
        with patch("agents.rag_agent.USE_CROSS_ENCODER_RERANK", True):
            out = rerank("topic", results, top_n=1)
    mock_ce.assert_called_once()
    assert out[0]["rerank_score"] == 9.9


def test_rerank_empty_results_returns_empty():
    assert rerank("query", [], top_n=3) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
