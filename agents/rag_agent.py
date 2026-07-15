"""
agents/rag_agent.py

Conversational RAG Agent

On each user query:
  1. Retrieve top-k relevant chunks from the vector DB
  2. Re-rank them with a real cross-encoder (sentence-transformers
     CrossEncoder); if the model can't be loaded (e.g. not downloaded /
     offline), falls back to a lexical-overlap blend rather than crashing
  3. Build a prompt with retrieved context + conversation history
  4. Generate a grounded response with inline citations via local Ollama
  5. Never hallucinate: if nothing relevant is retrieved, say so explicitly
  6. Manage conversation memory, summarizing older turns when needed
"""

import re
from typing import List, Dict, Any, Optional

import requests

from config import (
    TOP_K,
    RERANK_TOP_N,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_REQUEST_TIMEOUT,
    USE_CROSS_ENCODER_RERANK,
    CROSS_ENCODER_MODEL,
)
from database.vector_store import VectorStore
from memory.chat_memory import ChatMemory
from logging_config import get_logger

logger = get_logger(__name__)

# Below this similarity score, we treat retrieval as "nothing relevant found".
# (This is the vector cosine-similarity score, not the cross-encoder score -
# the two use different scales, so relevance gating stays anchored to
# retrieval, and re-ranking only reorders within the retrieved set.)
MIN_RELEVANCE_SCORE = 0.15

_cross_encoder = None
_cross_encoder_load_failed = False


def _get_cross_encoder():
    """Lazily load the cross-encoder once per process; cache failures too."""
    global _cross_encoder, _cross_encoder_load_failed
    if _cross_encoder is not None or _cross_encoder_load_failed:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder '%s'...", CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder ready.")
    except Exception as e:
        logger.warning("Could not load cross-encoder (%s); falling back to lexical re-rank.", e)
        _cross_encoder_load_failed = True
    return _cross_encoder

_SYSTEM_PROMPT = """You are a research assistant answering questions using ONLY the provided source excerpts.

Rules:
- Only use information present in the SOURCES section below.
- Every factual claim must include an inline citation like [1], [2] referring to the source number.
- If the sources do not contain enough information to answer, say so explicitly instead of guessing.
- Do not use outside knowledge beyond what's in the sources.
- Be concise and directly answer the question first, then add supporting detail.
"""


def _call_ollama(messages: List[Dict[str, str]]) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        return (
            "I couldn't reach the local Ollama model right now. "
            "Make sure `ollama serve` is running and the model is pulled."
        )


def _lexical_rerank(query: str, results: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """
    Fallback re-ranker (used only if the cross-encoder can't be loaded):
    blends vector similarity score with keyword overlap.
    """
    query_terms = set(re.findall(r"\w+", query.lower()))

    def overlap_score(text: str) -> float:
        text_terms = set(re.findall(r"\w+", text.lower()))
        if not query_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)

    for r in results:
        r["rerank_score"] = 0.7 * r["score"] + 0.3 * overlap_score(r["content"])

    results.sort(key=lambda r: r["rerank_score"], reverse=True)
    return results[:top_n]


def _cross_encoder_rerank(
    query: str, results: List[Dict[str, Any]], top_n: int
) -> Optional[List[Dict[str, Any]]]:
    """Real cross-encoder re-rank. Returns None if the model isn't available."""
    model = _get_cross_encoder()
    if model is None:
        return None

    pairs = [(query, r["content"]) for r in results]
    scores = model.predict(pairs)

    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)

    results.sort(key=lambda r: r["rerank_score"], reverse=True)
    return results[:top_n]


def rerank(query: str, results: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """Cross-encoder re-rank when available (config-gated), lexical fallback otherwise."""
    if not results:
        return []
    if USE_CROSS_ENCODER_RERANK:
        cross_encoded = _cross_encoder_rerank(query, results, top_n)
        if cross_encoded is not None:
            return cross_encoded
    return _lexical_rerank(query, results, top_n)


class RAGAgent:
    def __init__(self, store: VectorStore = None):
        self.store = store or VectorStore()

    def _summarize_history(self, memory: ChatMemory):
        """Collapse older turns into a running summary to stay within the token budget."""
        from config import CHAT_SUMMARY_KEEP_LAST_TURNS

        turns_to_summarize = memory.turns[:-CHAT_SUMMARY_KEEP_LAST_TURNS] if len(memory.turns) > CHAT_SUMMARY_KEEP_LAST_TURNS else []
        keep_turns = memory.turns[-CHAT_SUMMARY_KEEP_LAST_TURNS:]

        if not turns_to_summarize:
            return

        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns_to_summarize)
        prompt = [
            {"role": "system", "content": "Summarize the following conversation excerpt concisely, preserving key facts, entities, and user intent, in under 150 words."},
            {"role": "user", "content": (memory.summary + "\n" + transcript) if memory.summary else transcript},
        ]

        new_summary = _call_ollama(prompt)
        memory.apply_summary(new_summary, keep_turns)

    def chat(self, session_id: str, user_query: str) -> Dict[str, Any]:
        memory = ChatMemory(session_id=session_id)

        # 1. Retrieve
        raw_results = self.store.query(user_query, top_k=TOP_K)

        # 2. Re-rank
        reranked = rerank(user_query, raw_results, RERANK_TOP_N)

        # Filter out weak matches so we don't hallucinate off irrelevant chunks
        relevant = [r for r in reranked if r["score"] >= MIN_RELEVANCE_SCORE]

        if not relevant:
            answer = (
                "I don't have any relevant information in the knowledge base to answer that yet. "
                "Try running the ingestion pipeline first, or ask about a topic that's been indexed."
            )
            memory.add_turn("user", user_query)
            memory.add_turn("assistant", answer)
            return {"answer": answer, "sources": []}

        # 3. Build prompt
        sources_block_lines = []
        sources_meta = []
        for i, r in enumerate(relevant, start=1):
            meta = r["metadata"]
            sources_block_lines.append(f"[{i}] (from {meta.get('url', 'unknown')}): {r['content']}")
            sources_meta.append(
                {
                    "index": i,
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "retrieved_at": meta.get("retrieved_at", ""),
                    "score": round(r["score"], 3),
                }
            )

        sources_block = "\n\n".join(sources_block_lines)

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(memory.get_context_messages())
        messages.append(
            {
                "role": "user",
                "content": f"SOURCES:\n{sources_block}\n\nQUESTION: {user_query}",
            }
        )

        # 4. Generate
        answer = _call_ollama(messages)

        # 5. Update memory + summarize if needed
        memory.add_turn("user", user_query)
        memory.add_turn("assistant", answer)

        if memory.needs_summarization():
            self._summarize_history(memory)

        return {"answer": answer, "sources": sources_meta}
