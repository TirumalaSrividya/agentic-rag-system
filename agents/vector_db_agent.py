"""
agents/vector_db_agent.py

VectorDB Agent

Responsibilities:
1. Chunk scraped documents using CHUNK_SIZE / CHUNK_OVERLAP
2. Generate embeddings (delegated to VectorStore)
3. Deduplicate against existing DB entries (by content hash)
4. Upsert new chunks with metadata
5. Enforce retention policy (purge expired documents)
6. Output an indexing report (chunk counts + dedup stats)
"""

from typing import List, Dict, Any

from config import CHUNK_SIZE, CHUNK_OVERLAP, RETENTION_DAYS
from database.vector_store import VectorStore, hash_content
from logging_config import get_logger

logger = get_logger(__name__)


class VectorDBAgent:
    def __init__(self, store: VectorStore = None):
        self.store = store or VectorStore()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Simple word-based sliding-window chunker (prototype-grade, no external deps)."""
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(chunk_size - overlap, 1)

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            if start + chunk_size >= len(words):
                break

        return chunks

    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        documents: list of dicts like
            {"url": ..., "title": ..., "content": ..., "retrieved_at": ..., "snippet": ...}

        Returns an indexing report.
        """
        total_chunks_seen = 0
        total_chunks_added = 0
        docs_processed = 0
        docs_skipped_empty = 0

        for doc in documents:
            content = doc.get("content", "")
            if not content or len(content.strip()) < 50:
                docs_skipped_empty += 1
                continue

            chunks = self.chunk_text(content)
            if not chunks:
                docs_skipped_empty += 1
                continue

            metadatas = []
            for idx, chunk in enumerate(chunks):
                metadatas.append(
                    {
                        "url": doc.get("url", ""),
                        "title": doc.get("title", ""),
                        "retrieved_at": doc.get("retrieved_at", ""),
                        "published_date": doc.get("published_date") or "",
                        "chunk_index": idx,
                        "content_hash": hash_content(chunk),
                    }
                )

            total_chunks_seen += len(chunks)
            added = self.store.upsert_chunks(chunks, metadatas)
            total_chunks_added += added
            docs_processed += 1

            logger.info(
                "Indexed doc %s: %d chunks total, %d new (deduped %d)",
                doc.get("url", "unknown"),
                len(chunks),
                added,
                len(chunks) - added,
            )

        purged = self.store.purge_expired(RETENTION_DAYS)

        report = {
            "documents_processed": docs_processed,
            "documents_skipped_empty_or_short": docs_skipped_empty,
            "chunks_seen": total_chunks_seen,
            "chunks_added": total_chunks_added,
            "chunks_deduped": total_chunks_seen - total_chunks_added,
            "chunks_purged_retention": purged,
            "total_chunks_in_db": self.store.count(),
        }

        logger.info("Indexing report: %s", report)
        return report
