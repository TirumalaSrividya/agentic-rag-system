"""
database/vector_store.py

Thin wrapper around ChromaDB (persistent, local) used by VectorDBAgent
and RAGAgent. Handles:
  - collection creation
  - embedding via sentence-transformers
  - upsert with metadata
  - content-hash based dedup lookup
  - similarity search
  - retention-based purge
"""

import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import VECTOR_DB_PATH, EMBEDDING_MODEL
from logging_config import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "web_intelligence"


def hash_content(text: str) -> str:
    """Stable content hash used for dedup checks."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("VectorStore ready. Existing chunk count: %d", self.collection.count())

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def content_exists(self, content_hash: str) -> bool:
        """Check if a chunk with this content hash is already indexed."""
        try:
            result = self.collection.get(where={"content_hash": content_hash}, limit=1)
            return len(result.get("ids", [])) > 0
        except Exception as e:
            logger.warning("Dedup lookup failed, assuming not present: %s", e)
            return False

    def upsert_chunks(self, chunks: List[str], metadatas: List[Dict[str, Any]]) -> int:
        """
        Upserts chunks that are not already present (by content hash).
        Returns the number of NEW chunks actually added.
        """
        new_chunks, new_metas, new_ids = [], [], []

        for chunk, meta in zip(chunks, metadatas):
            content_hash = meta["content_hash"]
            if self.content_exists(content_hash):
                continue
            new_chunks.append(chunk)
            new_metas.append(meta)
            new_ids.append(content_hash)  # hash doubles as a stable unique id

        if not new_chunks:
            return 0

        embeddings = self.embed(new_chunks)

        self.collection.upsert(
            ids=new_ids,
            documents=new_chunks,
            embeddings=embeddings,
            metadatas=new_metas,
        )

        return len(new_chunks)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embed([query_text])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            output.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # cosine distance -> similarity
                }
            )

        return output

    def purge_expired(self, retention_days: int) -> int:
        """Deletes chunks older than retention_days based on retrieved_at metadata."""
        cutoff = datetime.now() - timedelta(days=retention_days)

        all_items = self.collection.get(include=["metadatas"])
        ids_to_delete = []

        for _id, meta in zip(all_items.get("ids", []), all_items.get("metadatas", [])):
            retrieved_at = meta.get("retrieved_at")
            if not retrieved_at:
                continue
            try:
                ts = datetime.fromisoformat(retrieved_at)
            except ValueError:
                continue
            if ts < cutoff:
                ids_to_delete.append(_id)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info("Purged %d expired chunks older than %d days.", len(ids_to_delete), retention_days)

        return len(ids_to_delete)

    def count(self) -> int:
        return self.collection.count()
