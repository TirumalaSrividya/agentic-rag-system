"""
memory/chat_memory.py

Conversation memory management for the multi-turn chat interface.

- Keeps full turn history in memory (and persisted to disk per session
  so sessions can be resumed).
- When the raw history grows past CHAT_MEMORY_MAX_TOKENS (approximated
  by word count for this prototype), older turns are collapsed into a
  running summary via the LLM, keeping only the last N turns verbatim.
- Exposes get_context_messages() used by RAGAgent to build the prompt.
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from config import SESSION_FOLDER, CHAT_MEMORY_MAX_TOKENS, CHAT_SUMMARY_KEEP_LAST_TURNS
from logging_config import get_logger

logger = get_logger(__name__)

os.makedirs(SESSION_FOLDER, exist_ok=True)


def _approx_tokens(text: str) -> int:
    # Rough approximation: ~1.3 tokens per word. Good enough for a prototype budget check.
    return int(len(text.split()) * 1.3)


class ChatMemory:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.summary: str = ""
        self.turns: List[Dict[str, str]] = []  # [{"role": "user"/"assistant", "content": "..."}]
        self._load()

    # ---------- persistence ----------

    @property
    def _path(self) -> str:
        return os.path.join(SESSION_FOLDER, f"{self.session_id}.json")

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.summary = data.get("summary", "")
                self.turns = data.get("turns", [])
            logger.info("Resumed session %s with %d turns.", self.session_id, len(self.turns))

    def _save(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "summary": self.summary,
                    "turns": self.turns,
                    "updated_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    @staticmethod
    def list_sessions() -> List[str]:
        if not os.path.exists(SESSION_FOLDER):
            return []
        return [f.replace(".json", "") for f in os.listdir(SESSION_FOLDER) if f.endswith(".json")]

    # ---------- turn management ----------

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        self._save()

    def needs_summarization(self) -> bool:
        raw_text = self.summary + " ".join(t["content"] for t in self.turns)
        return _approx_tokens(raw_text) > CHAT_MEMORY_MAX_TOKENS and len(self.turns) > CHAT_SUMMARY_KEEP_LAST_TURNS

    def apply_summary(self, new_summary: str, keep_last_turns: List[Dict[str, str]]):
        """Called by RAGAgent after it generates a summary of the older turns."""
        self.summary = new_summary
        self.turns = keep_last_turns
        self._save()
        logger.info("Session %s summarized. %d turns retained verbatim.", self.session_id, len(self.turns))

    def get_context_messages(self) -> List[Dict[str, str]]:
        """
        Returns messages suitable for feeding into the LLM prompt:
        an optional summary pseudo-turn followed by recent raw turns.
        """
        messages = []
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation: {self.summary}",
                }
            )
        messages.extend(self.turns)
        return messages
