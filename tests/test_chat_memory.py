"""
tests/test_chat_memory.py

Covers memory/chat_memory.py:
- needs_summarization() triggers once the token budget is exceeded
- apply_summary() collapses old turns and keeps the last N verbatim
- session persistence: turns survive a save/reload roundtrip
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import shutil
import tempfile
from unittest.mock import patch

import pytest

import config
from memory.chat_memory import ChatMemory


@pytest.fixture
def temp_session_folder():
    tmp = tempfile.mkdtemp()
    with patch("memory.chat_memory.SESSION_FOLDER", tmp):
        yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_needs_summarization_false_when_under_budget(temp_session_folder):
    mem = ChatMemory(session_id="short_session")
    mem.turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    with patch("memory.chat_memory.CHAT_MEMORY_MAX_TOKENS", 3000):
        assert mem.needs_summarization() is False


def test_needs_summarization_true_when_over_budget(temp_session_folder):
    mem = ChatMemory(session_id="long_session")
    mem.turns = [{"role": "user", "content": "word " * 500} for _ in range(10)]
    with patch("memory.chat_memory.CHAT_MEMORY_MAX_TOKENS", 100):
        with patch("memory.chat_memory.CHAT_SUMMARY_KEEP_LAST_TURNS", 2):
            assert mem.needs_summarization() is True


def test_apply_summary_keeps_last_n_turns_verbatim(temp_session_folder):
    mem = ChatMemory(session_id="session_x")
    mem.turns = [{"role": "user", "content": f"turn {i}"} for i in range(6)]
    keep = mem.turns[-2:]

    mem.apply_summary("summary of older turns", keep)

    assert mem.summary == "summary of older turns"
    assert mem.turns == keep


def test_session_persists_across_reload(temp_session_folder):
    mem = ChatMemory(session_id="persisted_session")
    mem.add_turn("user", "hello")
    mem.add_turn("assistant", "hi there")

    reloaded = ChatMemory(session_id="persisted_session")
    assert reloaded.turns == mem.turns


def test_get_context_messages_includes_summary_when_present(temp_session_folder):
    mem = ChatMemory(session_id="ctx_session")
    mem.summary = "earlier discussion about X"
    mem.turns = [{"role": "user", "content": "follow up"}]

    messages = mem.get_context_messages()

    assert messages[0]["role"] == "system"
    assert "earlier discussion about X" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "follow up"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
