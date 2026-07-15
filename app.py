"""
app.py

Streamlit conversational interface on top of RAGAgent.

- Session management: start new / resume previous sessions (persisted via ChatMemory).
- Displays retrieved source documents alongside each response.
- Follow-up questions resolve using conversation history (handled inside RAGAgent/ChatMemory).
- Never hallucinates: RAGAgent explicitly says when nothing relevant was found.
"""

import uuid

import streamlit as st

from config import TOPIC
from agents.rag_agent import RAGAgent
from memory.chat_memory import ChatMemory

st.set_page_config(page_title="Agentic RAG Chat", layout="wide")


@st.cache_resource
def get_rag_agent() -> RAGAgent:
    return RAGAgent()


rag_agent = get_rag_agent()

# ---------- Sidebar: session management ----------
st.sidebar.title("Sessions")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.display_history = []

if st.sidebar.button("+ New session"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.display_history = []
    st.rerun()

existing_sessions = ChatMemory.list_sessions()
if existing_sessions:
    chosen = st.sidebar.selectbox(
        "Resume a session",
        options=["(current)"] + existing_sessions,
    )
    if chosen != "(current)" and chosen != st.session_state.session_id:
        st.session_state.session_id = chosen
        resumed = ChatMemory(session_id=chosen)
        st.session_state.display_history = resumed.turns.copy()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Topic: **{TOPIC}**")
st.sidebar.caption(f"Session: `{st.session_state.session_id[:8]}...`")

if st.sidebar.button("Run ingestion now"):
    with st.spinner("Running ingestion pipeline (this can take a while)..."):
        from ingestion import run_pipeline
        report = run_pipeline()
    st.sidebar.success(f"Ingested {report['indexing']['chunks_added']} new chunks.")

# ---------- Main chat ----------
st.title("Agentic RAG — Chat")
st.caption("Ask questions about the ingested knowledge base. Answers are grounded and cited.")

if "display_history" not in st.session_state:
    st.session_state.display_history = []

for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(f"**[{s['index']}]** [{s['title'] or s['url']}]({s['url']}) — score {s['score']}")

user_input = st.chat_input("Ask something about the knowledge base...")

if user_input:
    st.session_state.display_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag_agent.chat(st.session_state.session_id, user_input)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"**[{s['index']}]** [{s['title'] or s['url']}]({s['url']}) — score {s['score']}")

    st.session_state.display_history.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
