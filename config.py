# -----------------------------
# Topic Configuration
# -----------------------------
TOPIC = "Advancements in AI in the medical field"

# Number of articles to fetch
MAX_SOURCES_PER_RUN = 10

# Number of sub-queries to decompose TOPIC into (spec: 5-10)
NUM_SUBQUERIES = 8

# --- Search: rate limiting / quota (gap #4) ---
SEARCH_RATE_LIMIT_SEC = 1.5      # minimum delay between successive search queries
SEARCH_MAX_RETRIES = 3           # retries on failure/rate-limit before giving up on a query
SEARCH_BACKOFF_BASE_SEC = 2.0    # exponential backoff base between retries

# --- Search: recency filter (gap #2) ---
RECENCY_DAYS = 14                # articles older than this (by detected publish date) are dropped
                                  # articles with no detectable publish date are kept (can't penalize
                                  # a source just because it lacks clean metadata)

# --- Post-scrape relevance filter (gap #5) ---
MIN_RELEVANCE_KEYWORD_OVERLAP = 0.08  # fraction of topic keywords that must appear in scraped content

# ChromaDB Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5
RERANK_TOP_N = 5  # how many of the top_k survive re-ranking and go into the prompt

# Retention Policy
RETENTION_DAYS = 30

# Simple token/article budget
MAX_ARTICLES = 10

# Analyzer / ingestion token budget (approx, word-count based for prototype simplicity)
MAX_INGESTION_TOKEN_BUDGET = 20000

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Re-ranking (gap #6) ---
USE_CROSS_ENCODER_RERANK = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------
# Ollama (local LLM) Settings
# -----------------------------
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT = 120

# -----------------------------
# Conversational Memory Settings
# -----------------------------
# Approx token budget for raw (non-summarized) history kept in a session.
CHAT_MEMORY_MAX_TOKENS = 3000
# Once history exceeds the budget, older turns are collapsed into a running summary.
CHAT_SUMMARY_KEEP_LAST_TURNS = 4

# ChromaDB Path
VECTOR_DB_PATH = "./vector_store"

# Reports
REPORT_FOLDER = "./reports"

# Logs
LOG_FOLDER = "./logs"

# Chat session persistence (for resuming conversations)
SESSION_FOLDER = "./memory/sessions"


