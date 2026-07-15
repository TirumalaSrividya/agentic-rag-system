## Agentic RAG with Scheduled Web Intelligence & Conversational Interface 

This project implements an Agentic Retrieval-Augmented Generation system that automatically discovers, filters, and ingests the latest web content into a ChromaDB vector database using AI agents.

It provides a conversational interface powered by Ollama  enabling users to ask questions and receive context-aware, grounded responses with source citations based on the continuously updated knowledge base.

## Project Folder Structure
```
agentic-rag/
│
├── agents/
│   ├── __init__.py
│   ├── search_agent.py          
│   ├── vector_db_agent.py       
│   └── rag_agent.py             
│
├── database/
│   ├── __init__.py
│   └── vector_store.py          # ChromaDB Wrapper
│
├── tools/
│   ├── __init__.py
│   ├── search.py                # Web Search Tool
│   └── scraper.py               # Article Scraper
│
├── memory/                      # Session Conversation history
│
├── vector_store/                # ChromaDB Storage
│
├── reports/                     # Daily ingestion reports
│   └── ingestion_report_*.json
│
├── logs/
│   └── agentic_rag.log
│
├── tests/
│   ├── __init__.py
│   ├── test_chat_memory.py
│   ├── test_ingestion_filters.py
│   ├── test_rag_agent.py
│   ├── test_scraper.py
│   ├── test_search_agent.py
│   ├── test_search_tool.py
│   └── test_vector_db_agent.py
│
├── app.py                      
├── ingestion.py                
├── config.py                    # Runtime Configuration
├── logging_config.py            
├── README.md
└── requirements.txt
```

## High Level Architecture
```

                               +-------------------------------+
                               |      Runtime Configuration    |
                               +-------------------------------+
                               |                               |Topic                           |
                               Chunksize                        Chunk Overlap                   |
                               Top K Retreival                 |
                               Token Budget                    |
                               RetentionDays                   ||------------------------------| 
                                                |
                                                |
                                    Manual Search Trigger
                                                |
                                                ▼
========================================================================================
                           DAILY INGESTION PIPELINE
========================================================================================

                +-------------------------+
                |     Search Agent        |
                |-------------------------|
                | • Generate 5–10 Queries |
                | • Search Latest Articles|
                | • Rank & Deduplicate    |
                +------------+------------+
                             |
                             ▼
                +-------------------------+
                |      Search Tool        |
                |-------------------------|
                | • Web Search            |
                | • Fetch URLs            |
                +------------+------------+
                             |
                             ▼
                +-------------------------+
                |     Scraper Tool        |
                |-------------------------|
                | • Extract Article       |
                | • Published Date        |
                | • Content               |
                | • Relevance Filter      |
                | • Skip Failed URLs      |
                +------------+------------+
                             |
                             ▼
                +-------------------------+
                |     VectorDB Agent      |
                |-------------------------|
                | • Chunk Documents       |
                | • Generate Embeddings   |
                | • Deduplicate Chunks    |
                | • Apply Retention       |
                | • Store Metadata        |
                +------------+------------+
                             |
                +------------+------------+
                |                         |
                ▼                         ▼
     +--------------------+    +----------------------+
     | ChromaDB           |    | Daily JSON Report    |
     | Vector Store       |    | Ingestion Report     |
     +--------------------+    +----------------------+

========================================================================================
                      CONVERSATIONAL RAG PIPELINE
========================================================================================

                    User Query
                         |
                         ▼
               +----------------------+
               |     Streamlit UI     |
               +----------+-----------+
                          |
                          ▼
               +----------------------+
               | Conversational       |
               | RAG Agent            |
               +----------+-----------+
                          |
                          ▼
               +----------------------+
               | ChromaDB             |
               | Retrieve Top-K       |
               +----------+-----------+
                          |
                Retrieved Documents
                          |
                 Re-ranking & Memory
                          |
                Prompt Construction
                          |
                          ▼
               +----------------------+
               | Ollama (Llama 3.2)   |
               +----------+-----------+
                          |
                          ▼
              Grounded Response
              + Inline Citations
```

## SET UP

## 1. Clone Repository
```
git clone <repo_url>
cd Agentic_Rag
```
## 2. Create Virtual Environment

## Windows
```
python -m venv .venv
.venv\Scripts\activate
```
## Linux / macOS
```
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
pip install ddgs   # your search.py imports this but it wasn't in requirements.txt
```


## 4. Start Ollama locally

```bash
ollama serve
ollama pull llama3.2
```


## 5. Run the ingestion pipeline once, manually

```bash
python ingestion.py 
```


## 6. Launch the chat interface

```bash
streamlit run app.py
```



