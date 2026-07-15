## Agentic RAG with Scheduled Web Intelligence & Conversational Interface 

- This project implements an Agentic Retrieval-Augmented Generation system that automatically discovers, filters, and ingests the latest web content into a ChromaDB vector   database using AI agents.

- It provides a conversational interface powered by Ollama enabling users to ask questions and receive context-aware, responses with source citations based on the         continuously updated knowledge base.

## Folder Structure
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

Ingestion Pipeline 

<img width="600" height="700" alt="image" src="https://github.com/user-attachments/assets/87eba671-e979-4bc5-97de-7349a4d155e3" />


Conversational Pipeline 

<img width="600" height="700" alt="image" src="https://github.com/user-attachments/assets/018e285b-5f8c-48ff-9198-e7bc3d8f2320" />



## Set Up

## 1. Clone Repository
```
git clone https://github.com/TirumalaSrividya/agentic-rag-system
cd Agentic_Rag
```


## 2. Install dependencies

```bash
pip install -r requirements.txt
pip install ddgs   # your search.py imports this but it wasn't in requirements.txt
```


## 3. Start Ollama locally

```bash
ollama serve
ollama pull llama3.2
```


## 4. Run the ingestion pipeline once, manually

```bash
python ingestion.py 
```


## 5. Launch the chat interface

```bash
streamlit run app.py
```



