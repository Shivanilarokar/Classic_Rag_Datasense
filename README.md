# Classic RAG System: PDF + Milvus + OpenAI

A production-ready Retrieval-Augmented Generation (RAG) system that answers employee questions using only company internal PDFs. Built with FastAPI, Milvus vector database, and OpenAI APIs.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Component Overview](#component-overview)
- [Data Flow](#data-flow)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Installation & Setup](#installation--setup)
- [Running the System](#running-the-system)
- [API Endpoints](#api-endpoints)
- [Features](#features)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Milvus instance (Zilliz Cloud or self-hosted)
- `uv` package manager

### 1. Clone & Install

```bash
git clone <repository>
cd RAG_Architect_Assignment1
uv sync
```

### 2. Configure Environment

Create `.env` file:

```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-ada-002
OPENAI_CHAT_MODEL=gpt-4o-mini
EMBED_DIM=1536

MILVUS_URI=https://your-milvus-cluster.zillizcloud.com
MILVUS_TOKEN=your-token-here
MILVUS_COLLECTION=rag_chunks

CHUNK_SIZE=500
CHUNK_OVERLAP=80
TOP_K=5
RERANK_TOP_N=5
DEFAULT_NAMESPACE=org:default:public
```

### 3. Run the API

```powershell
$env:PYTHONPATH="src"
uv run uvicorn main:app --reload --port 8000
```

### 4. Upload a PDF

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/upload" `
  -F "file=@data/documents/HR_Policies.pdf" `
  -F "user_id=u1" -F "org_id=o1" -F "scope=org"
```

### 5. Ask a Question

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ask" `
  -H "Content-Type: application/json" `
  -d '{"question":"What is the leave policy?"}'
```

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                       │
│                           (main.py)                                │
├─────────────────────────────────────────────────────────────────┤
│   /documents/upload          │           /ask                      │
│   (Upload PDFs)              │       (Ask Questions)               │
└──────────┬────────────────────┴──────────┬───────────────────────┘
           │                               │
           │                               │
      ┌────▼───────────────────────────────▼──────┐
      │      RagPipeline Orchestrator             │
      │    (src/Rag_pipeline/pipeline.py)         │
      └────┬──────────────────────────────┬───────┘
           │                              │
      ┌────▼──────────────┐      ┌───────▼────────────┐
      │   INGESTION       │      │  RETRIEVAL & GEN   │
      │   Components      │      │   Components       │
      └────┬──────────────┘      └───────┬────────────┘
           │                             │
    ┌──────┴─────────┬──────────┐   ┌───┴──────┬─────────┐
    │                │          │   │          │         │
┌───▼────┐  ┌──────▼──┐  ┌────▼────┐ ┌─────▼───┐ ┌─────▼──────┐
│Document│  │   PDF   │  │OpenAI   │ │Retriever│ │LLMReranker │
│Loader  │  │ Chunker │  │Embedder │ │         │ │            │
└────────┘  └─────────┘  └─────────┘ └─────────┘ └────────────┘
    │           │           │             │           │
    │           │           └─────┬───────┴─┬─────────┘
    │           │                 │         │
┌───▼──────┐    │          ┌──────▼──┐  ┌──▼──────────┐
│ SQLite   │    │          │ Milvus  │  │Answer       │
│Metadata  │    │          │Vector   │  │Generator    │
│DB       │    │          │Database │  │             │
└──────────┘    │          └─────────┘  └─────────────┘
                │
            ┌───▼─────────┐
            │ uploads/    │
            │ (PDF files) │
            └─────────────┘
```

### Architecture Layers

#### 1. **API Layer** (`main.py`)
- FastAPI application exposing REST endpoints
- Handles file uploads and question requests
- Manages request validation and response formatting

#### 2. **Orchestration Layer** (`pipeline.py`)
- Coordinates all pipeline components
- Manages ingestion and retrieval workflows
- Connects embedding, storage, and generation services

#### 3. **Ingestion Components**
| Component | Purpose | File |
|-----------|---------|------|
| **PDFIngestionLoader** | PDF ingestion, deduplication, versioning, lifecycle management | `ingestion/Document_loader.py` |
| **PDFChunker** | Text splitting with overlap awareness | `ingestion/Chunker.py` |

#### 4. **Embedding Component** (`Embedding/openai_embedding.py`)
- Converts text chunks and queries into vector embeddings
- Uses OpenAI Embeddings API
- Maintains consistent vector dimensions

#### 5. **Vector Storage** (`vectordbstorage/milvustore.py`)
- Manages Milvus vector database operations
- Schema management with metadata fields
- Similarity search with filtering

#### 6. **Retrieval Component** (`retrieval/retriever.py`)
- Embeds user questions
- Performs vector similarity search in Milvus
- Applies namespace and lifecycle filters

#### 7. **Ranking Component** (`Generation/reranker.py`)
- Reorders retrieved chunks by relevance
- Uses LLM-based relevance scoring
- Provides reasoning for ranking decisions

#### 8. **Generation Component** (`Generation/generation.py`)
- Builds context from retrieved chunks
- Generates grounded answers with citations
- Enforces strict context-only responses

---

## 🔄 Data Flow Diagrams

### Upload Flow: `/documents/upload`

```
User Upload
    │
    ▼
┌─────────────────────────────────┐
│ API receives PDF + metadata     │
│ (user_id, org_id, scope, etc)  │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Derive namespace (scope-based)   │
│ from org_id, user_id, team_id   │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Save PDF to uploads/<namespace>/ │
│ Compute SHA256 checksum         │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Extract text & pages via pypdf   │
│ Normalize & validate content     │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Check for duplicates (checksum)  │
│ If found: return DUPLICATE status│
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Create/update SQLite records:    │
│ - documents (namespace, doc_key) │
│ - document_versions (version_id) │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Chunk text page-by-page using    │
│ RecursiveCharacterTextSplitter   │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Embed all chunks with OpenAI API │
│ (Returns vector embeddings)      │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Upsert to Milvus:                │
│ - vectors + text                 │
│ - metadata (namespace, doc_id,   │
│   version_id, page, lifecycle)   │
└──────────────┬──────────────────┘
               │
               ▼
Return: {status, namespace, doc_id, 
         version_id, chunks_indexed}
```

### Ask Flow: `/ask`

```
User Question
    │
    ▼
┌──────────────────────────────────┐
│ Select namespace:                │
│ DEFAULT_NAMESPACE or latest DB   │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Embed question with OpenAI       │
│ (Returns query vector)           │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Search Milvus (Top-K):           │
│ - COSINE similarity              │
│ - Filters: namespace, active,    │
│   latest versions                │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Rerank chunks using LLM:         │
│ - Judge relevance to question    │
│ - Return top-N with reasoning    │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Build context from chunks:       │
│ - Create labeled context blocks  │
│ - Include page numbers           │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Generate answer via LLM:         │
│ - Use system prompt (context-only│
│ - Force inline citations         │
│ - No invented facts              │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│ Format response:                 │
│ - Extract citations              │
│ - Deduplicate sources            │
│ - Build source/page mappings     │
└──────────────┬──────────────────┘
               │
               ▼
Return: {answer, sources: [{source, 
         citations: [{ref, page}]}]}


## ⚙️ Component Details

### 1. Configuration (`config.py`)

Centralized settings for the entire system:

```python
@dataclass
class ragConfig:
    # OpenAI
    openai_key: str
    embed_model: str = "text-embedding-ada-002"
    embedding_dim: int = 1536
    chat_model: str = "gpt-4o-mini"
    
    # Milvus
    milvus_url: str = ""
    milvus_api_key: str = ""
    milvus_collection: str = "rag_chunks"
    
    # RAG parameters
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k_results: int = 5
    rerank_top_n: int = 5
    default_namespace: str = ""
```

**Load from environment:**
```bash
config = ragConfig.load_from_env()
```

### 2. Ingestion Components

#### Document Loader
- **File**: `ingestion/Document_loader.py`
- **Key Methods**:
  - `ingest_pdf()` - Main entry point
  - `_load_pdf()` - Extract text using pypdf
  - `_save_file()` - Store PDF locally
  - `_find_duplicate_version()` - Checksum-based deduplication

#### PDF Chunker
- **File**: `ingestion/Chunker.py`
- **Strategy**: RecursiveCharacterTextSplitter
- **Features**:
  - Page-aware chunking (preserves page numbers)
  - Configurable chunk size and overlap
  - Maintains character indices for citations

### 3. Vector Database (`milvustore.py`)

**Schema Fields:**
```
- chunk_pk (VARCHAR, Primary Key)
- embedding (FLOAT_VECTOR, dim=1536)
- namespace (VARCHAR)
- doc_id (VARCHAR)
- version_id (VARCHAR)
- page (INT64)
- text (VARCHAR)
- lifecycle (VARCHAR) - ACTIVE/ARCHIVED/DELETED
- is_latest (BOOL)
- checksum_sha256 (VARCHAR)
```

**Key Operations:**
- `connect()` - Establish Milvus connection
- `ensure_collection()` - Create/validate schema
- `upsert_chunks()` - Write vectors + metadata
- `search()` - Vector similarity with filters

### 4. Retrieval (`retriever.py`)

```python
# Workflow:
1. Embed question → query_vector
2. Call milvus.search(query_vector, namespace, top_k)
3. Apply filters: namespace, lifecycle==ACTIVE, is_latest==true
4. Return RetrievedChunk objects with metadata
```

### 5. Reranking (`reranker.py`)

**LLM Reranking Process:**
1. Format retrieved chunks as passages
2. Prompt LLM with ranking criteria
3. LLM returns JSON: `{"order": [indices], "reasons": {}}`
4. Reorder and deduplicate results

**Ranking Criteria:**
- Direct answer relevance
- Specificity (concrete facts vs. generic)
- Coverage (multi-part questions)
- Terminology match
- Redundancy reduction

### 6. Answer Generation (`generation.py`)

**System Prompt Enforces:**
- Use ONLY provided context
- Cite using chunk tokens: `[C:doc_id:chunk_pk]`
- No external knowledge
- No invented facts
- Direct, concise answers

**Context Format:**
```
[C:doc_id:chunk_pk] doc_id=... version_id=... page=... score=...
<chunk content trimmed to 1200 chars>
```

---

## 🔐 Multi-Tenancy & Isolation

### Namespace Derivation

```python
scope = "private"  → org:{org_id}:user:{user_id}
scope = "team"     → org:{org_id}:team:{team_id}
scope = "org"      → org:{org_id}:public
```

**Isolation Guarantees:**
- Milvus filters by namespace in all searches
- SQLite separates documents by namespace
- Cross-tenant leakage is impossible

### Document Lifecycle

| Status | Milvus Filter | Meaning |
|--------|---------------|---------|
| ACTIVE | lifecycle=="ACTIVE" | Include in retrieval |
| ARCHIVED | excluded | Hidden but preserved |
| DELETED | excluded | Soft-delete marker |

### Versioning

- Upload same `doc_key` → creates new `version_id`
- Old versions marked: `is_latest=0`
- Retrieval uses: `is_latest=true` filter
- Duplicate detection: `checksum_sha256` within namespace

---

## 🛠️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/Shivanilarokar/Classic_Rag_Datasense
cd RAG_ARCHITECT_ASSIGNMENT1
```

### Step 2: Install Dependencies

```bash
# Install uv (Python package manager)
pip install uv

# Sync dependencies
uv sync

# If lock file issues occur:
uv lock
uv sync
```

### Step 3: Configure Environment

Create `.env` file in project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-key-here
OPENAI_EMBED_MODEL=text-embedding-ada-002
OPENAI_CHAT_MODEL=gpt-4o-mini
EMBED_DIM=1536

# Milvus Configuration (Zilliz Cloud example)
MILVUS_URI=https://your-cluster.zillizcloud.com
MILVUS_TOKEN=your-api-token
MILVUS_DATABASE=default
MILVUS_COLLECTION=rag_chunks

# RAG Parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=80
TOP_K=5
RERANK_TOP_N=5

# Optional: Set default namespace to avoid /ask namespace selection
DEFAULT_NAMESPACE=org:default:public
```

### Step 4: Verify Installation

```powershell
$env:PYTHONPATH="src"
uv run python -c "from Rag_pipeline.config import ragConfig; print('✓ Imports OK')"
```

---

## ▶️ Running the System

### Option A: Run FastAPI Server (Production)

```powershell
# Set Python path
$env:PYTHONPATH="src"

# Start server
uv run uvicorn main:app --reload --port 8000

# Server will be available at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

### Option B: Run Local Demo (Quick Test)

```powershell
$env:PYTHONPATH="src"
python -m Rag_pipeline.pipeline
```

This script:
1. Finds all PDFs in `data/documents/`
2. Ingests them
3. Asks a sample question
4. Prints the answer

---

## 📡 API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "config_ok": true,
  "config_errors": [],
  "milvus": {
    "connected": true,
    "collection": "rag_chunks",
    "collection_exists": true,
    "vector_dim": 1536
  },
  "embed_model": "text-embedding-ada-002",
  "chat_model": "gpt-4o-mini",
  "collection": "rag_chunks",
  "top_k": 5,
  "rerank_top_n": 5,
  "default_namespace": "org:default:public"
}
```

### 2. Upload PDF

**Endpoint:** `POST /documents/upload`

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/documents/upload" \
  -F "file=@path/to/document.pdf" \
  -F "user_id=u123" \
  -F "org_id=o999" \
  -F "scope=org" \
  -F "team_id=t456" \        # Required if scope=team
  -F "doc_key=my-document"   # Optional, auto-derived if missing
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | ✓ | PDF file (must be .pdf) |
| `user_id` | String | ✓ | User identifier |
| `org_id` | String | ✓ | Organization identifier |
| `scope` | String | ✓ | `private`, `team`, or `org` |
| `team_id` | String | ✗ | Required if scope=`team` |
| `doc_key` | String | ✗ | Document key (auto-derived from filename) |

**Response (Success):**
```json
{
  "status": "INDEXED",
  "namespace": "org:o999:public",
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "version_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "doc_key": "document-name",
  "checksum": "a1b2c3d4e5f6...",
  "file_path": "uploads/org_o999_public/uuid_document.pdf",
  "chunks_indexed": 42
}
```

**Response (Duplicate):**
```json
{
  "status": "DUPLICATE",
  "namespace": "org:o999:public",
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "version_id": "existing-version-id",
  "doc_key": "document-name",
  "checksum": "a1b2c3d4e5f6...",
  "file_path": "uploads/org_o999_public/uuid_document.pdf"
}
```

### 3. Ask Question

**Endpoint:** `POST /ask`

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the leave policy?"
  }'
```

**Response:**
```json
{
  "answer": "Based on the company leave policy, employees are entitled to...\n\n[C:doc_id:chunk_pk] Additional details about...",
  "sources": [
    {
      "source": "HR_Policies.pdf",
      "citations": [
        {
          "ref": "C:doc_id:chunk_pk",
          "page": 3
        },
        {
          "ref": "C:doc_id:chunk_pk_2",
          "page": 5
        }
      ]
    },
    {
      "source": "Employee_Handbook.pdf",
      "citations": [
        {
          "ref": "C:doc_id2:chunk_pk",
          "page": 12
        }
      ]
    }
  ]
}
```



### ✅ Features Implemented

- [x] PDF ingestion with deduplication
- [x] Page-aware text chunking
- [x] OpenAI embeddings integration
- [x] Milvus vector storage with metadata
- [x] Vector similarity search
- [x] LLM-based reranking
- [x] Grounded answer generation with citations
- [x] Multi-tenancy with namespace isolation
- [x] Document versioning
- [x] Lifecycle management (ACTIVE/ARCHIVED/DELETED)
- [x] SQLite metadata tracking
- [x] FastAPI REST API


---

**Last Updated**: 2025
**Python Version**: 3.11+

**Builted with ❤️ 

** connect me on : https://www.linkedin.com/in/shivanilarokar/

