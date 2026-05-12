# GTM Intelligence — Multi-Agent Industry Research Platform

A production multi-agent system that takes an industry or market as input and autonomously produces a structured GTM intelligence report. Built with Python asyncio, FastAPI, and a custom ReAct orchestration framework.

---

## Architecture

```
User Query
    │
    ▼
Smart Router (Supervisor LLM)
    │ industry research?          │ other question
    ▼                             ▼
Research Pipeline            Direct LLM reply
    │
    ▼
┌─────────────────────────────────────────┐
│           Supervisor (ReAct Hub)         │
│        THINK → ACT, dynamic routing      │
│        6 hard-constraint safety nets     │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
  [missing    [approved]  [needs
   data]                  revision]
       │          │          │
       ▼          ▼          ▼
  Researcher   Writer    Analyst ──► Critic
  (4-dim       (GTM       (TAM/        │
  parallel)   Report)    PESTEL)   VERDICT
                                      │
                              ┌───────┴───────┐
                         APPROVED        NEEDS_REVISION
                              │                │
                           Writer         Supervisor
                                         (re-route)

         ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄
         Synthesizer (optional, when user
         knowledge base exists)
         Web findings ↔ RAG reconciliation
         ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄
```

### Agents

| Agent | Role |
|---|---|
| **Supervisor** | ReAct engine — THINK → ACT routing hub. Dispatches tasks, enforces 6 Python-level hard constraints, manages dual research budget. |
| **Researcher (Alex)** | Parallel web search + scrape across 4 dimensions (market, competitive, technology, regulatory). Tags findings as `[Data]` / `[Estimate]` / `[Claim]`. |
| **Synthesizer (Jordan)** | *Optional.* Reconciles web findings against user-uploaded documents. Cosine similarity pre-filter (≥ 0.42) before LLM-level conflict detection. |
| **Analyst (Jamie)** | Structured analysis — TAM/SAM/SOM, PESTEL, Porter's Five Forces. |
| **Critic (Morgan)** | Quality review — returns `VERDICT: APPROVED` or `VERDICT: NEEDS_REVISION` with specific feedback. |
| **Writer** | Produces the final GTM report and competitive battle cards in Markdown. |

---

## Key Design Decisions

### ReAct + Dual Research Budget
The Supervisor runs a THINK → ACT loop rather than a fixed pipeline. Research calls are budgeted independently before and after analysis (pre-analyst: 3 calls, post-analyst: 2 calls), so the Supervisor can fetch missing data without restarting the full pipeline.

### 6 Hard Constraints
Six Python-level deterministic guards override LLM routing decisions for known failure modes — e.g. `REJECT_DATA` always routes back to Researcher regardless of what the LLM says, and `MAX_ROUNDS` enforces a hard loop ceiling.

### Semantic Cache
ChromaDB-backed topic similarity lookup: if a semantically similar topic has been researched before, the cached report is returned immediately (sub-second), skipping the full pipeline (~60s). Cache entries are only written for successfully completed reports.

### Embedding Pre-filter for Synthesizer
Before passing web/RAG chunk pairs to the LLM for conflict analysis, cosine similarity is computed via `all-MiniLM-L6-v2`. Only pairs above threshold 0.42 are forwarded, reducing irrelevant LLM calls significantly.

### Graceful Degradation
When web search returns no usable results, the Researcher returns `[RESEARCH: UNAVAILABLE]`. The Supervisor routes directly to the Analyst with an `[Estimate]` flag rather than retrying indefinitely.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API & Streaming | FastAPI + WebSocket (SSE) |
| Reverse Proxy | Nginx |
| LLM Backends | Doubao (Volcano Engine) · DeepSeek |
| Vector Store | ChromaDB + LlamaIndex |
| Embeddings | `all-MiniLM-L6-v2` (local, HuggingFace) |
| Concurrency | Python asyncio + ThreadPoolExecutor |
| Storage | ChromaDB (vectors) · JSON (chat history) · Markdown (reports) |

---

## Project Structure

```
gtm-team/
├── app.py                  # FastAPI app, WebSocket handler, REST endpoints
├── rag_mgr.py              # RAG ingestion, retrieval, multi-format parsing
├── semantic_cache.py       # ChromaDB-backed semantic cache
├── team/
│   ├── agent.py            # Base Agent class (OpenAI-compatible)
│   ├── orchestrator.py     # ReAct supervisor loop, pipeline orchestration
│   ├── personas.py         # Agent configurations (system prompts, models)
│   ├── skills.py           # Web search, scrape, document import wrappers
│   └── telemetry.py        # Per-agent timing and token estimation
├── static/
│   ├── index.html          # Chat UI
│   ├── rag_inspector.html  # Knowledge base inspector
│   └── architecture.html  # System architecture diagram
├── scripts/                # Offline utilities (KB builder, benchmarks)
├── test_data/              # Sample documents for RAG testing
└── test_routing.py         # Supervisor routing accuracy tests (16 cases)
```

---

## Running Locally

**Requirements:** Python 3.11+, API keys for Doubao and/or DeepSeek.

```bash
# Install dependencies
pip install fastapi uvicorn websockets chromadb llama-index \
            sentence-transformers openai pymupdf pdfplumber \
            openpyxl pandas

# Set API keys
export VOLC_API_KEY=your_key
export VOLC_API_BASE=https://ark.cn-beijing.volces.com/api/v3
export DS_API_KEY=your_deepseek_key

# Start server
uvicorn app:app --host 0.0.0.0 --port 8091

# Open http://localhost:8091
```

### Upload documents to the knowledge base

POST a file to `/api/upload` (form fields: `file`, `session_id`) to ingest PDF, Excel, CSV, or Markdown into the user namespace. The Synthesizer will automatically cross-reference it during research.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `WS` | `/ws/{session_id}` | Main chat WebSocket |
| `GET` | `/api/sessions` | List past research sessions |
| `GET` | `/api/cache` | List semantic cache entries |
| `DELETE` | `/api/cache/{id}` | Delete a cache entry |
| `GET` | `/api/rag` | List knowledge base documents |
| `POST` | `/api/upload` | Upload document to RAG |
| `POST` | `/api/import/url` | Scrape URL into RAG |
| `GET` | `/api/reports` | List generated reports |
| `GET` | `/api/reports/{filename}` | Download a report |
