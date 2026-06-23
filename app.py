"""
GTM Intelligence Platform — FastAPI + WebSocket multi-agent backend.
"""
import asyncio
import json
import queue
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).parent / "static"
REPORTS_DIR = Path.home() / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

HISTORY_FILE  = Path.home() / ".openclaw" / "gtm_chat_history_web.json"
SESSIONS_FILE = Path.home() / ".openclaw" / "gtm_sessions_web.json"
MAX_HISTORY   = 100
MAX_SESSIONS  = 30


def _load_history() -> list:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_history(history: list):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(
            json.dumps(history[-MAX_HISTORY:], ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_sessions() -> list:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_session(session_id: str, topic: str, user_msg: str, report: str, team_messages: list = None):
    sessions = _load_sessions()
    sessions.append({
        "id":            session_id,
        "topic":         topic,
        "user_msg":      user_msg,
        "timestamp":     datetime.now().isoformat(),
        "team_messages": (team_messages or [])[-120:],
        "report":        report,
    })
    try:
        SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSIONS_FILE.write_text(
            json.dumps(sessions[-MAX_SESSIONS:], ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


CHAT_HISTORY: list = _load_history()

app = FastAPI(title="GTM Intelligence Platform")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ── Session store ─────────────────────────────────────────────────────────────

class Session:
    def __init__(self, sid: str, ws: WebSocket):
        self.sid = sid
        self.ws = ws
        self.queue: queue.Queue = queue.Queue()
        self.history: list = []   # supervisor chat history
        self.alive = True
        self.stop_event: threading.Event = threading.Event()

    async def send(self, data: dict):
        try:
            await self.ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            self.alive = False


SESSIONS: dict[str, Session] = {}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/{sid}")
async def websocket_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    session = Session(sid, ws)
    SESSIONS[sid] = session

    # Send persisted history on connect
    await ws.send_text(json.dumps({"type": "history", "messages": CHAT_HISTORY}, ensure_ascii=False))

    # Background thread → WebSocket bridge
    async def pump_queue():
        while session.alive:
            try:
                msg = session.queue.get_nowait()
                await session.send(msg)
            except queue.Empty:
                await asyncio.sleep(0.05)
            except Exception:
                break

    pump_task = asyncio.create_task(pump_queue())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "ping":
                await session.send({"type": "pong"})
                continue

            if data.get("type") == "stop":
                session.stop_event.set()
                await session.send({"type": "stopped"})
                continue

            if data.get("type") == "chat":
                user_msg = data.get("message", "").strip()
                if user_msg:
                    session.stop_event.clear()
                    await session.send({"type": "user_message", "content": user_msg})
                    threading.Thread(
                        target=_run_agent,
                        args=(session, user_msg, None, None),
                        daemon=True,
                    ).start()

    except WebSocketDisconnect:
        pass
    finally:
        session.alive = False
        pump_task.cancel()
        SESSIONS.pop(sid, None)


def _run_agent(
    session: Session,
    user_msg: str,
    doc_content: Optional[str],
    doc_filename: Optional[str],
):
    from agents.graph import GRAPH

    q = session.queue

    def emit_status(msg: str):
        q.put({"type": "status", "message": msg})

    try:
        emit_status("Processing...")

        rag_ctx = ""

        initial_state = {
            "user_message": user_msg,
            "history": session.history[-10:],
            "supervisor_response": "",
            "is_research": False,
            "topic": "",
            "rag_context": rag_ctx,
            "doc_content": doc_content or "",
            "report_content": "",
            "team_messages": [],
            "route": "",
            # ReAct loop state (supervisor_init_node will set these for research routes)
            "rnd": 0,
            "researcher_calls": 0,
            "post_analyst_calls": 0,
            "analyst_called": False,
            "validator_called": False,
            "revision_count": 0,
            "next_action": "",
            "next_param": "",
            "stale_report": "",
            "stale_entry_id": "",
            "stale_age": 0,
            "user_rag_chunks": [],
            "workspace": [],
        }

        config = {
            "configurable": {
                "queue": q,
                "session_id": session.sid,
            }
        }

        result = GRAPH.invoke(initial_state, config=config)

        if session.stop_event.is_set():
            return  # user cancelled — discard result silently

        now_ts = datetime.now().strftime("%H:%M")

        # Update chat history
        supervisor_resp = result.get("supervisor_response", "")
        if supervisor_resp:
            session.history.append({"role": "user", "content": user_msg})
            session.history.append({"role": "assistant", "content": supervisor_resp})
            CHAT_HISTORY.append({"role": "user",       "content": user_msg,       "ts": now_ts})
            CHAT_HISTORY.append({"role": "supervisor",  "content": supervisor_resp, "ts": now_ts})
            _save_history(CHAT_HISTORY)
            q.put({"type": "supervisor_response", "content": supervisor_resp})

        # Send report if available
        report        = result.get("report_content", "")
        topic         = result.get("topic", "")
        raw_team_msgs = result.get("team_messages", [])

        # Wrap raw TeamMsg objects into the same envelope the WebSocket emits,
        # so the frontend loadSession() can render them identically to live chat.
        from agents.graph import AGENT_META as _AGENT_META
        team_messages = [
            {"type": "team_chat", "msg": m, "meta": _AGENT_META.get(m.get("agent", ""), {})}
            for m in raw_team_msgs
        ]

        if report:
            q.put({
                "type": "report_ready",
                "content": report,
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
            })
            _save_session(
                session_id=f"res_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session.sid[-6:]}",
                topic=topic or user_msg,
                user_msg=user_msg,
                report=report,
                team_messages=team_messages,
            )

        q.put({"type": "done"})

    except Exception as e:
        import traceback
        q.put({"type": "error", "message": str(e), "trace": traceback.format_exc()[-500:]})


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    from rag.manager import ingest_document, extract_pdf_text

    content_bytes = await file.read()
    filename = file.filename or "upload.pdf"

    session = SESSIONS.get(session_id)

    def progress(msg: str):
        if session:
            session.queue.put({"type": "upload_progress", "message": msg, "filename": filename})

    def run_ingest():
        from rag.manager import extract_text_for_ingest
        progress(f"Extracting text from {filename}...")
        text = extract_text_for_ingest(filename, content_bytes)

        if text.startswith("PDF extraction error"):
            if session:
                session.queue.put({"type": "upload_error", "message": text})
            return

        progress(f"Chunking and embedding {len(text.split())} words...")
        result = ingest_document(filename, text)

        if session:
            session.queue.put({
                "type": "upload_done",
                "filename": filename,
                "chunks": result.get("chunks", 0),
                "words": result.get("words", 0),
            })

    threading.Thread(target=run_ingest, daemon=True).start()
    return {"status": "processing", "filename": filename}


@app.post("/api/analyze-doc")
async def analyze_doc(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Upload a document and trigger the analyst pipeline on it."""
    from rag.manager import extract_pdf_text, ingest_document

    content_bytes = await file.read()
    filename = file.filename or "document.pdf"

    session = SESSIONS.get(session_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)

    text = extract_pdf_text(content_bytes) if filename.lower().endswith(".pdf") else content_bytes.decode("utf-8", errors="ignore")

    def run():
        _run_agent(session, f"Analyze this document: {filename}", text[:8000], filename)

    threading.Thread(target=run, daemon=True).start()
    return {"status": "processing", "filename": filename}


@app.get("/api/documents")
def list_documents():
    from rag.manager import list_documents as _list
    return {"documents": _list()}


@app.delete("/api/documents/{filename:path}")
def delete_document(filename: str):
    from rag.manager import delete_document as _delete
    return _delete(filename)


@app.get("/api/reports")
def list_reports():
    reports = []
    for p in sorted(REPORTS_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)[:20]:
        reports.append({
            "filename": p.name,
            "size": p.stat().st_size,
            "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        })
    return {"reports": reports}


@app.get("/api/reports/{filename}")
def get_report(filename: str):
    path = REPORTS_DIR / filename
    if not path.exists() or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"content": path.read_text(encoding="utf-8")}


@app.get("/rag", response_class=HTMLResponse)
def rag_inspector():
    return (STATIC / "rag_inspector.html").read_text(encoding="utf-8")


@app.get("/api/rag/docs")
def rag_docs():
    from rag.manager import list_documents
    return {"documents": list_documents()}


@app.get("/api/rag/query")
def rag_query(q: str, n: int = 8):
    from rag.manager import _get_collection, _get_bm25, _rrf, _tokenize, SIM_THRESHOLD
    col = _get_collection()
    total = col.count()
    if total == 0:
        return {"results": [], "query": q, "cutoff": SIM_THRESHOLD}

    fetch_n = min(n * 2, total)

    # Dense
    dense_res = col.query(
        query_texts=[q],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids   = dense_res.get("ids", [[]])[0]
    dense_dists = dense_res.get("distances", [[]])[0]
    dense_docs  = dense_res.get("documents", [[]])[0]
    dense_metas = dense_res.get("metadatas", [[]])[0]
    id_to_dense = {id_: (1.0 - dist, doc, meta)
                   for id_, dist, doc, meta in zip(dense_ids, dense_dists, dense_docs, dense_metas)}

    # Sparse
    sparse_ids = []
    try:
        bm25, corpus = _get_bm25()
        if bm25:
            scores = bm25.get_scores(_tokenize(q))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for idx in ranked[:fetch_n]:
                sparse_ids.append(corpus[idx]["id"])
    except Exception:
        pass

    fused_ids = _rrf(dense_ids, sparse_ids)[:n] if sparse_ids else dense_ids[:n]

    corpus_map = {}
    if sparse_ids:
        try:
            bm25, corpus = _get_bm25()
            corpus_map = {e["id"]: (e["text"], e["meta"]) for e in corpus}
        except Exception:
            pass

    results = []
    for id_ in fused_ids:
        if id_ in id_to_dense:
            score, text, meta = id_to_dense[id_]
        elif id_ in corpus_map:
            text, meta = corpus_map[id_]
            score = 0.0
        else:
            continue
        results.append({
            "score": round(score, 4),
            "filename": meta.get("filename", "?"),
            "chunk_index": meta.get("chunk_index"),
            "text": text,
        })

    return {"results": results, "query": q, "cutoff": SIM_THRESHOLD}


@app.get("/api/rag/chunks")
def rag_chunks(filename: str):
    from rag.manager import _get_collection
    col = _get_collection()
    res = col.get(where={"filename": filename}, include=["documents", "metadatas"])
    chunks = [
        {"text": text, **meta}
        for text, meta in zip(res.get("documents", []), res.get("metadatas", []))
    ]
    chunks.sort(key=lambda x: x.get("chunk_index") or 0)
    return {"chunks": chunks, "filename": filename}


@app.get("/api/rag/stats")
def rag_stats():
    from rag.manager import list_documents, _get_collection, SIM_THRESHOLD
    docs = list_documents()
    total_chunks = sum(d["chunks"] for d in docs)
    return {
        "total_docs": len(docs),
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": round(total_chunks / len(docs), 1) if docs else 0,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "similarity_cutoff": SIM_THRESHOLD,
    }


@app.get("/api/cache")
def list_cache():
    try:
        import sys
        sys.path.insert(0, str(Path.home() / "gtm-team"))
        from semantic_cache import list_entries
        return {"entries": list_entries()}
    except Exception as e:
        return {"entries": [], "error": str(e)}


@app.delete("/api/cache/{entry_id}")
def delete_cache_entry(entry_id: str):
    try:
        import sys
        sys.path.insert(0, str(Path.home() / "gtm-team"))
        from semantic_cache import _get_col
        col = _get_col()
        res = col.get(ids=[entry_id], include=["metadatas"])
        if res["metadatas"]:
            Path(res["metadatas"][0].get("report_file", "")).unlink(missing_ok=True)
        col.delete(ids=[entry_id])
        return {"deleted": entry_id}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/sessions")
def list_sessions():
    sessions = _load_sessions()
    return {"sessions": [
        {"id": s["id"], "topic": s["topic"],
         "user_msg": s.get("user_msg", ""), "timestamp": s["timestamp"]}
        for s in reversed(sessions)
    ]}


@app.get("/api/sessions/{session_id:path}")
def get_session(session_id: str):
    for s in _load_sessions():
        if s["id"] == session_id:
            return s
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/history/clear")
def clear_history():
    global CHAT_HISTORY
    CHAT_HISTORY = []
    _save_history(CHAT_HISTORY)
    return {"status": "ok"}


@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS)}
