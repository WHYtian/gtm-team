"""
GTM Team — OpenClaw-powered multi-agent research platform.
FastAPI + WebSocket. Agents powered by _llm.py from OpenClaw workspace.
"""
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).parent / "static"
REPORTS_DIR = Path.home() / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

HISTORY_FILE  = Path.home() / ".openclaw" / "gtm_chat_history.json"
SESSIONS_FILE = Path.home() / ".openclaw" / "gtm_sessions.json"
MAX_HISTORY  = 100
MAX_SESSIONS = 30


def _load_history() -> list:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_history(history: list):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(history[-MAX_HISTORY:], ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_sessions() -> list:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_session(session_id: str, topic: str, user_msg: str,
                  team_messages: list, report: str):
    sessions = _load_sessions()
    sessions.append({
        "id":            session_id,
        "topic":         topic,
        "user_msg":      user_msg,
        "timestamp":     datetime.now().isoformat(),
        "team_messages": team_messages[-120:],
        "report":        report,
    })
    try:
        SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSIONS_FILE.write_text(
            json.dumps(sessions[-MAX_SESSIONS:], ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


CHAT_HISTORY: list = _load_history()

app = FastAPI(title="GTM Team — OpenClaw Multi-Agent")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.on_event("startup")
async def _warmup():
    """Pre-load the embedding model in a background thread so first RAG query is instant."""
    import concurrent.futures
    def _load():
        try:
            from rag_mgr import _get_index
            _get_index()
        except Exception:
            pass
    loop = asyncio.get_event_loop()
    loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _load)


# ── Session store ─────────────────────────────────────────────────────────────

class Session:
    def __init__(self, sid: str, ws: WebSocket):
        self.sid = sid
        self.ws = ws
        self.q: asyncio.Queue = asyncio.Queue()
        self.history: list = []
        self.alive = True
        self.current_task: asyncio.Task | None = None

    async def send(self, data: dict):
        try:
            await self.ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            self.alive = False


SESSIONS: dict[str, Session] = {}


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    session = Session(sid, ws)
    SESSIONS[sid] = session

    async def pump():
        while session.alive:
            try:
                msg = session.q.get_nowait()
                await session.send(msg)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.05)
            except Exception:
                break

    pump_task = asyncio.create_task(pump())

    # Send persisted history on connect
    await session.send({"type": "history", "messages": CHAT_HISTORY})

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
                if session.current_task and not session.current_task.done():
                    session.current_task.cancel()
                await session.send({"type": "stopped"})
                continue

            if data.get("type") == "chat":
                msg = data.get("message", "").strip()
                if msg:
                    await session.send({"type": "user_message", "content": msg})
                    task = asyncio.create_task(_handle_chat(session, msg))
                    session.current_task = task

    except WebSocketDisconnect:
        pass
    finally:
        session.alive = False
        pump_task.cancel()
        SESSIONS.pop(sid, None)


async def _handle_chat(session: Session, user_msg: str):
    from team.agent import Agent, AgentCallError
    from team.personas import SUPERVISOR
    from team.orchestrator import run_research

    q = session.q
    now = lambda: datetime.now().strftime("%H:%M")

    async def emit_error(msg: str):
        await q.put({"type": "error", "message": msg})

    async def emit_status(msg: str):
        await q.put({"type": "status", "message": msg})

    try:
        await emit_status("Thinking...")

        # ── Supervisor routing ────────────────────────────────────────────────
        supervisor = Agent(**SUPERVISOR)
        try:
            route_resp = await asyncio.wait_for(
                supervisor.speak(user_msg, max_tokens=200, remember=False),
                timeout=30,
            )
        except (AgentCallError, asyncio.TimeoutError) as e:
            await emit_error(
                f"Supervisor is unavailable right now ({type(e).__name__}: {e}). "
                "Please try again in a moment."
            )
            await q.put({"type": "done"})
            return

        if not route_resp or not route_resp.strip():
            await emit_error("Supervisor returned an empty response. Please rephrase your question.")
            await q.put({"type": "done"})
            return

        if route_resp.strip().startswith("TASK:RESEARCH"):
            topic = user_msg
            for line in route_resp.split("\n"):
                if line.startswith("TOPIC:"):
                    topic = line[6:].strip()
                    break

            # ── Research pipeline ─────────────────────────────────────────────
            try:
                result = await run_research(topic, q)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await emit_error(f"Research pipeline failed unexpectedly: {e}")
                await q.put({"type": "done"})
                return

            report = result.get("report", "")
            if not report or report == "Report generation incomplete.":
                await emit_error(
                    "Research completed but the report could not be generated. "
                    "The team chat above shows what was collected."
                )

            # Persist session regardless of report quality
            session_id = f"res_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session.sid[-6:]}"
            _save_session(
                session_id=session_id,
                topic=topic,
                user_msg=user_msg,
                team_messages=result.get("team_messages", []),
                report=report,
            )

            if report and report != "Report generation incomplete.":
                resp_text = (
                    f"Your GTM Intelligence Report on **{topic}** is ready. "
                    "Check the **Report** tab for the full analysis."
                )
                await q.put({
                    "type": "report_ready",
                    "content": report,
                    "topic": result["topic"],
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                resp_text = (
                    f"Research on **{topic}** finished but the report writer did not produce output. "
                    "See Team Chat for collected findings."
                )

            CHAT_HISTORY.append({"role": "user",       "content": user_msg,   "ts": now()})
            CHAT_HISTORY.append({"role": "supervisor",  "content": resp_text,  "ts": now()})
            _save_history(CHAT_HISTORY)
            await q.put({"type": "supervisor_response", "content": resp_text})

        else:
            GTM_INVITE = (
                "\n\n---\n"
                "*Want a full GTM intelligence report? Just name an industry or market. Examples:*\n"
                "- *Cloud CRM software in North America*\n"
                "- *China new energy vehicle (NEV) market*\n"
                "- *Global HR SaaS competitive landscape*\n"
                "- *Southeast Asia ride-hailing industry*"
            )
            display_resp = route_resp + GTM_INVITE
            CHAT_HISTORY.append({"role": "user",       "content": user_msg,    "ts": now()})
            CHAT_HISTORY.append({"role": "supervisor",  "content": display_resp, "ts": now()})
            _save_history(CHAT_HISTORY)
            await q.put({"type": "supervisor_response", "content": display_resp})

        await q.put({"type": "done"})

    except asyncio.CancelledError:
        raise  # let asyncio handle clean cancellation
    except Exception as e:
        # Last-resort catch — always unblock the frontend
        try:
            await q.put({"type": "error", "message": f"Unexpected server error: {e}"})
            await q.put({"type": "done"})
        except Exception:
            pass


# ── REST: file upload ─────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_to_kb(file: UploadFile = File(...), session_id: str = Form(...)):
    """Add any supported document (PDF/TXT/XLSX/CSV/JSON/ZIP) to RAG knowledge base."""
    content = await file.read()
    filename = file.filename or "upload"
    session = SESSIONS.get(session_id)
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    async def run():
        try:
            from rag_mgr import extract_pdf_text, ingest_document
            from team.skills import doc_import_bytes

            if session:
                await session.q.put({"type": "upload_progress", "message": f"Processing {filename}...", "filename": filename})

            if ext == "pdf":
                text = extract_pdf_text(content)
            elif ext in ("xlsx", "csv", "json", "zip"):
                parsed = await doc_import_bytes(filename, content)
                if "error" in parsed:
                    if session:
                        await session.q.put({"type": "upload_error", "message": parsed["error"]})
                    return
                text = parsed.get("text", "")
            else:
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    text = content.decode("latin-1")

            result = ingest_document(filename, text, namespace="user")
            if session:
                if "error" in result:
                    await session.q.put({"type": "upload_error", "message": result["error"]})
                else:
                    await session.q.put({"type": "upload_done", "filename": filename, "chunks": result.get("chunks", 0), "words": result.get("words", 0)})
        except Exception as e:
            if session:
                await session.q.put({"type": "upload_error", "message": str(e)})

    asyncio.create_task(run())
    return {"status": "processing", "filename": filename}


@app.post("/api/import/files")
async def import_files(files: list[UploadFile] = File(...), namespace: str = Form("user")):
    """Batch import multiple files of any supported format into RAG."""
    results = []

    async def process_one(file: UploadFile) -> dict:
        from rag_mgr import extract_pdf_text, ingest_document
        from team.skills import doc_import_bytes

        content = await file.read()
        filename = file.filename or "import"
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        try:
            if ext == "pdf":
                text = extract_pdf_text(content)
            elif ext in ("xlsx", "csv", "json", "zip"):
                parsed = await doc_import_bytes(filename, content)
                if "error" in parsed:
                    return {"filename": filename, "status": "error", "error": parsed["error"]}
                text = parsed.get("text", "")
            else:
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    text = content.decode("latin-1")

            result = ingest_document(filename, text, namespace=namespace)
            ok = "error" not in result
            return {"filename": filename, "status": ("ok" if ok else "error"), **result}
        except Exception as e:
            return {"filename": filename, "status": "error", "error": str(e)}

    results = await asyncio.gather(*[process_one(f) for f in files])
    return {"files": len(results), "results": results}


@app.post("/api/import/url")
async def import_url(url: str = Form(...), namespace: str = Form("user"), filename: str = Form("")):
    """Scrape a URL and ingest its content into RAG."""
    from team.skills import web_scrape
    from rag_mgr import ingest_document

    text = await web_scrape(url)
    if not text or len(text) < 50:
        return {"status": "error", "error": "URL returned insufficient content", "url": url}

    fname = filename or url.rstrip("/").rsplit("/", 1)[-1][:80] or "webpage"
    result = ingest_document(fname + ".txt", text, source_type="url", namespace=namespace)
    return {"status": "ok", "url": url, **result}


@app.post("/api/rag/migrate")
def migrate_namespace():
    """One-time: add namespace metadata to existing chunks."""
    try:
        from rag_mgr import migrate_existing_namespace
        return migrate_existing_namespace()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/analyze-doc")
async def analyze_doc(file: UploadFile = File(...), session_id: str = Form(...)):
    """Trigger full analyst pipeline on an uploaded document."""
    content = await file.read()
    filename = file.filename or "document"
    session = SESSIONS.get(session_id)
    if not session:
        return JSONResponse({"error": "session not found"}, 404)

    async def run():
        from team.orchestrator import run_doc_analysis
        try:
            from rag_mgr import extract_pdf_text
            text = extract_pdf_text(content) if filename.lower().endswith(".pdf") else content.decode("utf-8", errors="ignore")
        except Exception:
            text = content.decode("utf-8", errors="ignore")

        result = await run_doc_analysis(text, filename, session.q)
        await session.q.put({"type": "supervisor_response", "content": f"Document **{filename}** analysis complete."})
        await session.q.put({"type": "report_ready", "content": result["report"], "topic": result["topic"], "timestamp": datetime.now().isoformat()})
        await session.q.put({"type": "done"})

    asyncio.create_task(run())
    return {"status": "processing", "filename": filename}


# ── REST: misc ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/rag", response_class=HTMLResponse)
def rag_inspector():
    return (STATIC / "rag_inspector.html").read_text(encoding="utf-8")


# ── RAG Inspector API ─────────────────────────────────────────────────────────

@app.get("/api/rag/docs")
def rag_docs():
    try:
        from rag_mgr import list_documents
        return {"documents": list_documents()}
    except Exception as e:
        return {"documents": [], "error": str(e)}


@app.get("/api/rag/query")
def rag_query(q: str, n: int = 8):
    """Return raw retrieval results with similarity scores."""
    try:
        from rag_mgr import _get_index, SIMILARITY_CUTOFF
        idx = _get_index()
        retriever = idx.as_retriever(similarity_top_k=n)
        nodes = retriever.retrieve(q)
        results = []
        for node in nodes:
            results.append({
                "score": round(node.score or 0, 4),
                "filename": node.metadata.get("filename", "?"),
                "chunk_index": node.metadata.get("chunk_index"),
                "text": node.text,
            })
        return {"results": results, "query": q, "cutoff": SIMILARITY_CUTOFF}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/api/rag/chunks")
def rag_chunks(filename: str):
    """Return all chunks for a specific document."""
    try:
        import chromadb
        from chromadb.config import Settings
        from pathlib import Path
        DB_PATH = Path.home() / ".openclaw" / "rag_db"
        client = chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))
        col = client.get_collection("gtm_llamaindex")
        res = col.get(where={"filename": filename}, include=["documents", "metadatas"])
        chunks = []
        for text, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            chunks.append({"text": text, "chunk_index": meta.get("chunk_index"), **meta})
        chunks.sort(key=lambda x: x.get("chunk_index") or 0)
        return {"chunks": chunks, "filename": filename}
    except Exception as e:
        return {"chunks": [], "error": str(e)}


@app.get("/api/rag/stats")
def rag_stats():
    try:
        from rag_mgr import list_documents, EMBED_MODEL, SIMILARITY_CUTOFF
        docs = list_documents()
        total_chunks = sum(d["chunks"] for d in docs)
        return {
            "total_docs": len(docs),
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": round(total_chunks / len(docs), 1) if docs else 0,
            "embedding_model": EMBED_MODEL,
            "similarity_cutoff": SIMILARITY_CUTOFF,
            "docs": docs,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/documents")
def list_docs(namespace: str = ""):
    try:
        from rag_mgr import list_documents
        ns = namespace if namespace in ("user", "platform") else None
        return {"documents": list_documents(namespace=ns)}
    except Exception:
        return {"documents": []}


@app.get("/api/rag/namespaces")
def rag_namespaces():
    try:
        from rag_mgr import list_namespaces
        return {"namespaces": list_namespaces()}
    except Exception as e:
        return {"namespaces": [], "error": str(e)}


@app.delete("/api/documents/{filename:path}")
def del_doc(filename: str):
    try:
        from rag_mgr import delete_document
        return delete_document(filename)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/reports")
def list_reports():
    reports = [
        {"filename": p.name, "size": p.stat().st_size, "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat()}
        for p in sorted(REPORTS_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)[:20]
    ]
    return {"reports": reports}


@app.get("/api/cache")
def list_cache():
    try:
        from semantic_cache import list_entries
        return {"entries": list_entries()}
    except Exception as e:
        return {"entries": [], "error": str(e)}


@app.delete("/api/cache/{entry_id}")
def delete_cache_entry(entry_id: str):
    try:
        from semantic_cache import _get_col
        from pathlib import Path
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
        for s in reversed(sessions)   # newest first
    ]}


@app.get("/api/sessions/{session_id:path}")
def get_session(session_id: str):
    for s in _load_sessions():
        if s["id"] == session_id:
            return s
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS), "backend": "openclaw-native"}
