"""
RAG Manager — LlamaIndex + ChromaDB + HuggingFace local embeddings.

Improvements over the old hand-rolled version:
- SentenceSplitter: chunks on sentence boundaries, not arbitrary word counts
- similarity_cutoff: irrelevant chunks are filtered out before reaching agents
- pypdf backend: more robust PDF extraction than PyPDF2
- Per-query retrieval: callers can pass specific queries per research dimension
"""
import io
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from llama_index.core import VectorStoreIndex, StorageContext, Settings as LISettings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

DB_PATH = Path.home() / ".openclaw" / "rag_db"
DB_PATH.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512       # tokens (SentenceSplitter respects sentence boundaries)
CHUNK_OVERLAP = 64
SIMILARITY_CUTOFF = 0.35   # discard chunks below this cosine similarity

_index: Optional[VectorStoreIndex] = None
_embed: Optional[HuggingFaceEmbedding] = None


def _get_embed():
    global _embed
    if _embed is None:
        _embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        LISettings.embed_model = _embed
        LISettings.llm = None   # no LLM needed — we handle synthesis ourselves
    return _embed


def _get_index() -> VectorStoreIndex:
    global _index
    if _index is None:
        _get_embed()
        chroma = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        col = chroma.get_or_create_collection("gtm_llamaindex")
        store = ChromaVectorStore(chroma_collection=col)
        ctx = StorageContext.from_defaults(vector_store=store)
        _index = VectorStoreIndex([], storage_context=ctx)
    return _index


# ── Ingestion ─────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF using pypdf (more robust than PyPDF2)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i+1}]\n{text.strip()}")
        return "\n\n".join(pages)
    except Exception as e:
        return f"PDF extraction error: {e}"


def ingest_document(filename: str, content: str, source_type: str = "upload") -> dict:
    """
    Chunk with SentenceSplitter (respects sentence boundaries),
    embed, and store in ChromaDB.
    """
    if not content or not content.strip():
        return {"error": "empty content"}

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    doc = Document(
        text=content,
        metadata={"filename": filename, "source_type": source_type},
    )
    nodes = splitter.get_nodes_from_documents([doc])

    # Attach metadata to each node for later filtering
    for i, node in enumerate(nodes):
        node.metadata["chunk_index"] = i
        node.metadata["total_chunks"] = len(nodes)

    idx = _get_index()
    idx.insert_nodes(nodes)

    return {
        "status": "ok",
        "filename": filename,
        "chunks": len(nodes),
        "words": len(content.split()),
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def query_rag(
    query: str,
    n_results: int = 5,
    filename_filter: Optional[str] = None,
) -> str:
    """
    Retrieve relevant chunks. Applies similarity_cutoff to drop noise.
    Returns formatted string ready to inject into agent prompts.
    """
    idx = _get_index()

    retriever_kwargs = dict(similarity_top_k=n_results)
    if filename_filter:
        retriever_kwargs["filters"] = {"filename": filename_filter}

    retriever = idx.as_retriever(**retriever_kwargs)
    nodes = retriever.retrieve(query)

    # Filter by similarity threshold
    relevant = [n for n in nodes if n.score is not None and n.score >= SIMILARITY_CUTOFF]
    if not relevant:
        relevant = nodes[:2] if nodes else []   # fallback: top-2 regardless

    if not relevant:
        return ""

    parts = []
    for n in relevant:
        fname = n.metadata.get("filename", "?")
        page = n.metadata.get("chunk_index", "")
        label = f"[{fname}" + (f" · chunk {page}" if page != "" else "") + "]"
        parts.append(f"{label}\n{n.text}")

    return "\n\n---\n\n".join(parts)


def query_rag_multi(queries: list[str], n_per_query: int = 3) -> str:
    """
    Run multiple targeted queries and merge results (deduped).
    Used by orchestrator to query each research dimension separately.
    """
    seen_ids = set()
    all_parts = []

    idx = _get_index()
    retriever = idx.as_retriever(similarity_top_k=n_per_query)

    for q in queries:
        nodes = retriever.retrieve(q)
        for n in nodes:
            if n.score is not None and n.score < SIMILARITY_CUTOFF:
                continue
            nid = n.node_id
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            fname = n.metadata.get("filename", "?")
            all_parts.append(f"[{fname}]\n{n.text}")

    return "\n\n---\n\n".join(all_parts)


# ── Management ────────────────────────────────────────────────────────────────

def list_documents() -> list[dict]:
    chroma = chromadb.PersistentClient(
        path=str(DB_PATH), settings=Settings(anonymized_telemetry=False)
    )
    try:
        col = chroma.get_collection("gtm_llamaindex")
        res = col.get(include=["metadatas"])
        seen: dict = {}
        for m in res.get("metadatas", []):
            fn = m.get("filename", "?")
            seen.setdefault(fn, {"filename": fn, "chunks": 0, "source_type": m.get("source_type", "upload")})
            seen[fn]["chunks"] += 1
        return list(seen.values())
    except Exception:
        return []


def delete_document(filename: str) -> dict:
    chroma = chromadb.PersistentClient(
        path=str(DB_PATH), settings=Settings(anonymized_telemetry=False)
    )
    try:
        col = chroma.get_collection("gtm_llamaindex")
        res = col.get(where={"filename": filename}, include=["metadatas"])
        ids = res.get("ids", [])
        if ids:
            col.delete(ids=ids)
        global _index
        _index = None   # reset index cache after deletion
        return {"status": "ok", "deleted": len(ids)}
    except Exception as e:
        return {"error": str(e)}
