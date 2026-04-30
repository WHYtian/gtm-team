"""
RAG Manager — LlamaIndex + ChromaDB + HuggingFace local embeddings.

v2: namespace support for user/platform data isolation, batch ingest, metadata-filtered retrieval.
"""
import io
import re
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
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SIMILARITY_CUTOFF = 0.25

# ── Namespace constants ─────────────────────────────────────────────────────────
NAMESPACE_USER     = "user"
NAMESPACE_PLATFORM = "platform"
VALID_NAMESPACES   = {NAMESPACE_USER, NAMESPACE_PLATFORM}

_index: Optional[VectorStoreIndex] = None
_embed: Optional[HuggingFaceEmbedding] = None


def _get_embed():
    global _embed
    if _embed is None:
        _embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        LISettings.embed_model = _embed
        LISettings.llm = None
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


def _get_chroma_client():
    return chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


# ── PDF extraction ──────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> str:
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


# ── Ingestion ───────────────────────────────────────────────────────────────────

def _text_quality_ratio(text: str) -> float:
    """Return fraction of printable characters (0.0 = all binary, 1.0 = all text)."""
    if not text:
        return 0.0
    printable = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    return printable / len(text)


# Chunks below this threshold are considered binary garbage
MIN_TEXT_QUALITY = 0.85


def _tabular_to_sentences(text: str) -> str:
    """Convert markdown table text into natural language sentences for better embeddings.

    Markdown tables embed poorly with sentence-transformer models because:
    - Pipe characters add noise
    - SentenceSplitter cuts mid-table, orphaning values from their column headers

    Converts each data row into a self-contained sentence like:
      "Year: 2025. Market: Global Cloud Software. Size_Billion_USD: 285. Source: IDC."
    so every chunk is independently interpretable.

    Returns the original text unchanged if it doesn't look like a markdown table.
    """
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    # Detect markdown table: needs a header row and a separator row (|---|)
    table_lines = [ln for ln in lines if ln.startswith('|')]
    if len(table_lines) < 3:
        return text  # not a markdown table

    sep_idx = next(
        (i for i, ln in enumerate(table_lines)
         if re.match(r'^\|[\s\-|]+\|$', ln)), None)
    if sep_idx is None:
        return text

    # Parse header
    header_row = table_lines[sep_idx - 1] if sep_idx > 0 else table_lines[0]
    headers = [h.strip() for h in header_row.strip('|').split('|')]

    # Convert each data row to a sentence
    sentences = []
    for ln in table_lines[sep_idx + 1:]:
        cells = [c.strip() for c in ln.strip('|').split('|')]
        if not any(cells):
            continue
        parts = []
        for h, c in zip(headers, cells):
            if c:
                parts.append(f"{h}: {c}")
        if parts:
            sentences.append(". ".join(parts) + ".")

    if not sentences:
        return text

    # Keep non-table lines (e.g. title/notes above the table) as-is
    prose_lines = [ln for ln in lines if not ln.startswith('|')]
    prefix = "\n".join(prose_lines).strip()
    body = "\n".join(sentences)
    return (prefix + "\n\n" + body).strip() if prefix else body


def _dedup_document(filename: str, namespace: str) -> int:
    """Delete all chunks for a given filename+namespace combo. Returns count deleted."""
    chroma = _get_chroma_client()
    try:
        col = chroma.get_collection("gtm_llamaindex")
        res = col.get(
            where={"$and": [{"filename": filename}, {"namespace": namespace}]},
            include=["metadatas"],
        )
        ids = res.get("ids", [])
        if ids:
            col.delete(ids=ids)
            global _index
            _index = None
        return len(ids)
    except Exception:
        return 0


def ingest_document(
    filename: str,
    content: str,
    source_type: str = "upload",
    namespace: str = NAMESPACE_USER,
) -> dict:
    """
    Chunk with SentenceSplitter, embed, store in ChromaDB.

    Args:
        filename: document identifier
        content: raw text content
        source_type: "upload", "scraped", "api", etc.
        namespace: "user" or "platform" — logical data isolation
    """
    if not content or not content.strip():
        return {"error": "empty content"}

    if namespace not in VALID_NAMESPACES:
        return {"error": f"invalid namespace: {namespace}"}

    # Reject binary garbage before ingestion
    quality = _text_quality_ratio(content)
    if quality < MIN_TEXT_QUALITY:
        return {
            "error": f"content appears to be binary (text quality: {quality:.1%}, min: {MIN_TEXT_QUALITY:.0%})",
            "filename": filename,
        }

    # Convert markdown tables → natural language sentences before chunking.
    # Structured files (CSV/XLSX/JSON) arrive as markdown tables from doc_import_bytes;
    # sentence form embeds far better with all-MiniLM-L6-v2.
    content = _tabular_to_sentences(content)

    # Deduplicate: replace existing document with same filename + namespace
    _dedup_document(filename, namespace)

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    doc = Document(
        text=content,
        metadata={
            "filename": filename,
            "source_type": source_type,
            "namespace": namespace,
        },
    )
    nodes = splitter.get_nodes_from_documents([doc])

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
        "namespace": namespace,
    }


def ingest_batch(
    documents: list[dict],
    namespace: str = NAMESPACE_USER,
) -> list[dict]:
    """
    Batch ingest multiple documents in parallel.

    Args:
        documents: list of {"filename": str, "content": str, "source_type": str}
        namespace: namespace for all documents in batch
    """
    results = []
    for doc in documents:
        result = ingest_document(
            filename=doc["filename"],
            content=doc["content"],
            source_type=doc.get("source_type", "upload"),
            namespace=namespace,
        )
        results.append(result)
    return results


# ── Retrieval ───────────────────────────────────────────────────────────────────

def query_rag(
    query: str,
    n_results: int = 5,
    filename_filter: Optional[str] = None,
    namespace_filter: Optional[str] = None,
) -> str:
    """
    Retrieve relevant chunks with optional namespace filtering.

    Args:
        query: search query
        n_results: max chunks to return
        filename_filter: restrict to a specific document
        namespace_filter: "user", "platform", or None (all)
    """
    idx = _get_index()

    retriever = idx.as_retriever(similarity_top_k=n_results * 3)  # oversample then filter
    nodes = retriever.retrieve(query)

    # ── Filtering ────────────────────────────────────────────────────────────
    relevant = []
    for n in nodes:
        # Filename filter
        if filename_filter and n.metadata.get("filename") != filename_filter:
            continue
        # Namespace filter
        if namespace_filter and n.metadata.get("namespace") != namespace_filter:
            continue
        # Similarity cutoff
        if n.score is not None and n.score < SIMILARITY_CUTOFF:
            continue
        relevant.append(n)

    if not relevant:
        relevant = nodes[:2] if nodes else []

    if not relevant:
        return ""

    parts = []
    for n in relevant[:n_results]:
        fname = n.metadata.get("filename", "?")
        ns = n.metadata.get("namespace", "?")
        label = f"[{fname}" + (f" · {ns}" if ns else "") + "]"
        parts.append(f"{label}\n{n.text}")

    return "\n\n---\n\n".join(parts)


def query_rag_multi(
    queries: list[str],
    n_per_query: int = 3,
    namespace_filter: Optional[str] = None,
) -> str:
    """
    Run multiple targeted queries and merge deduped results.
    Supports namespace filtering across all queries.
    """
    seen_ids = set()
    all_parts = []

    idx = _get_index()
    retriever = idx.as_retriever(similarity_top_k=n_per_query * 3)

    for q in queries:
        nodes = retriever.retrieve(q)
        for n in nodes:
            if n.score is not None and n.score < SIMILARITY_CUTOFF:
                continue
            if namespace_filter and n.metadata.get("namespace") != namespace_filter:
                continue
            nid = n.node_id
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            fname = n.metadata.get("filename", "?")
            all_parts.append(f"[{fname}]\n{n.text}")

    return "\n\n---\n\n".join(all_parts)


# ── Management ──────────────────────────────────────────────────────────────────

def list_documents(namespace: Optional[str] = None) -> list[dict]:
    chroma = _get_chroma_client()
    try:
        col = chroma.get_collection("gtm_llamaindex")
        res = col.get(include=["metadatas"])
        seen: dict = {}
        for m in res.get("metadatas", []):
            ns = m.get("namespace", "?")
            if namespace and ns != namespace:
                continue
            fn = m.get("filename", "?")
            key = f"{ns}:{fn}"
            seen.setdefault(key, {
                "filename": fn,
                "namespace": ns,
                "chunks": 0,
                "source_type": m.get("source_type", "?"),
            })
            seen[key]["chunks"] += 1
        return list(seen.values())
    except Exception:
        return []


def list_namespaces() -> list[dict]:
    """Return stats per namespace."""
    docs = list_documents()
    stats: dict[str, dict] = {}
    for d in docs:
        ns = d["namespace"]
        if ns not in stats:
            stats[ns] = {"namespace": ns, "documents": 0, "chunks": 0}
        stats[ns]["documents"] += 1
        stats[ns]["chunks"] += d["chunks"]
    return list(stats.values())


def delete_document(filename: str, namespace: Optional[str] = None) -> dict:
    chroma = _get_chroma_client()
    try:
        col = chroma.get_collection("gtm_llamaindex")
        where = {"filename": filename}
        if namespace:
            where["namespace"] = namespace
        res = col.get(where=where, include=["metadatas"])
        ids = res.get("ids", [])
        if ids:
            col.delete(ids=ids)
        global _index
        _index = None
        return {"status": "ok", "deleted": len(ids)}
    except Exception as e:
        return {"error": str(e)}


def migrate_existing_namespace() -> dict:
    """
    One-time migration: add namespace to existing chunks that lack it.
    Uses source_type to infer: scraped → platform, upload → user.

    Updates BOTH the ChromaDB top-level metadata AND the nested
    _node_content JSON that LlamaIndex reads for node.metadata.
    Idempotent — safe to run multiple times.
    """
    import json as _json
    chroma = _get_chroma_client()
    try:
        col = chroma.get_collection("gtm_llamaindex")
        res = col.get(include=["metadatas"])
        updated = 0
        for doc_id, meta in zip(res.get("ids", []), res.get("metadatas", [])):
            if not meta:
                continue

            # Check if already fully migrated (both top-level and _node_content)
            has_top_ns = bool(meta.get("namespace"))
            has_inner_ns = False
            if "_node_content" in meta:
                try:
                    nc = _json.loads(meta["_node_content"])
                    has_inner_ns = bool(nc.get("metadata", {}).get("namespace"))
                except Exception:
                    pass

            if has_top_ns and has_inner_ns:
                continue

            ns = NAMESPACE_PLATFORM if meta.get("source_type") == "scraped" else NAMESPACE_USER

            # Update the _node_content JSON blob (what LlamaIndex reads)
            if "_node_content" in meta:
                try:
                    nc = _json.loads(meta["_node_content"])
                    nc.setdefault("metadata", {})["namespace"] = ns
                    meta["_node_content"] = _json.dumps(nc)
                except Exception:
                    pass

            meta["namespace"] = ns
            col.update(ids=[doc_id], metadatas=[meta])
            updated += 1
        global _index
        _index = None
        return {"status": "ok", "migrated": updated}
    except Exception as e:
        return {"error": str(e)}
