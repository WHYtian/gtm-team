"""
RAG Manager — LlamaIndex + ChromaDB + HuggingFace local embeddings.

v2: namespace support for user/platform data isolation, batch ingest, metadata-filtered retrieval.
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
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SIMILARITY_CUTOFF = 0.35

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
