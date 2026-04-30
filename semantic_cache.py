"""
Semantic query cache for GTM research.

Lookup returns:
  {"hit": "fresh", ...}  — age <= FRESH_DAYS and similarity >= threshold
  {"hit": "stale", ...}  — age >  FRESH_DAYS and similarity >= threshold
  None                   — no match
"""
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import chromadb
from chromadb.config import Settings

CACHE_DIR = Path.home() / ".openclaw" / "research_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path.home() / ".openclaw" / "rag_db"
COLLECTION = "gtm_query_cache"
SIMILARITY_THRESHOLD = 0.75
FRESH_DAYS = 30

_embed_model = None


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model


def _embed(text: str) -> list[float]:
    return _get_model().encode(text, show_progress_bar=False).tolist()


def _get_col():
    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def lookup(query: str) -> dict | None:
    """Return a cache hit dict or None."""
    try:
        col = _get_col()
        if col.count() == 0:
            return None

        emb = _embed(query)
        res = col.query(
            query_embeddings=[emb],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        if not res["ids"][0]:
            return None

        # ChromaDB cosine collection returns distance = 1 - similarity
        distance = res["distances"][0][0]
        similarity = 1.0 - distance
        if similarity < SIMILARITY_THRESHOLD:
            return None

        meta = res["metadatas"][0][0]
        report_file = Path(meta["report_file"])
        if not report_file.exists():
            return None

        report = report_file.read_text(encoding="utf-8")
        cached_at = datetime.fromisoformat(meta["cached_at"])
        age_days = (datetime.now() - cached_at).days
        hit_type = "fresh" if age_days <= FRESH_DAYS else "stale"

        return {
            "hit": hit_type,
            "report": report,
            "topic": meta["topic"],
            "age_days": age_days,
            "similarity": round(similarity, 3),
            "cached_at": meta["cached_at"],
            "entry_id": res["ids"][0][0],
        }
    except Exception:
        return None


def store(query: str, topic: str, report: str, old_entry_id: str | None = None) -> None:
    """Save or replace a cache entry."""
    try:
        col = _get_col()

        if old_entry_id:
            try:
                # Remove old report file before deleting the DB entry
                old_res = col.get(ids=[old_entry_id], include=["metadatas"])
                if old_res["metadatas"]:
                    Path(old_res["metadatas"][0].get("report_file", "")).unlink(missing_ok=True)
                col.delete(ids=[old_entry_id])
            except Exception:
                pass

        emb = _embed(query)
        entry_id = str(uuid.uuid4())
        report_file = CACHE_DIR / f"{entry_id}.md"
        report_file.write_text(report, encoding="utf-8")

        col.add(
            ids=[entry_id],
            embeddings=[emb],
            documents=[query],
            metadatas=[{
                "topic": topic,
                "cached_at": datetime.now().isoformat(),
                "report_file": str(report_file),
                "query": query,
            }],
        )
    except Exception:
        pass


def list_entries() -> list[dict]:
    """Return all cache entries (for inspection)."""
    try:
        col = _get_col()
        res = col.get(include=["documents", "metadatas"])
        entries = []
        for eid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            cached_at = datetime.fromisoformat(meta["cached_at"])
            age_days = (datetime.now() - cached_at).days
            entries.append({
                "id": eid,
                "query": doc,
                "topic": meta["topic"],
                "cached_at": meta["cached_at"],
                "age_days": age_days,
                "report_file": meta["report_file"],
                "fresh": age_days <= FRESH_DAYS,
            })
        entries.sort(key=lambda x: x["cached_at"], reverse=True)
        return entries
    except Exception:
        return []
