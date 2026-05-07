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

def _humanize_col(name: str) -> str:
    """
    Expand abbreviated column headers for better embedding similarity.

    Examples:
      "2026f"   → "2026 forecast"
      "2026f_2" → "2026 forecast pct change"
      "2026f_3" → "2026 forecast vs prior"
      "2025"    → "2025"  (plain years unchanged)
    """
    m = re.fullmatch(r'(\d{4})f', name)
    if m:
        return f"{m.group(1)} forecast"
    m = re.fullmatch(r'(\d{4})f_2', name)
    if m:
        return f"{m.group(1)} forecast pct change"
    m = re.fullmatch(r'(\d{4})f_(\d+)', name)
    if m:
        return f"{m.group(1)} forecast vs prior"
    return name


def _looks_like_unit(s: str) -> bool:
    """True if s looks like a physical/currency unit (e.g. '$/bbl', '$/mt', '$/kg', '%')."""
    if not s:
        return False
    return len(s) <= 12 and ('/' in s or s.startswith('$') or s == '%'
                              or s.lower() in ('percent', 'index', 'mt', 'kg', 'bbl',
                                               'mmbtu', 'cum', 'toz', 'dmt'))


def _table_rows_to_nl(headers: list | None, rows: list) -> str:
    """
    Core table → NL conversion used by both PDF and markdown table paths.

    Design:
    - Unit column ("Unit" / "Units") is folded into the item name: "Name (unit)"
      so queries like "copper price $/mt" naturally find the right row.
    - Duplicate column names (e.g. three "2026f" for value/pct-change/difference)
      get positional suffixes: 2026f, 2026f_2, 2026f_3.
    - When headers are absent (cross-page tables), each row is serialised as
      "name (unit): val, val, val." preserving data while avoiding nonsensical
      header=value pairings.

    Output per row (when headers known):
      "Crude oil, Brent ($/bbl): 2024=80.7, 2025=69.0, 2026f=86.0, 2027f=70.0."
    Output per row (no headers):
      "Crude oil, Brent ($/bbl): 80.7, 69.0, 86.0, 70.0."
    """
    if not rows:
        return ""

    num_cols = max(len(r) for r in rows)

    # ── Build/normalize headers ───────────────────────────────────────────────
    if headers and any(h for h in headers if h is not None and str(h).strip()):
        hdrs = [str(h).strip() if h is not None else "" for h in headers]
        while len(hdrs) < num_cols:
            hdrs.append("")
        has_headers = True
    else:
        hdrs = [""] * num_cols
        has_headers = False

    # Deduplicate then humanize: "2026f" → "2026 forecast", "2026f_2" → "2026 forecast pct change"
    seen: dict = {}
    unique_hdrs: list = []
    for h in hdrs:
        k = h.lower()
        if k and k in seen:
            seen[k] += 1
            deduped = f"{h}_{seen[k]}"
        else:
            if k:
                seen[k] = 1
            deduped = h
        unique_hdrs.append(_humanize_col(deduped))

    # Detect "unit" column — by header name when headers exist,
    # or by value pattern when headers are absent (e.g. cross-page tables)
    unit_col: int | None = next(
        (i for i, h in enumerate(hdrs) if h.lower() in ("unit", "units")),
        None,
    )
    if unit_col is None and not has_headers and num_cols >= 2:
        # Heuristic: if every row's second cell looks like a unit string → it's the unit col
        col1_vals = [str(r[1]).strip() for r in rows
                     if r is not None and len(r) > 1 and r[1] is not None]
        if col1_vals and all(_looks_like_unit(v) or not v for v in col1_vals):
            unit_col = 1
    name_col = 0  # first column is always the item name/label

    # ── Convert rows ──────────────────────────────────────────────────────────
    sentences: list = []
    for row in rows:
        if not row or not any(c is not None and str(c).strip() for c in row):
            continue
        # Pad row to header length
        row = list(row) + [None] * max(0, num_cols - len(row))

        name = str(row[name_col]).strip() if row[name_col] is not None else ""
        unit = (
            str(row[unit_col]).strip()
            if unit_col is not None and row[unit_col] is not None
            else ""
        )
        subject = f"{name} ({unit})" if name and unit else name

        if has_headers:
            # "col=value" pairs, skipping name and unit columns
            pairs = []
            for i, (h, c) in enumerate(zip(unique_hdrs, row)):
                if i == name_col or i == unit_col:
                    continue
                c_str = str(c).strip() if c is not None else ""
                if c_str and h:
                    pairs.append(f"{h}={c_str}")
                elif c_str:
                    pairs.append(c_str)
            if subject and pairs:
                sentences.append(f"{subject}: {', '.join(pairs)}.")
            elif subject:
                sentences.append(f"{subject}.")
        else:
            # No headers: list values after subject, preserving order
            values = []
            for i, c in enumerate(row):
                if i == name_col or i == unit_col:
                    continue
                c_str = str(c).strip() if c is not None else ""
                if c_str:
                    values.append(c_str)
            if subject and values:
                sentences.append(f"{subject}: {', '.join(values)}.")
            elif subject:
                sentences.append(f"{subject}.")

    return "\n".join(sentences)


def _first_row_is_header(row: list) -> bool:
    """
    Return True if the row looks like column headers rather than data values.
    Heuristic: a header row has no decimal-point numbers (years like "2024"
    are fine as headers; values like "105.1" or "-12.3" are data).
    """
    values = [str(c).strip() for c in row if c is not None and str(c).strip()]
    return not any("." in v for v in values)


def _pdf_table_to_nl(table_data: list) -> str:
    """Convert pdfplumber table data to NL via _table_rows_to_nl."""
    if not table_data:
        return ""
    if len(table_data) >= 2 and _first_row_is_header(table_data[0]):
        return _table_rows_to_nl(table_data[0], table_data[1:])
    # Headerless (e.g. cross-page continuation): treat all rows as data
    return _table_rows_to_nl(None, table_data)


def _is_chart_noise(data: list) -> bool:
    """True only for clearly misdetected chart/figure regions (not real table data)."""
    if not data:
        return True
    total = sum(len(row) for row in data)
    if total == 0:
        return True
    non_empty = sum(
        1 for row in data for cell in row
        if cell is not None and str(cell).strip()
    )
    # Completely empty table (chart border lines detected as table)
    if non_empty == 0:
        return True
    # Chart axis labels: cell with 3+ newlines mixed with digits (e.g. "0\n20\n40\n60")
    for row in data:
        for cell in row:
            if cell is None:
                continue
            s = str(cell)
            if s.count('\n') >= 3 and any(c.isdigit() for c in s):
                return True
    return False


def _merge_table_fragments(table_data_pairs: list) -> list:
    """
    Reconstruct logical tables from pdfplumber fragments via two-phase merge.

    Phase 1 — horizontal: fragments at the same y-position (within 5px tolerance)
    are concatenated left-to-right to form complete rows.

    Phase 2 — vertical: adjacent rows that share the same x-span (within 20px)
    are stacked into a single logical table. This handles PDFs that encode one
    wide table as separate per-column-group objects (common in financial reports).
    """
    if not table_data_pairs:
        return []

    # ── Phase 1: horizontal merge within same y-position ────────────────────
    pairs = sorted(table_data_pairs, key=lambda p: (p[0].bbox[1], p[0].bbox[0]))
    y_groups, current = [], [pairs[0]]
    for pair in pairs[1:]:
        if abs(pair[0].bbox[1] - current[-1][0].bbox[1]) <= 5:
            current.append(pair)
        else:
            y_groups.append(current)
            current = [pair]
    y_groups.append(current)

    # Build rows with their combined x-span
    rows: list[tuple[float, float, list]] = []
    for group in y_groups:
        group.sort(key=lambda p: p[0].bbox[0])
        x0 = min(p[0].bbox[0] for p in group)
        x1 = max(p[0].bbox[2] for p in group)
        max_sub_rows = max(len(d) for _, d in group)
        for ri in range(max_sub_rows):
            merged_row = []
            for _, data in group:
                merged_row.extend(
                    data[ri] if ri < len(data) else [None] * len(data[0])
                )
            rows.append((x0, x1, merged_row))

    # ── Phase 2: vertical merge into logical tables ───────────────────────────
    # Two rows belong to the same table when their LEFT edges match (±20px).
    # We intentionally ignore the right edge so that narrow section-header rows
    # (e.g. "Energy" spanning only 6 of 10 columns) don't break the chain
    # between the INDEXES and PRICES blocks within the same wide table.
    tables: list[list[list]] = []
    current_tbl = [rows[0]]
    for row in rows[1:]:
        if abs(row[0] - current_tbl[-1][0]) <= 20:
            current_tbl.append(row)
        else:
            tables.append([r[2] for r in current_tbl])
            current_tbl = [row]
    tables.append([r[2] for r in current_tbl])

    return tables


def _chart_regions(fitz_page) -> list:
    """
    Return bounding boxes of chart/figure areas by clustering vector drawing objects.

    Charts in PDFs are composed of many small vector shapes (lines, bars, axes).
    Clustering their bboxes identifies the chart plot area so we can exclude
    chart-interior text (axis tick labels, legend text) from the extracted prose.
    """
    import fitz as _fitz
    drawings = fitz_page.get_drawings()
    page_w = fitz_page.rect.width
    # Ignore full-page-width lines (section separators) and tiny dots
    rects = [
        d['rect'] for d in drawings
        if d['rect'].width > 8 and d['rect'].height > 8
        and d['rect'].width < page_w * 0.94
    ]
    if not rects:
        return []
    # Greedy merge: expand each rect by 10px and union with any overlapping rect
    taken = [False] * len(rects)
    clusters = []
    for i, r in enumerate(rects):
        if taken[i]:
            continue
        taken[i] = True
        cluster = _fitz.Rect(r)
        changed = True
        while changed:
            changed = False
            for j in range(len(rects)):
                if taken[j]:
                    continue
                probe = _fitz.Rect(cluster.x0 - 10, cluster.y0 - 10,
                                   cluster.x1 + 10, cluster.y1 + 10)
                if probe.intersects(rects[j]):
                    cluster |= rects[j]
                    taken[j] = True
                    changed = True
        clusters.append(cluster)
    # Return only chart-sized regions (ignore small decorative elements)
    return [c for c in clusters if c.width > 60 and c.height > 60]


def _sort_reading_order(blocks: list, page_width: float) -> list:
    """Sort text blocks in reading order, handling 2-column layouts."""
    if not blocks:
        return blocks
    mid = page_width / 2
    left = [b for b in blocks if (b[0] + b[2]) / 2 < mid]
    right = [b for b in blocks if (b[0] + b[2]) / 2 >= mid]
    # 2-column: both sides have content AND no single block spans >60% of page width
    if (len(left) >= 2 and len(right) >= 2
            and all((b[2] - b[0]) < page_width * 0.6 for b in blocks)):
        return sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
    return sorted(blocks, key=lambda b: b[1])


def _process_page(fz_page, pl_page) -> str:
    """
    Extract one page using PyMuPDF for layout + pdfplumber for tables.

    Steps:
    1. PyMuPDF: get text blocks, detect chart regions from vector drawings
    2. Filter out header/footer (by y-position) and chart-interior text
    3. Sort blocks in reading order (2-column aware)
    4. pdfplumber: find tables, discard garbage (chart misdetections), merge fragments
    5. Compose: prose text (table areas masked out) + NL table sentences
    """
    page_h = fz_page.rect.height
    page_w = fz_page.rect.width
    header_cut = page_h * 0.06   # top 6%  → running headers / page titles
    footer_cut = page_h * 0.94   # bottom 6% → page numbers / footnote rules

    raw_blocks = fz_page.get_text("blocks")  # (x0,y0,x1,y1,text,bno,type)
    chart_bboxes = _chart_regions(fz_page)

    def _in_chart(b) -> bool:
        for cb in chart_bboxes:
            if (b[0] >= cb.x0 - 5 and b[2] <= cb.x1 + 5
                    and b[1] >= cb.y0 - 5 and b[3] <= cb.y1 + 5):
                return True
        return False

    text_blocks = [
        b for b in raw_blocks
        if b[6] == 0             # text block (not raster image)
        and b[4].strip()         # non-empty
        and b[1] >= header_cut   # below running header
        and b[3] <= footer_cut   # above footer
        and not _in_chart(b)     # not inside a chart/figure area
    ]
    text_blocks = _sort_reading_order(text_blocks, page_w)

    # pdfplumber table pipeline
    raw_tables = pl_page.find_tables()
    valid_pairs = []
    for t in raw_tables:
        data = t.extract()
        if not _is_chart_noise(data):
            valid_pairs.append((t, data))
    merged_tables = _merge_table_fragments(valid_pairs)
    table_bboxes_pl = [t.bbox for t, _ in valid_pairs]

    def _in_table(b) -> bool:
        for x0, top, x1, bottom in table_bboxes_pl:
            if (b[0] >= x0 - 5 and b[2] <= x1 + 5
                    and b[1] >= top - 5 and b[3] <= bottom + 5):
                return True
        return False

    prose_parts = [
        b[4].strip().replace('\n', ' ')
        for b in text_blocks
        if not _in_table(b) and b[4].strip()
    ]
    prose = "\n\n".join(prose_parts)
    table_nls = [nl for data in merged_tables if (nl := _pdf_table_to_nl(data))]

    parts = ([prose] if prose else []) + table_nls
    return "\n\n".join(parts)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF using PyMuPDF (layout analysis) + pdfplumber (tables).

    PyMuPDF handles: column reading order, header/footer filtering, chart region
    detection (via vector drawing clustering).
    pdfplumber handles: structured table extraction, with garbage filtering and
    fragment merging for tables split across column groups.
    Falls back to pypdf if either library is unavailable.
    """
    try:
        import fitz
        import pdfplumber as _plumber
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_out = []
        with _plumber.open(io.BytesIO(pdf_bytes)) as pdf_pl:
            for idx, (fz_pg, pl_pg) in enumerate(zip(doc, pdf_pl.pages)):
                text = _process_page(fz_pg, pl_pg)
                if text.strip():
                    pages_out.append(f"[Page {idx + 1}]\n{text.strip()}")
        return "\n\n".join(pages_out)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: pypdf plain-text extraction
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

    # Parse data rows
    data_rows = []
    for ln in table_lines[sep_idx + 1:]:
        cells = [c.strip() for c in ln.strip('|').split('|')]
        if any(cells):
            data_rows.append(cells)

    body = _table_rows_to_nl(headers, data_rows)
    if not body:
        return text

    # Keep non-table lines (e.g. title/notes above the table) as-is
    prose_lines = [ln for ln in lines if not ln.startswith('|')]
    prefix = "\n".join(prose_lines).strip()
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

    retriever = idx.as_retriever(similarity_top_k=n_results * 6)  # oversample: table NL chunks rank lower than prose
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
    retriever = idx.as_retriever(similarity_top_k=n_per_query * 6)

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
