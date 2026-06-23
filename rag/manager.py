"""
RAG Manager: PDF ingestion + ChromaDB vector search.
Hybrid retrieval: dense (BGE) + sparse (BM25) fused with RRF.
"""
import hashlib
import io
import os
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

DB_PATH = Path.home() / ".openclaw" / "rag_db"
DB_PATH.mkdir(parents=True, exist_ok=True)

SIM_THRESHOLD = 0.62  # cosine similarity cutoff for RAG retrieval (BGE-calibrated)

_client: Optional[chromadb.PersistentClient] = None
_collection = None
_embed_fn = None

# BM25 state
_bm25_index = None
_bm25_corpus: list[dict] = []   # [{"id": ..., "text": ..., "meta": ...}]
_bm25_dirty = True               # rebuild needed after ingest/delete

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "do", "does", "did", "have", "has", "had",
    "what", "which", "who", "how", "that", "this", "it", "its",
    "not", "no", "nor", "so", "yet", "both", "either", "whether",
    "i", "we", "you", "he", "she", "they", "my", "our", "their",
}

def _tokenize(text: str) -> list[str]:
    return [w for w in text.lower().split() if w not in _STOPWORDS]


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils import embedding_functions
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
    return _embed_fn


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name="gtm_documents",
            embedding_function=_get_embed_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_bm25():
    """Lazy-load BM25 index from ChromaDB. Rebuilds when _bm25_dirty is True."""
    global _bm25_index, _bm25_corpus, _bm25_dirty
    if _bm25_index is not None and not _bm25_dirty:
        return _bm25_index, _bm25_corpus

    from rank_bm25 import BM25Okapi

    col = _get_collection()
    res = col.get(include=["documents", "metadatas"])
    ids       = res.get("ids", [])
    documents = res.get("documents", [])
    metadatas = res.get("metadatas", [])

    _bm25_corpus = [
        {"id": id_, "text": text, "meta": meta}
        for id_, text, meta in zip(ids, documents, metadatas)
    ]
    tokenized = [_tokenize(doc) for doc in documents]
    _bm25_index = BM25Okapi(tokenized) if tokenized else None
    _bm25_dirty = False
    return _bm25_index, _bm25_corpus


def _rrf(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion — returns doc IDs ranked by combined score."""
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── Table → Natural Language helpers (pdfplumber pipeline) ────────────────────

# Maps lowercased column header → expanded label.
# Fix 1: keep abbreviation alongside expansion for headers users query by acronym.
# Fix 2: add CAC/LTV (was missing entirely).
# Fix 3: use shorter expansion form to avoid sentence-length dilution.
# Fix 4: fix "yoy" missing the word "growth".
_COL_ABBREVS = {
    "rev.": "revenue", "rev": "revenue",
    "yoy%": "year-over-year growth pct", "yoy": "year-over-year growth",
    "qoq%": "quarter-over-quarter growth pct", "qoq": "quarter-over-quarter growth",
    "mom%": "month-over-month growth pct", "mom": "month-over-month growth",
    "cagr": "CAGR compound annual growth rate",           # keep acronym + expansion
    "cagr(3yr)": "3-year CAGR compound annual growth rate",
    "tam": "total addressable market TAM", "sam": "serviceable addressable market SAM",
    "som": "serviceable obtainable market SOM",
    "nrr%": "net revenue retention NRR pct", "nrr": "net revenue retention NRR",
    "arr": "annual recurring revenue ARR", "mrr": "monthly recurring revenue MRR",
    "arpu": "average revenue per user ARPU",
    "cac": "customer acquisition cost CAC", "ltv": "lifetime value LTV",
    "cac/ltv": "customer acquisition cost to lifetime value ratio CAC/LTV",
    "p/e": "price-to-earnings ratio P/E",
    "adj.ebitda": "adjusted EBITDA", "ebitda": "EBITDA",
    "mkt.shr(%)": "market share pct", "mkt.shr": "market share",
    "mkt.cap": "market cap", "mkt.cap(b$)": "market cap billion usd",
    "rev.(b$)": "revenue billion usd", "rev.(m$)": "revenue million usd",
    "iaas%": "IaaS revenue pct", "paas%": "PaaS revenue pct", "saas%": "SaaS revenue pct",
    "reg.": "region",
    "churn%": "churn rate pct",
}

def _humanize_col(name: str) -> str:
    # Expand year abbreviations: "2026f" → "2026 forecast"
    m = re.fullmatch(r'(\d{4})f', name)
    if m: return f"{m.group(1)} forecast"
    m = re.fullmatch(r'(\d{4})f_2', name)
    if m: return f"{m.group(1)} forecast pct change"
    m = re.fullmatch(r'(\d{4})f_(\d+)', name)
    if m: return f"{m.group(1)} forecast vs prior"
    # Try exact match first
    key = name.strip().lower()
    if key in _COL_ABBREVS:
        return _COL_ABBREVS[key]
    # Strip trailing parenthetical unit: "TAM ($B)" → base="TAM", unit="$B"
    m2 = re.match(r'^(.+?)\s*\(([^)]+)\)\s*$', name.strip())
    if m2:
        base, unit = m2.group(1).strip(), m2.group(2).strip()
        expanded = _COL_ABBREVS.get(base.lower())
        if expanded:
            return f"{expanded} ({unit})"
    # Strip trailing % or suffix: "YoY Δ" → try "yoy"
    base_plain = re.sub(r'[\s%Δ△*†‡#]+$', '', name).strip()
    expanded = _COL_ABBREVS.get(base_plain.lower())
    if expanded:
        return expanded
    return name


def _looks_like_unit(s: str) -> bool:
    if not s: return False
    return len(s) <= 12 and ('/' in s or s.startswith('$') or s == '%'
        or s.lower() in ('percent', 'index', 'mt', 'kg', 'bbl', 'mmbtu', 'cum', 'toz', 'dmt'))


def _table_rows_to_nl(headers: list | None, rows: list) -> str:
    if not rows:
        return ""
    num_cols = max(len(r) for r in rows)
    if headers and any(h for h in headers if h is not None and str(h).strip()):
        hdrs = [str(h).strip() if h is not None else "" for h in headers]
        while len(hdrs) < num_cols:
            hdrs.append("")
        has_headers = True
    else:
        hdrs = [""] * num_cols
        has_headers = False

    seen: dict = {}
    unique_hdrs: list = []
    for h in hdrs:
        k = h.lower()
        if k and k in seen:
            seen[k] += 1
            deduped = f"{h}_{seen[k]}"
        else:
            if k: seen[k] = 1
            deduped = h
        unique_hdrs.append(_humanize_col(deduped))

    unit_col: int | None = next(
        (i for i, h in enumerate(hdrs) if h.lower() in ("unit", "units")), None)
    if unit_col is None and not has_headers and num_cols >= 2:
        col1_vals = [str(r[1]).strip() for r in rows if r is not None and len(r) > 1 and r[1] is not None]
        if col1_vals and all(_looks_like_unit(v) or not v for v in col1_vals):
            unit_col = 1
    name_col = 0

    sentences: list = []
    for row in rows:
        if not row or not any(c is not None and str(c).strip() for c in row):
            continue
        row = list(row) + [None] * max(0, num_cols - len(row))
        name = str(row[name_col]).strip() if row[name_col] is not None else ""
        unit = str(row[unit_col]).strip() if unit_col is not None and row[unit_col] is not None else ""
        subject = f"{name} ({unit})" if name and unit else name

        if has_headers:
            pairs = []
            for i, (h, c) in enumerate(zip(unique_hdrs, row)):
                if i == name_col or i == unit_col: continue
                c_str = str(c).strip() if c is not None else ""
                if c_str in ("N/A", "n/a", "-", "", "NULL", "null", "None", "none"): continue
                if c_str and h: pairs.append(f"{h}={c_str}")
                elif c_str: pairs.append(c_str)
            if subject and pairs: sentences.append(f"{subject}: {', '.join(pairs)}.")
            elif subject: sentences.append(f"{subject}.")
        else:
            values = [str(c).strip() for i, c in enumerate(row) if i != name_col and i != unit_col and c is not None and str(c).strip()]
            if subject and values: sentences.append(f"{subject}: {', '.join(values)}.")
            elif subject: sentences.append(f"{subject}.")

    return "\n".join(sentences)


def _first_row_is_header(row: list) -> bool:
    values = [str(c).strip() for c in row if c is not None and str(c).strip()]
    return not any("." in v for v in values)


def _pdf_table_to_nl(table_data: list) -> str:
    if not table_data: return ""
    if len(table_data) >= 2 and _first_row_is_header(table_data[0]):
        return _table_rows_to_nl(table_data[0], table_data[1:])
    return _table_rows_to_nl(None, table_data)


def _is_chart_noise(data: list) -> bool:
    if not data: return True
    non_empty = sum(1 for row in data for cell in row if cell is not None and str(cell).strip())
    if non_empty == 0: return True
    for row in data:
        for cell in row:
            if cell is None: continue
            s = str(cell)
            if s.count('\n') >= 3 and any(c.isdigit() for c in s):
                return True
    return False


def _merge_table_fragments(table_data_pairs: list) -> list:
    if not table_data_pairs: return []
    pairs = sorted(table_data_pairs, key=lambda p: (p[0].bbox[1], p[0].bbox[0]))
    y_groups, current = [], [pairs[0]]
    for pair in pairs[1:]:
        if abs(pair[0].bbox[1] - current[-1][0].bbox[1]) <= 5:
            current.append(pair)
        else:
            y_groups.append(current)
            current = [pair]
    y_groups.append(current)

    rows: list[tuple[float, float, list]] = []
    for group in y_groups:
        group.sort(key=lambda p: p[0].bbox[0])
        x0 = min(p[0].bbox[0] for p in group)
        x1 = max(p[0].bbox[2] for p in group)
        max_sub_rows = max(len(d) for _, d in group)
        for ri in range(max_sub_rows):
            merged_row = []
            for _, data in group:
                merged_row.extend(data[ri] if ri < len(data) else [None] * len(data[0]))
            rows.append((x0, x1, merged_row))

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
    import fitz as _fitz
    drawings = fitz_page.get_drawings()
    page_w = fitz_page.rect.width
    rects = [d['rect'] for d in drawings
             if d['rect'].width > 8 and d['rect'].height > 8
             and d['rect'].width < page_w * 0.94]
    if not rects: return []
    taken = [False] * len(rects)
    clusters = []
    for i, r in enumerate(rects):
        if taken[i]: continue
        taken[i] = True
        cluster = _fitz.Rect(r)
        changed = True
        while changed:
            changed = False
            for j in range(len(rects)):
                if taken[j]: continue
                probe = _fitz.Rect(cluster.x0 - 10, cluster.y0 - 10, cluster.x1 + 10, cluster.y1 + 10)
                if probe.intersects(rects[j]):
                    cluster |= rects[j]
                    taken[j] = True
                    changed = True
        clusters.append(cluster)
    return [c for c in clusters if c.width > 60 and c.height > 60]


def _sort_reading_order(blocks: list, page_width: float) -> list:
    if not blocks: return blocks
    mid = page_width / 2
    left  = [b for b in blocks if (b[0] + b[2]) / 2 < mid]
    right = [b for b in blocks if (b[0] + b[2]) / 2 >= mid]
    if (len(left) >= 2 and len(right) >= 2
            and all((b[2] - b[0]) < page_width * 0.6 for b in blocks)):
        return sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
    return sorted(blocks, key=lambda b: b[1])


def _process_page(fz_page, pl_page, table_converter=None) -> str:
    """Extract one page: PyMuPDF for prose layout, pdfplumber for tables.
    table_converter: callable(table_data) -> str, defaults to _pdf_table_to_nl.
    """
    if table_converter is None:
        table_converter = _pdf_table_to_nl

    page_h = fz_page.rect.height
    page_w = fz_page.rect.width
    header_cut = page_h * 0.06
    footer_cut = page_h * 0.94

    raw_blocks  = fz_page.get_text("blocks")
    chart_bboxes = _chart_regions(fz_page)

    def _in_chart(b) -> bool:
        return any(b[0] >= cb.x0 - 5 and b[2] <= cb.x1 + 5
                   and b[1] >= cb.y0 - 5 and b[3] <= cb.y1 + 5
                   for cb in chart_bboxes)

    text_blocks = [b for b in raw_blocks
                   if b[6] == 0 and b[4].strip()
                   and b[1] >= header_cut and b[3] <= footer_cut
                   and not _in_chart(b)]
    text_blocks = _sort_reading_order(text_blocks, page_w)

    raw_tables = pl_page.find_tables()
    valid_pairs = []
    for t in raw_tables:
        data = t.extract()
        if not _is_chart_noise(data):
            valid_pairs.append((t, data))
    merged_tables    = _merge_table_fragments(valid_pairs)
    table_bboxes_pl  = [t.bbox for t, _ in valid_pairs]

    def _in_table(b) -> bool:
        return any(b[0] >= x0 - 5 and b[2] <= x1 + 5
                   and b[1] >= top - 5 and b[3] <= bottom + 5
                   for x0, top, x1, bottom in table_bboxes_pl)

    prose_parts = [b[4].strip().replace('\n', ' ')
                   for b in text_blocks if not _in_table(b) and b[4].strip()]
    prose     = "\n\n".join(prose_parts)
    table_nls = [nl for data in merged_tables if (nl := table_converter(data))]

    parts = ([prose] if prose else []) + table_nls
    return "\n\n".join(parts)


# ── LLM table converter ────────────────────────────────────────────────────────

_LLM_TABLE_PROMPT = """\
Convert this table to indexed sentences for semantic search. \
Use EXACTLY the format below — no prose, no filler words.

OUTPUT FORMAT (strict):
  <row_label>: <expanded_col1>=<val1>, <expanded_col2>=<val2>, ...

Rules:
1. One line per data row.
2. Expand abbreviated column headers into full English — this is the only change you make.
   DO NOT add words like "has", "is", "with", "a", "the", "of", "operates".
   Examples of correct expansion:
     Rev.(B$)    → revenue billion usd
     YoY%        → year-over-year growth pct
     CAGR(3yr)   → 3-year CAGR
     Mkt.Shr(%)  → market share pct
     P/E         → price-to-earnings ratio
     TAM ($B)    → total addressable market billion usd
     SAM ($B)    → serviceable addressable market billion usd
     SOM ($B)    → serviceable obtainable market billion usd
     NRR%        → net revenue retention pct
     Churn%      → churn rate pct
     CAC/LTV     → customer acquisition cost to lifetime value ratio
     IaaS%       → IaaS revenue pct
     PaaS%       → PaaS revenue pct
     SaaS%       → SaaS revenue pct
     ARR         → annual recurring revenue
     MRR         → monthly recurring revenue
     Reg.        → region
     Adj.EBITDA  → adjusted EBITDA
     QoQ%        → quarter-over-quarter growth pct
     Entry ($/seat/mo) → entry price per seat per month
3. Copy all numeric values, units, years, and proper nouns EXACTLY as-is.
4. Skip fields where the value is N/A or empty.
5. No preamble, no numbering, no trailing commentary.

Example:
  Input headers: Company | Reg. | Rev.(B$) | YoY% | Mkt.Shr(%)
  Input row:     Amazon AWS | NA | 107.6 | 19.0 | 31.0
  Output:        Amazon AWS: region=NA, revenue billion usd=107.6, year-over-year growth pct=19.0, market share pct=31.0.

Table:
{table_text}"""


def _table_to_nl_llm(table_data: list) -> str:
    """LLM-based table → NL conversion. Falls back to rule-based on error."""
    if not table_data:
        return ""
    lines = []
    for row in table_data:
        cells = [str(c).strip() if c is not None else "" for c in row]
        lines.append(" | ".join(cells))
    table_text = "\n".join(lines)
    try:
        from agents.llm import chat
        result = chat(
            messages=[{"role": "user", "content": _LLM_TABLE_PROMPT.format(table_text=table_text)}],
            max_tokens=800,
            temperature=0.1,
        )
        return result.strip()
    except Exception:
        return _pdf_table_to_nl(table_data)


# Prompt asking LLM to paraphrase the already-expanded rule sentences into
# natural prose — one paraphrase per line, same row order, no new data.
_LLM_SUPPLEMENT_PROMPT = """\
Each line below is a structured data sentence extracted from a table.
Write one natural language paraphrase for each line so that users searching \
in conversational English can find it.

Rules:
- One output line per input line, same order
- Keep ALL numbers, units, and proper nouns exactly as-is
- Write in plain English prose without "key=value" notation
- Do not add information not present in the input
- No preamble, no numbering

Input:
{rule_sentences}"""


def _table_to_nl_hybrid(table_data: list) -> str:
    """
    Hybrid table → NL: rule sentences + LLM sentences from the same raw table,
    interleaved row by row.

      Rule line  — high keyword density, expanded abbreviations, exact numbers
      LLM line   — natural language form for conversational queries

    Both are derived independently from the original table data so neither
    is a degraded version of the other.
    Falls back to rule-only if LLM call fails.
    """
    rule_out = _pdf_table_to_nl(table_data)
    if not rule_out.strip():
        return rule_out

    rule_lines = [l for l in rule_out.splitlines() if l.strip()]

    try:
        llm_out   = _table_to_nl_llm(table_data)   # same raw table, independent prompt
        llm_lines = [l.strip() for l in llm_out.splitlines() if l.strip()]
    except Exception:
        return rule_out

    # Interleave: rule[i] then llm[i] for each row
    combined = []
    for i, rule_line in enumerate(rule_lines):
        combined.append(rule_line)
        if i < len(llm_lines):
            combined.append(llm_lines[i])

    return "\n".join(combined)


def extract_pdf_text(pdf_bytes: bytes, use_llm_tables: bool = True,
                     use_hybrid_tables: bool = False) -> str:
    """
    Extract text from PDF using PyMuPDF (layout) + pdfplumber (tables).
    Falls back to pypdf if libraries unavailable.

    use_llm_tables:    use LLM for table→NL conversion (default: True, best recall)
    use_hybrid_tables: interleave rule + LLM lines (higher doc count, marginal gain)
    """
    if use_hybrid_tables:
        table_converter = _table_to_nl_hybrid
    elif use_llm_tables:
        table_converter = _table_to_nl_llm
    else:
        table_converter = _pdf_table_to_nl
    try:
        import fitz
        import pdfplumber as _plumber
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_out = []
        with _plumber.open(io.BytesIO(pdf_bytes)) as pdf_pl:
            for idx, (fz_pg, pl_pg) in enumerate(zip(doc, pdf_pl.pages)):
                text = _process_page(fz_pg, pl_pg, table_converter=table_converter)
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
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
        return "\n\n".join(pages)
    except Exception as e:
        return f"PDF extraction error: {e}"


def extract_csv_text(csv_bytes: bytes) -> str:
    """Convert CSV table to natural language using LLM."""
    import csv
    try:
        text = csv_bytes.decode("utf-8", errors="ignore")
        reader = list(csv.reader(io.StringIO(text)))
        if not reader:
            return text
        return _table_to_nl_llm(reader)
    except Exception:
        return csv_bytes.decode("utf-8", errors="ignore")


def extract_xlsx_text(xlsx_bytes: bytes) -> str:
    """Convert each XLSX sheet to natural language using LLM."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), data_only=True)
        parts = []
        for sheet in wb.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                rows.append([str(c) if c is not None else "" for c in row])
            rows = [r for r in rows if any(c.strip() for c in r)]
            if rows:
                parts.append(f"Sheet: {sheet.title}\n{_table_to_nl_llm(rows)}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"XLSX extraction error: {e}"


def extract_json_text(json_bytes: bytes) -> str:
    """Convert JSON to readable text; treat list-of-dicts as a table."""
    import json as _json
    try:
        data = _json.loads(json_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return json_bytes.decode("utf-8", errors="ignore")

    if isinstance(data, list) and data and isinstance(data[0], dict):
        headers = list(data[0].keys())
        rows = [headers] + [[str(row.get(h, "")) for h in headers] for row in data]
        return _table_to_nl_llm(rows)

    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                headers = list(v[0].keys())
                rows = [headers] + [[str(r.get(h, "")) for h in headers] for r in v]
                lines.append(f"{k}:\n{_table_to_nl_llm(rows)}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    return str(data)


def extract_md_text(md_bytes: bytes) -> str:
    """Keep prose as-is; convert markdown tables to natural language using LLM."""
    text = md_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        if (lines[i].strip().startswith("|") and
                i + 1 < len(lines) and re.match(r"^\|[\s\-:|]+\|", lines[i + 1])):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = [
                [c.strip() for c in ln.strip("|").split("|")]
                for ln in table_lines
                if not re.match(r"^\|[\s\-:|]+\|", ln)
            ]
            if rows:
                result.append(_table_to_nl_llm(rows))
        else:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def extract_text_for_ingest(filename: str, content_bytes: bytes) -> str:
    """Route file bytes to the appropriate extractor by extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_pdf_text(content_bytes)
    if ext == ".csv":
        return extract_csv_text(content_bytes)
    if ext in (".xlsx", ".xls"):
        return extract_xlsx_text(content_bytes)
    if ext == ".json":
        return extract_json_text(content_bytes)
    if ext in (".md", ".markdown"):
        return extract_md_text(content_bytes)
    return content_bytes.decode("utf-8", errors="ignore")


def ingest_document(filename: str, content: str, source_type: str = "upload") -> dict:
    global _bm25_dirty
    col = _get_collection()
    chunks = _chunk_text(content)
    if not chunks:
        return {"error": "No content to ingest"}

    ids = []
    documents = []
    metadatas = []

    file_hash = hashlib.md5(content.encode()).hexdigest()[:8]

    for i, chunk in enumerate(chunks):
        doc_id = f"{file_hash}_{i}"
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "source_type": source_type,
        })

    col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    _bm25_dirty = True  # invalidate BM25 index

    return {
        "status": "ok",
        "filename": filename,
        "chunks": len(chunks),
        "words": len(content.split()),
    }


def query_rag(query: str, n_results: int = 5, filename_filter: Optional[str] = None) -> str:
    col = _get_collection()
    total = col.count()
    if total == 0:
        return ""

    fetch_n = min(n_results * 4, total)  # oversample; threshold will trim

    # ── Dense retrieval (BGE vectors via ChromaDB) ────────────────────────────
    try:
        where = {"filename": filename_filter} if filename_filter else None
        dense_res = col.query(
            query_texts=[query],
            n_results=fetch_n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        raw_ids   = dense_res.get("ids",       [[]])[0]
        raw_dists = dense_res.get("distances", [[]])[0]
        raw_docs  = dense_res.get("documents", [[]])[0]
        raw_metas = dense_res.get("metadatas", [[]])[0]
    except Exception as e:
        return f"RAG query error: {e}"

    # Apply similarity threshold — discard results below SIM_THRESHOLD.
    # ChromaDB returns cosine *distance* (0=identical, 2=opposite); convert to similarity.
    dense_ids: list[str] = []
    dense_docs: dict[str, tuple] = {}
    for id_, dist, doc, meta in zip(raw_ids, raw_dists, raw_docs, raw_metas):
        similarity = 1.0 - dist
        if similarity >= SIM_THRESHOLD:
            dense_ids.append(id_)
            dense_docs[id_] = (doc, meta)

    if not dense_ids:
        return ""  # no relevant content in the knowledge base for this query

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────
    # BM25 is used only to rerank the already-filtered dense candidates.
    # It must not introduce new candidates that failed the similarity threshold.
    sparse_ids: list[str] = []
    try:
        bm25, corpus = _get_bm25()
        if bm25 is not None:
            scores = bm25.get_scores(_tokenize(query))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            dense_id_set = set(dense_ids)
            for idx in ranked:
                entry = corpus[idx]
                if filename_filter and entry["meta"].get("filename") != filename_filter:
                    continue
                # Only include IDs that already passed the similarity threshold
                if entry["id"] in dense_id_set:
                    sparse_ids.append(entry["id"])
                if len(sparse_ids) >= fetch_n:
                    break
    except Exception:
        pass  # BM25 failure degrades to dense-only

    # ── RRF fusion ────────────────────────────────────────────────────────────
    if sparse_ids:
        fused_ids = _rrf(dense_ids, sparse_ids)[:n_results]
    else:
        fused_ids = dense_ids[:n_results]

    parts = []
    for id_ in fused_ids:
        if id_ not in dense_docs:
            continue
        doc, meta = dense_docs[id_]
        fname = meta.get("filename", "unknown")
        parts.append(f"[Source: {fname}]\n{doc}")

    return "\n\n---\n\n".join(parts)


def list_documents() -> list[dict]:
    col = _get_collection()
    try:
        results = col.get(include=["metadatas"])
        seen = {}
        for meta in results.get("metadatas", []):
            fname = meta.get("filename", "unknown")
            if fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "chunks": 0,
                    "source_type": meta.get("source_type", "upload"),
                }
            seen[fname]["chunks"] += 1
        return list(seen.values())
    except Exception:
        return []


def delete_document(filename: str) -> dict:
    global _bm25_dirty
    col = _get_collection()
    try:
        results = col.get(where={"filename": filename}, include=["metadatas"])
        ids = results.get("ids", [])
        if ids:
            col.delete(ids=ids)
            _bm25_dirty = True  # invalidate BM25 index
        return {"status": "ok", "deleted": len(ids)}
    except Exception as e:
        return {"error": str(e)}
