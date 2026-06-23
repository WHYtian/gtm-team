"""
Comprehensive table conversion test:
  1. Parse CSV / Markdown / PDF into raw tables
  2. Rule-based vs LLM conversion — diff output
  3. Data fidelity: every number in source must appear in output
  4. Semantic search: embed both into temp ChromaDB collections, compare recall
"""
import io, re, sys, csv, math
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── colour helpers ──────────────────────────────────────────────────────────────
G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; B = "\033[34m"; RESET = "\033[0m"
def ok(s):   return f"{G}✓ {s}{RESET}"
def bad(s):  return f"{R}✗ {s}{RESET}"
def warn(s): return f"{Y}⚠ {s}{RESET}"
def hdr(s):  return f"\n{B}{'='*68}\n  {s}\n{'='*68}{RESET}"

# ── extract numbers from text ──────────────────────────────────────────────────
def extract_numbers(text: str) -> set[str]:
    """Pull every numeric token (integers, decimals, negatives) from text."""
    return set(re.findall(r'-?\d+\.?\d*', text.replace(',', '')))

def fidelity(source_table_text: str, nl_output: str, label: str):
    nums_src = extract_numbers(source_table_text)
    # ignore 1-digit numbers — too noisy (row indices, etc.)
    nums_src = {n for n in nums_src if len(n.replace('-','').replace('.','')) > 1}
    nums_out = extract_numbers(nl_output)
    missing  = nums_src - nums_out
    coverage = (len(nums_src - missing) / len(nums_src) * 100) if nums_src else 100.0
    status   = ok(f"{label}: {coverage:.0f}% numeric coverage") if not missing \
               else warn(f"{label}: {coverage:.0f}% coverage — missing: {sorted(missing)[:8]}")
    print(status)
    return coverage, missing

# ── CSV parsing ────────────────────────────────────────────────────────────────
def csv_to_table(path: Path) -> list[list]:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            rows.append(row)
    return rows

# ── Markdown table parsing ─────────────────────────────────────────────────────
def md_to_tables(md_text: str) -> list[tuple[str, list[list]]]:
    """Return list of (title_hint, table_rows) for each markdown table found."""
    lines = md_text.splitlines()
    tables = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s\-:|]+\|', lines[i+1]):
            # found table header + separator
            title = ""
            for j in range(i-1, max(i-4, -1), -1):
                s = lines[j].strip()
                if s and not s.startswith('|'):
                    title = s.lstrip('#').strip()
                    break
            headers = [c.strip() for c in lines[i].strip('|').split('|')]
            rows = [headers]
            k = i + 2
            while k < len(lines) and lines[k].strip().startswith('|'):
                cells = [c.strip() for c in lines[k].strip('|').split('|')]
                rows.append(cells)
                k += 1
            tables.append((title, rows))
            i = k
        else:
            i += 1
    return tables

# ── LLM and rule-based converters ─────────────────────────────────────────────
from rag.manager import _table_rows_to_nl, _table_to_nl_llm

def rule_convert(table_rows: list[list]) -> str:
    if not table_rows:
        return ""
    return _table_rows_to_nl(table_rows[0], table_rows[1:])

def llm_convert(table_rows: list[list]) -> str:
    return _table_to_nl_llm(table_rows)

# ── Semantic search test ───────────────────────────────────────────────────────
def run_search_test(rule_docs: list[str], llm_docs: list[str], queries: list[str]):
    """
    Embed both sets into temporary in-memory ChromaDB collections.
    Run queries, compare top-1 score and which document is retrieved.
    """
    import chromadb
    from chromadb.utils import embedding_functions

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )
    client = chromadb.EphemeralClient()
    col_rule = client.create_collection("rule_test", embedding_function=ef,
                                         metadata={"hnsw:space": "cosine"})
    col_llm  = client.create_collection("llm_test",  embedding_function=ef,
                                         metadata={"hnsw:space": "cosine"})

    for i, doc in enumerate(rule_docs):
        col_rule.add(ids=[f"r{i}"], documents=[doc])
    for i, doc in enumerate(llm_docs):
        col_llm.add(ids=[f"l{i}"], documents=[doc])

    print(f"\n{'Query':<42} {'Rule score':>11} {'LLM score':>11}  {'Winner'}")
    print("-" * 72)
    rule_wins = llm_wins = ties = 0
    for q in queries:
        r_res = col_rule.query(query_texts=[q], n_results=1,
                               include=["distances"])
        l_res = col_llm.query(query_texts=[q],  n_results=1,
                               include=["distances"])
        r_sim = round(1 - r_res["distances"][0][0], 4) if r_res["distances"][0] else 0
        l_sim = round(1 - l_res["distances"][0][0], 4) if l_res["distances"][0] else 0
        diff  = l_sim - r_sim
        if abs(diff) < 0.005:
            winner = "tie"
            ties += 1
        elif diff > 0:
            winner = f"{G}LLM +{diff:.4f}{RESET}"
            llm_wins += 1
        else:
            winner = f"{Y}Rule +{-diff:.4f}{RESET}"
            rule_wins += 1
        q_short = q[:40] + ".." if len(q) > 40 else q
        print(f"  {q_short:<42} {r_sim:>11.4f} {l_sim:>11.4f}  {winner}")

    print(f"\n  Summary → Rule wins: {rule_wins}, LLM wins: {llm_wins}, Ties: {ties}")
    return rule_wins, llm_wins, ties

# ── Side-by-side diff ─────────────────────────────────────────────────────────
def show_diff(rule_out: str, llm_out: str):
    rule_lines = rule_out.strip().splitlines()
    llm_lines  = llm_out.strip().splitlines()
    max_len    = max(len(rule_lines), len(llm_lines))
    changed    = 0
    for i in range(max_len):
        r = rule_lines[i] if i < len(rule_lines) else "<missing>"
        l = llm_lines[i]  if i < len(llm_lines)  else "<missing>"
        if r != l:
            print(f"  R: {Y}{r}{RESET}")
            print(f"  L: {G}{l}{RESET}")
            changed += 1
        # else: identical, skip
    if changed == 0:
        print(f"  {ok('Outputs are identical')}")
    else:
        print(f"  ({changed}/{max_len} rows differ)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    all_rule_docs: list[str] = []
    all_llm_docs:  list[str] = []

    # ── 1. CSV ─────────────────────────────────────────────────────────────────
    print(hdr("1. CSV  —  cloud_vendors.csv"))
    csv_path = ROOT / "test_data/cloud_vendors.csv"
    csv_table = csv_to_table(csv_path)
    raw_text  = csv_path.read_text()

    print(f"  Rows: {len(csv_table)-1} data rows, {len(csv_table[0])} columns")
    print(f"  Headers: {csv_table[0]}\n")

    rule_csv = rule_convert(csv_table)
    llm_csv  = llm_convert(csv_table)

    print("--- RULE-BASED ---")
    print(rule_csv)
    print("\n--- LLM-BASED ---")
    print(llm_csv)

    print("\n--- DIFF (only changed rows) ---")
    show_diff(rule_csv, llm_csv)

    print("\n--- DATA FIDELITY ---")
    fidelity(raw_text, rule_csv, "Rule")
    fidelity(raw_text, llm_csv,  "LLM ")

    # split into per-row docs for search
    all_rule_docs.extend(rule_csv.splitlines())
    all_llm_docs.extend(llm_csv.splitlines())

    # ── 2. Markdown ────────────────────────────────────────────────────────────
    print(hdr("2. Markdown  —  saas_market_report.md"))
    md_path  = ROOT / "test_data/saas_market_report.md"
    md_text  = md_path.read_text()
    md_tables = md_to_tables(md_text)
    print(f"  Found {len(md_tables)} table(s)\n")

    for title, rows in md_tables:
        print(f"  >> {title}  ({len(rows)-1} rows × {len(rows[0])} cols)")
        rule_md = rule_convert(rows)
        llm_md  = llm_convert(rows)

        print("  --- RULE-BASED ---")
        print(rule_md)
        print("  --- LLM-BASED ---")
        print(llm_md)
        print("  --- DIFF ---")
        show_diff(rule_md, llm_md)
        print("  --- DATA FIDELITY ---")
        tbl_raw = "\n".join("|".join(r) for r in rows)
        fidelity(tbl_raw, rule_md, "Rule")
        fidelity(tbl_raw, llm_md,  "LLM ")
        print()

        all_rule_docs.extend(rule_md.splitlines())
        all_llm_docs.extend(llm_md.splitlines())

    # ── 3. PDF ─────────────────────────────────────────────────────────────────
    print(hdr("3. PDF  —  idc_cloud_forecast_2025.pdf"))
    pdf_path = Path.home() / "gtm-team/test_data/idc_cloud_forecast_2025.pdf"
    if pdf_path.exists():
        from rag.manager import extract_pdf_text, _is_chart_noise, _merge_table_fragments
        import fitz, pdfplumber

        pdf_bytes = pdf_path.read_bytes()
        all_pdf_tables = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf_pl:
            for fz_pg, pl_pg in zip(doc, pdf_pl.pages):
                raw_tables = pl_pg.find_tables()
                valid_pairs = [(t, t.extract()) for t in raw_tables
                               if not _is_chart_noise(t.extract())]
                all_pdf_tables.extend(_merge_table_fragments(valid_pairs))

        print(f"  Found {len(all_pdf_tables)} table(s)\n")
        for i, tbl in enumerate(all_pdf_tables):
            rows = [r for r in tbl if any(c is not None and str(c).strip() for c in r)]
            print(f"  >> Table {i+1}  ({len(rows)} rows)")
            rule_pdf = _table_rows_to_nl(rows[0] if rows else [], rows[1:] if len(rows)>1 else [])
            llm_pdf  = _table_to_nl_llm(tbl)
            print("  --- RULE-BASED ---"); print(rule_pdf)
            print("  --- LLM-BASED ---"); print(llm_pdf)
            print("  --- DIFF ---"); show_diff(rule_pdf, llm_pdf)
            tbl_raw = "\n".join("|".join(str(c or "") for c in r) for r in tbl)
            print("  --- DATA FIDELITY ---")
            fidelity(tbl_raw, rule_pdf, "Rule")
            fidelity(tbl_raw, llm_pdf,  "LLM ")
            print()
            all_rule_docs.extend(rule_pdf.splitlines())
            all_llm_docs.extend(llm_pdf.splitlines())
    else:
        print(f"  {warn('PDF not found, skipping')}")

    # ── 4. Semantic search comparison ──────────────────────────────────────────
    print(hdr("4. Semantic Search  —  same queries against both collections"))

    queries = [
        # CSV queries (test abbreviation expansion)
        "AWS annual revenue in billion dollars 2025",
        "Azure year over year growth percentage",
        "cloud vendor market share percentage",
        "Google Cloud price-to-earnings ratio",
        "which company has highest CAGR three year",
        "Huawei Cloud financial performance",
        "IaaS PaaS SaaS revenue breakdown",
        # Markdown queries
        "CRM total addressable market size",
        "SaaS net revenue retention by segment",
        "customer acquisition cost lifetime value ratio",
        "Asia Pacific growth rate forecast",
        "security SaaS net revenue retention",
        "ERP enterprise pricing per seat",
        "BI analytics year over year growth",
        # PDF queries
        "IaaS cloud spending forecast 2028",
        "AI cloud services compound annual growth rate",
        "total cloud market size 2025",
    ]

    rule_wins, llm_wins, ties = run_search_test(
        [d for d in all_rule_docs if d.strip()],
        [d for d in all_llm_docs  if d.strip()],
        queries,
    )

    # ── 5. Final verdict ───────────────────────────────────────────────────────
    print(hdr("5. Verdict"))
    total = rule_wins + llm_wins + ties
    print(f"  Rule-based wins : {rule_wins}/{total}")
    print(f"  LLM wins        : {llm_wins}/{total}")
    print(f"  Ties            : {ties}/{total}")
    if llm_wins > rule_wins + 2:
        print(f"\n  {ok('LLM conversion meaningfully improves semantic search recall.')}")
    elif rule_wins > llm_wins + 2:
        print(f"\n  {warn('Rule-based is better — LLM adds noise or misses structure.')}")
    else:
        print(f"\n  {warn('Results are comparable. LLM overhead may not be justified for clean tables.')}")


if __name__ == "__main__":
    main()
