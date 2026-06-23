"""Three-way semantic search: Rule vs LLM vs Hybrid."""
import csv, io, re, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rag.manager import _table_rows_to_nl, _table_to_nl_llm, _table_to_nl_hybrid

G = "\033[32m"; Y = "\033[33m"; B = "\033[34m"; C = "\033[36m"; RESET = "\033[0m"

def load_csv(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.reader(f): rows.append(r)
    return rows

def md_tables(md):
    lines = md.splitlines(); tables = []; i = 0
    while i < len(lines):
        if lines[i].strip().startswith('|') and i+1 < len(lines) and re.match(r'^\|[\s\-:|]+\|', lines[i+1]):
            title = next((lines[j].lstrip('#').strip() for j in range(i-1,max(i-4,-1),-1)
                          if lines[j].strip() and not lines[j].strip().startswith('|')), "")
            rows = [[c.strip() for c in ln.strip('|').split('|')] for ln in lines[i:] if ln.strip().startswith('|')]
            rows = [r for r in rows if not re.match(r'^[\s\-:|]+$', ''.join(r))]
            tables.append((title, rows)); k = i+1
            while k < len(lines) and lines[k].strip().startswith('|'): k += 1
            i = k
        else: i += 1
    return tables

def search_test(rule_docs, llm_docs, hybrid_docs, queries):
    import chromadb
    from chromadb.utils import embedding_functions
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    client = chromadb.EphemeralClient()
    cols = {}
    for name, docs in [("rule", rule_docs), ("llm", llm_docs), ("hybrid", hybrid_docs)]:
        c = client.create_collection(name, embedding_function=ef, metadata={"hnsw:space": "cosine"})
        valid = [d for d in docs if d.strip()]
        c.add(ids=[f"{name}_{i}" for i in range(len(valid))], documents=valid)
        cols[name] = c

    print(f"\n{'Query':<45} {'Rule':>7} {'LLM':>7} {'Hybrid':>7}  Winner")
    print("-" * 78)
    wins = {"rule": 0, "llm": 0, "hybrid": 0, "tie": 0}
    for q in queries:
        scores = {}
        for name, col in cols.items():
            res = col.query(query_texts=[q], n_results=1, include=["distances"])
            scores[name] = round(1 - res["distances"][0][0], 4) if res["distances"][0] else 0
        best = max(scores, key=scores.get)
        others = [v for k, v in scores.items() if k != best]
        if all(abs(scores[best] - o) < 0.005 for o in others):
            winner = "tie"; wins["tie"] += 1
        else:
            colors = {"rule": Y, "llm": C, "hybrid": G}
            winner = f"{colors[best]}{best}{RESET}"; wins[best] += 1
        q_s = (q[:43] + "..") if len(q) > 45 else q
        print(f"  {q_s:<45} {scores['rule']:>7.4f} {scores['llm']:>7.4f} {scores['hybrid']:>7.4f}  {winner}")

    print(f"\n  {Y}Rule{RESET}: {wins['rule']}  {C}LLM{RESET}: {wins['llm']}  {G}Hybrid{RESET}: {wins['hybrid']}  tie: {wins['tie']}")
    return wins

def main():
    rule_docs = []; llm_docs = []; hybrid_docs = []

    # CSV
    csv_table = load_csv(ROOT / "test_data/cloud_vendors.csv")
    for docs, fn in [(rule_docs, _table_rows_to_nl), (llm_docs, _table_to_nl_llm), (hybrid_docs, _table_to_nl_hybrid)]:
        out = fn(csv_table) if fn != _table_rows_to_nl else _table_rows_to_nl(csv_table[0], csv_table[1:])
        docs.extend([l for l in out.splitlines() if l.strip()])

    # Markdown
    for title, rows in md_tables((ROOT / "test_data/saas_market_report.md").read_text()):
        for docs, fn in [(rule_docs, _table_rows_to_nl), (llm_docs, _table_to_nl_llm), (hybrid_docs, _table_to_nl_hybrid)]:
            out = fn(rows) if fn != _table_rows_to_nl else _table_rows_to_nl(rows[0], rows[1:])
            docs.extend([l for l in out.splitlines() if l.strip()])

    # PDF
    pdf = Path.home() / "gtm-team/test_data/idc_cloud_forecast_2025.pdf"
    if pdf.exists():
        import fitz, pdfplumber
        from rag.manager import _is_chart_noise, _merge_table_fragments, _pdf_table_to_nl
        pdf_bytes = pdf.read_bytes()
        tables = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as p:
            for fz, pl in zip(doc, p.pages):
                vp = [(t, t.extract()) for t in pl.find_tables() if not _is_chart_noise(t.extract())]
                tables.extend(_merge_table_fragments(vp))
        for t in tables:
            for docs, fn in [(rule_docs, _pdf_table_to_nl), (llm_docs, _table_to_nl_llm), (hybrid_docs, _table_to_nl_hybrid)]:
                docs.extend([l for l in fn(t).splitlines() if l.strip()])

    print(f"{B}Docs per collection — Rule: {len(rule_docs)}, LLM: {len(llm_docs)}, Hybrid: {len(hybrid_docs)}{RESET}")

    queries = [
        "AWS annual revenue in billion dollars 2025",
        "Azure year over year growth percentage",
        "cloud vendor market share percentage",
        "Google Cloud price-to-earnings ratio",
        "which company has highest CAGR three year",
        "Huawei Cloud financial performance",
        "IaaS PaaS SaaS revenue breakdown",
        "CRM total addressable market size",
        "SaaS net revenue retention by segment",
        "customer acquisition cost lifetime value ratio",
        "Asia Pacific growth rate forecast",
        "security SaaS net revenue retention",
        "ERP enterprise pricing per seat",
        "BI analytics year over year growth",
        "IaaS cloud spending forecast 2028",
        "AI cloud services compound annual growth rate",
        "total cloud market size 2025",
    ]
    wins = search_test(rule_docs, llm_docs, hybrid_docs, queries)

if __name__ == "__main__":
    main()
