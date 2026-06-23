"""
Compare rule-based vs LLM table→NL conversion on a real PDF.
Usage: python test_table_conversion.py
"""
import io, sys
from pathlib import Path

PDF_PATH = Path.home() / "gtm-team/test_data/idc_cloud_forecast_2025.pdf"

def extract_tables_from_pdf(pdf_bytes: bytes) -> list[list[list]]:
    """Return all tables (as 2D lists) extracted from the PDF."""
    import fitz
    import pdfplumber as _plumber
    sys.path.insert(0, str(Path(__file__).parent))
    from rag.manager import _is_chart_noise, _merge_table_fragments

    all_tables = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    with _plumber.open(io.BytesIO(pdf_bytes)) as pdf_pl:
        for fz_pg, pl_pg in zip(doc, pdf_pl.pages):
            raw_tables = pl_pg.find_tables()
            valid_pairs = []
            for t in raw_tables:
                data = t.extract()
                if not _is_chart_noise(data):
                    valid_pairs.append((t, data))
            merged = _merge_table_fragments(valid_pairs)
            all_tables.extend(merged)
    return all_tables


def run():
    sys.path.insert(0, str(Path(__file__).parent))
    from rag.manager import _pdf_table_to_nl, _table_to_nl_llm

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        sys.exit(1)

    pdf_bytes = PDF_PATH.read_bytes()
    tables = extract_tables_from_pdf(pdf_bytes)
    print(f"Found {len(tables)} table(s) in PDF.\n")

    # Test the first 3 non-trivial tables
    tested = 0
    for i, table in enumerate(tables):
        rows = [r for r in table if any(c is not None and str(c).strip() for c in r)]
        if len(rows) < 2:
            continue

        print("=" * 70)
        print(f"TABLE {i+1}  ({len(rows)} rows × {max(len(r) for r in rows)} cols)")
        print("-" * 70)
        print("RAW:")
        for row in rows[:5]:
            print("  " + " | ".join(str(c or "").strip()[:20] for c in row))
        if len(rows) > 5:
            print(f"  ... ({len(rows)-5} more rows)")

        print("\n--- RULE-BASED ---")
        rule_out = _pdf_table_to_nl(table)
        print(rule_out[:800] + ("..." if len(rule_out) > 800 else ""))

        print("\n--- LLM-BASED ---")
        llm_out = _table_to_nl_llm(table)
        print(llm_out[:800] + ("..." if len(llm_out) > 800 else ""))

        print()
        tested += 1
        if tested >= 3:
            break

    if tested == 0:
        print("No non-trivial tables found in this PDF.")

    # Also test the full extract_pdf_text comparison (first 500 chars of each)
    print("\n" + "=" * 70)
    print("FULL PDF EXTRACTION COMPARISON (first 600 chars)")
    print("=" * 70)
    from rag.manager import extract_pdf_text

    print("\n--- RULE-BASED extract_pdf_text ---")
    rule_full = extract_pdf_text(pdf_bytes, use_llm_tables=False)
    print(rule_full[:600])

    print("\n--- LLM extract_pdf_text ---")
    llm_full = extract_pdf_text(pdf_bytes, use_llm_tables=True)
    print(llm_full[:600])


if __name__ == "__main__":
    run()
