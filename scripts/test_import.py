#!/usr/bin/env python3
"""End-to-end test for multi-format import + RAG namespace isolation."""
import sys, asyncio, json, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_DATA = Path(__file__).parent.parent / "test_data"
PASS, FAIL = 0, 0


def asyncio_run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def run_test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print("  \033[32mOK\033[0m " + name)
    except Exception as e:
        FAIL += 1
        print("  \033[31mFAIL\033[0m " + name + ": " + str(e)[:120])
        traceback.print_exc()


def check(r, **kwargs):
    if "error" in r:
        raise AssertionError("Import error: " + str(r.get("error")))
    for key, val in kwargs.items():
        if key == "format":
            assert r.get("format") == val, f"Expected format {val}, got {r.get('format')}"
        elif key == "text_contains":
            assert val in r.get("text", ""), f"Missing '{val}' in output"
        elif key == "text_contains_any":
            assert any(s in r.get("text", "") for s in val), f"Missing any of {val}"
        elif key == "min_chars":
            assert r.get("chars", 0) >= val, f"Too few chars: {r.get('chars')} vs {val}"
        elif key == "min_zip_files":
            assert r.get("metadata", {}).get("files_inside", 0) >= val, f"Too few ZIP files"


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Document Import Skill
# ═══════════════════════════════════════════════════════════════════════════

def test_skill_import():
    from team.skills import doc_import

    def test_csv():
        r = asyncio_run(doc_import(str(TEST_DATA / "global_saas_market_2025.csv")))
        check(r, format="csv", text_contains="SaaS_Market_Size_Billion_USD", min_chars=500)

    def test_xlsx():
        r = asyncio_run(doc_import(str(TEST_DATA / "cloud_industry_stats.xlsx")))
        check(r, format="xlsx", text_contains_any=["Competitive Landscape", "Growth Forecasts"], min_chars=100)

    def test_json():
        r = asyncio_run(doc_import(str(TEST_DATA / "cloud_industry_stats.json")))
        check(r, format="json", text_contains="IDC")

    def test_markdown():
        r = asyncio_run(doc_import(str(TEST_DATA / "gartner_cloud_market_summary_2025.md")))
        check(r, format="markdown", text_contains="Gartner")

    def test_pdf():
        r = asyncio_run(doc_import(str(TEST_DATA / "idc_cloud_forecast_2025.pdf")))
        check(r, format="pdf", min_chars=50)
        assert r.get("pages", 0) >= 1, f"Expected pages, got {r.get('pages')}"

    def test_zip():
        r = asyncio_run(doc_import(str(TEST_DATA / "industry_reports_bundle.zip")))
        check(r, format="zip", min_zip_files=2)

    def test_unsupported():
        r = asyncio_run(doc_import(str(TEST_DATA / "nonexistent.fake")))
        assert "error" in r, "Should reject unsupported format"

    run_test("CSV import", test_csv)
    run_test("XLSX import", test_xlsx)
    run_test("JSON import", test_json)
    run_test("Markdown import", test_markdown)
    run_test("PDF import", test_pdf)
    run_test("ZIP import", test_zip)
    run_test("Unsupported format", test_unsupported)


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: RAG Namespace Ingestion
# ═══════════════════════════════════════════════════════════════════════════

def test_rag_ingestion():
    from rag_mgr import ingest_document, ingest_batch, list_documents, list_namespaces
    from rag_mgr import delete_document

    # Clean up test docs from previous runs
    for fn in ["test_user_doc.txt", "test_platform_doc.txt", "test_batch_1.txt", "test_batch_2.txt"]:
        try:
            delete_document(fn)
        except Exception:
            pass

    def test_single_user():
        r = ingest_document("test_user_doc.txt",
            "SaaS CRM market reached $73.1 billion in 2025. Salesforce holds 22% market share.",
            namespace="user")
        assert r["status"] == "ok", f"Ingestion failed: {r}"
        assert r["namespace"] == "user"
        assert r["chunks"] >= 1

    def test_single_platform():
        r = ingest_document("test_platform_doc.txt",
            "Global cloud market forecast: $723B in 2025 per Gartner. IaaS growth at 23.7%.",
            namespace="platform")
        assert r["status"] == "ok"
        assert r["namespace"] == "platform"

    def test_batch():
        results = ingest_batch([
            {"filename": "test_batch_1.txt", "content": "AI cloud services: $35B in 2025, 38% CAGR to 2028. Source: IDC 2025.", "source_type": "test"},
            {"filename": "test_batch_2.txt", "content": "Enterprise SaaS: 85% of Fortune 500 use 3+ cloud providers. Source: Flexera 2025.", "source_type": "test"},
        ], namespace="user")
        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)

    def test_invalid_ns():
        r = ingest_document("bad.txt", "test", namespace="invalid")
        assert "error" in r, "Should reject invalid namespace"

    def test_list_filtered():
        user_docs = list_documents(namespace="user")
        user_names = [d["filename"] for d in user_docs]
        assert "test_user_doc.txt" in user_names, f"User doc missing: {user_names}"
        all_docs = list_documents()
        assert len(all_docs) >= len(user_docs)

    def test_list_namespaces():
        ns_list = list_namespaces()
        ns_names = [n["namespace"] for n in ns_list]
        assert "user" in ns_names
        assert "platform" in ns_names

    run_test("Single doc namespace=user", test_single_user)
    run_test("Single doc namespace=platform", test_single_platform)
    run_test("Batch ingestion", test_batch)
    run_test("Invalid namespace rejection", test_invalid_ns)
    run_test("list_documents namespace filter", test_list_filtered)
    run_test("list_namespaces", test_list_namespaces)


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: RAG Namespace-Isolated Retrieval
# ═══════════════════════════════════════════════════════════════════════════

def test_rag_retrieval():
    from rag_mgr import query_rag, query_rag_multi

    def test_user_filter():
        r = query_rag("AI cloud services market 2025", n_results=5, namespace_filter="user")
        assert r, "No results for user query"
        assert any(w in r for w in ["35B", "38%", "test_batch", "test_user"]), \
            f"Missed user data: {r[:200]}"

    def test_platform_filter():
        r = query_rag("cloud market Gartner forecast", n_results=5, namespace_filter="platform")
        assert r, "No results for platform query"
        assert any(w in r.lower() for w in ["gartner", "723"]), \
            f"Missed platform data: {r[:200]}"

    def test_no_filter():
        r = query_rag("SaaS market cloud 2025", n_results=10)
        assert r, "No results at all"

    def test_multi():
        r = query_rag_multi(["cloud market size", "AI services growth"], n_per_query=3, namespace_filter="user")
        assert r, "query_rag_multi returned nothing"

    run_test("query_rag namespace_filter=user", test_user_filter)
    run_test("query_rag namespace_filter=platform", test_platform_filter)
    run_test("query_rag no filter", test_no_filter)
    run_test("query_rag_multi with filter", test_multi)


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def test_full_pipeline():
    from rag_mgr import extract_pdf_text, ingest_document, query_rag

    def test_csv_pipeline():
        csv_text = (TEST_DATA / "global_saas_market_2025.csv").read_text()
        r = ingest_document("pipeline_csv.txt", csv_text, source_type="csv", namespace="user")
        assert r["status"] == "ok" and r["chunks"] >= 1, f"CSV ingest: {r}"
        ret = query_rag("SaaS CRM market size 2025", n_results=3, namespace_filter="user")
        assert ret, "Could not retrieve CSV data"
        assert "73.1" in ret or "CRM" in ret, f"Wrong content: {ret[:200]}"

    def test_pdf_pipeline():
        pdf_bytes = (TEST_DATA / "idc_cloud_forecast_2025.pdf").read_bytes()
        text = extract_pdf_text(pdf_bytes)
        assert "IDC" in text or "372" in text or "Cloud" in text, f"PDF extraction: {text[:200]}"
        r = ingest_document("pipeline_pdf.txt", text, source_type="pdf", namespace="user")
        assert r["status"] == "ok", f"PDF ingest: {r}"

    def test_bulk_import():
        from team.skills import doc_import
        total = 0
        files = [f for f in TEST_DATA.glob("*") if f.suffix.lower() in (".csv", ".xlsx", ".json", ".md", ".pdf")][:5]
        for fp in files:
            r = asyncio_run(doc_import(str(fp)))
            if "error" in r:
                print(f"      Skip {fp.name}: {r['error']}")
                continue
            result = ingest_document(fp.name, r["text"], source_type=r.get("format", "import"), namespace="user")
            if result.get("status") == "ok":
                total += result.get("chunks", 0)
        assert total > 0, f"No chunks from {len(files)} files"
        print(f"      {total} chunks from {len(files)} files")

    run_test("CSV pipeline", test_csv_pipeline)
    run_test("PDF pipeline", test_pdf_pipeline)
    run_test("Bulk import all formats", test_bulk_import)


# ═══════════════════════════════════════════════════════════════════════════

def main():
    global PASS, FAIL
    print("=" * 60)
    print("GTM Multi-Format Import — Test Suite")
    print("=" * 60)

    print("\n[1] Document Import Skill (format parsing)")
    test_skill_import()

    print("\n[2] RAG Namespace Ingestion")
    test_rag_ingestion()

    print("\n[3] RAG Namespace-Isolated Retrieval")
    test_rag_retrieval()

    print("\n[4] Full Pipeline (parse → ingest → retrieve)")
    test_full_pipeline()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
