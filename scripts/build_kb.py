#!/usr/bin/env python3
"""
Build the GTM knowledge base by scraping public market research sources.
Run once: python3 scripts/build_kb.py
"""
import sys, json, subprocess, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SKILLS = Path.home() / ".openclaw/workspace/skills"
PYTHON = sys.executable

TOPICS = [
    # (label, queries)
    ("AI & Machine Learning Market", [
        "artificial intelligence market size 2024 2025 growth forecast",
        "AI software market share companies revenue enterprise",
        "generative AI market trends investment 2024",
    ]),
    ("SaaS & Cloud Software Market", [
        "SaaS market size growth rate 2024 forecast",
        "cloud software market share Salesforce Microsoft SAP",
        "B2B SaaS metrics ARR churn NRR benchmarks 2024",
    ]),
    ("Electric Vehicle Market", [
        "electric vehicle EV market size growth 2024 2030",
        "EV charging infrastructure market players Tesla BYD",
        "electric vehicle adoption barriers regulations incentives",
    ]),
    ("Digital Health & MedTech", [
        "digital health market size 2024 telehealth AI diagnostics",
        "healthcare technology investment funding trends 2024",
        "medtech regulation FDA approval digital therapeutics",
    ]),
    ("Cybersecurity Market", [
        "cybersecurity market size growth 2024 enterprise spending",
        "cybersecurity vendors market share Palo Alto CrowdStrike",
        "zero trust security market trends threats 2024",
    ]),
    ("Fintech & Payments", [
        "fintech market size 2024 payments digital banking",
        "neobank digital payments market share growth",
        "fintech regulation open banking trends 2024",
    ]),
    ("E-commerce & Retail Tech", [
        "ecommerce market size 2024 global growth",
        "retail technology AI personalization supply chain",
        "cross-border ecommerce trends platforms 2024",
    ]),
    ("Climate Tech & Clean Energy", [
        "climate tech market size investment 2024",
        "renewable energy solar wind market growth forecast",
        "carbon market ESG investment trends 2024",
    ]),
]

def search(query, n=3):
    r = subprocess.run(
        [PYTHON, str(SKILLS / "web-search/scripts/web_search.py"), query, str(n)],
        capture_output=True, text=True, timeout=20,
        env={**os.environ, "HOME": str(Path.home())}
    )
    try:
        return json.loads(r.stdout.strip()) if r.stdout.strip() else []
    except:
        return []

def scrape(url):
    r = subprocess.run(
        [PYTHON, str(SKILLS / "web-scrape/scripts/web_scrape.py"), url],
        capture_output=True, text=True, timeout=25,
        env={**os.environ, "HOME": str(Path.home())}
    )
    try:
        d = json.loads(r.stdout.strip()) if r.stdout.strip() else {}
        return d.get("text") or d.get("raw", "")
    except:
        return ""

def build():
    from rag_mgr import ingest_document

    total_chunks = 0
    total_docs = 0

    for topic_label, queries in TOPICS:
        print(f"\n{'='*50}")
        print(f"Topic: {topic_label}")
        topic_text_parts = []

        for query in queries:
            print(f"  🔍 {query[:60]}...")
            results = search(query, n=3)
            if not results:
                print("     no results")
                continue

            items = results if isinstance(results, list) else results.get("results", [])
            for item in items[:2]:
                url = item.get("url") or item.get("href", "")
                title = item.get("title", "")
                snippet = item.get("body") or item.get("snippet", "")

                if url:
                    text = scrape(url)
                    if text and len(text) > 200:
                        topic_text_parts.append(f"## {title}\nSource: {url}\n\n{text[:3000]}")
                        print(f"     ✓ scraped {len(text)} chars — {title[:50]}")
                    elif snippet:
                        topic_text_parts.append(f"## {title}\n\n{snippet}")
                        print(f"     ~ snippet only — {title[:50]}")
                elif snippet:
                    topic_text_parts.append(f"## {title}\n\n{snippet}")

            time.sleep(0.5)

        if topic_text_parts:
            combined = f"# {topic_label}\n\n" + "\n\n---\n\n".join(topic_text_parts)
            filename = topic_label.lower().replace(" ", "_").replace("&", "and") + ".txt"
            result = ingest_document(filename, combined, source_type="scraped")
            total_chunks += result.get("chunks", 0)
            total_docs += 1
            print(f"  ✅ Ingested: {result['chunks']} chunks, {result['words']} words")
        else:
            print(f"  ⚠️  No content collected")

    print(f"\n{'='*50}")
    print(f"Done. {total_docs} topics, {total_chunks} total chunks in knowledge base.")

if __name__ == "__main__":
    build()
