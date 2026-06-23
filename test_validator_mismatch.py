"""
Test suite: validator (Jordan) data reconciliation — geographic and scope mismatch detection.

Tests whether the LLM-based synthesizer correctly identifies cases where web search findings
and RAG chunks cannot be directly compared because they cover different geographies, regions,
currencies, or market definitions.

Usage:
    python test_validator_mismatch.py [--fix]    # --fix tests the improved prompt

Structure:
    Each test case is a dict with:
        id          — short identifier
        category    — REGION / SCOPE / CURRENCY / TIME / LEGITIMATE
        description — what mismatch (or match) is being tested
        web_finding — simulated web search finding
        rag_chunk   — simulated RAG chunk
        should_flag — True if LLM should refuse/warn to directly compare them
        keywords    — strings that MUST appear in LLM output for a correct response
"""

import sys
import os
import re
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.llm import chat, MODEL_V3
from agents.prompts import DATA_SYNTHESIZER_SYSTEM

# ── Improved prompt that adds explicit REGION MISMATCH handling ───────────────

DATA_SYNTHESIZER_SYSTEM_FIXED = DATA_SYNTHESIZER_SYSTEM.replace(
    "  3. Scope check: if one figure covers a broader category, flag SCOPE MISMATCH and use the narrower figure.",
    """\
  3. Scope check: if one figure covers a broader category, flag SCOPE MISMATCH and use the narrower figure.
  4. Region check: if one figure is for a specific country/region (e.g. "US market", "China market",
     "APAC") and the other is for a different geography, flag REGION MISMATCH.
     Do NOT average or compare cross-regional figures — they measure different markets.
     Mark as [DO NOT COMPARE — different geographies] and report each separately.\
""",
).replace(
    "### ➕ SCOPE MISMATCHES",
    """\
### 🌍 REGION MISMATCHES
- **[metric]**: Web=[geography A] vs RAG=[geography B] → [DO NOT COMPARE — different geographies]. \\
  Report separately: Web=[value for geo A], RAG=[value for geo B].

### ➕ SCOPE MISMATCHES\
""",
)

# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── REGION mismatches ────────────────────────────────────────────────────
    {
        "id": "R1",
        "category": "REGION",
        "description": "US market vs China market — completely different geographies",
        "web_finding": (
            "[Data] US SaaS market size 2025: $280B — Gartner Cloud Software Report 2025 "
            "(https://gartner.com/cloud-2025). North America dominates with 44% of global spend."
        ),
        "rag_chunk": (
            "[Source: china_saas_market_2025.pdf]\n"
            "中国SaaS市场2025年规模约为人民币1,850亿元（约合260亿美元），"
            "同比增长22%。本土厂商占据78%市场份额。"
            "China SaaS market 2025: ~¥185B CNY (~$26B USD), YoY growth 22%."
        ),
        "should_flag": True,
        "keywords": ["region", "geo", "different", "US", "China", "DO NOT COMPARE", "mismatch"],
    },
    {
        "id": "R2",
        "category": "REGION",
        "description": "Global cloud market vs APAC cloud market",
        "web_finding": (
            "[Data] Global public cloud services revenue 2025: $591B — IDC Worldwide Cloud "
            "Spending Forecast 2025 (https://idc.com/cloud-forecast-2025)."
        ),
        "rag_chunk": (
            "[Source: apac_cloud_2024.md]\n"
            "APAC public cloud market reached $132B in 2024, growing at 28% YoY. "
            "Asia-Pacific excludes North America and EMEA."
        ),
        "should_flag": True,
        "keywords": ["scope", "region", "global", "APAC", "mismatch", "different"],
    },
    {
        "id": "R3",
        "category": "REGION",
        "description": "European fintech vs US fintech — same industry, different continents",
        "web_finding": (
            "[Data] US fintech market 2025: $312B total transaction value — CB Insights Fintech "
            "State of the Industry 2025 (https://cbinsights.com/fintech-2025)."
        ),
        "rag_chunk": (
            "[Source: eu_fintech_report.pdf]\n"
            "European fintech investment reached €18.4B in 2024. UK accounts for 35% of "
            "total European fintech deal volume. GDPR compliance is a key differentiator."
        ),
        "should_flag": True,
        "keywords": ["region", "US", "Europe", "different", "mismatch"],
    },
    {
        "id": "R4",
        "category": "REGION",
        "description": "Global EV sales vs China EV sales — China is a subset of global",
        "web_finding": (
            "[Data] Global EV sales 2025: 17.8 million units — IEA Global EV Outlook 2025 "
            "(https://iea.org/ev-outlook-2025). Represents 22% of all new car sales globally."
        ),
        "rag_chunk": (
            "[Source: china_ev_market.txt]\n"
            "China EV sales 2025: 11.2 million units, accounting for 63% of global EV volume. "
            "NEV penetration rate reached 45% in China."
        ),
        "should_flag": False,  # China IS a subset of global — these are additive, not conflicting
        "keywords": ["supplement", "subset", "China", "global", "63%"],
    },

    # ── CURRENCY mismatches ──────────────────────────────────────────────────
    {
        "id": "C1",
        "category": "CURRENCY",
        "description": "USD market size vs CNY market size — same market but different currencies without conversion",
        "web_finding": (
            "[Data] China cloud infrastructure market 2025: $35B USD — "
            "Canalys China Cloud Market Analysis Q4 2025 (https://canalys.com/china-cloud-2025)."
        ),
        "rag_chunk": (
            "[Source: alibaba_cloud_report.pdf]\n"
            "阿里云2025财年收入：人民币1,050亿元。中国云基础设施市场规模约2,500亿元人民币。"
            "Alibaba Cloud FY2025 revenue: ¥105B CNY. China cloud IaaS market: ~¥250B CNY."
        ),
        "should_flag": True,
        "keywords": ["currency", "CNY", "USD", "convert", "mismatch", "RMB"],
    },

    # ── TIME mismatches ──────────────────────────────────────────────────────
    {
        "id": "T1",
        "category": "TIME",
        "description": "2025 web data vs 2020 RAG data — 5-year gap in fast-moving market",
        "web_finding": (
            "[Data] Global AI market 2025: $391B — Grand View Research AI Market Report 2025 "
            "(https://grandviewresearch.com/ai-2025). CAGR 36.6% from 2022-2030."
        ),
        "rag_chunk": (
            "[Source: ai_market_2020_report.pdf]\n"
            "Global artificial intelligence market size was $27.23B in 2019, projected to "
            "reach $266.92B by 2027 at CAGR 33.2%. Report published Q3 2020."
        ),
        "should_flag": True,
        "keywords": ["outdated", "stale", "2020", "2025", "time", "recency", "year"],
    },
    {
        "id": "T2",
        "category": "TIME",
        "description": "2025 web data vs 2024 RAG data — 1-year gap, acceptable",
        "web_finding": (
            "[Data] Global cybersecurity market 2025: $217.9B — MarketsandMarkets Cybersecurity "
            "Report 2025 (https://marketsandmarkets.com/cyber-2025)."
        ),
        "rag_chunk": (
            "[Source: cybersecurity_market.txt]\n"
            "Global cybersecurity market reached $202.7B in 2024, growing at 12.3% YoY. "
            "Enterprise segment accounts for 67% of total spend."
        ),
        "should_flag": False,  # 1-year gap is fine, should confirm with recency note
        "keywords": ["confirm", "2025", "2024", "recent", "web"],
    },

    # ── SCOPE mismatches (existing functionality) ────────────────────────────
    {
        "id": "S1",
        "category": "SCOPE",
        "description": "Entire cloud market vs SaaS-only sub-segment — broader category error",
        "web_finding": (
            "[Data] Global cloud computing market 2025: $912B — IDC Cloud Computing Market 2025 "
            "(https://idc.com/cloud-2025). Includes IaaS, PaaS, SaaS, and managed services."
        ),
        "rag_chunk": (
            "[Source: saas_and_cloud_software_market.txt]\n"
            "Global SaaS market 2025: $317B, growing at 18% YoY. Pure-play software "
            "subscriptions only; excludes IaaS infrastructure and PaaS platforms."
        ),
        "should_flag": True,
        "keywords": ["scope", "SaaS", "cloud", "broader", "sub-segment", "mismatch", "IaaS"],
    },
    {
        "id": "S2",
        "category": "SCOPE",
        "description": "Total healthcare IT vs digital health sub-segment",
        "web_finding": (
            "[Data] Global healthcare IT market 2025: $660B — Mordor Intelligence Healthcare IT "
            "Report 2025 (https://mordorintelligence.com/healthcare-it-2025). "
            "Includes EHR, medical devices, hospital IT infrastructure."
        ),
        "rag_chunk": (
            "[Source: digital_health_and_medtech.txt]\n"
            "USA digital health market 2025: $166B. Covers telemedicine, health apps, "
            "wearables, and AI diagnostics — excludes traditional hospital IT infrastructure."
        ),
        "should_flag": True,
        "keywords": ["scope", "scope mismatch", "broader", "sub-segment", "excludes"],
    },

    # ── LEGITIMATE matches (should confirm, not flag) ────────────────────────
    {
        "id": "L1",
        "category": "LEGITIMATE",
        "description": "Both describe global SaaS market 2025 — should confirm",
        "web_finding": (
            "[Data] Global SaaS market 2025: $317B — Statista Cloud Software Market 2025 "
            "(https://statista.com/cloud-software-2025). CAGR 18% projected to 2030."
        ),
        "rag_chunk": (
            "[Source: saas_and_cloud_software_market.txt]\n"
            "Global SaaS market grew to $315B in 2025, with enterprise SaaS representing "
            "65% of total. Top vendors: Salesforce, Microsoft, SAP."
        ),
        "should_flag": False,
        "keywords": ["confirm", "agree", "consistent", "315", "317"],
    },
    {
        "id": "L2",
        "category": "LEGITIMATE",
        "description": "Both describe global AI market — slight value difference, should confirm",
        "web_finding": (
            "[Data] Global AI market size 2025: $391B — Grand View Research "
            "(https://grandviewresearch.com/ai-2025). Includes software, hardware, services."
        ),
        "rag_chunk": (
            "[Source: ai_and_machine_learning_market.txt]\n"
            "AI market worldwide projected at $407B for 2025, with generative AI accounting "
            "for $36B. CAGR of 36% through 2030."
        ),
        "should_flag": False,
        "keywords": ["confirm", "consistent", "391", "407", "within"],
    },
]


# ── Test runner ───────────────────────────────────────────────────────────────

def _build_synth_input(web_finding: str, rag_chunk: str) -> str:
    """Build the same input format the validator_node sends to Jordan."""
    return (
        f"Research topic: SaaS and cloud software market 2025\n\n"
        f"OVERLAP PAIRS (embedding similarity ≥ 0.62):\n"
        f"\n[OVERLAP — sim=0.71]\n"
        f"  Web finding : {web_finding}\n"
        f"  RAG chunk   : {rag_chunk}\n\n"
        f"RAG SUPPLEMENTS (imported data not covered by web search):\n"
        f"(none)"
    )


def _check_response(response: str, case: dict) -> dict:
    """
    Evaluate the LLM response against expected behavior.
    Returns a result dict with pass/fail and details.
    """
    resp_lower = response.lower()

    if case["should_flag"]:
        # LLM should warn, refuse to compare, or flag mismatch
        flag_signals = [
            "region mismatch", "scope mismatch", "do not compare",
            "different geograph", "different countr", "different region",
            "cannot be directly compared", "should not be compared",
            "outdated", "stale data", "currency mismatch",
            "⚠️", "region", "mismatch", "conflict",
        ]
        flagged = any(s.lower() in resp_lower for s in flag_signals)
        # Also check that it did NOT just mark as CONFIRMED without caveats
        blindly_confirmed = (
            "✅ confirmed" in resp_lower
            and not any(w in resp_lower for w in ["caveat", "note", "however", "but", "region", "scope"])
        )
        passed = flagged and not blindly_confirmed
        detail = "correctly flagged mismatch" if passed else (
            "blindly confirmed despite mismatch" if blindly_confirmed
            else "did not flag mismatch"
        )
    else:
        # LLM should confirm or supplement without unnecessary warnings
        confirmed = any(w in resp_lower for w in ["confirm", "consistent", "agree", "supplement"])
        passed = confirmed
        detail = "correctly confirmed/supplemented" if passed else "failed to confirm valid match"

    # Check expected keywords
    found_kw = [kw for kw in case["keywords"] if kw.lower() in resp_lower]
    kw_coverage = len(found_kw) / len(case["keywords"]) if case["keywords"] else 1.0

    return {
        "passed": passed,
        "detail": detail,
        "kw_coverage": kw_coverage,
        "found_keywords": found_kw,
        "response_snippet": response[:300].replace("\n", " "),
    }


def run_tests(use_fixed_prompt: bool = False) -> None:
    prompt_label = "FIXED prompt" if use_fixed_prompt else "ORIGINAL prompt"
    system_prompt = DATA_SYNTHESIZER_SYSTEM_FIXED if use_fixed_prompt else DATA_SYNTHESIZER_SYSTEM

    print(f"\n{'='*70}")
    print(f"  Validator Mismatch Test Suite — {prompt_label}")
    print(f"  {len(TEST_CASES)} test cases")
    print(f"{'='*70}\n")

    results = []
    for case in TEST_CASES:
        synth_input = _build_synth_input(case["web_finding"], case["rag_chunk"])
        try:
            response = chat(
                [{"role": "user", "content": synth_input}],
                system=system_prompt,
                model=MODEL_V3,
                max_tokens=600,
                temperature=0.1,
            )
        except Exception as e:
            response = f"[LLM ERROR: {e}]"

        result = _check_response(response, case)
        result["case"] = case
        result["response"] = response
        results.append(result)

        icon = "PASS" if result["passed"] else "FAIL"
        expect = "FLAG" if case["should_flag"] else "CONFIRM"
        print(f"[{icon}] [{case['id']}] [{case['category']:10s}] (expect {expect}) {case['description']}")
        print(f"       {result['detail']} | keywords hit: {len(result['found_keywords'])}/{len(case['keywords'])}: {result['found_keywords']}")
        print(f"       Response: \"{result['response_snippet']}...\"")
        print()

    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_cat: dict = {}
    for r in results:
        cat = r["case"]["category"]
        by_cat.setdefault(cat, {"pass": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r["passed"]:
            by_cat[cat]["pass"] += 1

    print(f"{'─'*70}")
    print(f"Overall: {passed}/{total} PASSED")
    for cat, s in sorted(by_cat.items()):
        print(f"  {cat:12s}: {s['pass']}/{s['total']}")

    # Show failures in detail
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{'─'*70}")
        print("FAILURES — full LLM response:\n")
        for r in failures:
            c = r["case"]
            print(f"[{c['id']}] {c['description']}")
            print(f"  Expected: {'FLAG mismatch' if c['should_flag'] else 'CONFIRM match'}")
            print(f"  LLM output:\n{r['response'][:600]}")
            print()

    return passed, total


if __name__ == "__main__":
    use_fixed = "--fix" in sys.argv

    if "--both" in sys.argv:
        print("\n>>> Testing ORIGINAL prompt first...\n")
        orig_pass, orig_total = run_tests(use_fixed_prompt=False)
        print("\n>>> Testing FIXED prompt...\n")
        fix_pass, fix_total = run_tests(use_fixed_prompt=True)
        print(f"\n{'='*70}")
        print(f"  ORIGINAL: {orig_pass}/{orig_total}  →  FIXED: {fix_pass}/{fix_total}")
        delta = fix_pass - orig_pass
        print(f"  Improvement: +{delta} cases" if delta > 0 else f"  No improvement" if delta == 0 else f"  Regression: {delta}")
        print(f"{'='*70}")
    else:
        run_tests(use_fixed_prompt=use_fixed)
