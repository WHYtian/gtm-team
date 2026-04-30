#!/usr/bin/env python3
"""
Multi-model benchmark for GTM agent role assignment.
Tests 4 dimensions matching actual agent tasks:
  T1 — Structured output   (Supervisor DEEP DIVE: / Analyst CONFLICT: format)
  T2 — Analytical reasoning (Analyst framework application)
  T3 — Critical thinking    (Critic finding weaknesses)
  T4 — Long-form writing    (Writer coherent structured report section)
"""
import time
from openai import OpenAI

# ── Clients ───────────────────────────────────────────────────────────────────

VOLC_KEY  = "ark-f1d21ed1-6d5d-4188-9310-bf90426a5229-893c2"
VOLC_URL  = "https://ark.cn-beijing.volces.com/api/v3"
DS_KEY    = "sk-691d2c142f7d447e89caec4a95570acc"
DS_URL    = "https://api.deepseek.com"

volc   = OpenAI(api_key=VOLC_KEY,  base_url=VOLC_URL)
ds     = OpenAI(api_key=DS_KEY,    base_url=DS_URL)

MODELS = [
    ("doubao-seed-2-0-lite-260215",  volc,  "Doubao Seed 2.0 Lite"),
    ("doubao-seed-2-0-mini-260215",  volc,  "Doubao Seed 2.0 Mini"),
    ("doubao-seed-1-8-251228",       volc,  "Doubao Seed 1.8"),
    ("doubao-seed-2-0-pro-260215",   volc,  "Doubao Seed 2.0 Pro"),
    ("doubao-1-5-pro-32k-250115",    volc,  "Doubao 1.5 Pro 32k"),
    ("doubao-v3-2-251201",           volc,  "Doubao V3 (DeepSeek V3)"),
    ("deepseek-v4-flash",            ds,    "DeepSeek V4 Flash"),
    ("deepseek-v4-pro",              ds,    "DeepSeek V4 Pro"),
]

# ── Test prompts ──────────────────────────────────────────────────────────────

T1_STRUCTURED = """You are a research quality controller. Review these dimension summaries for the topic "AI chip market 2024":

[market_overview]: The AI chip market is growing rapidly. Some sources suggest strong demand.
CONFIDENCE: 2/5 — no specific revenue figures cited, vague

[competitive_landscape]: NVIDIA dominates with ~80% market share in data center GPUs.
CONFIDENCE: 4/5 — specific market share figures, named companies

[technology_trends]: New architectures are emerging in the AI chip space.
CONFIDENCE: 1/5 — completely vague, no named technologies or companies

[regulatory_env]: Export controls are affecting the market.
CONFIDENCE: 3/5 — mentions topic but no specific regulations named

Flag dimensions needing deeper research. For each, write ONE line in this EXACT format:
DEEP DIVE: [dim_key] | reason: [one sentence] | search: [targeted search query]

Valid dim_key values: market_overview, competitive_landscape, technology_trends, regulatory_env
If all are fine: ALL ADEQUATE
Maximum 2 flags. Be strict."""

T2_REASONING = """Analyze this market data using Porter's Five Forces framework:

- Global cybersecurity market: $250B in 2024, growing 14% CAGR
- Top players: CrowdStrike (18% share), Palo Alto (15%), Fortinet (12%)
- New entrants: 400+ startups in 2023, avg Series A $25M
- Customer switching cost: high (avg 18-month migration)
- Key substitute threat: in-house SOC teams at Fortune 500
- Supplier power: semiconductor shortage affecting hardware vendors

Provide a concise Porter's Five Forces analysis (under 250 words). Rate each force: Low/Medium/High pressure."""

T3_CRITICAL = """Jamie (analyst) wrote this about the SaaS market:

"The SaaS market is projected to reach $1 trillion by 2030, driven by cloud adoption.
Microsoft and Salesforce lead the market. The main growth driver is digital transformation.
AI integration is becoming a key differentiator. The market shows strong fundamentals."

Identify 3 specific weaknesses in this analysis. Be concrete and harsh."""

T4_WRITING = """Write the Executive Summary section of a GTM Intelligence Report on "B2B SaaS Security Tools Market".

Use this data:
- Market size: $18.5B (2024), forecast $42B by 2029 (CAGR 17.8%)
- Top vendors: CrowdStrike, Okta, Zscaler, Wiz (fastest growing, $500M ARR in 3 years)
- Buyer: CISO + IT Security teams at companies 500+ employees
- Key trend: consolidation — buyers reducing vendor count from avg 45 to 25 tools

Format:
## Executive Summary
[3-5 bullet points, each specific and data-backed]

Be precise. Use the numbers provided."""

TESTS = [
    ("T1_structured_output",  T1_STRUCTURED, 400,
     "structured",
     ["DEEP DIVE:", "market_overview", "technology_trends", "regulatory_env"]),
    ("T2_analytical_reasoning", T2_REASONING, 500,
     "reasoning",
     ["Porter", "Five Forces", "High", "Medium", "Low"]),
    ("T3_critical_thinking",  T3_CRITICAL,  350,
     "critical",
     ["$1 trillion", "Microsoft", "Salesforce", "weakness"]),
    ("T4_long_form_writing",  T4_WRITING,   400,
     "writing",
     ["Executive Summary", "$18.5B", "42B", "CAGR", "Wiz"]),
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_test(client, model_id, prompt, max_tokens):
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        elapsed = time.time() - t0
        content = (r.choices[0].message.content or "").strip()
        tokens  = r.usage.completion_tokens if r.usage else len(content.split())
        return content, elapsed, tokens, None
    except Exception as e:
        return "", time.time() - t0, 0, str(e)[:120]


def score_structured(output: str, keywords: list[str]) -> int:
    """0-3: count how many expected keywords appear."""
    return sum(1 for kw in keywords if kw.lower() in output.lower())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}   # model_name → {test_id → {output, latency, tokens, error, kw_score}}

    for model_id, client, label in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing: {label} ({model_id})")
        results[label] = {}

        for test_id, prompt, max_tok, category, kw in TESTS:
            print(f"  {test_id}...", end="", flush=True)
            output, latency, tokens, error = run_test(client, model_id, prompt, max_tok)
            kw_score = score_structured(output, kw) if not error else 0
            results[label][test_id] = {
                "output": output[:600],
                "latency": round(latency, 2),
                "tokens": tokens,
                "kw_score": kw_score,
                "max_kw": len(kw),
                "error": error,
            }
            if error:
                print(f" FAIL ({error[:50]})")
            else:
                print(f" {latency:.1f}s  {tokens}tok  kw={kw_score}/{len(kw)}")

    # ── Print markdown report ─────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("GENERATING MARKDOWN REPORT")
    print("="*60)

    md = ["# Model Benchmark Report — GTM Agent Role Assignment\n"]
    md.append(f"*Tests run on {time.strftime('%Y-%m-%d %H:%M')}*\n")

    # Summary table
    md.append("## Summary Table\n")
    md.append("| Model | T1 Structured | T2 Reasoning | T3 Critical | T4 Writing | Avg Latency |")
    md.append("|-------|:---:|:---:|:---:|:---:|:---:|")

    test_ids = [t[0] for t in TESTS]
    for label in results:
        row = [f"**{label}**"]
        latencies = []
        for tid in test_ids:
            r = results[label].get(tid, {})
            if r.get("error"):
                row.append("❌")
            else:
                pct = round(r["kw_score"] / r["max_kw"] * 100)
                row.append(f"{pct}% ({r['latency']}s)")
                latencies.append(r["latency"])
        avg_lat = f"{sum(latencies)/len(latencies):.1f}s" if latencies else "N/A"
        row.append(avg_lat)
        md.append("| " + " | ".join(row) + " |")

    # Per-test output excerpts
    for test_id, prompt, _, category, kw in TESTS:
        md.append(f"\n## {test_id.replace('_',' ').title()}\n")
        md.append(f"**Task type:** {category} | **Keywords checked:** `{'`, `'.join(kw)}`\n")
        md.append("<details><summary>Show prompt</summary>\n\n```\n" + prompt[:300] + "...\n```\n</details>\n")
        for label in results:
            r = results[label].get(test_id, {})
            if r.get("error"):
                md.append(f"### {label}\n**ERROR:** {r['error']}\n")
            else:
                pct = round(r["kw_score"] / r["max_kw"] * 100)
                md.append(
                    f"### {label}\n"
                    f"**Score:** {r['kw_score']}/{r['max_kw']} ({pct}%) | "
                    f"**Latency:** {r['latency']}s | **Tokens:** {r['tokens']}\n\n"
                    f"```\n{r['output'][:500]}\n```\n"
                )

    report = "\n".join(md)
    out = "/home/admin/gtm-team/scripts/benchmark_results.md"
    with open(out, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {out}")
    return results


if __name__ == "__main__":
    main()
