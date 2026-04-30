#!/usr/bin/env python3
"""
LLM-as-Judge for Supervisor (T1), Analyst (T2), Writer (T4).
Each role has task-specific scoring dimensions.
"""
import time, json, re
from openai import OpenAI

VOLC_KEY = "ark-f1d21ed1-6d5d-4188-9310-bf90426a5229-893c2"
VOLC_URL = "https://ark.cn-beijing.volces.com/api/v3"
DS_KEY   = "sk-691d2c142f7d447e89caec4a95570acc"
DS_URL   = "https://api.deepseek.com"

volc = OpenAI(api_key=VOLC_KEY, base_url=VOLC_URL)
ds   = OpenAI(api_key=DS_KEY,   base_url=DS_URL)

CANDIDATES = [
    ("doubao-seed-2-0-lite-260215", volc, "Seed 2.0 Lite"),
    ("doubao-seed-2-0-mini-260215", volc, "Seed 2.0 Mini"),
    ("doubao-seed-1-8-251228",      volc, "Seed 1.8"),
    ("doubao-seed-2-0-pro-260215",  volc, "Seed 2.0 Pro"),
    ("doubao-1-5-pro-32k-250115",   volc, "1.5 Pro 32k ★"),
    ("deepseek-v3-2-251201",        volc, "DS-V3"),
]

JUDGES = [
    ("doubao-1-5-pro-32k-250115",   volc, "1.5 Pro"),
    ("deepseek-v3-2-251201",        volc, "DS-V3"),
    ("doubao-seed-2-0-mini-260215", volc, "Seed Mini"),
    ("doubao-seed-2-0-pro-260215",  volc, "Seed Pro"),
]

# ── Task prompts ──────────────────────────────────────────────────────────────

TASKS = {
    "supervisor": {
        "label": "Supervisor — T1 Structured Output",
        "prompt": """You are a research quality controller. Review these dimension summaries for the topic "AI chip market 2024":

[market_overview]: The AI chip market is growing rapidly. Some sources suggest strong demand.
CONFIDENCE: 2/5 — no specific revenue figures cited, vague

[competitive_landscape]: NVIDIA dominates with ~80% market share in data center GPUs. AMD gaining traction with MI300X, Intel lagging with Gaudi.
CONFIDENCE: 4/5 — specific market share and named companies

[technology_trends]: New architectures are emerging in the AI chip space.
CONFIDENCE: 1/5 — completely vague, no named technologies or companies

[regulatory_env]: US export controls are significantly affecting Nvidia's sales to China, impacting ~25% of revenue.
CONFIDENCE: 4/5 — specific policy and impact figures

Flag dimensions needing deeper research. For each, write ONE line in this EXACT format:
DEEP DIVE: [dim_key] | reason: [one sentence] | search: [targeted search query]

Valid dim_key values: market_overview, competitive_landscape, technology_trends, regulatory_env
If all are fine: ALL ADEQUATE. Maximum 2 flags. Be strict.""",
        "max_tokens": 300,
        "dimensions": ["format_compliance", "accuracy", "query_quality", "conciseness"],
        "judge_prompt": """You are evaluating a research quality assessment output for the topic "AI chip market 2024".

The CORRECT answer is: market_overview (CONFIDENCE 2/5) and technology_trends (CONFIDENCE 1/5) should be flagged. competitive_landscape and regulatory_env (both 4/5) should NOT be flagged.

OUTPUT TO EVALUATE:
{output}

Score on 4 dimensions (1-5):
1. **format_compliance** — Does it strictly follow "DEEP DIVE: dim_key | reason: ... | search: ..." format for each flag? (1=wrong format, 5=perfect)
2. **accuracy** — Does it flag the RIGHT dimensions (market_overview and/or technology_trends) and NOT flag the good ones? (1=wrong dims flagged, 5=exactly right)
3. **query_quality** — Are the suggested search queries specific, targeted, and actionable? (1=vague, 5=precise and useful)
4. **conciseness** — Is the output appropriately brief without padding? (1=verbose/padded, 5=tight and precise)

Respond ONLY with valid JSON:
{{"format_compliance": X, "accuracy": X, "query_quality": X, "conciseness": X, "comment": "one sentence"}}"""
    },

    "analyst": {
        "label": "Analyst — T2 Analytical Reasoning",
        "prompt": """Analyze this market data using Porter's Five Forces framework:

- Global cybersecurity market: $250B in 2024, growing 14% CAGR
- Top players: CrowdStrike (18% share), Palo Alto (15%), Fortinet (12%)
- New entrants: 400+ startups in 2023, avg Series A $25M
- Customer switching cost: high (avg 18-month migration)
- Key substitute threat: in-house SOC teams at Fortune 500
- Supplier power: semiconductor shortage affecting hardware vendors

Provide a concise Porter's Five Forces analysis (under 250 words). Rate each force: Low/Medium/High pressure.""",
        "max_tokens": 500,
        "dimensions": ["framework_coverage", "evidence_use", "rating_quality", "conciseness"],
        "judge_prompt": """You are evaluating a Porter's Five Forces analysis of the cybersecurity market.

Available data provided to the analyst:
- $250B market, 14% CAGR
- CrowdStrike 18%, Palo Alto 15%, Fortinet 12%
- 400+ startups, $25M avg Series A
- 18-month customer migration cost
- In-house SOC teams as substitute
- Semiconductor shortage affecting hardware vendors

OUTPUT TO EVALUATE:
{output}

Score on 4 dimensions (1-5):
1. **framework_coverage** — Does it cover all 5 forces (New Entrants, Supplier Power, Buyer Power, Substitutes, Rivalry)? (1=missing forces, 5=all 5 covered)
2. **evidence_use** — Does it cite the specific data provided (numbers, company names, switching costs)? (1=ignores data, 5=uses all key data points)
3. **rating_quality** — Are the High/Medium/Low ratings justified with reasoning? (1=ratings without justification, 5=each rating clearly explained)
4. **conciseness** — Is it under 250 words and well-structured? (1=bloated, 5=tight and clear)

Respond ONLY with valid JSON:
{{"framework_coverage": X, "evidence_use": X, "rating_quality": X, "conciseness": X, "comment": "one sentence"}}"""
    },

    "writer": {
        "label": "Writer — T4 Long-form Report",
        "prompt": """Write the Executive Summary section of a GTM Intelligence Report on "B2B SaaS Security Tools Market".

Use this data:
- Market size: $18.5B (2024), forecast $42B by 2029 (CAGR 17.8%)
- Top vendors: CrowdStrike, Okta, Zscaler, Wiz (fastest growing, $500M ARR in 3 years)
- Buyer: CISO + IT Security teams at companies 500+ employees
- Key trend: consolidation — buyers reducing vendor count from avg 45 to 25 tools

Format:
## Executive Summary
[3-5 bullet points, each specific and data-backed]

Be precise. Use the numbers provided.""",
        "max_tokens": 400,
        "dimensions": ["data_accuracy", "structure", "insight_quality", "readability"],
        "judge_prompt": """You are evaluating an Executive Summary for a B2B SaaS Security Tools Market report.

Required data that MUST appear: $18.5B (2024), $42B (2029), CAGR 17.8%, Wiz $500M ARR, consolidation from 45 to 25 tools, CISO buyers.

OUTPUT TO EVALUATE:
{output}

Score on 4 dimensions (1-5):
1. **data_accuracy** — Are all key numbers correctly cited ($18.5B, $42B, 17.8%, $500M ARR, 45→25 tools)? (1=numbers missing/wrong, 5=all correct)
2. **structure** — Does it use the correct ## Executive Summary header and 3-5 bullet points? (1=wrong format, 5=perfect structure)
3. **insight_quality** — Do the bullet points go beyond restating data to offer strategic context? (1=pure data dump, 5=data + strategic meaning)
4. **readability** — Is it professional, clear, and free of filler? (1=verbose/jargon, 5=crisp and executive-ready)

Respond ONLY with valid JSON:
{{"data_accuracy": X, "structure": X, "insight_quality": X, "readability": X, "comment": "one sentence"}}"""
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_output(model_id, client, prompt, max_tokens):
    try:
        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.3,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: {e}"

def judge(judge_mid, judge_client, judge_prompt_template, output_text):
    try:
        r = judge_client.chat.completions.create(
            model=judge_mid,
            messages=[{"role": "user", "content": judge_prompt_template.format(output=output_text[:800])}],
            max_tokens=200, temperature=0.1,
        )
        raw = (r.choices[0].message.content or "").strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except:
        return None

def avg_scores(score_list, dims):
    if not score_list:
        return {d: 0 for d in dims}, 0
    avgs = {d: round(sum(s[d] for s in score_list if d in s) / len(score_list), 1) for d in dims}
    total = round(sum(avgs.values()), 1)
    return avgs, total

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for role_key, task in TASKS.items():
        print(f"\n{'='*70}")
        print(f"ROLE: {task['label']}")
        print('='*70)

        # Collect outputs
        print("Collecting outputs...")
        outputs = {}
        for mid, client, label in CANDIDATES:
            print(f"  {label}...", end="", flush=True)
            t0 = time.time()
            text = get_output(mid, client, task["prompt"], task["max_tokens"])
            elapsed = time.time() - t0
            outputs[label] = {"text": text, "latency": round(elapsed, 1)}
            status = "ERROR" if text.startswith("ERROR") else f"{elapsed:.1f}s {len(text.split())}w"
            print(f" {status}")

        # Judge each output
        print("\nScoring...")
        scores = {label: [] for label in outputs}
        dims = task["dimensions"]

        for jmid, jclient, jlabel in JUDGES:
            print(f"  Judge: {jlabel}")
            for clabel, cdata in outputs.items():
                if cdata["text"].startswith("ERROR"):
                    continue
                result = judge(jmid, jclient, task["judge_prompt"], cdata["text"])
                if result:
                    is_self = jlabel.split()[0] in clabel
                    result["judge"] = jlabel
                    result["self"] = is_self
                    scores[clabel].append(result)
                    total = sum(result.get(d, 0) for d in dims)
                    flag = " ← self" if is_self else ""
                    print(f"    {clabel:<22} {total}/20  {result.get('comment','')[:55]}{flag}")

        # Aggregate (exclude self-scores)
        rows = []
        for label, sc_list in scores.items():
            non_self = [s for s in sc_list if not s.get("self")] or sc_list
            avgs, total = avg_scores(non_self, dims)
            rows.append({
                "label": label,
                **avgs,
                "total": total,
                "latency": outputs[label]["latency"],
                "n": len(non_self),
            })
        rows.sort(key=lambda r: r["total"], reverse=True)

        # Print table
        print(f"\n  {'Model':<22} " + " ".join(f"{d[:4]:>5}" for d in dims) + f" {'Tot':>5} {'Lat':>7}")
        print("  " + "-"*68)
        for r in rows:
            star = " ★" if "★" in r["label"] else "  "
            scores_str = " ".join(f"{r[d]:>5}" for d in dims)
            print(f"  {r['label']:<22}{star} {scores_str} {r['total']:>5} {r['latency']:>6}s")

        all_results[role_key] = {"task": task, "outputs": outputs, "rows": rows}

    # ── Markdown report ───────────────────────────────────────────────────────
    md = ["# Role-Specific Model Evaluation: LLM-as-Judge\n"]
    md.append("Supervisor (T1)、Analyst (T2)、Writer (T4) 三个角色的横向评分。\n")
    md.append("评估方式：4 个裁判模型对每个候选输出打分，排除自评后取平均。★ = 当前选定模型。\n")

    for role_key, data in all_results.items():
        task = data["task"]
        rows = data["rows"]
        dims = task["dimensions"]
        outputs = data["outputs"]

        md.append(f"\n## {task['label']}\n")
        md.append("| 排名 | 模型 | " + " | ".join(d.replace("_"," ").title() for d in dims) + " | 总分/20 | 延迟 |")
        md.append("|:---:|------|" + "|".join([":---:"]*len(dims)) + "|:---:|:---:|")

        for i, r in enumerate(rows, 1):
            star = " ★" if "★" in r["label"] else ""
            scores_str = " | ".join(str(r[d]) for d in dims)
            md.append(f"| {i} | **{r['label']}{star}** | {scores_str} | **{r['total']}** | {r['latency']}s |")

        md.append("\n### 输出摘录\n")
        for label, cdata in outputs.items():
            if not cdata["text"].startswith("ERROR"):
                md.append(f"**{label}** ({cdata['latency']}s)\n```\n{cdata['text'][:500]}\n```\n")

    # Final recommendation table
    md.append("\n## 最终推荐汇总\n")
    md.append("| 角色 | 推荐模型 | 理由 |")
    md.append("|------|---------|------|")
    for role_key, data in all_results.items():
        rows = data["rows"]
        top = rows[0]
        selected = next((r for r in rows if "★" in r["label"]), None)
        role_name = role_key.title()
        if selected and selected["label"] == top["label"]:
            md.append(f"| {role_name} | {top['label']} ✅ | 当前选定，排名第一 |")
        elif selected:
            gap = round(top["total"] - selected["total"], 1)
            lat_diff = round(top["latency"] - selected["latency"], 1)
            md.append(f"| {role_name} | 当前: {selected['label']} | #{rows.index(selected)+1}名，比第一低{gap}分，快{lat_diff}s |")

    out = "/home/admin/gtm-team/scripts/role_judge_results.md"
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"\n\nMarkdown saved: {out}")

if __name__ == "__main__":
    main()
