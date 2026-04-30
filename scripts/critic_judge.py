#!/usr/bin/env python3
"""
LLM-as-Judge: evaluate T3 critic outputs across all models.
Each judge scores every critic response on 4 dimensions (1-5).
Self-scoring included — we'll see if models are biased toward their own output.
"""
import time, json
from openai import OpenAI

VOLC_KEY = "ark-f1d21ed1-6d5d-4188-9310-bf90426a5229-893c2"
VOLC_URL = "https://ark.cn-beijing.volces.com/api/v3"
DS_KEY   = "sk-691d2c142f7d447e89caec4a95570acc"
DS_URL   = "https://api.deepseek.com"

volc = OpenAI(api_key=VOLC_KEY, base_url=VOLC_URL)
ds   = OpenAI(api_key=DS_KEY,   base_url=DS_URL)

# Models that can reliably produce structured JSON scores
JUDGES = [
    ("doubao-1-5-pro-32k-250115", volc, "1.5 Pro"),
    ("deepseek-v3-2-251201",      volc, "DS-V3"),
    ("doubao-seed-2-0-mini-260215", volc, "Seed Mini"),
    ("doubao-seed-2-0-pro-260215",  volc, "Seed Pro"),
]

T3_PROMPT = """Jamie (analyst) wrote this about the SaaS market:

"The SaaS market is projected to reach $1 trillion by 2030, driven by cloud adoption. \
Microsoft and Salesforce lead the market. The main growth driver is digital transformation. \
AI integration is becoming a key differentiator. The market shows strong fundamentals."

Identify 3 specific weaknesses in this analysis. Be concrete and harsh."""

CRITIC_MODELS = [
    ("doubao-seed-2-0-lite-260215", volc, "Seed 2.0 Lite"),
    ("doubao-seed-2-0-mini-260215", volc, "Seed 2.0 Mini ★"),
    ("doubao-seed-1-8-251228",      volc, "Seed 1.8"),
    ("doubao-seed-2-0-pro-260215",  volc, "Seed 2.0 Pro"),
    ("doubao-1-5-pro-32k-250115",   volc, "1.5 Pro 32k"),
    ("deepseek-v3-2-251201",        volc, "DS-V3"),
]

JUDGE_PROMPT = """You are an expert evaluator assessing the quality of market analysis critiques.

ORIGINAL ANALYSIS being critiqued:
"The SaaS market is projected to reach $1 trillion by 2030, driven by cloud adoption. \
Microsoft and Salesforce lead the market. The main growth driver is digital transformation. \
AI integration is becoming a key differentiator. The market shows strong fundamentals."

CRITIQUE TO EVALUATE:
{critique}

Score this critique on 4 dimensions (each 1-5):
1. **Specificity** — Does it point to specific missing data, named sources, or concrete numbers? (1=vague, 5=very specific)
2. **Validity** — Are the weaknesses actually real problems with the original analysis? (1=wrong/irrelevant, 5=all valid)
3. **Depth** — Does it go beyond surface observations to expose structural reasoning flaws? (1=shallow, 5=deep)
4. **Actionability** — Does the critique tell the analyst what to fix and how? (1=no guidance, 5=clear concrete suggestions)

Respond ONLY with valid JSON, no explanation:
{{"specificity": X, "validity": X, "depth": X, "actionability": X, "comment": "one sentence"}}"""


def get_critique(model_id, client):
    try:
        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": T3_PROMPT}],
            max_tokens=600, temperature=0.3,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def judge_critique(judge_model, judge_client, critique_text):
    try:
        r = judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(critique=critique_text[:800])}],
            max_tokens=200, temperature=0.1,
        )
        raw = (r.choices[0].message.content or "").strip()
        # Parse JSON
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return None
    except Exception as e:
        return None


def main():
    # Step 1: collect critiques
    print("Collecting critiques from each model...")
    critiques = {}
    for mid, client, label in CRITIC_MODELS:
        print(f"  {label}...", end="", flush=True)
        t0 = time.time()
        text = get_critique(mid, client)
        elapsed = time.time() - t0
        critiques[label] = {"text": text, "latency": round(elapsed, 1)}
        print(f" {elapsed:.1f}s  {len(text.split())} words")

    # Step 2: each judge scores all critiques
    print("\nRunning LLM-as-Judge scoring...")
    scores = {label: [] for label in critiques}

    for jmid, jclient, jlabel in JUDGES:
        print(f"\n  Judge: {jlabel}")
        for clabel, cdata in critiques.items():
            if cdata["text"].startswith("ERROR"):
                continue
            result = judge_critique(jmid, jclient, cdata["text"])
            if result:
                result["judge"] = jlabel
                result["self"] = (jlabel.replace(" ★","") == clabel.replace(" ★",""))
                scores[clabel].append(result)
                total = result["specificity"] + result["validity"] + result["depth"] + result["actionability"]
                flag = " ← self" if result["self"] else ""
                print(f"    {clabel:<20} {total}/20  {result.get('comment','')[:60]}{flag}")
            else:
                print(f"    {clabel:<20} parse failed")

    # Step 3: aggregate and report
    print("\n\n" + "="*70)
    print("RESULTS")
    print("="*70)

    rows = []
    for label, sc_list in scores.items():
        if not sc_list:
            continue
        non_self = [s for s in sc_list if not s.get("self")]
        if not non_self:
            non_self = sc_list
        avg = lambda key: round(sum(s[key] for s in non_self) / len(non_self), 1)
        row = {
            "label": label,
            "specificity": avg("specificity"),
            "validity": avg("validity"),
            "depth": avg("depth"),
            "actionability": avg("actionability"),
            "total": round(sum(avg(k) for k in ["specificity","validity","depth","actionability"]), 1),
            "latency": critiques[label]["latency"],
            "n_judges": len(non_self),
        }
        rows.append(row)

    rows.sort(key=lambda r: r["total"], reverse=True)

    print(f"\n{'Model':<22} {'Spec':>5} {'Valid':>6} {'Depth':>6} {'Act':>5} {'Total':>6} {'Latency':>8} {'Judges':>7}")
    print("-"*70)
    for r in rows:
        star = " ★" if "★" in r["label"] else "  "
        print(f"{r['label']:<22}{star} {r['specificity']:>4} {r['validity']:>6} {r['depth']:>6} {r['actionability']:>5} {r['total']:>6} {r['latency']:>7}s {r['n_judges']:>6}")

    print("\n★ = currently selected Critic model")

    # Markdown report
    md = ["# Critic 质量评估：LLM-as-Judge\n"]
    md.append("评估方式：让 4 个模型（1.5 Pro、DS-V3、Seed Mini、Seed Pro）分别对每个模型的 T3 批判输出打分，排除自评分后取平均。\n")
    md.append("评分维度（各 1-5）：**Specificity**（具体性）、**Validity**（有效性）、**Depth**（深度）、**Actionability**（可操作性）\n")
    md.append("## 综合排名\n")
    md.append("| 排名 | 模型 | 具体性 | 有效性 | 深度 | 可操作性 | 总分/20 | 延迟 |")
    md.append("|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|")
    for i, r in enumerate(rows, 1):
        star = " ★" if "★" in r["label"] else ""
        md.append(f"| {i} | **{r['label']}{star}** | {r['specificity']} | {r['validity']} | {r['depth']} | {r['actionability']} | **{r['total']}** | {r['latency']}s |")
    md.append("\n★ = 原始基准测试选定的模型\n")

    # Self-bias analysis
    md.append("## 自评偏差分析\n")
    md.append("| 模型 | 自评总分 | 他评平均 | 偏差 |")
    md.append("|------|:---:|:---:|:---:|")
    for label, sc_list in scores.items():
        self_s  = [s for s in sc_list if s.get("self")]
        other_s = [s for s in sc_list if not s.get("self")]
        if self_s and other_s:
            self_t  = sum(s["specificity"]+s["validity"]+s["depth"]+s["actionability"] for s in self_s) / len(self_s)
            other_t = sum(s["specificity"]+s["validity"]+s["depth"]+s["actionability"] for s in other_s) / len(other_s)
            bias = round(self_t - other_t, 1)
            bias_str = f"+{bias}" if bias > 0 else str(bias)
            md.append(f"| {label} | {round(self_t,1)} | {round(other_t,1)} | {bias_str} |")

    # Critique excerpts
    md.append("\n## 各模型批判内容摘录\n")
    for label, cdata in critiques.items():
        md.append(f"### {label}\n```\n{cdata['text'][:600]}\n```\n")

    out = "/home/admin/gtm-team/scripts/critic_judge_results.md"
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"\nMarkdown saved: {out}")


if __name__ == "__main__":
    main()
