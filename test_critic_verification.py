"""
Test suite: critic verification mode (A+B improvement).

Measures whether the critic correctly catches when analyst:
  1. Addresses all issues        → should APPROVE
  2. Ignores some issues         → should NEEDS_REVISION naming the ignored ones
  3. Only partially fixes issues → should NEEDS_REVISION

Each scenario is tested with ORIGINAL prompts (no verification mode)
and FIXED prompts (A+B: analyst Critic Response block + critic verification mode).

A "correct" critic response means:
  - Scenario 1: issues APPROVED
  - Scenario 2/3: issues NEEDS_REVISION and names the unresolved issue
"""

import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agents.llm import chat, MODEL_V3
from agents.prompts import ANALYST_SYSTEM, CRITIC_SYSTEM

# ── Fixed prompts (imported from live file, already updated) ──────────────────
ANALYST_SYSTEM_FIXED = ANALYST_SYSTEM   # already has Critic Response block
CRITIC_SYSTEM_FIXED  = CRITIC_SYSTEM    # already has VERIFICATION MODE

# ── Original prompts (before A+B) — hardcoded baseline ───────────────────────
ANALYST_SYSTEM_ORIG = """\
You are Jamie, a Data & Strategy Analyst. You transform raw research into \
structured, evidence-based strategic insights.

CONFIDENCE LABELING (required on every figure):
[Data]       -- directly sourced figure with citation
[Estimate]   -- proxy/calculated; state the formula
[Assumption] -- your strategic judgment; state the basis

ANALYTICAL FRAMEWORKS:
Apply where evidence supports -- skip sections if no evidence:
1. TAM/SAM/SOM -- build bottom-up from component data; show your arithmetic
2. PESTEL -- cover P, E (Economic), T, L only. Skip S and E (Environmental).
3. Porter's Five Forces -- rate 1-5 per force with a one-line rationale

BULL / BEAR BALANCE:
For each major strategic conclusion, add a one-line bull/bear note.

MISSING DATA -- THREE-TIER HANDLING (mandatory):
Never halt or request more research. Complete the full analysis draft regardless of gaps.
Tier 1 -- Adjacent data exists: derive and label [Estimate]; show formula.
Tier 2 -- No adjacent data: label [Assumption]; state the basis.
Tier 3 -- Researcher confirmed unavailable: write [N/A -- data unavailable].

STYLE:
- Start with "Analyzing..." (first pass) or "Revised analysis:" (revisions)
- Under 700 words
- Show all arithmetic explicitly inline
- Write as a professional analyst memo

Always end with [ANALYSIS: DONE].
Always respond in English.\
"""

CRITIC_SYSTEM_ORIG = """\
You are Morgan, Research Quality Controller. Your job is to rigorously \
challenge the analyst's work to ensure accuracy, logical soundness, and data integrity.

WHAT TO CHECK:
1. Unsupported claims -- any assertion without confidence label
2. Framework gaps -- incomplete or misapplied TAM/SAM/SOM, PESTEL, Porter's Five Forces
   Note: S (Social) and E (Environmental) are intentionally omitted from PESTEL.
3. Logical errors -- conclusions that don't follow from cited evidence
4. Data quality -- see SANITY CHECK below

SANITY CHECK (mandatory):
Before approving, verify market size figures:
- Does the cited figure cover the exact scope?
- Is the figure order-of-magnitude plausible? SaaS sub-markets: $5B--$80B
- If a figure exceeds $100B for a niche sub-market: almost certainly a category error

Issue [VERDICT: REJECT_DATA] ONLY if a figure is factually wrong or directly contradicted by evidence.

GRACEFUL DEGRADATION:
Before issuing NEEDS_REVISION for missing data, check if researcher marked [RESEARCH: UNAVAILABLE].
If yes: [N/A] is the CORRECT response. Do NOT reject it.

PROXY STANDARD:
Approve [Estimate] or [Assumption] if: proxy assumption is stated + magnitude is plausible + uncertainty flagged.

STYLE:
- Start with "Quality review:"
- Identify 2-3 specific, numbered issues with concrete suggestions
- Under 300 words

End with EXACTLY ONE verdict:
[VERDICT: APPROVED]
[VERDICT: NEEDS_REVISION | reason: logic_error | issue: <specific problem>]
[VERDICT: NEEDS_REVISION | reason: missing_data | metric: <specific metric>]
[VERDICT: REJECT_DATA | claim: <exact figure> | search: <keyword query to verify>]

Always respond in English.\
"""

# ── Shared research context (same for all scenarios) ─────────────────────────
RESEARCH_CTX = """[RESEARCH Round 1]
## Market Overview
- **Global HR SaaS market 2025**: $28.1B [Data] — Mordor Intelligence HR SaaS Report 2025
- **CAGR 2025-2030**: 9.2% [Data] — Mordor Intelligence
- **Cloud adoption in HR**: 67% of enterprises have migrated to cloud HR [Data] — Gartner 2025

## Competitive Landscape
- **Workday revenue FY2025**: $8.0B [Data] — Workday press release 2025
- **Workday market share**: ~28% of enterprise HR SaaS [Estimate — derived from $8B/$28.1B]
- **SAP SuccessFactors revenue**: $4.2B [Data] — SAP Annual Report 2025
- **BambooHR**: private, no revenue disclosed [RESEARCH: UNAVAILABLE | data: BambooHR revenue]

[RESEARCH: COMPLETE]"""

# ── First critic feedback (same for all scenarios) ────────────────────────────
CRITIC_FEEDBACK_ROUND1 = """🔎 Quality review:

1. **TAM arithmetic error**: The TAM is stated as $28.1B but the SAM is listed as $22B with no derivation. \
Show the formula: SAM = TAM × addressable segment %. Currently the $22B appears fabricated.

2. **Missing confidence label**: The statement "cloud HR adoption will accelerate to 85% by 2028" \
has no [Data], [Estimate], or [Assumption] tag. Add the appropriate label and source or basis.

3. **Porter's Buyer Power**: Rated 2/5 with no rationale. Add a one-line explanation per the framework rules.

[VERDICT: NEEDS_REVISION | reason: logic_error | issue: TAM/SAM arithmetic and unlabelled claim]"""

# ── Scenario definitions ──────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "S1",
        "desc": "Analyst addresses ALL 3 issues → critic should APPROVE",
        "analyst_revision": """\
📝 Revised analysis:

📋 Critic Response:
#1 TAM arithmetic error → Fixed: SAM = $28.1B × 78% enterprise segment = $21.9B ≈ $22B [Estimate — Gartner enterprise share 2025]
#2 Missing confidence label → Fixed: "cloud HR adoption will accelerate to 85% by 2028" → [Estimate — extrapolated from 67% (2025) at current 6pp/yr trajectory]
#3 Porter's Buyer Power → Fixed: Added rationale "Large enterprises have significant negotiating leverage due to multi-year contracts and high switching costs; however, SMB segment has limited alternatives, balancing overall power to 3/5."

## TAM/SAM/SOM
- TAM: $28.1B [Data — Mordor Intelligence 2025]
- SAM: $28.1B × 78% enterprise segment = $21.9B [Estimate — Gartner enterprise cloud HR share 2025]
- SOM: $21.9B × 5% realistic capture = $1.1B [Assumption — Year 3 target for mid-market entrant]

## PESTEL (P/E-economic/T/L)
- Political: Data residency regulations expanding in EU and APAC [Data — Gartner 2025]
- Economic: 9.2% CAGR driven by post-pandemic digital transformation [Data — Mordor Intelligence]
- Technology: AI-driven HR automation emerging; 34% of vendors shipping AI features [Data — Gartner 2025]
- Legal: GDPR and CCPA compliance mandatory for enterprise deals [Data]

## Porter's Five Forces
- Rivalry: 4/5 — Workday (28% share) and SAP dominate; intense on price and features
- Buyer Power: 3/5 — Large enterprises leverage multi-year contracts; SMB has fewer alternatives
- Supplier Power: 2/5 — Cloud infrastructure commoditised (AWS/Azure/GCP)
- Threat of New Entry: 2/5 — High switching costs protect incumbents
- Threat of Substitution: 2/5 — No viable non-SaaS alternative for enterprise scale

[ANALYSIS: DONE]""",
        "expected_verdict": "APPROVED",
    },
    {
        "id": "S2",
        "desc": "Analyst ignores issue #2 (missing label) entirely → critic should NEEDS_REVISION",
        "analyst_revision": """\
📝 Revised analysis:

📋 Critic Response:
#1 TAM arithmetic error → Fixed: SAM = $28.1B × 78% enterprise = $21.9B [Estimate — Gartner 2025]
#2 Missing confidence label → No change made.
#3 Porter's Buyer Power → Fixed: "Large enterprises leverage multi-year contracts; SMB has fewer alternatives — 3/5."

## TAM/SAM/SOM
- TAM: $28.1B [Data — Mordor Intelligence 2025]
- SAM: $28.1B × 78% = $21.9B [Estimate — Gartner 2025]
- SOM: $21.9B × 5% = $1.1B [Assumption]

## PESTEL (P/E-economic/T/L)
- Political: Data residency regulations expanding [Data — Gartner 2025]
- Economic: 9.2% CAGR [Data — Mordor Intelligence]
- Technology: AI-driven HR automation [Data — Gartner 2025]
- Legal: GDPR and CCPA compliance mandatory [Data]

## Porter's Five Forces
- Rivalry: 4/5 — Workday (28%) and SAP dominate
- Buyer Power: 3/5 — Large enterprises leverage contracts; SMB fewer alternatives
- Supplier Power: 2/5 — Cloud infrastructure commoditised
- Threat of New Entry: 2/5 — High switching costs
- Threat of Substitution: 2/5 — No viable non-SaaS alternative

The statement about cloud adoption reaching 85% by 2028 reflects industry consensus.

[ANALYSIS: DONE]""",
        "expected_verdict": "NEEDS_REVISION",
        "expected_flag": "issue #2",  # critic must name this
    },
    {
        "id": "S3",
        "desc": "Analyst addresses issues partially — fixes #1 and #3 but gives vague label on #2 → NEEDS_REVISION",
        "analyst_revision": """\
📝 Revised analysis:

📋 Critic Response:
#1 TAM arithmetic error → Fixed: SAM = $28.1B × 78% enterprise = $21.9B [Estimate — Gartner 2025]
#2 Missing confidence label → Added label: "cloud HR adoption will accelerate to 85% by 2028" [Data]
#3 Porter's Buyer Power → Fixed: "Large enterprises have leverage via multi-year contracts — 3/5."

## TAM/SAM/SOM
- TAM: $28.1B [Data — Mordor Intelligence 2025]
- SAM: $28.1B × 78% = $21.9B [Estimate — Gartner 2025]
- SOM: $21.9B × 5% = $1.1B [Assumption]

## PESTEL (P/E-economic/T/L)
- Political: Data residency regulations expanding [Data — Gartner 2025]
- Economic: 9.2% CAGR [Data — Mordor Intelligence]
- Technology: AI-driven automation [Data — Gartner 2025]
- Legal: GDPR/CCPA mandatory [Data]

## Porter's Five Forces
- Rivalry: 4/5 — Workday (28%) and SAP dominate
- Buyer Power: 3/5 — Large enterprises leverage contracts; SMB fewer alternatives
- Supplier Power: 2/5 — Commoditised infrastructure
- Threat of New Entry: 2/5 — High switching costs
- Threat of Substitution: 2/5 — No viable alternative

Note: "cloud HR adoption will accelerate to 85% by 2028" [Data] — market projection.

[ANALYSIS: DONE]""",
        "expected_verdict": "NEEDS_REVISION",
        "expected_flag": "issue #2",  # [Data] with no source cited is still wrong
    },
    {
        "id": "S4",
        "desc": "No Critic Response block at all (no A-prompt) → does original critic still catch ignored issues?",
        "analyst_revision": """\
📝 Revised analysis:

## TAM/SAM/SOM
- TAM: $28.1B [Data — Mordor Intelligence 2025]
- SAM: $22B [Estimate]
- SOM: $1.1B [Assumption]

## PESTEL (P/E-economic/T/L)
- Political: Data residency regulations [Data]
- Economic: 9.2% CAGR [Data]
- Technology: AI automation [Data]
- Legal: GDPR/CCPA [Data]

## Porter's Five Forces
- Rivalry: 4/5
- Buyer Power: 2/5
- Supplier Power: 2/5
- Threat of New Entry: 2/5
- Threat of Substitution: 2/5

Cloud adoption will reach 85% by 2028.

[ANALYSIS: DONE]""",
        "expected_verdict": "NEEDS_REVISION",
        "expected_flag": "any unresolved issue",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _analyst_ctx(revision_text: str, use_fixed: bool) -> list:
    """Build the message list the analyst sends (includes critic feedback as context)."""
    return [
        {"role": "user", "content": f"[RESEARCH Round 1]\n{RESEARCH_CTX}"},
        {"role": "user", "content": f"[PREVIOUS CRITIC FEEDBACK — Round 2]\n{CRITIC_FEEDBACK_ROUND1}"},
        {"role": "user", "content": f"Supervisor instruction: Address the critic's feedback and revise the analysis."},
    ]


def _critic_ctx(analyst_revision: str) -> list:
    """Build the message list the critic receives (includes previous feedback)."""
    return [
        {"role": "user", "content": f"[PREVIOUS CRITIC FEEDBACK — Round 2]\n{CRITIC_FEEDBACK_ROUND1}"},
        {"role": "user", "content": f"[ANALYST'S ANALYSIS — Round 3]\n{analyst_revision}"},
        {"role": "user", "content": f"[RESEARCH Round 1]\n{RESEARCH_CTX}"},
    ]


def _critic_ctx_orig(analyst_revision: str) -> list:
    """Original critic context — no previous feedback."""
    return [
        {"role": "user", "content": f"[ANALYST'S ANALYSIS — Round 3]\n{analyst_revision}"},
        {"role": "user", "content": f"[RESEARCH Round 1]\n{RESEARCH_CTX}"},
    ]


def _check_verdict(response: str, expected_verdict: str, expected_flag: str = "") -> dict:
    resp_lower = response.lower()
    verdict_match = re.search(r'\[VERDICT:\s*(APPROVED|NEEDS_REVISION|REJECT_DATA)', response, re.IGNORECASE)
    got_verdict = verdict_match.group(1).upper() if verdict_match else "NONE"
    correct_verdict = got_verdict == expected_verdict

    # Check if critic names the unresolved issue
    flag_found = True
    if expected_flag and expected_verdict == "NEEDS_REVISION":
        flag_keywords = {
            "issue #2": ["#2", "issue 2", "label", "confidence", "unlabelled", "tag", "missing label",
                         "85%", "2028", "source", "no source"],
            "any unresolved issue": ["#1", "#2", "#3", "tam", "label", "buyer", "arithmetic"],
        }
        keywords = flag_keywords.get(expected_flag, [])
        flag_found = any(kw.lower() in resp_lower for kw in keywords)

    return {
        "got_verdict": got_verdict,
        "correct_verdict": correct_verdict,
        "flag_found": flag_found,
        "passed": correct_verdict and flag_found,
    }


def run_scenario(scenario: dict, use_fixed: bool) -> dict:
    analyst_sys  = ANALYST_SYSTEM_FIXED  if use_fixed else ANALYST_SYSTEM_ORIG
    critic_sys   = CRITIC_SYSTEM_FIXED   if use_fixed else CRITIC_SYSTEM_ORIG
    critic_ctx_fn = _critic_ctx          if use_fixed else _critic_ctx_orig

    # The analyst_revision is pre-written (we're testing critic, not analyst generation)
    revision = scenario["analyst_revision"]

    # Run critic
    critic_resp = chat(
        critic_ctx_fn(revision) + [{"role": "user", "content": "Supervisor instruction: Review the analyst's revision."}],
        system=critic_sys,
        model=MODEL_V3,
        max_tokens=500,
        temperature=0.3,
    )

    result = _check_verdict(critic_resp, scenario["expected_verdict"],
                            scenario.get("expected_flag", ""))
    result["response"] = critic_resp
    return result


def run_all(use_fixed: bool) -> list:
    label = "FIXED (A+B)" if use_fixed else "ORIGINAL"
    print(f"\n{'='*65}")
    print(f"  {label} prompts")
    print(f"{'='*65}")
    results = []
    for sc in SCENARIOS:
        r = run_scenario(sc, use_fixed)
        results.append((sc, r))
        icon = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{icon}] {sc['id']}: {sc['desc']}")
        print(f"  Expected: {sc['expected_verdict']}  |  Got: {r['got_verdict']}  |  "
              f"Flag found: {r['flag_found']}")
        print(f"  Critic: \"{r['response'][:220].replace(chr(10),' ')}...\"")
    return results


if __name__ == "__main__":
    print("Testing critic verification behaviour before and after A+B changes\n")
    print("Scenarios:")
    for sc in SCENARIOS:
        print(f"  {sc['id']}: {sc['desc']}")

    orig_results = run_all(use_fixed=False)
    fix_results  = run_all(use_fixed=True)

    orig_pass = sum(1 for _, r in orig_results if r["passed"])
    fix_pass  = sum(1 for _, r in fix_results  if r["passed"])

    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  ORIGINAL : {orig_pass}/{len(SCENARIOS)} passed")
    print(f"  FIXED A+B: {fix_pass}/{len(SCENARIOS)} passed")
    delta = fix_pass - orig_pass
    if delta > 0:
        print(f"  Improvement: +{delta} scenario(s) correctly handled")
    elif delta == 0:
        print(f"  No change in pass rate (check qualitative diff in responses)")
    else:
        print(f"  Regression: {delta} (investigate)")

    print("\nPer-scenario diff:")
    for (sc, or_), (_, fr) in zip(orig_results, fix_results):
        o = "PASS" if or_["passed"] else "FAIL"
        f = "PASS" if fr["passed"] else "FAIL"
        change = "  (same)" if o == f else f"  ← {'improved' if f == 'PASS' else 'regressed'}"
        print(f"  {sc['id']}: ORIG={o}  FIXED={f}{change}")
