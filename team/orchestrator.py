"""
ReAct Supervisor orchestrator — Supervisor-Driven Dynamic Routing.

Architecture:
  Supervisor LLM decides every action freely at every step.
  No phase state machine, no forced transitions.

  Single hard safety net:
    - researcher_calls >= MAX and analyst has never run → force CALL_ANALYST
    - MAX_ROUNDS exhausted → emergency writer fallback

  Everything else is the supervisor's call.
"""
import asyncio
import re
from datetime import datetime
from pathlib import Path

from team.agent import Agent, AgentCallError
from team.personas import ANALYST, CRITIC, RESEARCHER, WRITER
from team.skills import gather_dimension, web_search, web_scrape

REPORTS_DIR = Path.home() / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MAX_ROUNDS                  = 40  # absolute backstop — emergency writer fires if hit
MAX_RESEARCHER_CALLS        = 3   # pre-analyst exploratory ceiling
MAX_POST_ANALYST_RESEARCHER = 2   # post-analyst targeted gap-fill ceiling (independent budget)

_SUPV = "doubao-seed-2-0-pro-260215"

RESEARCH_DIMS = [
    ("market_overview",       "market size growth revenue forecast"),
    ("competitive_landscape", "competitors market share key players"),
    ("technology_trends",     "technology innovation disruption future"),
    ("regulatory_env",        "regulations compliance policy legal"),
]


# ── Supervisor prompts ────────────────────────────────────────────────────────

REACT_SYSTEM = """\
You are the GTM Intelligence Supervisor. You decide every action — there is no fixed pipeline.

TEAM:
  CALL_RESEARCHER — Alex: self-decomposes your directive into 6-8 parallel searches; retries any that find nothing
  CALL_ANALYST    — Jamie: TAM/SAM/SOM, PESTEL, Porter's Five Forces; labels [Data]/[Estimate]/[Assumption]/[N/A]
  CALL_CRITIC     — Morgan: quality review
  CALL_WRITER     — Report Writer: final GTM Intelligence Report with Competitive Battle Cards

━━━ DATA RECENCY ━━━
The current year is 2026. Prioritize 2025/2026 data. Fall back to 2024 or earlier only when unavailable.

━━━ RESEARCHER BUDGETS ━━━
You have TWO independent researcher budgets:
  Pre-analyst  (exploratory): 3 calls — used before analyst runs; covers broad dimensions.
  Post-analyst (gap-fill):    2 calls — used after analyst or critic flags specific missing data;
                                        targeted single-metric or single-company searches only.

━━━ NATURAL PROGRESSION ━━━
1. RESEARCH — initial 4-dim search runs automatically (uses 1 pre-analyst call).
   You have 2 more pre-analyst follow-up calls.
   ADVANCEMENT RULE: Call CALL_ANALYST only when workspace contains BOTH:
     (a) a cited dollar market size or revenue figure (e.g. "$9.4B", "CAGR 14%"), AND
     (b) at least one named competitor with a specific quantitative metric (revenue, market share %, user count).
   If only (a) is satisfied → one pre-analyst follow-up targeting competitive data.
   If only (b) is satisfied → one pre-analyst follow-up targeting market size.
   Once BOTH conditions are met, advance to analyst immediately.

2. CALL_ANALYST — runs full frameworks, labels every figure. Handles missing data with [N/A] or [Estimate].

3. CALL_CRITIC — mandatory before writer. One quality pass.
   After APPROVED or minor concerns → CALL_WRITER directly.
   After NEEDS_REVISION → follow CRITIC ROUTING below.

4. CALL_WRITER — final report. Done.

━━━ ANTI-LOOP RULE (STRICT) ━━━
Researcher internally retries any query that finds nothing. So [RESEARCH: UNAVAILABLE] means \
two attempts failed — the data is not publicly accessible.
- NEVER search for a metric already marked [RESEARCH: UNAVAILABLE] in the workspace.
- CALL_ANALYST immediately; it will write [N/A — data unavailable] for that metric.
If researcher returns [RESEARCH: WEAK] but has any citable finding → sufficient, advance to analyst.

━━━ CRITIC ROUTING ━━━
After [VERDICT: NEEDS_REVISION]:
  reason: logic_error  → CALL_ANALYST to fix the framework or arithmetic
  reason: missing_data, AND metric NOT UNAVAILABLE in workspace → CALL_RESEARCHER (post-analyst budget)
  reason: missing_data, AND metric IS UNAVAILABLE in workspace → CALL_ANALYST (label [N/A], do not re-search)

After [VERDICT: REJECT_DATA]:
  → CALL_RESEARCHER (post-analyst budget) to find an alternative source for the rejected figure.
  → If post-analyst budget exhausted → CALL_ANALYST to reclassify as [Assumption] or [N/A].

━━━ FOLLOW-UP SEARCH STRATEGY ━━━
When using a follow-up researcher call, choose the angle most likely to yield new data:
  Macro→Micro  — search total market first, then drill: "global cloud software 2025" → "cloud CRM segment share"
  Framework gap — target a specific missing piece: "Salesforce revenue 2025 market share" for Porter's rivalry data
  Lateral proxy — find an adjacent metric to derive from: "cloud CRM total users 2025 ARPU" to estimate revenue

━━━ ACT FORMAT ━━━
ACT: CALL_RESEARCHER | task: [broad directive covering a full dimension, e.g. "competitive landscape: top 5 cloud CRM vendors revenue pricing positioning 2025"]
ACT: CALL_ANALYST    | task: [frameworks to apply + gaps to estimate or label N/A]
ACT: CALL_CRITIC     | task: [what to focus on]
ACT: CALL_WRITER     | task: write final report

━━━ RESEARCHER TASK RULES ━━━
- Provide a DIRECTIVE (topic + angle). Researcher self-decomposes into queries automatically.
- All sources are fair game — paywalled sources (Gartner, IDC, Forrester, Statista, Mordor Intelligence, etc.)
  often expose headline figures in search snippets; researcher will cite them with (snippet only) notation.
- NAMED COMPANIES RULE: if the workspace already lists specific company names, include those names
  explicitly in the directive. Do NOT use generic phrases like "top 5 vendors" or "top N players" —
  these generate un-searchable queries. Use actual names: "Workday, BambooHR, Gusto revenue 2025".

━━━ DATA CONFLICT ━━━
Two rounds with 10×+ different figures → note in THINK, instruct analyst to use conservative figure and flag conflict.

━━━ BIAS: PROGRESS OVER PERFECTION ━━━
[N/A] and [Estimate] in a report are honest and valuable. Don't hold up the pipeline chasing data \
that isn't freely available.

Respond in the same language as the topic.\
"""

REACT_PROMPT = """\
Research topic: {topic}
Round: {rnd}/{max_rounds}
Pre-analyst researcher: {research_count}/{max_researcher_calls} | Post-analyst gap-fills: {post_analyst_count}/{max_post_analyst}{budget_warn}
Analyst called: {analyst_called} | Analyst revisions: {revision_count} | Critic called: {critic_called}

Previously searched (do NOT repeat):
{searched_queries}

WORKSPACE (oldest → newest):
{workspace}

THINK then ACT:\
"""


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_react(text: str) -> tuple[str, str, str]:
    """Returns (think_text, ACTION_KEYWORD, param)."""
    think_m = re.search(
        r'THINK\s*:\s*(.*?)(?=\nACT\s*:|\Z)', text, re.DOTALL | re.IGNORECASE)
    think = think_m.group(1).strip() if think_m else ""

    act_m = re.search(
        r'ACT\s*:\s*(CALL_RESEARCHER|CALL_ANALYST|CALL_CRITIC|CALL_WRITER|DONE)'
        r'(?:\s*\|\s*(?:queries|query|task|reason)\s*:\s*(.+?))?(?:\n|$)',
        text, re.DOTALL | re.IGNORECASE)

    if not act_m:
        for kw in ("CALL_RESEARCHER", "CALL_ANALYST", "CALL_CRITIC", "CALL_WRITER", "DONE"):
            if kw.lower().replace("_", " ") in text.lower():
                return think, kw, ""
        return think, "CALL_ANALYST", "Continue analysis with available research data."

    action = act_m.group(1).upper()
    param  = (act_m.group(2) or "").strip()[:800]
    return think, action, param


# ── Workspace helpers ─────────────────────────────────────────────────────────

def _researcher_digest(output: str) -> str:
    """Compact key-findings digest for supervisor workspace view.

    Extracts every confidence-tagged finding line and the final signal so the
    supervisor always sees what was found regardless of how long the raw output is.
    """
    lines = []
    for line in output.split('\n'):
        stripped = line.strip()
        if re.search(r'\[(Data|Estimate|Claim)\]', stripped, re.IGNORECASE) and len(stripped) > 10:
            lines.append('  • ' + stripped[:180])
    sig_match = re.search(r'\[RESEARCH[^\]]*\]', output, re.IGNORECASE)
    sig = sig_match.group(0) if sig_match else ""
    digest = '\n'.join(lines[:10])
    if digest:
        return f"KEY FINDINGS:\n{digest}\nSIGNAL: {sig}"
    return f"SIGNAL: {sig}" if sig else "(no findings)"


def _workspace_text(workspace: list) -> str:
    if not workspace:
        return "(empty — no work done yet)"
    n = len(workspace)
    parts = []
    for i, w in enumerate(workspace):
        if w["agent"] == "researcher":
            # Always show a compact digest — never truncate researcher findings
            out = _researcher_digest(w["output"])
        elif i >= n - 3:
            limit = 1000 if w["agent"] == "critic" else 700
            out = w["output"][:limit] + ("..." if len(w["output"]) > limit else "")
        else:
            limit = 180
            out = w["output"][:limit] + ("..." if len(w["output"]) > limit else "")
        sig = f"\n  └─ signal: {w['signal']}" if w.get("signal") else ""
        parts.append(
            f"[Round {w['round']}] {w['agent'].upper()}{sig}\n"
            f"  Task: {w['task'][:120]}\n"
            f"  Output: {out}")
    return "\n\n".join(parts)


def _build_ctx_for(agent_id: str, workspace: list) -> list[dict]:
    """
    Build curated extra_context for analyst / critic.
    Full outputs are passed — no truncation — to prevent context blindness.
    The lightweight [:600] fallback is retained only for the supervisor routing path.
    """
    if agent_id == "analyst":
        msgs = []
        for r in (w for w in workspace if w["agent"] == "researcher"):
            msgs.append({"role": "user",
                         "content": f"[RESEARCH Round {r['round']}]\n{r['output']}"})
        # Validator reconciliation (if it ran) — inject before critic feedback so analyst
        # knows which figures are confirmed vs conflicted vs RAG-only supplements
        for v in (w for w in workspace if w["agent"] == "validator"):
            msgs.append({"role": "user",
                         "content": f"[DATA RECONCILIATION]\n{v['output']}"})
        critics = [w for w in workspace if w["agent"] == "critic"]
        if critics:
            c = critics[-1]
            msgs.append({"role": "user",
                         "content": f"[CRITIC FEEDBACK — Round {c['round']}]\n{c['output']}"})
        analysts = [w for w in workspace if w["agent"] == "analyst"]
        if analysts:
            a = analysts[-1]
            msgs.append({"role": "user",
                         "content": f"[YOUR PREVIOUS ANALYSIS — Round {a['round']}]\n{a['output']}"})
        return msgs

    elif agent_id == "critic":
        msgs = []
        analysts = [w for w in workspace if w["agent"] == "analyst"]
        if analysts:
            a = analysts[-1]
            msgs.append({"role": "user",
                         "content": f"[ANALYST'S ANALYSIS — Round {a['round']}]\n{a['output']}"})
        for r in (w for w in workspace if w["agent"] == "researcher"):
            msgs.append({"role": "user",
                         "content": f"[RESEARCH Round {r['round']}]\n{r['output']}"})
        return msgs

    # Supervisor routing fallback — keep lightweight
    return [{"role": "user",
             "content": f"[{w['agent'].upper()} — Round {w['round']}]\n{w['output'][:600]}"}
            for w in workspace[-5:]]


def _has_findings(text: str) -> bool:
    """True if researcher output contains at least one confidence-tagged finding."""
    return bool(re.search(r'\[(Data|Estimate|Claim)\]', text, re.IGNORECASE))


# ── Data Synthesizer helpers ──────────────────────────────────────────────────

# Files to exclude from synthesis (test / non-data uploads)
_SYNTH_SKIP_FILES = re.compile(
    r'^(test_|fe_path|ng_test|test_upload|test_single)', re.IGNORECASE)

SIM_THRESHOLD = 0.42  # tuned: 100% recall on true-match pairs, 0 false noise at this value


def _extract_finding_texts(workspace: list) -> list[str]:
    """Extract all confidence-tagged finding lines from researcher workspace entries.

    Returns deduplicated list of clean strings (URLs stripped, max 220 chars each).
    """
    seen, findings = set(), []
    for w in workspace:
        if w["agent"] != "researcher":
            continue
        for line in w["output"].split('\n'):
            sl = line.strip()
            if not re.search(r'\[(Data|Estimate|Claim)\]', sl, re.IGNORECASE):
                continue
            # Strip URLs and artifacts
            clean = re.sub(r'\(https?://[^\)]+\)', '', sl)
            clean = re.sub(r'\(not provided\)', '', clean)
            clean = re.sub(r'^[-•*\s]+', '', clean).strip()[:220]
            if len(clean) < 20:
                continue
            key = clean[:80].lower()
            if key not in seen:
                seen.add(key)
                findings.append(clean)
    return findings


def _get_user_rag_chunks() -> list[dict]:
    """Return all meaningful user-namespace RAG chunks (test files excluded)."""
    try:
        from rag_mgr import _get_index
        idx = _get_index()
        col = idx._vector_store._collection
        res = col.get(where={"namespace": "user"}, include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(res["documents"], res["metadatas"]):
            fname = meta.get("filename", "")
            if _SYNTH_SKIP_FILES.match(fname):
                continue
            text = doc.strip()
            if len(text) < 40:
                continue
            chunks.append({
                "text":      text[:400],
                "filename":  fname,
                "chunk_idx": meta.get("chunk_index", 0),
            })
        return chunks
    except Exception:
        return []


def _embed_pair_match(
    findings: list[str],
    rag_chunks: list[dict],
    threshold: float = SIM_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """Embedding-based pre-filter: returns (matched_pairs, rag_supplements).

    matched_pairs  — RAG chunks whose max cosine similarity to any finding ≥ threshold.
                     Each entry includes the best-matching finding text and the sim score.
    rag_supplements — RAG chunks with max_sim < threshold (researcher didn't find this).

    Uses the already-loaded HuggingFace embedding model — no LLM calls, pure numpy.
    """
    if not findings or not rag_chunks:
        return [], rag_chunks

    try:
        import numpy as np
        from rag_mgr import _get_embed

        embed = _get_embed()
        f_embs = np.array(embed.get_text_embedding_batch(findings))   # (N, D)
        r_embs = np.array(embed.get_text_embedding_batch(
            [c["text"] for c in rag_chunks]))                          # (M, D)

        # Cosine similarity matrix (N, M)
        f_norm = f_embs / (np.linalg.norm(f_embs, axis=1, keepdims=True) + 1e-9)
        r_norm = r_embs / (np.linalg.norm(r_embs, axis=1, keepdims=True) + 1e-9)
        sim_matrix = f_norm @ r_norm.T  # (N, M)

        matched, supplements = [], []
        for j, chunk in enumerate(rag_chunks):
            col_sims = sim_matrix[:, j]
            best_f_idx = int(np.argmax(col_sims))
            max_sim = float(col_sims[best_f_idx])
            if max_sim >= threshold:
                matched.append({
                    **chunk,
                    "best_finding": findings[best_f_idx],
                    "sim":          round(max_sim, 3),
                })
            else:
                supplements.append(chunk)
        return matched, supplements
    except Exception:
        return [], rag_chunks


# ── Web helpers ───────────────────────────────────────────────────────────────

def _get_rag(topic: str) -> str:
    try:
        from rag_mgr import query_rag_multi
        return query_rag_multi([f"{topic} {dq}" for _, dq in RESEARCH_DIMS], n_per_query=3)
    except Exception:
        return ""


async def _search_with_sources(query: str) -> dict:
    results = await web_search(query, max_results=4)
    if not results:
        return {"text": "", "sources": [], "query": query}

    sources, snippets, urls = [], [], []
    for r in results[:4]:
        url     = r.get("url") or r.get("href", "")
        title   = r.get("title") or r.get("name", "")
        snippet = r.get("body") or r.get("snippet", "")
        sources.append({"url": url, "title": title[:80], "snippet": snippet[:200]})
        if url:
            urls.append(url)
        if snippet:
            snippets.append(snippet)

    scraped = await asyncio.gather(*[web_scrape(u) for u in urls[:2]]) if urls[:2] else []
    texts   = [t for t in scraped if t and len(t) > 100]
    body    = ("\n\n---\n\n".join(texts) if texts else "\n".join(snippets))[:3000]
    return {"text": body, "sources": sources, "query": query}


async def _search_with_retry(query: str) -> dict:
    result = await _search_with_sources(query)
    if len(result.get("text", "")) >= 200:
        return result

    _stop = {
        "the","a","an","in","of","for","on","at","to","with","and","or","is","are",
        "was","were","about","how","what","why","when","which","where","their","its",
        "that","this","these","those","by","from","as","be","have","has","had",
    }
    keywords = [w for w in query.split() if w.lower() not in _stop][:10]
    short_q  = " ".join(keywords)

    if short_q and short_q.lower() != query.lower():
        result2 = await _search_with_sources(short_q)
        if len(result2.get("text", "")) >= 200:
            result2["retried"] = True
            result2["original_query"] = query
            return result2

    result["limited"] = True
    return result


def _fmt_sources(sources: list) -> str:
    seen, lines = set(), []
    for s in sources:
        url = s.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        title   = s.get("title") or url[:60]
        snippet = s.get("snippet", "")[:100]
        lines.append(f"  • [{title}]({url}){': ' + snippet if snippet else ''}")
        if len(lines) >= 8:
            break
    return "\n".join(lines) if lines else "  (no URLs available)"


# ── Main research loop ────────────────────────────────────────────────────────

async def run_research(topic: str, q: asyncio.Queue) -> dict:

    react_supervisor = Agent(
        agent_id="supervisor",
        name="Supervisor",
        color="#00d4aa",
        avatar="S",
        model=_SUPV,
        temperature=0.3,
        system_prompt=REACT_SYSTEM,
    )

    researcher = Agent(**RESEARCHER)
    analyst    = Agent(**ANALYST)
    critic     = Agent(**CRITIC)
    writer     = Agent(**WRITER)

    agents_map     = {"researcher": researcher, "analyst": analyst,
                      "critic": critic,         "writer": writer}
    phase_map      = {"researcher": "research", "analyst": "analysis",
                      "critic": "critique",     "writer": "writing"}
    max_tokens_map = {"researcher": 900, "analyst": 1800, "critic": 900, "writer": 3000}
    team_messages: list[dict] = []

    loop = asyncio.get_event_loop()

    async def emit(agent, content, phase, is_think=False):
        msg = {
            "type": "team_chat",
            "msg": {
                "agent":    agent.agent_id,
                "content":  content,
                "ts":       datetime.now().strftime("%H:%M:%S"),
                "phase":    phase,
                "is_think": is_think,
            },
            "meta": {"name": agent.name, "color": agent.color, "avatar": agent.avatar},
        }
        team_messages.append(msg)
        await q.put(msg)

    async def emit_error(message: str, agent=None):
        agent = agent or react_supervisor
        err_msg = {
            "type": "team_chat",
            "msg": {
                "agent":    agent.agent_id,
                "content":  f"⚠️ **Error:** {message}",
                "ts":       datetime.now().strftime("%H:%M:%S"),
                "phase":    "error",
                "is_think": False,
            },
            "meta": {"name": agent.name, "color": "#ef4444", "avatar": "!"},
        }
        team_messages.append(err_msg)
        await q.put(err_msg)

    # ── Semantic cache ────────────────────────────────────────────────────────
    from semantic_cache import lookup as cache_lookup, store as cache_store
    cache_hit = await loop.run_in_executor(None, cache_lookup, topic)
    if cache_hit and cache_hit["hit"] == "fresh":
        await emit(react_supervisor,
            f"⚡ **Cache hit** — report from {cache_hit['age_days']} day(s) ago "
            f"(similarity {cache_hit['similarity']}).", "routing")
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in topic)[:40]
        (REPORTS_DIR / f"gtm_{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md").write_text(
            cache_hit["report"], encoding="utf-8")
        return {"report": cache_hit["report"], "topic": topic}

    stale_report = stale_entry_id = stale_age = None
    if cache_hit and cache_hit["hit"] == "stale":
        await emit(react_supervisor,
            f"📅 Previous report found ({cache_hit['age_days']} days old). "
            "Running fresh research with historical context.", "routing")
        stale_report, stale_entry_id, stale_age = (
            cache_hit["report"], cache_hit["entry_id"], cache_hit["age_days"])

    # ── State ─────────────────────────────────────────────────────────────────
    workspace:                    list[dict] = []
    rag_context                   = ""
    researcher_calls              = 0
    post_analyst_researcher_calls = 0
    analyst_called                = False
    validator_called              = False
    revision_count                = 0
    final_report                  = ""

    await emit(react_supervisor,
        f"🚀 **Starting: {topic}**\n\n"
        "I'll reason about quality at every step and decide what the team works on next.",
        "routing")

    # ── ReAct loop ────────────────────────────────────────────────────────────
    for rnd in range(1, MAX_ROUNDS + 1):

        if any(w["agent"] == "writer" for w in workspace):
            break

        # ── Determine action ──────────────────────────────────────────────────

        if researcher_calls == 0:
            # Round 1 is always the initial 4-dim search — no need to ask supervisor
            action    = "CALL_RESEARCHER"
            param     = "initial"
            think_txt = ""
            await emit(react_supervisor,
                "▶ Calling **Alex (Researcher)** — Initial broad research across 4 dimensions",
                "routing")

        elif researcher_calls >= MAX_RESEARCHER_CALLS and not analyst_called:
            # Hard safety: pre-analyst budget exhausted but analyst never ran — force it
            action    = "CALL_ANALYST"
            param     = ("Pre-analyst researcher budget exhausted. Analyse all collected data using "
                         "TAM/SAM/SOM, PESTEL, Porter's Five Forces. "
                         "Label every figure [Data], [Estimate], or [Assumption].")
            think_txt = ""
            await emit(react_supervisor,
                f"⏰ Pre-analyst researcher budget ({MAX_RESEARCHER_CALLS} calls) exhausted — routing to analyst.",
                "routing")

        else:
            # ── Supervisor decides freely ─────────────────────────────────────
            if analyst_called:
                post_remaining = MAX_POST_ANALYST_RESEARCHER - post_analyst_researcher_calls
                budget_warn = (f" ⚠️ {post_remaining} post-analyst gap-fill(s) left"
                               if post_remaining <= 1 else "")
            else:
                pre_remaining = MAX_RESEARCHER_CALLS - researcher_calls
                budget_warn = (f" ⚠️ {pre_remaining} pre-analyst call(s) left — consider advancing"
                               if pre_remaining <= 1 else "")

            critic_called = any(w["agent"] == "critic" for w in workspace)

            searched_parts = []
            for w in workspace:
                if w["agent"] == "researcher" and w.get("task") and w["task"] != "initial":
                    searched_parts.append(f"  • {w['task'][:120]}")
            searched_block = "\n".join(searched_parts) or "  (none yet)"

            try:
                react_raw = await react_supervisor.speak(
                    REACT_PROMPT.format(
                        topic=topic,
                        rnd=rnd,
                        max_rounds=MAX_ROUNDS,
                        budget_warn=budget_warn,
                        research_count=researcher_calls,
                        max_researcher_calls=MAX_RESEARCHER_CALLS,
                        post_analyst_count=post_analyst_researcher_calls,
                        max_post_analyst=MAX_POST_ANALYST_RESEARCHER,
                        analyst_called="Yes" if analyst_called else "No",
                        revision_count=revision_count,
                        critic_called="Yes" if critic_called else "No",
                        searched_queries=searched_block,
                        workspace=_workspace_text(workspace),
                    ),
                    max_tokens=600, remember=False)
            except AgentCallError as e:
                await emit_error(f"Supervisor failed: {e}")
                # Sensible fallback based on current state
                if not analyst_called:
                    action, param = "CALL_ANALYST", "Analyse all available research."
                elif not critic_called:
                    action, param = "CALL_CRITIC", "Review the analysis."
                else:
                    action, param = "CALL_WRITER", "Write the final report."
                think_txt = ""
            else:
                think_txt, action, param = _parse_react(react_raw)
                # Hard constraint 1: enforce researcher budget limits
                if action == "CALL_RESEARCHER":
                    pre_exhausted  = not analyst_called and researcher_calls >= MAX_RESEARCHER_CALLS
                    post_exhausted = analyst_called and post_analyst_researcher_calls >= MAX_POST_ANALYST_RESEARCHER
                    if pre_exhausted:
                        action = "CALL_ANALYST"
                        param  = "Pre-analyst researcher budget exhausted — proceeding to analysis."
                    elif post_exhausted:
                        action = "CALL_ANALYST"
                        param  = ("Post-analyst gap-fill budget exhausted. "
                                  "Label any remaining missing metrics as [N/A] or derive [Estimate].")
                # Hard constraint 2: critic must run after every analyst before writer
                if action == "CALL_WRITER" and analyst_called:
                    last_analyst_rnd = max(
                        (w["round"] for w in workspace if w["agent"] == "analyst"), default=0)
                    last_critic_rnd = max(
                        (w["round"] for w in workspace if w["agent"] == "critic"), default=0)
                    if last_analyst_rnd > last_critic_rnd:
                        action = "CALL_CRITIC"
                        param  = "Review the latest analyst output before the final report is written."
                # Hard constraint 3: max 2 analyst revisions — block further analyst calls only
                # CALL_CRITIC is still allowed so the final revision gets reviewed
                if revision_count >= 2 and action == "CALL_ANALYST":
                    action = "CALL_WRITER"
                    param  = ("Analyst has been revised twice. Write the final report using the "
                              "best available analysis. Label any unresolved issues inline.")
                # Hard constraint 0: run Data Synthesizer before the very first analyst call
                # if user-uploaded RAG data exists (no-op if no user docs).
                if action == "CALL_ANALYST" and not analyst_called and not validator_called:
                    has_user_docs = await loop.run_in_executor(None, _get_user_rag_chunks)
                    if has_user_docs:
                        action = "CALL_VALIDATOR"
                        param  = "Reconcile web search findings with imported knowledge base."

            if think_txt:
                await emit(react_supervisor, think_txt, "thinking", is_think=True)

            _labels = {
                "CALL_RESEARCHER": f"Calling **Alex (Researcher)** — {param}",
                "CALL_ANALYST":    f"Calling **Jamie (Analyst)** — {param}",
                "CALL_CRITIC":     f"Calling **Morgan (Critic)** — {param}",
                "CALL_WRITER":     "Calling **Report Writer** — producing final report",
                "CALL_VALIDATOR":  "Calling **Jordan (Synthesizer)** — reconciling web vs imported data",
            }
            await emit(react_supervisor,
                f"▶ {_labels.get(action, action)}", "routing")

        if action == "DONE":
            break

        # ── Execute action ────────────────────────────────────────────────────

        sig    = ""
        output = ""

        # ── VALIDATOR (Data Synthesizer) ──────────────────────────────────────
        if action == "CALL_VALIDATOR":
            from team.personas import DATA_SYNTHESIZER
            synthesizer = Agent(**DATA_SYNTHESIZER)

            # Step 1: extract all researcher findings (full list, not digest-truncated)
            all_findings = _extract_finding_texts(workspace)

            # Step 2: get user RAG chunks (test files excluded) — run in executor (sync ChromaDB call)
            rag_chunks = await loop.run_in_executor(None, _get_user_rag_chunks)

            if all_findings and rag_chunks:
                # Step 3: embedding-based pre-filter (pure numpy, no LLM)
                loop = asyncio.get_event_loop()
                matched, supplements = await loop.run_in_executor(
                    None, _embed_pair_match, all_findings, rag_chunks, SIM_THRESHOLD)

                # Step 4: build compact LLM input (only matched pairs + supplements)
                pairs_block = ""
                for p in matched:
                    pairs_block += (
                        f"\n[OVERLAP — sim={p['sim']}]\n"
                        f"  Web finding : {p['best_finding'][:220]}\n"
                        f"  RAG chunk   : [{p['filename']}] {p['text'][:220]}\n"
                    )

                supps_block = ""
                for c in supplements[:15]:   # cap at 15 to avoid context overflow
                    supps_block += f"\n[RAG-ONLY — {c['filename']}]\n  {c['text'][:200]}\n"

                await emit(synthesizer,
                    f"🔄 **Data Reconciliation** — {len(matched)} overlap pairs, "
                    f"{len(supplements)} RAG supplements\n"
                    f"Embedding threshold: {SIM_THRESHOLD} | Findings scanned: {len(all_findings)} | "
                    f"RAG chunks: {len(rag_chunks)}",
                    "validation")

                synth_input = (
                    f"Research topic: {topic}\n\n"
                    f"OVERLAP PAIRS (embedding similarity ≥ {SIM_THRESHOLD}):\n"
                    f"{pairs_block or '(none — web and RAG cover different metrics)'}\n\n"
                    f"RAG SUPPLEMENTS (imported data not covered by web search):\n"
                    f"{supps_block or '(none)'}"
                )

                try:
                    result = await synthesizer.speak(
                        synth_input, max_tokens=800, remember=False)
                except AgentCallError as e:
                    await emit_error(f"Data Synthesizer failed: {e}", synthesizer)
                    result = "[SYNTHESIS: SKIPPED — error]"
            else:
                result = (
                    f"[SYNTHESIS: SKIPPED — "
                    f"{'no researcher findings yet' if not all_findings else 'no user RAG data'}]"
                )

            await emit(synthesizer, result, "validation")
            validator_called = True
            sig = "[SYNTHESIS: COMPLETE]" if "SYNTHESIS: COMPLETE" in result else ""
            workspace.append({
                "round": rnd, "agent": "validator",
                "task": "reconcile web vs RAG", "output": result, "signal": sig,
            })
            # Immediately route to analyst (validator is not a terminal action)
            continue

        # ── WRITER ───────────────────────────────────────────────────────────
        if action == "CALL_WRITER":
            _writer_ctx = []
            for _r in (w for w in workspace if w["agent"] == "researcher"):
                _writer_ctx.append({"role": "user",
                                    "content": f"[RESEARCH Round {_r['round']}]\n{_r['output']}"})
            for w in workspace:
                if w["agent"] == "analyst":
                    _writer_ctx.append({"role": "user",
                                        "content": f"[ANALYST]\n{w['output']}"})
                elif w["agent"] == "critic":
                    _writer_ctx.append({"role": "user",
                                        "content": f"[CRITIC]\n{w['output']}"})

            rag_note = f"\n\nKNOWLEDGE BASE:\n{rag_context[:800]}" if rag_context else ""
            try:
                result = await writer.speak(
                    f"Supervisor instruction: {param}{rag_note}",
                    extra_context=_writer_ctx, max_tokens=3000)
            except AgentCallError as e:
                await emit_error(f"Report Writer failed: {e}", writer)
                result = ""
            if result:
                await emit(writer, result, "writing")
                final_report = result
                sig = "[REPORT: COMPLETE]" if re.search(r'\[REPORT', result, re.IGNORECASE) else ""
                workspace.append({
                    "round": rnd, "agent": "writer",
                    "task": param, "output": result, "signal": sig})
                await emit(react_supervisor,
                    "✅ Report written. Research pipeline complete.", "complete")
                break
            await emit_error("Writer call failed — will retry.", react_supervisor)
            continue

        # ── RESEARCHER ────────────────────────────────────────────────────────
        elif action == "CALL_RESEARCHER":
            researcher_calls += 1
            if analyst_called:
                post_analyst_researcher_calls += 1

            if researcher_calls == 1:
                # Initial parallel search across 4 dimensions
                await emit(researcher,
                    f"🔍 **Initial research: *{topic}***\n"
                    "Launching parallel searches across 4 dimensions...", "research")

                dim_tasks  = [gather_dimension(topic, dk, dq) for dk, dq in RESEARCH_DIMS]
                rag_task   = loop.run_in_executor(None, _get_rag, topic)
                dim_results, rag_ctx = await asyncio.gather(
                    asyncio.gather(*dim_tasks), rag_task)

                rag_context = rag_ctx
                summaries: dict[str, str] = {}

                async def _summarize_dim(dr: dict) -> tuple[str, str]:
                    dim, raw = dr["dimension"], dr["text"]

                    async def _do_dim_summarize(content: str) -> str:
                        return await researcher.speak(
                            f'Summarise web data about "{topic}" — {dim.replace("_", " ")}.\n\n'
                            f'{content[:3000]}\n\n'
                            "Use TEMPLATE A (Key Findings + Synthesis + Gaps + Confidence). "
                            "Start immediately with ## — no preamble.",
                            max_tokens=550, remember=False)

                    try:
                        s = await _do_dim_summarize(raw)
                    except AgentCallError as e:
                        await emit_error(f"Researcher failed on {dim}: {e}", researcher)
                        return dim, f"⚠️ Researcher error for {dim}: {e}"

                    # Semantic retry: if no citable findings, try a different search angle
                    if not _has_findings(s):
                        try:
                            alt_q_raw = await researcher.speak(
                                f'Search for "{topic} — {dim.replace("_", " ")}" found no citable data.\n'
                                f'Suggest ONE alternative search query (keywords only, max 12 words) '
                                f'to find {dim.replace("_", " ")} data for "{topic}" from a completely different angle.',
                                max_tokens=80, remember=False)
                            alt_q = (alt_q_raw.strip().split('\n')[0]
                                     .lstrip("•-0123456789.) ").strip()[:200])
                        except AgentCallError:
                            alt_q = ""

                        if alt_q:
                            alt_dr = await gather_dimension(topic, dim, alt_q)
                            if alt_dr.get("text"):
                                try:
                                    s2 = await _do_dim_summarize(alt_dr["text"])
                                    if _has_findings(s2):
                                        s = s2  # retry found citable data
                                except AgentCallError:
                                    pass  # keep original s

                    return dim, s

                dim_summaries = await asyncio.gather(*[_summarize_dim(dr) for dr in dim_results])
                for dim, s in dim_summaries:
                    summaries[dim] = s
                    await emit(researcher,
                        f"✅ **{dim.replace('_', ' ').title()}**\n\n{s}", "research")

                if rag_ctx:
                    summaries["knowledge_base"] = rag_ctx
                    await emit(researcher,
                        f"📚 **Knowledge Base** ({len(rag_ctx.split())} words):\n{rag_ctx[:400]}...",
                        "research")

                output = "\n\n".join(
                    f"**{k.replace('_', ' ').title()}**:\n{v}"
                    for k, v in summaries.items())

                try:
                    sig_raw = await researcher.speak(
                        f"Review summaries for \"{topic}\" and emit a signal:\n\n{output[:2000]}\n\n"
                        "Use TEMPLATE C. Reply with ONLY the signal line.",
                        max_tokens=200, remember=False)
                except AgentCallError as e:
                    sig_raw = "[RESEARCH: WEAK | gaps: signal check failed]"
                    await emit_error(f"Researcher signal check failed: {e}", researcher)
                sig_match = re.search(r'\[RESEARCH[^\]]*\]', sig_raw, re.IGNORECASE)
                sig = sig_match.group(0) if sig_match else "[RESEARCH: WEAK | gaps: signal unclear]"
                output += f"\n\n{sig}"
                await emit(researcher,
                    f"📊 Initial research complete. Signal: `{sig}`", "research")

                task_label = "initial"

            else:
                # Follow-up search: researcher self-decomposes supervisor's directive
                directive = param.strip() or topic

                last_critic = next(
                    (w for w in reversed(workspace) if w["agent"] == "critic"), None)
                critic_ctx = ""
                if last_critic and workspace and workspace[-1]["agent"] == "critic":
                    critic_ctx = f"\nCritic concern: {last_critic['output'][-400:]}\n"

                # Step 1: researcher decomposes directive into specific sub-queries
                try:
                    decompose_raw = await researcher.speak(
                        f"Research topic: {topic}\n"
                        f"Research directive: {directive}\n\n"
                        f"List specific web search queries to cover this directive.\n"
                        f"ENTITY COVERAGE RULE (mandatory):\n"
                        f"  1. Count every distinct entity in the directive: each named company, "
                        f"industry, or specific metric counts as one entity.\n"
                        f"  2. Generate EXACTLY ONE query per entity — the single most searchable "
                        f"query for it (e.g. 'Workday revenue 2025'). "
                        f"Never generate 2 queries for the same entity.\n"
                        f"  3. Add market-level queries (size, CAGR) ONLY if the directive "
                        f"explicitly requests market size or growth rate data.\n"
                        f"  4. Total queries = number of entities + market queries (if requested). "
                        f"No arbitrary cap — ensure every entity is covered.\n"
                        f"Example: directive names 6 companies → 6 queries, one per company.\n"
                        f"Include '2025' or '2026' in queries where current data is needed.\n"
                        f"Format: one search query per line, keywords only (max 12 words each), "
                        f"no bullets or numbers.",
                        max_tokens=400, remember=False)
                    queries = [
                        ln.strip().lstrip("•-0123456789.) ")
                        for ln in decompose_raw.split("\n")
                        if ln.strip() and not ln.strip().startswith("#") and len(ln.strip()) > 5
                    ][:16]  # raised from 8 → 16 to accommodate large entity lists
                except AgentCallError:
                    queries = []
                if not queries:
                    queries = [directive]

                n = len(queries)
                await emit(researcher,
                    f"🔍 **Search #{researcher_calls} — {n} queries for: _{directive}_**\n\n" +
                    "\n".join(f"  • *{q}*" for q in queries), "research")

                # Step 2: run queries in batches of 4 (parallel within each batch)
                all_sr_list: list[dict] = []
                for i in range(0, len(queries), 4):
                    batch = queries[i:i + 4]
                    batch_results = await asyncio.gather(*[_search_with_retry(q) for q in batch])
                    all_sr_list.extend(batch_results)

                async def _summarize_query(sr: dict) -> str:
                    query   = sr["query"]
                    text    = sr.get("text", "")
                    sources = _fmt_sources(sr.get("sources", []))
                    notes   = ""
                    if sr.get("limited"):
                        notes = "\n\n⚠️ Limited data. Mark ⚠️ LIMITED SOURCE."
                    if not text:
                        notes = "\n\n❌ Zero results."
                    if not text and not sr.get("sources"):
                        # No results at all — go straight to semantic retry
                        text, sources, notes = "", "", ""

                    async def _do_query_summarize(q: str, t: str, src: str,
                                                  extra: str = "") -> str:
                        return await researcher.speak(
                            f"Research task: {q}{critic_ctx}\n\n"
                            f"Sources:\n{src}\n\n"
                            f"Content:\n{t[:1800]}{extra}\n\n"
                            "Use TEMPLATE B (## 🔍 — <topic> / **Found** / **Not found** / "
                            "**Plausibility** / **Summary**). "
                            "Tag each finding [Data], [Estimate], or [Claim]. "
                            "Cite source URLs. Prefer 2025 sources; note year for all figures. "
                            "End with signal.",
                            max_tokens=550, remember=False)

                    s = ""
                    if text or sources:
                        try:
                            s = await _do_query_summarize(query, text, sources, notes)
                        except AgentCallError as e:
                            await emit_error(f"Researcher failed on '{query}': {e}", researcher)
                            s = (f"## 🔍 — {query}\n\n**Found:** Error\n\n"
                                 f"[RESEARCH: WEAK | gaps: summarisation error]")
                            return s

                    # Semantic retry: if no citable findings, try a different search angle
                    if not _has_findings(s):
                        try:
                            alt_q_raw = await researcher.speak(
                                f'Search for "{query}" found no citable data.\n'
                                f'Suggest ONE alternative search query (keywords only, max 12 words) '
                                f'to find this from a completely different angle.',
                                max_tokens=60, remember=False)
                            alt_q = (alt_q_raw.strip().split('\n')[0]
                                     .lstrip("•-0123456789.) ").strip()[:200])
                        except AgentCallError:
                            alt_q = ""

                        if alt_q:
                            alt_sr = await _search_with_retry(alt_q)
                            if alt_sr.get("text"):
                                alt_src = _fmt_sources(alt_sr.get("sources", []))
                                try:
                                    s2 = await _do_query_summarize(alt_q, alt_sr["text"], alt_src)
                                    if _has_findings(s2):
                                        return s2  # retry found citable data
                                except AgentCallError:
                                    pass

                        # Both attempts exhausted — return definitive UNAVAILABLE
                        return (f"## 🔍 — {query}\n\n**Found:** Nothing after retry\n\n"
                                f"**Not found:** {query}\n\n**Plausibility:** N/A\n\n"
                                f"**Summary:** No citable data found after two search attempts.\n\n"
                                f"[RESEARCH: UNAVAILABLE | data: {query}]")

                    return s

                # Step 3: summarise in serial batches of 4 (mirrors web search batching)
                query_summaries: list[str] = []
                for i in range(0, len(all_sr_list), 4):
                    batch_sums = await asyncio.gather(
                        *[_summarize_query(sr) for sr in all_sr_list[i:i + 4]])
                    query_summaries.extend(batch_sums)

                all_outputs, sig = [], ""
                for s in query_summaries:
                    await emit(researcher, s, "research")
                    all_outputs.append(s)
                    sm = re.search(r'\[RESEARCH[^\]]*\]', s, re.IGNORECASE)
                    if sm:
                        sig = sm.group(0)

                output = "\n\n".join(all_outputs)
                if not output.strip():
                    output = (f"No data found for directive: {directive}. "
                              "[RESEARCH: UNAVAILABLE | data: all queries returned empty]")
                    await emit(researcher,
                        f"⚠️ No results for any query in this dimension.\n\n"
                        f"`[RESEARCH: UNAVAILABLE | data: {directive[:80]}]`",
                        "research")

        # ── ANALYST / CRITIC ──────────────────────────────────────────────────
        else:
            agent_id   = {"CALL_ANALYST": "analyst", "CALL_CRITIC": "critic"}[action]
            agent      = agents_map[agent_id]
            exec_phase = phase_map[agent_id]
            ctx        = _build_ctx_for(agent_id, workspace)

            extra = ""
            if agent_id == "analyst":
                if stale_report and researcher_calls <= 2:
                    extra += (f"\n\nHISTORICAL CONTEXT (prev report, {stale_age}d old):\n"
                              f"{stale_report[:1500]}")
                if rag_context:
                    extra += f"\n\nKNOWLEDGE BASE:\n{rag_context[:800]}"

            try:
                result = await agent.speak(
                    f"Supervisor instruction: {param}{extra}",
                    extra_context=ctx,
                    max_tokens=max_tokens_map[agent_id],
                    remember=False)
                await emit(agent, result, exec_phase)
                output = result
            except AgentCallError as e:
                err_txt = f"{agent.name} call failed: {e}"
                await emit_error(err_txt, agent)
                output = f"[ERROR: {err_txt}]"
                workspace.append({
                    "round": rnd, "agent": agent_id,
                    "task": param, "output": output, "signal": "",
                })
                continue

            if agent_id == "analyst":
                am = re.search(r'\[ANALYSIS[^\]]*\]', output, re.IGNORECASE)
                sig = am.group(0) if am else ""
                if analyst_called:
                    # Only count as a quality revision when critic gave feedback before this call.
                    # Researcher → Analyst (data incorporation) does not consume a revision slot.
                    last_agent = next(
                        (w["agent"] for w in reversed(workspace)
                         if w["agent"] != "researcher"), None)
                    if last_agent == "critic":
                        revision_count += 1
                analyst_called = True
            else:  # critic
                cm = re.search(r'\[VERDICT[^\]]*\]', output, re.IGNORECASE)
                sig = cm.group(0) if cm else ""

        workspace.append({
            "round":  rnd,
            "agent":  "researcher" if action == "CALL_RESEARCHER" else agent_id,
            "task":   ("initial" if researcher_calls == 1 else param)
                      if action == "CALL_RESEARCHER" else param,
            "output": output,
            "signal": sig,
        })

    # ── Emergency writer fallback ─────────────────────────────────────────────
    if not final_report and any(w["agent"] == "analyst" for w in workspace):
        await emit(react_supervisor,
            "⚠️ Loop exhausted — emergency writer call.", "routing")
        ctx = [{"role": "user",
                "content": f"[{w['agent'].upper()}]\n{w['output'][:600]}"}
               for w in workspace[-6:]]
        rag_note = f"\n\nKNOWLEDGE BASE:\n{rag_context[:800]}" if rag_context else ""
        try:
            emergency_report = await writer.speak(
                f"Write the complete GTM Intelligence Report for: {topic}. "
                f"Use all available research and analysis data.{rag_note}",
                extra_context=ctx, max_tokens=3000)
            final_report = emergency_report
            await emit(writer, emergency_report, "writing")
            workspace.append({
                "round": MAX_ROUNDS, "agent": "writer",
                "task": "emergency fallback", "output": emergency_report, "signal": ""})
        except AgentCallError as e:
            await emit_error(f"Emergency writer also failed: {e}")

    if not final_report:
        writer_ws = [w for w in workspace if w["agent"] == "writer"]
        final_report = writer_ws[-1]["output"] if writer_ws else "Report generation incomplete."

    # ── Summary ───────────────────────────────────────────────────────────────
    counts: dict[str, int] = {}
    for w in workspace:
        counts[w["agent"]] = counts.get(w["agent"], 0) + 1
    stats = " · ".join(f"{a} ×{n}" for a, n in counts.items())
    await emit(react_supervisor,
        f"🎉 **Complete — {topic}**\n\n{len(workspace)} rounds · {stats}", "complete")

    safe   = "".join(c if c.isalnum() or c in "-_ " else "_" for c in topic)[:40]
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    (REPORTS_DIR / f"gtm_{safe}_{ts_str}.md").write_text(final_report, encoding="utf-8")
    await loop.run_in_executor(None, lambda: cache_store(topic, topic, final_report, stale_entry_id))
    return {"report": final_report, "topic": topic, "team_messages": team_messages}


# ── Document analysis ─────────────────────────────────────────────────────────

async def run_doc_analysis(doc_text: str, filename: str, q: asyncio.Queue) -> dict:
    analyst    = Agent(**ANALYST)
    critic     = Agent(**CRITIC)
    supervisor = Agent(
        agent_id="supervisor", name="Supervisor",
        color="#00d4aa", avatar="S",
        model=_SUPV, temperature=0.3,
        system_prompt=REACT_SYSTEM,
    )

    async def emit(agent, content, phase, is_think=False):
        await q.put({
            "type": "team_chat",
            "msg": {
                "agent": agent.agent_id, "content": content,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "phase": phase, "is_think": is_think,
            },
            "meta": {"name": agent.name, "color": agent.color, "avatar": agent.avatar},
        })

    await emit(supervisor,
        f"📋 Document received: **{filename}**. Jamie analyses, Morgan reviews.", "routing")

    rag_ctx = ""
    try:
        from rag_mgr import query_rag
        rag_ctx = query_rag(doc_text[:500], n_results=3)
    except Exception:
        pass

    try:
        analysis = await analyst.speak(
            f"Analyse: **{filename}**\n\n{doc_text[:7000]}"
            + (f"\n\nKNOWLEDGE BASE:\n{rag_ctx[:800]}" if rag_ctx else "")
            + "\n\n## Overview\n## Key Findings\n## Strategic Implications\n## Recommendations",
            max_tokens=1500)
        await emit(analyst, analysis, "analysis")
    except AgentCallError as e:
        await emit(supervisor, f"❌ Analyst failed for '{filename}': {e}", "error")
        analysis = f"[ERROR: Analyst call failed: {e}]"
        await emit(analyst, analysis, "analysis")

    try:
        critique = await critic.speak(
            f"Jamie analysed '{filename}':\n{analysis}\n\nIdentify 2 key gaps. Under 200 words.",
            max_tokens=300)
        await emit(critic, critique, "critique")
    except AgentCallError as e:
        await emit(supervisor, f"❌ Critic failed for '{filename}': {e}", "error")
        critique = f"[ERROR: Critic call failed: {e}]"
        await emit(critic, critique, "critique")

    await emit(supervisor, "✅ Document analysis complete.", "complete")
    return {
        "report": f"{analysis}\n\n---\n\n**Quality Review (Morgan):**\n{critique}",
        "topic":  f"Analysis: {filename}",
    }
