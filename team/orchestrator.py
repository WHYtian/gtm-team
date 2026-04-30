"""
ReAct Supervisor orchestrator.

Supervisor is the intelligent brain — at every step:
  1. THINKS: reasons about current state, quality gaps, what's needed next
  2. ACTS:   calls one of four tools (researcher / analyst / critic / writer)

No hardcoded routing. Supervisor's LLM decides everything.
Safeguard: after MAX_REVISION_CYCLES analyst revisions, force writer.
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

MAX_ROUNDS          = 30
MAX_REVISION_CYCLES = 6  # analyst can be called this many extra times after first
MAX_RESEARCHER_CALLS = 5  # hard cap — supervisor must proceed to analyst after this

_SUPV = "doubao-seed-2-0-pro-260215"   # upgraded model for deep orchestration reasoning

RESEARCH_DIMS = [
    ("market_overview",       "market size growth revenue forecast"),
    ("competitive_landscape", "competitors market share key players"),
    ("technology_trends",     "technology innovation disruption future"),
    ("regulatory_env",        "regulations compliance policy legal"),
]

# ── Supervisor ReAct system prompt ────────────────────────────────────────────

REACT_SYSTEM = """\
You are the GTM Intelligence Supervisor. Orchestrate a research team to produce a \
high-quality, data-rich GTM Intelligence Report.

TEAM TOOLS:
  CALL_RESEARCHER — Alex: parallel web searches for market data, statistics, companies
  CALL_ANALYST    — Jamie: applies TAM/SAM/SOM, PESTEL, Porter's Five Forces
  CALL_CRITIC     — Morgan: rigorously challenges claims, identifies unsupported data
  CALL_WRITER     — Report Writer: produces the final structured GTM Intelligence Report

AT EACH STEP output in this exact format:
THINK: [2-4 sentences — what's done, specific gaps, why this action]
ACT: CALL_RESEARCHER | queries: q1 || q2 || q3
  OR CALL_ANALYST    | task: [what to analyze, which frameworks]
  OR CALL_CRITIC     | task: [what to critique]
  OR CALL_WRITER     | task: [write final report]
  OR DONE            | reason: [only if report already written]

RESEARCHER CALL RULES:
- First call ever: ACT: CALL_RESEARCHER | query: initial
- Follow-up calls: DECOMPOSE gaps into 2-4 parallel sub-queries using the || separator.
  ACT: CALL_RESEARCHER | queries: sub-query-1 || sub-query-2 || sub-query-3

⚠️ PARALLEL QUERY FORMAT — STRICTLY REQUIRED:
  CORRECT:   queries: SaaS CRM market size 2024 Gartner || top 5 CRM vendors market share || CRM CAGR 2024-2030
  WRONG:     queries: SaaS CRM market size 2024, top 5 vendors, CAGR forecast
  WRONG:     queries: What is the market size of SaaS CRM including top vendors and growth rate?
  Using commas or natural language means only ONE search fires — you waste the entire call.

QUERY WRITING RULES (every sub-query):
- Keywords only — metric + geography + year + company/source names — max 12 words
  ✓ "solid state battery market size 2024 2030 forecast"
  ✓ "Toyota Samsung solid state battery pilot production 2024"
  ✓ "solid electrolyte battery cost per kWh 2024"
  ✗ "What is the size of the solid-state battery market?" (question format — bad)
  ✗ "BloombergNEF solid state battery report 2024" (paywalled — won't work)
- For verification: [disputed claim] site:news/company/wiki
  ✓ "Toyota solid state battery 2027 launch EV production"

SEARCH SOURCE RULES — CRITICAL:
- Target ONLY freely accessible sources: Wikipedia, company press releases, news articles (Reuters, Bloomberg news — NOT Bloomberg terminal), government sites, NGO reports, industry association publications, vendor websites.
- Do NOT name paywalled databases in queries: Bloomberg Intelligence, Statista, S&P Global, Gartner, Forrester, McKinsey, IDC, Wood Mackenzie, IHS Markit — these will return zero results.
- If you need a market figure, search "[topic] market size [year] report" or "[topic] [metric] forecast [year] industry".

RESEARCHER BUDGET: Maximum 5 researcher calls total.
- After 5 calls, STOP and move to CALL_ANALYST — accept best available data.
- Never search the same data point more than twice with different source names.
  Check "Previously searched queries" in the prompt before each call.

DATA UNAVAILABILITY RULE:
- If a data point appears in "Previously searched queries" 2+ times with no result, it does NOT exist in free sources.
- When researcher returns [RESEARCH: UNAVAILABLE], accept a proxy/estimate — do NOT search again.
- Accept imperfect data; an analysis with clearly-labelled estimates beats an empty report.
- PIVOT RULE: once a metric fails 2 searches, do NOT rephrase and retry — spend remaining researcher calls on DIFFERENT dimensions (competitors, use-cases, pricing, geography). A report with 3 solid data pillars beats one that spent all 5 calls chasing 1 missing number.
- DATA CONFLICT ALERT: if two researcher rounds return figures for the same metric that differ by 10× or more (e.g. $5M vs $5B, or $30B vs $462B), flag this in your THINK step and instruct the analyst to use the more conservative, better-sourced figure and label the conflict explicitly.

POST-ANALYST RULE (important):
- Once CALL_ANALYST has been called, do NOT call CALL_RESEARCHER for general data gaps.
- The only exception: Critic issues [VERDICT: REJECT_DATA] for a SPECIFIC figure.
  In that case use the critic's "search:" query exactly — one targeted CALL_RESEARCHER only.
- After that single verification call, return to CALL_ANALYST or CALL_WRITER.

CRITIC FEEDBACK RULES:
- [VERDICT: APPROVED] → proceed to CALL_WRITER
- [VERDICT: NEEDS_REVISION] → CALL_ANALYST with specific revision instructions
- [VERDICT: REJECT_DATA] → one targeted CALL_RESEARCHER using the critic's search query, then CALL_ANALYST
- Do NOT call CALL_RESEARCHER after NEEDS_REVISION — that means logic, not missing data.

QUALITY STANDARDS before CALL_WRITER:
- Research: market sizes, CAGR %, named companies with data, source URLs (estimates labelled)
- Analysis: frameworks applied with evidence, clear strategic conclusions
- Critic's major concerns addressed or revision limit reached

THINK: 2-4 sentences. Respond in the same language as the topic.\
"""

REACT_PROMPT = """\
Research topic: {topic}
Round: {rnd}/{max_rounds}
Analyst called: {analyst_called} | Analyst revisions: {revision_count} (max: {max_revision_cycles})
Researcher calls: {research_count}/{max_researcher_calls}{researcher_budget_warning}

Previously searched queries (do NOT repeat these):
{searched_queries}

WORKSPACE (oldest → newest):
{workspace}

Your assessment and next action:\
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
        # Fallback: infer from keywords
        for kw in ("CALL_RESEARCHER", "CALL_ANALYST", "CALL_CRITIC", "CALL_WRITER", "DONE"):
            if kw.lower().replace("_", " ") in text.lower():
                return think, kw, ""
        return think, "CALL_ANALYST", "Continue with the analysis using research data."

    action = act_m.group(1).upper()
    param  = (act_m.group(2) or "").strip()[:800]
    return think, action, param


def _workspace_text(workspace: list) -> str:
    if not workspace:
        return "(empty — no work done yet)"
    n = len(workspace)
    parts = []
    for i, w in enumerate(workspace):
        if i >= n - 3:
            # Show more of recent entries; critic verdict must not be cut off
            limit = 1000 if w["agent"] == "critic" else 700
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
    Deliberately excludes each agent's own previous outputs to prevent repetition.
    """
    if agent_id == "analyst":
        msgs = []
        # ALL researcher rounds — first round has 4-dim baseline, later rounds fill gaps
        # Older rounds get shorter budget; most recent gets full 2000 chars
        researchers = [w for w in workspace if w["agent"] == "researcher"]
        if researchers:
            for i, r in enumerate(researchers):
                limit = 2000 if i == len(researchers) - 1 else 600
                msgs.append({"role": "user",
                             "content": f"[RESEARCH Round {r['round']}]\n{r['output'][:limit]}"})
        # Critic's most recent verdict — full output so analyst knows exactly what to fix
        critics = [w for w in workspace if w["agent"] == "critic"]
        if critics:
            c = critics[-1]
            msgs.append({"role": "user",
                         "content": f"[CRITIC FEEDBACK — Round {c['round']}]\n{c['output'][:1200]}"})
        # Analyst's own last analysis — so it knows what to revise (but NOT as assistant role)
        analysts = [w for w in workspace if w["agent"] == "analyst"]
        if analysts:
            a = analysts[-1]
            msgs.append({"role": "user",
                         "content": f"[YOUR PREVIOUS ANALYSIS — Round {a['round']}]\n{a['output'][:1000]}"})
        return msgs

    elif agent_id == "critic":
        msgs = []
        # Analyst's current analysis — full output (this is what critic evaluates)
        analysts = [w for w in workspace if w["agent"] == "analyst"]
        if analysts:
            a = analysts[-1]
            msgs.append({"role": "user",
                         "content": f"[ANALYST'S CURRENT ANALYSIS — Round {a['round']}]\n{a['output'][:2500]}"})
        # All researcher rounds so critic can cross-check contradictory figures
        researchers = [w for w in workspace if w["agent"] == "researcher"]
        if researchers:
            for i, r in enumerate(researchers):
                limit = 800 if i == len(researchers) - 1 else 300
                msgs.append({"role": "user",
                             "content": f"[RESEARCH Round {r['round']}]\n{r['output'][:limit]}"})
        # NOTE: deliberately NOT including critic's own previous verdicts → prevents repetition
        return msgs

    # Fallback
    return [{"role": "user",
             "content": f"[{w['agent'].upper()} — Round {w['round']}]\n{w['output'][:600]}"}
            for w in workspace[-5:]]


# ── Web helpers ───────────────────────────────────────────────────────────────

def _get_rag(topic: str) -> str:
    try:
        from rag_mgr import query_rag_multi
        return query_rag_multi([f"{topic} {dq}" for _, dq in RESEARCH_DIMS], n_per_query=3)
    except Exception:
        return ""


async def _search_with_sources(query: str) -> dict:
    """Search and return {text, sources, query}. Sources carry URL + title + snippet."""
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
    """
    Try query; if < 200 chars back, reformulate once (strip stop words, shorten).
    On second failure mark as limited so researcher can flag it in the report.
    """
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
    """Format source list for researcher prompt."""
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
    max_tokens_map = {"researcher": 900, "analyst": 1000, "critic": 900, "writer": 3000}
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
        """Send an error bubble into the team chat area."""
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
    workspace:       list[dict] = []
    rag_context      = ""
    researcher_calls = 0
    revision_count   = 0
    analyst_called   = False
    final_report     = ""

    await emit(react_supervisor,
        f"🚀 **Starting: {topic}**\n\n"
        "I'll reason about quality at every step and decide what the team works on next.",
        "routing")

    # ── ReAct loop ────────────────────────────────────────────────────────────
    for rnd in range(1, MAX_ROUNDS + 1):

        writer_done = any(w["agent"] == "writer" for w in workspace)

        # Force analyst when researcher budget exhausted but analyst hasn't run yet
        force_analyst = (
            researcher_calls >= MAX_RESEARCHER_CALLS
            and not analyst_called
            and not writer_done
        )

        # Force writer when revision cap hit or near round limit.
        # Researcher budget saturation alone does NOT force write — critic must run at least once first.
        critic_called = any(w["agent"] == "critic" for w in workspace)
        force_write = (
            not writer_done and analyst_called and (
                revision_count >= MAX_REVISION_CYCLES or
                rnd >= MAX_ROUNDS - 2 or
                (researcher_calls >= MAX_RESEARCHER_CALLS and critic_called)
            )
        )

        if force_analyst:
            think_txt = ""
            action    = "CALL_ANALYST"
            param     = ("Researcher call budget exhausted. Analyse all collected research data "
                         "using TAM/SAM/SOM, PESTEL, and Porter's Five Forces.")
            await emit(react_supervisor,
                f"⏰ **Round {rnd}:** Researcher budget ({MAX_RESEARCHER_CALLS} calls) exhausted — "
                "routing to analyst.", "routing")
        elif force_write:
            think_txt = ""
            action    = "CALL_WRITER"
            param     = ("Write the final GTM Intelligence Report using all collected research "
                         "and analysis. The analyst has revised sufficiently.")
            reason = (f"Analyst revised {revision_count} times" if revision_count >= MAX_REVISION_CYCLES
                      else f"researcher budget saturated" if researcher_calls >= MAX_RESEARCHER_CALLS
                      else f"approaching round limit ({rnd}/{MAX_ROUNDS})")
            await emit(react_supervisor,
                f"⏰ **Round {rnd}:** {reason} — proceeding to final report.", "routing")
        else:
            # Budget warning when running low and analyst not yet called
            budget_remaining = MAX_RESEARCHER_CALLS - researcher_calls
            budget_warn = (f" — ⚠️ ONLY {budget_remaining} CALL(S) LEFT, move to analyst soon"
                           if budget_remaining <= 1 and not analyst_called else "")

            # Build list of already-searched queries so supervisor avoids repeating them
            searched_parts = []
            for w in workspace:
                if w["agent"] == "researcher" and w.get("task") and w["task"] != "initial":
                    for sq in w["task"].split("||"):
                        sq = sq.strip()
                        if sq:
                            searched_parts.append(f"  • {sq[:120]}")
            searched_block = "\n".join(searched_parts) if searched_parts else "  (none yet — first follow-up)"

            # Ask Supervisor: THINK + ACT
            try:
                react_raw = await react_supervisor.speak(
                    REACT_PROMPT.format(
                        topic=topic,
                        rnd=rnd,
                        max_rounds=MAX_ROUNDS,
                        analyst_called="Yes" if analyst_called else "No",
                        revision_count=revision_count,
                        max_revision_cycles=MAX_REVISION_CYCLES,
                        research_count=researcher_calls,
                        max_researcher_calls=MAX_RESEARCHER_CALLS,
                        researcher_budget_warning=budget_warn,
                        searched_queries=searched_block,
                        workspace=_workspace_text(workspace),
                    ),
                    max_tokens=500, remember=False)
            except AgentCallError as e:
                await emit_error(f"Supervisor failed: {e}")
                # Fallback routing based on current state
                if not analyst_called and workspace:
                    react_raw = "THINK: Supervisor error — falling back.\nACT: CALL_ANALYST | task: Analyse all available research data."
                elif analyst_called and not any(w["agent"] == "writer" for w in workspace):
                    react_raw = "THINK: Supervisor error — proceeding to report.\nACT: CALL_WRITER | task: Write the final report with available data."
                else:
                    await emit_error("Supervisor unrecoverable — stopping research.")
                    break

            think_txt, action, param = _parse_react(react_raw)

            # Emit THINK (supervisor reasoning — styled differently in frontend)
            if think_txt:
                await emit(react_supervisor,
                    think_txt, "thinking", is_think=True)

        # Emit ACT decision (first researcher call always does initial 4-dim search)
        researcher_display = (
            "Calling **Alex (Researcher)** — Initial broad research across 4 dimensions"
            if action == "CALL_RESEARCHER" and researcher_calls == 0
            else f"Calling **Alex (Researcher)** — {param[:180]}"
        )
        action_desc = {
            "CALL_RESEARCHER": researcher_display,
            "CALL_ANALYST":    f"Calling **Jamie (Analyst)** — {param[:180]}",
            "CALL_CRITIC":     f"Calling **Morgan (Critic)** — {param[:180]}",
            "CALL_WRITER":     "Calling **Report Writer** — producing final report",
            "DONE":            f"Research complete — {param[:180]}",
        }
        await emit(react_supervisor,
            f"▶ {action_desc.get(action, action)}", "routing")

        if action == "DONE":
            break

        if action == "CALL_WRITER":
            # Give writer: analyst full output + critic full output + researcher summaries
            _writer_ctx = []
            for w in workspace:
                if w["agent"] == "analyst":
                    _writer_ctx.append({"role": "user",
                                        "content": f"[ANALYST]\n{w['output'][:1500]}"})
                elif w["agent"] == "critic":
                    _writer_ctx.append({"role": "user",
                                        "content": f"[CRITIC]\n{w['output'][:800]}"})
            # Add first researcher round (baseline) and last researcher round (gap-fill)
            _rs = [w for w in workspace if w["agent"] == "researcher"]
            if _rs:
                _writer_ctx.insert(0, {"role": "user",
                                       "content": f"[RESEARCH BASELINE]\n{_rs[0]['output'][:800]}"})
            if len(_rs) > 1:
                _writer_ctx.insert(1, {"role": "user",
                                       "content": f"[RESEARCH LATEST]\n{_rs[-1]['output'][:600]}"})
            ctx = _writer_ctx
            rag_note = f"\n\nKNOWLEDGE BASE:\n{rag_context[:800]}" if rag_context else ""
            try:
                result = await writer.speak(
                    f"Supervisor instruction: {param}{rag_note}",
                    extra_context=ctx, max_tokens=3000)
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
            # Writer failed — try once more next round via emergency fallback
            await emit_error("Writer call failed — will retry after loop.", react_supervisor)

        # Map action → agent_id
        agent_id = {
            "CALL_RESEARCHER": "researcher",
            "CALL_ANALYST":    "analyst",
            "CALL_CRITIC":     "critic",
        }.get(action)

        if not agent_id:
            await emit(react_supervisor, f"⚠️ Unknown action `{action}`. Stopping.", "routing")
            break

        phase = phase_map[agent_id]
        agent = agents_map[agent_id]

        # ── Execute agent ─────────────────────────────────────────────────────

        if agent_id == "researcher":
            researcher_calls += 1

            if researcher_calls == 1:
                # Initial parallel search across all 4 dimensions
                await emit(researcher,
                    f"🔍 **Initial research: *{topic}***\n"
                    "Launching parallel searches across 4 dimensions...", "research")

                dim_tasks  = [gather_dimension(topic, dk, dq) for dk, dq in RESEARCH_DIMS]
                rag_task   = loop.run_in_executor(None, _get_rag, topic)
                dim_results, rag_ctx = await asyncio.gather(
                    asyncio.gather(*dim_tasks), rag_task)

                rag_context = rag_ctx
                summaries: dict[str, str] = {}

                # Parallelize all 4 dimension LLM summarization calls
                async def _summarize_dim(dr: dict) -> tuple[str, str]:
                    dim, raw = dr["dimension"], dr["text"]
                    try:
                        s = await researcher.speak(
                            f'Summarise web data about "{topic}" — {dim.replace("_", " ")}.\n\n'
                            f'{raw[:3000]}\n\n'
                            "Use TEMPLATE A (table + summary + confidence). "
                            "Start immediately with ## — no preamble.",
                            max_tokens=550, remember=False)
                    except AgentCallError as e:
                        await emit_error(f"Researcher failed on {dim}: {e}", researcher)
                        s = f"⚠️ Researcher error for {dim}: {e}"
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
                # Extract clean signal line even if reasoning model adds extra text
                sig_match = re.search(r'\[RESEARCH[^\]]*\]', sig_raw, re.IGNORECASE)
                sig = sig_match.group(0) if sig_match else "[RESEARCH: WEAK | gaps: signal unclear]"
                output += f"\n\n{sig}"
                await emit(researcher,
                    f"📊 Initial research complete. Signal: `{sig}`", "research")

            else:
                # Parse multi-query param: "q1 || q2 || q3" or single query
                # is_verify branch removed — unified gap-fill handles both cases;
                # multi-query must work even when previous step was critic (REJECT_DATA flow)
                raw_queries = [q.strip() for q in param.split("||") if q.strip()]
                queries     = raw_queries[:4] if raw_queries else [param]

                # If previous workspace step was a critic REJECT_DATA, attach critic context
                last_critic = next(
                    (w for w in reversed(workspace) if w["agent"] == "critic"), None)
                critic_ctx  = ""
                if last_critic and workspace and workspace[-1]["agent"] == "critic":
                    critic_ctx = f"\nCritic concern: {last_critic['output'][-400:]}\n"

                n     = len(queries)
                label = f"{n} parallel sub-searches" if n > 1 else "targeted search"
                await emit(researcher,
                    f"🔍 **Search #{researcher_calls} — {label}:**\n\n" +
                    "\n".join(f"  • *{q}*" for q in queries), "research")

                # Web searches run in parallel
                sr_list = await asyncio.gather(*[_search_with_retry(q) for q in queries])

                # Per-query LLM summarization — each sub-search gets its own call and bubble
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
                        return (f"## 🔍 — {query}\n\n### Findings\n- ⚠️ No public data\n\n"
                                f"### Summary\nNo results found.\n\n"
                                f"[RESEARCH: UNAVAILABLE | data: {query}]")
                    try:
                        s = await researcher.speak(
                            f"Research task: {query}{critic_ctx}\n\n"
                            f"Sources:\n{sources}\n\n"
                            f"Content:\n{text[:1800]}{notes}\n\n"
                            "Use TEMPLATE B (## 🔍 — <topic> / ### Findings / ### Summary). "
                            "Cite source URLs. Mark uncertain data ⚠️. "
                            "End with signal: [RESEARCH: COMPLETE] / [RESEARCH: WEAK | gaps: ...] / [RESEARCH: UNAVAILABLE | data: ...]",
                            max_tokens=550, remember=False)
                    except AgentCallError as e:
                        await emit_error(f"Researcher failed on '{query}': {e}", researcher)
                        s = (f"## 🔍 — {query}\n\n### Findings\n- ⚠️ Error: {e}\n\n"
                             f"[RESEARCH: WEAK | gaps: summarisation error]")
                    return s

                query_summaries = await asyncio.gather(
                    *[_summarize_query(sr) for sr in sr_list])

                all_outputs, sig = [], ""
                for s in query_summaries:
                    await emit(researcher, s, "research")
                    all_outputs.append(s)
                    sm = re.search(r'\[RESEARCH[^\]]*\]', s, re.IGNORECASE)
                    if sm:
                        sig = sm.group(0)   # last signal wins; UNAVAILABLE > WEAK > COMPLETE

                output = "\n\n".join(all_outputs)
                if not output.strip():
                    output = (f"No data found for: {'; '.join(queries)}. "
                              "[RESEARCH: UNAVAILABLE | data: all queries returned empty]")
                    await emit(researcher,
                        f"⚠️ No results for any query.\n\n"
                        f"`[RESEARCH: UNAVAILABLE | data: {'; '.join(queries[:2])}...]`",
                        "research")

        else:
            # analyst / critic — always remember=False to prevent cross-call contamination;
            # pass curated workspace context so each call sees only what it needs
            ctx = _build_ctx_for(agent_id, workspace)

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
                await emit(agent, result, phase)
                output = result
            except AgentCallError as e:
                err_txt = f"{agent.name} call failed: {e}"
                await emit_error(err_txt, agent)
                output = f"[ERROR: {err_txt}]"
                sig    = ""
                workspace.append({
                    "round": rnd, "agent": agent_id,
                    "task": param, "output": output, "signal": sig,
                })
                continue   # supervisor will see the error in workspace next round

            # Extract signal tag
            if agent_id == "analyst":
                am = re.search(r'\[ANALYSIS[^\]]*\]', output, re.IGNORECASE)
                sig = am.group(0) if am else ""
                if analyst_called:
                    revision_count += 1
                analyst_called = True
            else:  # critic
                cm = re.search(r'\[VERDICT[^\]]*\]', output, re.IGNORECASE)
                sig = cm.group(0) if cm else ""

        workspace.append({
            "round": rnd, "agent": agent_id,
            "task": param, "output": output, "signal": sig,
        })

    # ── Emergency writer fallback if loop exhausted without a report ─────────
    if not final_report and analyst_called:
        await emit(react_supervisor,
            "⚠️ Loop exhausted — emergency writer call to produce final report.", "routing")
        ctx = [{"role": "user",
                "content": f"[{w['agent'].upper()}]\n{w['output'][:600]}"}
               for w in workspace[-6:]]
        rag_note = f"\n\nKNOWLEDGE BASE:\n{rag_context[:800]}" if rag_context else ""
        emergency_report = await writer.speak(
            f"Write the complete GTM Intelligence Report for: {topic}. "
            f"Use all available research and analysis data.{rag_note}",
            extra_context=ctx, max_tokens=3000)
        final_report = emergency_report
        await emit(writer, emergency_report, "writing")
        workspace.append({
            "round": MAX_ROUNDS, "agent": "writer",
            "task": "emergency fallback", "output": emergency_report, "signal": ""})

    # ── Extract final report ──────────────────────────────────────────────────
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
