"""
ReAct Supervisor orchestrator — Phase State Machine Edition.

Architecture:
  Explicit Phase enum replaces ad-hoc force conditions.
  - Phase defines which actions are VALID at each step
  - Supervisor LLM reasons and decides among valid actions
  - State transitions are deterministic based on agent signals
  - Hard caps per phase prevent deadlock; MAX_ROUNDS is the absolute backstop

Phases:
  RESEARCH  → gather data (supervisor decides: more research or advance)
  ANALYSIS  → analyst runs (forced; supervisor provides task)
  CRITIQUE  → critic runs (forced)
  REVISE    → post-critique (supervisor decides: revise or write)
  VERIFY    → one targeted search after REJECT_DATA (forced)
  WRITE     → writer runs (forced → DONE)
"""
import asyncio
import re
from datetime import datetime
from enum import Enum
from pathlib import Path

from team.agent import Agent, AgentCallError
from team.personas import ANALYST, CRITIC, RESEARCHER, WRITER
from team.skills import gather_dimension, web_search, web_scrape

REPORTS_DIR = Path.home() / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MAX_ROUNDS           = 40   # absolute backstop
MAX_REVISION_CYCLES  = 3    # analyst revisions after first analysis
MAX_RESEARCHER_CALLS = 7    # researcher call budget
MAX_VERIFY           = 1    # max REJECT_DATA verification cycles

_SUPV = "doubao-seed-2-0-pro-260215"

RESEARCH_DIMS = [
    ("market_overview",       "market size growth revenue forecast"),
    ("competitive_landscape", "competitors market share key players"),
    ("technology_trends",     "technology innovation disruption future"),
    ("regulatory_env",        "regulations compliance policy legal"),
]


# ── Phase State Machine ────────────────────────────────────────────────────────

class Phase(str, Enum):
    RESEARCH  = "research"
    ANALYSIS  = "analysis"
    CRITIQUE  = "critique"
    REVISE    = "revise"
    VERIFY    = "verify"
    WRITE     = "write"


def _valid_actions(phase: Phase, researcher_calls: int,
                   revision_count: int, verify_count: int) -> list[str]:
    """Return list of valid ACT keywords for the current phase."""
    if phase == Phase.RESEARCH:
        acts = ["CALL_ANALYST"]
        if researcher_calls < MAX_RESEARCHER_CALLS:
            acts.insert(0, "CALL_RESEARCHER")
        return acts
    elif phase == Phase.REVISE:
        acts = ["CALL_WRITER"]
        if revision_count < MAX_REVISION_CYCLES:
            acts.insert(0, "CALL_ANALYST")
        return acts
    elif phase == Phase.ANALYSIS:
        return ["CALL_ANALYST"]
    elif phase == Phase.CRITIQUE:
        return ["CALL_CRITIC"]
    elif phase == Phase.VERIFY:
        return ["CALL_RESEARCHER"]
    elif phase == Phase.WRITE:
        return ["CALL_WRITER"]
    return ["CALL_WRITER"]


def _next_phase(phase: Phase, action: str, signal: str,
                revision_count: int, verify_count: int) -> Phase:
    """Deterministic state transition after an action completes."""
    sig_up = signal.upper()
    if action == "CALL_RESEARCHER":
        return Phase.ANALYSIS if phase == Phase.VERIFY else Phase.RESEARCH
    elif action == "CALL_ANALYST":
        return Phase.CRITIQUE
    elif action == "CALL_CRITIC":
        if "APPROVED" in sig_up:
            return Phase.WRITE
        elif "REJECT_DATA" in sig_up and verify_count < MAX_VERIFY:
            return Phase.VERIFY
        else:
            return Phase.REVISE if revision_count < MAX_REVISION_CYCLES else Phase.WRITE
    elif action == "CALL_WRITER":
        return Phase.WRITE
    return phase


def _extract_verify_query(workspace: list) -> str:
    """Pull the critic's suggested search query from REJECT_DATA verdict."""
    for w in reversed(workspace):
        if w["agent"] == "critic":
            m = re.search(
                r'\[VERDICT:\s*REJECT_DATA[^|]*\|\s*claim:[^|]*\|\s*search:\s*([^\]]+)\]',
                w["output"], re.IGNORECASE)
            if m:
                return m.group(1).strip()
    return "verify disputed market data figure"


def _default_task(action: str, workspace: list, phase: Phase) -> str:
    """Generate a deterministic task description when supervisor is not consulted."""
    if action == "CALL_ANALYST":
        critics = [w for w in workspace if w["agent"] == "critic"]
        if critics and phase in (Phase.REVISE, Phase.ANALYSIS):
            fb = critics[-1]["output"][:400]
            return (f"Revise your analysis to address the critic's feedback. "
                    f"Label all figures [Data]/[Estimate]/[Assumption].\n\nCritic: {fb}")
        return ("Analyse all collected research using TAM/SAM/SOM, PESTEL, and Porter's Five Forces. "
                "Label every figure [Data], [Estimate], or [Assumption]. "
                "Present bull and bear perspectives for key conclusions.")
    elif action == "CALL_CRITIC":
        return ("Critique the analyst's analysis: unsupported claims, logical gaps, "
                "suspicious market size figures (check order-of-magnitude), missing evidence.")
    elif action == "CALL_WRITER":
        return "Write the complete GTM Intelligence Report including a Competitive Battle Cards section."
    return ""


def _describe_action(action: str, param: str) -> str:
    labels = {
        "CALL_RESEARCHER": f"Calling **Alex (Researcher)** — {param[:160]}",
        "CALL_ANALYST":    f"Calling **Jamie (Analyst)** — {param[:160]}",
        "CALL_CRITIC":     f"Calling **Morgan (Critic)** — {param[:160]}",
        "CALL_WRITER":     "Calling **Report Writer** — producing final report",
    }
    return labels.get(action, action)


# ── Supervisor prompts ─────────────────────────────────────────────────────────

REACT_SYSTEM = """\
You are the GTM Intelligence Supervisor. Orchestrate a research team to produce a \
high-quality, data-rich GTM Intelligence Report.

TEAM:
  CALL_RESEARCHER — Alex: parallel web searches for market data, statistics, companies
  CALL_ANALYST    — Jamie: TAM/SAM/SOM, PESTEL, Porter's Five Forces, [Data]/[Estimate]/[Assumption] labels
  CALL_CRITIC     — Morgan: challenges claims, checks data quality and order-of-magnitude
  CALL_WRITER     — Report Writer: final structured GTM Intelligence Report with Battle Cards

The current phase and valid actions are shown in each prompt — only output valid actions.

━━━ THINK STEP (required, structured) ━━━
Structure your THINK as:
  HAVE: What confirmed data do we have? (cite key figures and their sources)
  GAPS: What critical data is still missing for a complete GTM report?
  FEASIBLE: Can those gaps be filled with more research, or should we accept estimates?
  DECISION: Which action best moves toward a high-quality report right now?

━━━ ACT FORMAT ━━━
ACT: CALL_RESEARCHER | queries: q1 || q2 || q3
ACT: CALL_ANALYST    | task: [frameworks to apply + specific gaps to address]
ACT: CALL_CRITIC     | task: [what to focus on]
ACT: CALL_WRITER     | task: write final report

━━━ RESEARCHER QUERY RULES ━━━
- Keywords only: metric + geography + year — max 12 words per sub-query
- Separate parallel sub-queries with || (each fires as a separate search + bubble)
- Avoid paywalled sources: Statista, Gartner PDF, IDC, McKinsey, Forrester
- Target: Wikipedia, press releases, Reuters/Bloomberg news, government/NGO sites, vendor pages
- PIVOT RULE: if a metric failed 2+ prior searches → do NOT retry, search a DIFFERENT gap instead

━━━ DATA CONFLICT RULE ━━━
If two rounds return figures for the same metric differing by 10×+ (e.g. $5M vs $5B):
flag it in THINK, instruct analyst to use the conservative figure and label the conflict explicitly.

━━━ QUALITY BAR before CALL_WRITER ━━━
- Market size estimate (even proxied/labelled), growth rate, ≥3 named competitors
- At least one framework fully applied with evidence
- Critic has reviewed at least once

THINK: structured as HAVE / GAPS / FEASIBLE / DECISION (3-5 sentences total).
Respond in the same language as the topic.\
"""

REACT_PROMPT = """\
Research topic: {topic}
Round: {rnd}/{max_rounds}
Phase: **{phase}** | Valid actions this phase: {valid_actions}{budget_warn}
Analyst revisions used: {revision_count}/{max_revision_cycles} | Researcher calls: {research_count}/{max_researcher_calls}

Previously searched (do NOT repeat these):
{searched_queries}

WORKSPACE (oldest → newest):
{workspace}

Your structured assessment and next action (THINK then ACT, only use valid actions):\
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

def _workspace_text(workspace: list) -> str:
    if not workspace:
        return "(empty — no work done yet)"
    n = len(workspace)
    parts = []
    for i, w in enumerate(workspace):
        if i >= n - 3:
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
    Analyst sees ALL researcher rounds (early baseline + latest gap-fill).
    Critic sees full analyst output + all researcher rounds for cross-checking.
    """
    if agent_id == "analyst":
        msgs = []
        researchers = [w for w in workspace if w["agent"] == "researcher"]
        for i, r in enumerate(researchers):
            limit = 2000 if i == len(researchers) - 1 else 600
            msgs.append({"role": "user",
                         "content": f"[RESEARCH Round {r['round']}]\n{r['output'][:limit]}"})
        critics = [w for w in workspace if w["agent"] == "critic"]
        if critics:
            c = critics[-1]
            msgs.append({"role": "user",
                         "content": f"[CRITIC FEEDBACK — Round {c['round']}]\n{c['output'][:1200]}"})
        analysts = [w for w in workspace if w["agent"] == "analyst"]
        if analysts:
            a = analysts[-1]
            msgs.append({"role": "user",
                         "content": f"[YOUR PREVIOUS ANALYSIS — Round {a['round']}]\n{a['output'][:1000]}"})
        return msgs

    elif agent_id == "critic":
        msgs = []
        analysts = [w for w in workspace if w["agent"] == "analyst"]
        if analysts:
            a = analysts[-1]
            msgs.append({"role": "user",
                         "content": f"[ANALYST'S ANALYSIS — Round {a['round']}]\n{a['output'][:2500]}"})
        researchers = [w for w in workspace if w["agent"] == "researcher"]
        for i, r in enumerate(researchers):
            limit = 800 if i == len(researchers) - 1 else 300
            msgs.append({"role": "user",
                         "content": f"[RESEARCH Round {r['round']}]\n{r['output'][:limit]}"})
        return msgs

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
    workspace:        list[dict] = []
    rag_context       = ""
    researcher_calls  = 0
    revision_count    = 0
    verify_count      = 0
    first_analysis    = True   # True until analyst has run once
    final_report      = ""
    phase             = Phase.RESEARCH

    await emit(react_supervisor,
        f"🚀 **Starting: {topic}**\n\n"
        "I'll reason about quality at every step and decide what the team works on next.",
        "routing")

    # ── ReAct loop ────────────────────────────────────────────────────────────
    for rnd in range(1, MAX_ROUNDS + 1):

        if any(w["agent"] == "writer" for w in workspace):
            break

        valid = _valid_actions(phase, researcher_calls, revision_count, verify_count)

        # ── Determine action ──────────────────────────────────────────────────

        if phase == Phase.VERIFY:
            # Forced: targeted researcher call using critic's REJECT_DATA search query
            action    = "CALL_RESEARCHER"
            param     = _extract_verify_query(workspace)
            think_txt = ""
            await emit(react_supervisor,
                f"🔬 **Round {rnd}:** Verifying disputed claim — {param[:120]}", "routing")

        elif len(valid) == 1:
            # Forced: only one valid action (ANALYSIS / CRITIQUE / WRITE)
            action    = valid[0]
            param     = _default_task(action, workspace, phase)
            think_txt = ""
            await emit(react_supervisor,
                f"▶ **Round {rnd}:** {_describe_action(action, param)}", "routing")

        else:
            # Dynamic routing: supervisor decides among valid options
            # (RESEARCH with budget remaining, or REVISE phase)
            budget_remaining = MAX_RESEARCHER_CALLS - researcher_calls
            budget_warn = (f" | ⚠️ {budget_remaining} researcher call(s) left"
                           if budget_remaining <= 2 else "")

            searched_parts = []
            for w in workspace:
                if w["agent"] == "researcher" and w.get("task") and w["task"] != "initial":
                    for sq in w["task"].split("||"):
                        sq = sq.strip()
                        if sq:
                            searched_parts.append(f"  • {sq[:120]}")
            searched_block = "\n".join(searched_parts) or "  (none yet — first follow-up)"

            try:
                react_raw = await react_supervisor.speak(
                    REACT_PROMPT.format(
                        topic=topic,
                        rnd=rnd,
                        max_rounds=MAX_ROUNDS,
                        phase=phase.value,
                        valid_actions=" | ".join(valid),
                        budget_warn=budget_warn,
                        revision_count=revision_count,
                        max_revision_cycles=MAX_REVISION_CYCLES,
                        research_count=researcher_calls,
                        max_researcher_calls=MAX_RESEARCHER_CALLS,
                        searched_queries=searched_block,
                        workspace=_workspace_text(workspace),
                    ),
                    max_tokens=600, remember=False)
            except AgentCallError as e:
                await emit_error(f"Supervisor failed: {e}")
                action    = valid[0]
                param     = _default_task(action, workspace, phase)
                think_txt = ""
            else:
                think_txt, action, param = _parse_react(react_raw)
                # Enforce valid actions — never let supervisor escape the phase constraints
                if action not in valid:
                    action = valid[0]
                    param  = _default_task(action, workspace, phase)

            if think_txt:
                await emit(react_supervisor, think_txt, "thinking", is_think=True)
            await emit(react_supervisor,
                f"▶ {_describe_action(action, param)}", "routing")

        if action == "DONE":
            break

        # ── Execute action ────────────────────────────────────────────────────

        sig    = ""
        output = ""

        # ── WRITER ───────────────────────────────────────────────────────────
        if action == "CALL_WRITER":
            _writer_ctx = []
            _rs = [w for w in workspace if w["agent"] == "researcher"]
            if _rs:
                _writer_ctx.append({"role": "user",
                                    "content": f"[RESEARCH BASELINE]\n{_rs[0]['output'][:800]}"})
            if len(_rs) > 1:
                _writer_ctx.append({"role": "user",
                                    "content": f"[RESEARCH LATEST]\n{_rs[-1]['output'][:600]}"})
            for w in workspace:
                if w["agent"] == "analyst":
                    _writer_ctx.append({"role": "user",
                                        "content": f"[ANALYST]\n{w['output'][:1500]}"})
                elif w["agent"] == "critic":
                    _writer_ctx.append({"role": "user",
                                        "content": f"[CRITIC]\n{w['output'][:800]}"})

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
                    try:
                        s = await researcher.speak(
                            f'Summarise web data about "{topic}" — {dim.replace("_", " ")}.\n\n'
                            f'{raw[:3000]}\n\n'
                            "Use TEMPLATE A (Key Findings + Synthesis + Gaps + Confidence). "
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
                sig_match = re.search(r'\[RESEARCH[^\]]*\]', sig_raw, re.IGNORECASE)
                sig = sig_match.group(0) if sig_match else "[RESEARCH: WEAK | gaps: signal unclear]"
                output += f"\n\n{sig}"
                await emit(researcher,
                    f"📊 Initial research complete. Signal: `{sig}`", "research")

                task_label = "initial"

            else:
                # Follow-up search: one LLM call + bubble per sub-query (true parallel)
                raw_queries = [q.strip() for q in param.split("||") if q.strip()]
                queries     = raw_queries[:4] if raw_queries else [param]

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

                sr_list = await asyncio.gather(*[_search_with_retry(q) for q in queries])

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
                        return (f"## 🔍 — {query}\n\n**Found:** Nothing\n\n"
                                f"**Not found:** {query}\n\n**Plausibility:** N/A\n\n"
                                f"**Summary:** No results found.\n\n"
                                f"[RESEARCH: UNAVAILABLE | data: {query}]")
                    try:
                        s = await researcher.speak(
                            f"Research task: {query}{critic_ctx}\n\n"
                            f"Sources:\n{sources}\n\n"
                            f"Content:\n{text[:1800]}{notes}\n\n"
                            "Use TEMPLATE B (## 🔍 — <topic> / **Found** / **Not found** / "
                            "**Plausibility** / **Summary**). "
                            "Tag each finding [Data], [Estimate], or [Claim]. "
                            "Cite source URLs. End with signal.",
                            max_tokens=550, remember=False)
                    except AgentCallError as e:
                        await emit_error(f"Researcher failed on '{query}': {e}", researcher)
                        s = (f"## 🔍 — {query}\n\n**Found:** Error\n\n"
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
                        sig = sm.group(0)

                output = "\n\n".join(all_outputs)
                if not output.strip():
                    output = (f"No data found for: {'; '.join(queries)}. "
                              "[RESEARCH: UNAVAILABLE | data: all queries returned empty]")
                    await emit(researcher,
                        f"⚠️ No results for any query.\n\n"
                        f"`[RESEARCH: UNAVAILABLE | data: {'; '.join(queries[:2])}...]`",
                        "research")

                task_label = param

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
                sig    = ""
                workspace.append({
                    "round": rnd, "agent": agent_id,
                    "task": param, "output": output, "signal": sig,
                })
                continue

            if agent_id == "analyst":
                am = re.search(r'\[ANALYSIS[^\]]*\]', output, re.IGNORECASE)
                sig = am.group(0) if am else ""
                if first_analysis:
                    first_analysis = False
                else:
                    revision_count += 1
            else:  # critic
                cm = re.search(r'\[VERDICT[^\]]*\]', output, re.IGNORECASE)
                sig = cm.group(0) if cm else ""

            task_label = param

        # ── State transition ──────────────────────────────────────────────────
        new_phase = _next_phase(phase, action, sig, revision_count, verify_count)
        if action == "CALL_CRITIC" and "REJECT_DATA" in sig.upper():
            verify_count += 1
        phase = new_phase

        workspace.append({
            "round":  rnd,
            "agent":  "researcher" if action == "CALL_RESEARCHER" else agent_id,
            "task":   task_label if action == "CALL_RESEARCHER" else param,
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
