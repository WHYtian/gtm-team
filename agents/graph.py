"""
Multi-Agent GTM Research Graph — ReAct Supervisor-Driven Dynamic Routing.

Architecture mirrors gtm-team/orchestrator.py but implemented with LangGraph.
Supervisor LLM decides every action freely; hard safety nets override deterministically.

Loop: supervisor_decide → (researcher | analyst | validator | critic) → supervisor_decide
Exit: supervisor_decide → writer → finalize → END
"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
import operator
import queue

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agents.llm import chat, MODEL_PRO, MODEL_V3, MODEL_SUPV
from agents.prompts import (
    SUPERVISOR_SYSTEM,
    REACT_SYSTEM,
    REACT_PROMPT,
    RESEARCHER_SYSTEM,
    ANALYST_SYSTEM,
    CRITIC_SYSTEM,
    WRITER_SYSTEM,
    DATA_SYNTHESIZER_SYSTEM,
    DOCUMENT_ANALYSIS_SYSTEM,
)

SKILLS_DIR = Path.home() / ".openclaw/workspace/skills"
REPORTS_DIR = Path.home() / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RESEARCH_DIMENSIONS = [
    ("market_overview",       "market size growth revenue forecast"),
    ("competitive_landscape", "competitors market share key players"),
    ("technology_trends",     "technology innovation disruption future"),
    ("regulatory_env",        "regulations compliance policy legal"),
]

MAX_ROUNDS                  = 40
MAX_RESEARCHER_CALLS        = 3
MAX_POST_ANALYST_RESEARCHER = 2
SIM_THRESHOLD               = 0.62   # RAG retrieval: query → document
PAIR_MATCH_THRESHOLD        = 0.70   # cross-corpus: web finding → RAG chunk (stricter)

AGENT_META = {
    "supervisor":  {"name": "Supervisor",          "color": "#00d4aa", "avatar": "S"},
    "researcher":  {"name": "Alex · Researcher",   "color": "#f472b6", "avatar": "A"},
    "analyst":     {"name": "Jamie · Analyst",     "color": "#a78bfa", "avatar": "J"},
    "validator":   {"name": "Jordan · Synthesizer","color": "#34d399", "avatar": "Js"},
    "critic":      {"name": "Morgan · Critic",     "color": "#fbbf24", "avatar": "M"},
    "writer":      {"name": "Report Writer",       "color": "#38bdf8", "avatar": "W"},
    "system":      {"name": "System",              "color": "#64748b", "avatar": "⚙"},
}


# ── State ─────────────────────────────────────────────────────────────────────

class TeamMsg(TypedDict):
    agent: str
    content: str
    ts: str
    phase: str
    is_think: bool


class State(TypedDict):
    # Input
    user_message: str
    history: List[dict]
    doc_content: str
    rag_context: str

    # Routing
    route: str                   # "general" | "research" | "doc_analysis"
    supervisor_response: str
    is_research: bool
    topic: str

    # ReAct loop counters
    rnd: int
    researcher_calls: int
    post_analyst_calls: int
    analyst_called: bool
    validator_called: bool
    revision_count: int
    next_action: str             # supervisor decision for this iteration
    next_param: str              # task directive for the chosen agent
    stale_report: str
    stale_entry_id: str
    stale_age: int
    user_rag_chunks: List[dict]

    # Accumulating outputs (operator.add → append-only)
    workspace: Annotated[List[dict], operator.add]
    team_messages: Annotated[List[TeamMsg], operator.add]

    # Final output
    report_content: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _make_msg(agent: str, content: str, phase: str = "chat", is_think: bool = False) -> TeamMsg:
    return {"agent": agent, "content": content, "ts": now(), "phase": phase, "is_think": is_think}


def emit(q: Optional[queue.Queue], agent: str, content: str,
         phase: str = "chat", is_think: bool = False) -> TeamMsg:
    msg = _make_msg(agent, content, phase, is_think)
    if q:
        q.put({"type": "team_chat", "msg": msg, "meta": AGENT_META.get(agent, {})})
    return msg


def get_queue(config: RunnableConfig) -> Optional[queue.Queue]:
    return config.get("configurable", {}).get("queue")


# ── Skill / web helpers ───────────────────────────────────────────────────────

def _run_skill(script_rel: str, args: list, timeout: int = 40) -> dict:
    script = str(SKILLS_DIR / script_rel)
    if not Path(script).exists():
        return {"error": f"skill not found: {script_rel}"}
    try:
        r = subprocess.run(
            [sys.executable, script] + [str(a) for a in args],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "HOME": str(Path.home())},
        )
        out = r.stdout.strip()
        if not out:
            return {"error": r.stderr[:300] or "no output"}
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"raw": out[:3000]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def _web_search(query: str, max_results: int = 4) -> list:
    result = _run_skill("web-search/scripts/web_search.py", [query, max_results])
    if "error" in result:
        return []
    return result if isinstance(result, list) else result.get("results", [])


def _web_scrape(url: str) -> str:
    result = _run_skill("web-scrape/scripts/web_scrape.py", [url])
    return result.get("text") or result.get("raw", "")[:2000]


def _search_with_sources(query: str) -> dict:
    results = _web_search(query, max_results=4)
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

    scraped = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_web_scrape, u): u for u in urls[:2]}
        for f in as_completed(futs):
            try:
                t = f.result()
                if t and len(t) > 100:
                    scraped.append(t)
            except Exception:
                pass

    body = ("\n\n---\n\n".join(scraped) if scraped else "\n".join(snippets))[:3000]
    return {"text": body, "sources": sources, "query": query}


def _search_with_retry(query: str) -> dict:
    result = _search_with_sources(query)
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
        r2 = _search_with_sources(short_q)
        if len(r2.get("text", "")) >= 200:
            r2["retried"] = True
            r2["original_query"] = query
            return r2

    result["limited"] = True
    return result


def _gather_dimension(topic: str, dimension: str, query_suffix: str) -> dict:
    query = f"{topic} {query_suffix}"
    results = _web_search(query, max_results=3)
    if not results:
        return {"dimension": dimension, "text": f"No results found for {query}", "sources": []}

    sources = [
        {"url": r.get("url") or r.get("href", ""),
         "title": r.get("title") or r.get("name", ""),
         "snippet": r.get("body") or r.get("snippet", "")}
        for r in results[:3]
        if r.get("url") or r.get("href")
    ]
    urls     = [s["url"] for s in sources[:2] if s["url"]]
    snippets = [r.get("body") or r.get("snippet", "") for r in results[:3]]

    scraped = []
    if urls:
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(_web_scrape, u) for u in urls]
            for f in as_completed(futs):
                try:
                    t = f.result()
                    if t and len(t) > 100:
                        scraped.append(t)
                except Exception:
                    pass

    combined = "\n\n---\n\n".join(scraped) if scraped else "\n".join(snippets[:3])
    return {"dimension": dimension, "text": combined[:4000], "sources": sources}


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


# ── RAG helpers ───────────────────────────────────────────────────────────────

def _get_rag(topic: str) -> str:
    try:
        from rag.manager import query_rag
        queries = [f"{topic} {dq}" for _, dq in RESEARCH_DIMENSIONS]
        parts = []
        seen = set()
        for q in queries:
            result = query_rag(q, n_results=3)
            for block in result.split("\n\n---\n\n"):
                block = block.strip()
                if block and block not in seen:
                    seen.add(block)
                    parts.append(block)
        return "\n\n---\n\n".join(parts[:8])
    except Exception:
        return ""


def _get_user_rag_for_topic(topic: str) -> list:
    try:
        from rag.manager import query_rag
        queries = [f"{topic} {dq}" for _, dq in RESEARCH_DIMENSIONS]
        chunks = []
        seen = set()
        for q in queries:
            result = query_rag(q, n_results=3)
            for block in result.split("\n\n---\n\n"):
                block = block.strip()
                if not block:
                    continue
                m = re.match(r'^\[([^\]]+)\]\n(.*)', block, re.DOTALL)
                if m:
                    fname = m.group(1).strip()
                    text  = m.group(2).strip()
                    key   = fname + text[:50]
                    if key not in seen and len(text) >= 40:
                        seen.add(key)
                        chunks.append({"text": text[:400], "filename": fname})
        return chunks
    except Exception:
        return []


def _get_all_user_rag_chunks() -> list:
    try:
        from rag.manager import _get_collection
        col = _get_collection()
        res = col.get(include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            fname = meta.get("filename", "")
            text  = doc.strip() if doc else ""
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


def _embed_pair_match(findings: list, rag_chunks: list, threshold: float = PAIR_MATCH_THRESHOLD):
    if not findings or not rag_chunks:
        return [], rag_chunks
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer("BAAI/bge-small-en-v1.5")
        prefix = "Represent this sentence for searching relevant passages: "
        # normalize_embeddings=True already produces unit vectors; no manual re-norm needed
        f_embs = _st.encode([prefix + t for t in findings], normalize_embeddings=True)
        r_embs = _st.encode([c["text"] for c in rag_chunks], normalize_embeddings=True)
        sim    = f_embs @ r_embs.T   # cosine similarity (both sides already unit-normed)
        matched, supplements = [], []
        for j, chunk in enumerate(rag_chunks):
            col_sim = sim[:, j]
            best    = int(col_sim.argmax())
            max_sim = float(col_sim[best])
            if max_sim >= threshold:
                matched.append({**chunk, "best_finding": findings[best], "sim": round(max_sim, 3)})
            else:
                supplements.append(chunk)
        return matched, supplements
    except Exception:
        return [], rag_chunks


# ── Workspace helpers ─────────────────────────────────────────────────────────

def _researcher_digest(output: str) -> str:
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
            f"  Output: {out}"
        )
    return "\n\n".join(parts)


def _build_ctx_for(agent_id: str, workspace: list) -> list:
    if agent_id == "analyst":
        msgs = []
        for r in (w for w in workspace if w["agent"] == "researcher"):
            msgs.append({"role": "user", "content": f"[RESEARCH Round {r['round']}]\n{r['output']}"})
        for v in (w for w in workspace if w["agent"] == "validator"):
            msgs.append({"role": "user", "content": f"[DATA RECONCILIATION]\n{v['output']}"})
        critics = [w for w in workspace if w["agent"] == "critic"]
        if critics:
            c = critics[-1]
            msgs.append({"role": "user", "content": f"[PREVIOUS CRITIC FEEDBACK — Round {c['round']}]\n{c['output']}"})
        analysts = [w for w in workspace if w["agent"] == "analyst"]
        if analysts:
            a = analysts[-1]
            msgs.append({"role": "user", "content": f"[YOUR PREVIOUS ANALYSIS — Round {a['round']}]\n{a['output']}"})
        return msgs

    elif agent_id == "critic":
        msgs = []
        prior_critics = [w for w in workspace if w["agent"] == "critic"]
        # Inject previous critic feedback so Morgan enters verification mode on revision pass
        if prior_critics:
            last_c = prior_critics[-1]
            msgs.append({"role": "user", "content":
                f"[PREVIOUS CRITIC FEEDBACK — Round {last_c['round']}]\n{last_c['output']}"})
        analysts = [w for w in workspace if w["agent"] == "analyst"]
        if analysts:
            a = analysts[-1]
            msgs.append({"role": "user", "content": f"[ANALYST'S ANALYSIS — Round {a['round']}]\n{a['output']}"})
        for r in (w for w in workspace if w["agent"] == "researcher"):
            msgs.append({"role": "user", "content": f"[RESEARCH Round {r['round']}]\n{r['output']}"})
        return msgs

    return [{"role": "user",
             "content": f"[{w['agent'].upper()} — Round {w['round']}]\n{w['output'][:600]}"}
            for w in workspace[-5:]]


def _has_findings(text: str) -> bool:
    return bool(re.search(r'\[(Data|Estimate|Claim)\]', text, re.IGNORECASE))


def _extract_finding_texts(workspace: list) -> list:
    seen, findings = set(), []
    for w in workspace:
        if w["agent"] != "researcher":
            continue
        for line in w["output"].split('\n'):
            sl = line.strip()
            if not re.search(r'\[(Data|Estimate|Claim)\]', sl, re.IGNORECASE):
                continue
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


def _last_critic_verdict(workspace: list):
    for w in reversed(workspace):
        if w["agent"] == "critic":
            m = re.search(r'\[VERDICT:\s*(APPROVED|NEEDS_REVISION|REJECT_DATA)',
                          w["output"], re.IGNORECASE)
            return (m.group(1).upper() if m else ""), w
    return "", None


def _researcher_after(workspace: list, after_round: int) -> bool:
    return any(w["agent"] == "researcher" and w["round"] > after_round for w in workspace)


# ── ReAct parsing ─────────────────────────────────────────────────────────────

def _parse_react(text: str):
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
    param  = (act_m.group(2) or "").strip()
    return think, action, param


# ── LLM call wrapper ──────────────────────────────────────────────────────────

def _llm(system: str, prompt: str, model: str, max_tokens: int, temperature: float = 0.5) -> str:
    return chat(
        [{"role": "user", "content": prompt}],
        system=system,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _llm_ctx(system: str, prompt: str, extra_ctx: list, model: str,
             max_tokens: int, temperature: float = 0.5) -> str:
    msgs = list(extra_ctx) + [{"role": "user", "content": prompt}]
    return chat(msgs, system=system, model=model, max_tokens=max_tokens, temperature=temperature)


# ── Semantic cache ────────────────────────────────────────────────────────────

def _cache_lookup(topic: str):
    try:
        sys.path.insert(0, str(Path.home() / "gtm-team"))
        from semantic_cache import lookup
        return lookup(topic)
    except Exception:
        return None


def _cache_store(topic: str, report: str, old_entry_id: str | None = None):
    try:
        sys.path.insert(0, str(Path.home() / "gtm-team"))
        from semantic_cache import store
        store(topic, topic, report, old_entry_id)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════════════════════

def supervisor_init_node(state: State, config: RunnableConfig) -> dict:
    """Initial routing: decide general vs research vs doc_analysis."""
    q = get_queue(config)

    if state.get("doc_content"):
        emit(q, "supervisor", "📋 Document uploaded for analysis. Routing to Analyst team.", "routing")
        return {
            "route": "doc_analysis",
            "is_research": True,
            "topic": "Document Analysis",
            "team_messages": [_make_msg("supervisor",
                "📋 Document uploaded for analysis. Routing to Analyst team.", "routing")],
        }

    history = state.get("history", [])
    msgs = history + [{"role": "user", "content": state["user_message"]}]
    emit(q, "supervisor", "🤔 Analyzing your request...", "routing")

    response = chat(msgs, system=SUPERVISOR_SYSTEM, model=MODEL_SUPV, max_tokens=300, temperature=0.3)

    if response.startswith("TASK:RESEARCH"):
        topic = state["user_message"]
        for line in response.split("\n"):
            if line.startswith("TOPIC:"):
                topic = line[6:].strip()
                break

        announce = (
            f"🚀 **Starting: {topic}**\n\n"
            "I'll reason about quality at every step and decide what the team works on next."
        )
        emit(q, "supervisor", announce, "routing")

        # Semantic cache check — DISABLED for testing
        # cache_hit = _cache_lookup(topic)
        cache_hit = None
        stale_report = stale_entry_id = stale_age = ""

        return {
            "route": "research",
            "is_research": True,
            "topic": topic,
            "rnd": 0,
            "researcher_calls": 0,
            "post_analyst_calls": 0,
            "analyst_called": False,
            "validator_called": False,
            "revision_count": 0,
            "next_action": "",
            "next_param": "",
            "stale_report": stale_report,
            "stale_entry_id": stale_entry_id,
            "stale_age": stale_age,
            "user_rag_chunks": [],
            "workspace": [],
            "report_content": "",
            "team_messages": [_make_msg("supervisor", announce, "routing")],
        }
    else:
        return {
            "route": "general",
            "is_research": False,
            "supervisor_response": response,
            "team_messages": [],
        }


def direct_response_node(state: State, config: RunnableConfig) -> dict:
    return {}


# ── Supervisor decide (ReAct heart) ──────────────────────────────────────────

def supervisor_decide_node(state: State, config: RunnableConfig) -> dict:
    """
    Called once per iteration. Decides next_action and next_param via ReAct LLM,
    then applies hard safety constraints deterministically.
    """
    q = get_queue(config)
    workspace        = state.get("workspace", [])
    researcher_calls = state.get("researcher_calls", 0)
    post_analyst_calls = state.get("post_analyst_calls", 0)
    analyst_called   = state.get("analyst_called", False)
    validator_called = state.get("validator_called", False)
    revision_count   = state.get("revision_count", 0)
    rnd              = state.get("rnd", 0) + 1
    topic            = state["topic"]
    user_rag_chunks  = state.get("user_rag_chunks", [])

    # Emergency exit: writer already ran
    if any(w["agent"] == "writer" for w in workspace):
        return {"rnd": rnd, "next_action": "DONE", "next_param": "", "team_messages": []}

    # Hard: round 1 always goes to researcher
    if researcher_calls == 0:
        emit(q, "supervisor",
             "▶ Calling **Alex (Researcher)** — Initial broad research across 4 dimensions",
             "routing")
        return {
            "rnd": rnd,
            "next_action": "CALL_RESEARCHER",
            "next_param": "initial",
            "team_messages": [_make_msg("supervisor",
                "▶ Calling **Alex (Researcher)** — Initial broad research across 4 dimensions",
                "routing")],
        }

    # Hard: pre-analyst budget exhausted but analyst never ran
    if researcher_calls >= MAX_RESEARCHER_CALLS and not analyst_called:
        msg = f"⏰ Pre-analyst researcher budget ({MAX_RESEARCHER_CALLS} calls) exhausted — routing to analyst."
        emit(q, "supervisor", msg, "routing")
        param = ("Pre-analyst researcher budget exhausted. Analyse all collected data using "
                 "TAM/SAM/SOM, PESTEL, Porter's Five Forces. Label every figure [Data], [Estimate], or [Assumption].")
        return {
            "rnd": rnd,
            "next_action": "CALL_ANALYST",
            "next_param": param,
            "team_messages": [_make_msg("supervisor", msg, "routing")],
        }

    # Emergency: max rounds hit
    if rnd > MAX_ROUNDS:
        return {"rnd": rnd, "next_action": "CALL_WRITER",
                "next_param": "Write the final report using all available data.",
                "team_messages": []}

    # ── Supervisor LLM decides freely ────────────────────────────────────────
    critic_called = any(w["agent"] == "critic" for w in workspace)

    if analyst_called:
        post_remaining = MAX_POST_ANALYST_RESEARCHER - post_analyst_calls
        budget_warn = (f" ⚠️ {post_remaining} post-analyst gap-fill(s) left"
                       if post_remaining <= 1 else "")
    else:
        pre_remaining = MAX_RESEARCHER_CALLS - researcher_calls
        budget_warn = (f" ⚠️ {pre_remaining} pre-analyst call(s) left — consider advancing"
                       if pre_remaining <= 1 else "")

    searched_parts = []
    for w in workspace:
        if w["agent"] == "researcher" and w.get("task") and w["task"] != "initial":
            searched_parts.append(f"  • {w['task'][:120]}")
    searched_block = "\n".join(searched_parts) or "  (none yet)"

    react_raw = _llm(
        REACT_SYSTEM,
        REACT_PROMPT.format(
            topic=topic,
            rnd=rnd,
            max_rounds=MAX_ROUNDS,
            budget_warn=budget_warn,
            research_count=researcher_calls,
            max_researcher_calls=MAX_RESEARCHER_CALLS,
            post_analyst_count=post_analyst_calls,
            max_post_analyst=MAX_POST_ANALYST_RESEARCHER,
            analyst_called="Yes" if analyst_called else "No",
            revision_count=revision_count,
            critic_called="Yes" if critic_called else "No",
            validator_called="Yes" if validator_called else "No",
            searched_queries=searched_block,
            workspace=_workspace_text(workspace),
        ),
        model=MODEL_SUPV,
        max_tokens=600,
        temperature=0.3,
    )

    think_txt, action, param = _parse_react(react_raw)

    # ── Apply hard constraints ────────────────────────────────────────────────

    # Constraint: critic routing
    last_verdict, last_critic_w = _last_critic_verdict(workspace)
    if last_critic_w:
        # APPROVED → must go directly to writer, no detours
        if last_verdict == "APPROVED" and action != "CALL_WRITER":
            action = "CALL_WRITER"
            param  = "Critic approved. Write the final GTM Intelligence Report."

        followed_up = _researcher_after(workspace, last_critic_w["round"])
        if last_verdict == "REJECT_DATA" and not followed_up:
            if action != "CALL_RESEARCHER" and post_analyst_calls < MAX_POST_ANALYST_RESEARCHER:
                action = "CALL_RESEARCHER"
                sq_m = re.search(r'search:\s*([^\]\n]+)', last_critic_w["output"], re.IGNORECASE)
                param = (f"Find alternative source: {sq_m.group(1).strip()[:200]}"
                         if sq_m else "Find alternative source for the figure rejected by critic.")
        elif last_verdict == "NEEDS_REVISION" and not followed_up:
            reason_m = re.search(r'reason:\s*(\w+)', last_critic_w["output"], re.IGNORECASE)
            reason = reason_m.group(1).lower() if reason_m else ""
            if reason == "logic_error" and action == "CALL_RESEARCHER":
                action = "CALL_ANALYST"
                param = param or "Fix the logical errors in the framework or arithmetic flagged by critic."

    # Constraint: researcher budget limits
    if action == "CALL_RESEARCHER":
        pre_exhausted  = not analyst_called and researcher_calls >= MAX_RESEARCHER_CALLS
        post_exhausted = analyst_called and post_analyst_calls >= MAX_POST_ANALYST_RESEARCHER
        if pre_exhausted:
            action = "CALL_ANALYST"
            param  = "Pre-analyst researcher budget exhausted — proceeding to analysis."
        elif post_exhausted:
            action = "CALL_ANALYST"
            param  = ("Post-analyst gap-fill budget exhausted. "
                      "Label any remaining missing metrics as [N/A] or derive [Estimate].")

    # Constraint: force critic immediately after analyst (proactive)
    # Allow only CALL_CRITIC or CALL_RESEARCHER (post-analyst gap fill); block everything else.
    if analyst_called and action not in ("CALL_CRITIC", "CALL_RESEARCHER"):
        last_analyst_rnd = max((w["round"] for w in workspace if w["agent"] == "analyst"), default=0)
        last_critic_rnd  = max((w["round"] for w in workspace if w["agent"] == "critic"),  default=0)
        if last_analyst_rnd > last_critic_rnd:
            action = "CALL_CRITIC"
            param  = "Review the latest analyst output before the final report is written."

    # Constraint: max 2 analyst revisions
    if revision_count >= 2 and action == "CALL_ANALYST":
        action = "CALL_WRITER"
        param  = ("Analyst has been revised twice. Write the final report using the "
                  "best available analysis. Label any unresolved issues inline.")

    # Constraint: advancement rule — enforce (a) market size + (b) competitor metric before analyst
    if action == "CALL_ANALYST" and not analyst_called and researcher_calls < MAX_RESEARCHER_CALLS:
        ws_text = " ".join(w.get("output", "") for w in workspace if w["agent"] == "researcher")
        has_market_size = bool(re.search(
            r'\$[\d,.]+\s*[BMKTbmkt]|\d[\d,.]*\s*(?:billion|million|trillion)',
            ws_text, re.IGNORECASE,
        ))
        has_competitor_metric = bool(re.search(
            r'\d+\.?\d*\s*%.*(?:share|revenue|market)|(?:share|revenue|market).*\d+\.?\d*\s*%'
            r'|\$[\d,.]+[BMK].*(?:revenue|sales|arr|gmv)',
            ws_text, re.IGNORECASE,
        ))
        if not has_market_size:
            action = "CALL_RESEARCHER"
            param  = (f"Find market size with a cited dollar figure for '{topic}' — "
                      "search Statista, IBISWorld, Grand View Research, Mordor Intelligence.")
            emit(q, "supervisor",
                 "⚠️ Advancement blocked — no market size figure found. Re-routing to Researcher.",
                 "routing")
        elif not has_competitor_metric:
            action = "CALL_RESEARCHER"
            param  = (f"Find named competitors with revenue figures or market share % for '{topic}'. "
                      "Include actual company names and specific quantitative metrics.")
            emit(q, "supervisor",
                 "⚠️ Advancement blocked — no competitor metric found. Re-routing to Researcher.",
                 "routing")

    # Constraint: after validator runs, block any further pre-analyst researcher calls.
    # Validator is the final step before analyst — new research would bypass the reconciliation.
    if validator_called and not analyst_called and action == "CALL_RESEARCHER":
        action = "CALL_ANALYST"
        param  = "Synthesizer has completed data reconciliation. Proceed with full analysis."

    # Constraint: validator before first analyst call if user RAG data exists
    if action == "CALL_ANALYST" and not analyst_called and not validator_called:
        if user_rag_chunks:
            action = "CALL_VALIDATOR"
            param  = "Reconcile web search findings with imported knowledge base."

    # Emit think + routing announcement
    msgs_out: List[TeamMsg] = []
    if think_txt:
        emit(q, "supervisor", think_txt, "thinking", is_think=True)
        msgs_out.append(_make_msg("supervisor", think_txt, "thinking", is_think=True))

    _labels = {
        "CALL_RESEARCHER": f"▶ Calling **Alex (Researcher)** — {param}",
        "CALL_ANALYST":    f"▶ Calling **Jamie (Analyst)** — {param}",
        "CALL_CRITIC":     f"▶ Calling **Morgan (Critic)** — {param}",
        "CALL_WRITER":     "▶ Calling **Report Writer** — producing final report",
        "CALL_VALIDATOR":  "▶ Calling **Jordan (Synthesizer)** — reconciling web vs imported data",
    }
    label = _labels.get(action, action)
    emit(q, "supervisor", label, "routing")
    msgs_out.append(_make_msg("supervisor", label, "routing"))

    return {
        "rnd": rnd,
        "next_action": action,
        "next_param": param,
        "team_messages": msgs_out,
    }


# ── Researcher node ───────────────────────────────────────────────────────────

def researcher_node(state: State, config: RunnableConfig) -> dict:
    q = get_queue(config)
    topic            = state["topic"]
    researcher_calls = state.get("researcher_calls", 0)
    analyst_called   = state.get("analyst_called", False)
    param            = state.get("next_param", "")
    rnd              = state.get("rnd", 1)
    msgs_out: List[TeamMsg] = []

    new_researcher_calls = researcher_calls + 1
    new_post_analyst     = state.get("post_analyst_calls", 0) + (1 if analyst_called else 0)

    sig    = ""
    output = ""
    task_label = param

    # ── INITIAL 4-dim search ──────────────────────────────────────────────────
    if researcher_calls == 0:
        m = _make_msg("researcher",
            f"🔍 **Initial research: *{topic}***\nLaunching parallel searches across 4 dimensions...",
            "research")
        emit(q, "researcher", m["content"], "research")
        msgs_out.append(m)

        dim_results = {}
        rag_ctx     = ""
        user_rag_chunks: list = []

        def _do_dim(dk, dq):
            return _gather_dimension(topic, dk, dq)

        with ThreadPoolExecutor(max_workers=4) as ex:
            dim_futs = {ex.submit(_do_dim, dk, dq): (dk, dq) for dk, dq in RESEARCH_DIMENSIONS}
            for f in as_completed(dim_futs):
                dk, _ = dim_futs[f]
                try:
                    dim_results[dk] = f.result()
                except Exception as e:
                    dim_results[dk] = {"dimension": dk, "text": f"Error: {e}"}

        # Fetch RAG in parallel with dim research (we do it after since ThreadPoolExecutor is done)
        try:
            rag_ctx = _get_rag(topic)
        except Exception:
            rag_ctx = ""

        try:
            user_rag_chunks = _get_user_rag_for_topic(topic)
        except Exception:
            user_rag_chunks = []

        summaries = {}
        _res_c = {"findings": 0, "unavail": 0, "retries": 0}

        def _summarize_dim(dr):
            dim, raw = dr["dimension"], dr["text"]
            src_block = _fmt_sources(dr.get("sources", []))

            def _do_summarize(content, sources_str=src_block):
                return _llm(
                    RESEARCHER_SYSTEM,
                    f'Summarise web data about "{topic}" — {dim.replace("_", " ")}.\n\n'
                    f'Sources:\n{sources_str}\n\n'
                    f'Content:\n{content[:3000]}\n\n'
                    "Use TEMPLATE A (Key Findings + Synthesis + Gaps + Confidence). "
                    "Start immediately with ## — no preamble.",
                    model=MODEL_PRO, max_tokens=550, temperature=0.3,
                )

            s = _do_summarize(raw)

            if _has_findings(s):
                _res_c["findings"] += 1
                return dim, s

            # Semantic retry
            _res_c["retries"] += 1
            try:
                alt_q_raw = _llm(
                    RESEARCHER_SYSTEM,
                    f'Search for "{topic} — {dim.replace("_", " ")}" found no citable data.\n'
                    f'Suggest ONE alternative search query (keywords only, max 12 words) '
                    f'to find {dim.replace("_", " ")} data for "{topic}" from a completely different angle.',
                    model=MODEL_PRO, max_tokens=80, temperature=0.3,
                )
                alt_q = (alt_q_raw.strip().split('\n')[0].lstrip("•-0123456789.) ").strip()[:200])
            except Exception:
                alt_q = ""

            if alt_q:
                alt_dr = _gather_dimension(topic, dim, alt_q)
                if alt_dr.get("text"):
                    s2 = _do_summarize(alt_dr["text"], _fmt_sources(alt_dr.get("sources", [])))
                    if _has_findings(s2):
                        _res_c["findings"] += 1
                        return dim, s2
            _res_c["unavail"] += 1
            return dim, s

        with ThreadPoolExecutor(max_workers=4) as ex:
            sum_futs = {ex.submit(_summarize_dim, dr): dk for dk, dr in dim_results.items()}
            for f in as_completed(sum_futs):
                try:
                    dim, s = f.result()
                    summaries[dim] = s
                    m2 = _make_msg("researcher",
                        f"✅ **{dim.replace('_', ' ').title()}**\n\n{s}", "research")
                    emit(q, "researcher", m2["content"], "research")
                    msgs_out.append(m2)
                except Exception as e:
                    pass

        if rag_ctx:
            m3 = _make_msg("researcher",
                f"📚 **Knowledge Base** ({len(rag_ctx.split())} words):\n{rag_ctx[:400]}...", "research")
            emit(q, "researcher", m3["content"], "research")
            msgs_out.append(m3)
            summaries["knowledge_base"] = rag_ctx

        output = "\n\n".join(
            f"**{k.replace('_', ' ').title()}**:\n{v}"
            for k, v in summaries.items()
        )

        # Signal check
        try:
            sig_raw = _llm(
                RESEARCHER_SYSTEM,
                f"Review summaries for \"{topic}\" and emit a signal:\n\n{output[:2000]}\n\n"
                "Use TEMPLATE C. Reply with ONLY the signal line.",
                model=MODEL_PRO, max_tokens=200, temperature=0.3,
            )
        except Exception:
            sig_raw = "[RESEARCH: WEAK | gaps: signal check failed]"
        sig_match = re.search(r'\[RESEARCH[^\]]*\]', sig_raw, re.IGNORECASE)
        sig = sig_match.group(0) if sig_match else "[RESEARCH: WEAK | gaps: signal unclear]"
        output += f"\n\n{sig}"

        m4 = _make_msg("researcher",
            f"📊 Initial research complete. Signal: `{sig}`", "research")
        emit(q, "researcher", m4["content"], "research")
        msgs_out.append(m4)
        task_label = "initial"

        new_entries = [{"round": rnd, "agent": "researcher",
                        "task": "initial", "output": output, "signal": sig}]

        return {
            "researcher_calls": new_researcher_calls,
            "post_analyst_calls": new_post_analyst,
            "user_rag_chunks": user_rag_chunks,
            "workspace": new_entries,
            "team_messages": msgs_out,
        }

    # ── FOLLOW-UP search ──────────────────────────────────────────────────────
    directive = param.strip() or topic
    m5 = _make_msg("researcher",
        f"🔍 **Follow-up research #{new_researcher_calls} — directive:** _{directive}_",
        "research")
    emit(q, "researcher", m5["content"], "research")
    msgs_out.append(m5)

    last_critic = next((w for w in reversed(state.get("workspace", [])) if w["agent"] == "critic"), None)
    critic_ctx_str = ""
    if last_critic and state.get("workspace") and state["workspace"][-1]["agent"] == "critic":
        critic_ctx_str = f"\nCritic concern: {last_critic['output'][-400:]}\n"

    # Decompose directive into queries
    try:
        decompose_raw = _llm(
            RESEARCHER_SYSTEM,
            f"Research topic: {topic}\n"
            f"Research directive: {directive}\n\n"
            f"List specific web search queries to cover this directive.\n"
            f"ENTITY COVERAGE RULE (mandatory):\n"
            f"  1. Count every distinct entity in the directive.\n"
            f"  2. Generate EXACTLY ONE query per entity.\n"
            f"  3. Add market-level queries ONLY if directive explicitly requests market size.\n"
            f"  4. Hard cap: maximum 8 queries total.\n"
            f"Include '2025' or '2026' in queries where current data is needed.\n"
            f"Format: one search query per line, keywords only (max 12 words each), no bullets.",
            model=MODEL_PRO, max_tokens=400, temperature=0.3,
        )
        queries = [
            ln.strip().lstrip("•-0123456789.) ")
            for ln in decompose_raw.split("\n")
            if ln.strip() and not ln.strip().startswith("#") and len(ln.strip()) > 5
        ][:8]
    except Exception:
        queries = []

    if not queries:
        queries = [directive]

    m6 = _make_msg("researcher",
        f"🔍 **{len(queries)} queries:**\n" + "\n".join(f"  • *{q}*" for q in queries),
        "research")
    emit(q, "researcher", m6["content"], "research")
    msgs_out.append(m6)

    # Run queries in batches of 4
    all_sr: list = []
    for i in range(0, len(queries), 4):
        batch = queries[i:i+4]
        batch_results = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(_search_with_retry, bq): bq for bq in batch}
            for f in as_completed(futs):
                bq = futs[f]
                try:
                    batch_results[bq] = f.result()
                except Exception as e:
                    batch_results[bq] = {"text": "", "sources": [], "query": bq}
        for bq in batch:
            all_sr.append(batch_results.get(bq, {"text": "", "sources": [], "query": bq}))

    _res_c2 = {"findings": 0, "unavail": 0, "retries": 0}

    def _summarize_query(sr):
        query   = sr["query"]
        text    = sr.get("text", "")
        sources = _fmt_sources(sr.get("sources", []))
        notes   = ""
        if sr.get("limited"):
            notes = "\n\n⚠️ Limited data. Mark ⚠️ LIMITED SOURCE."
        if not text:
            notes = "\n\n❌ Zero results."

        def _do_q_sum(q_str, t, src, extra=""):
            return _llm(
                RESEARCHER_SYSTEM,
                f"Research task: {q_str}{critic_ctx_str}\n\n"
                f"Sources:\n{src}\n\n"
                f"Content:\n{t[:1800]}{extra}\n\n"
                "Use TEMPLATE B. Tag each finding [Data], [Estimate], or [Claim]. "
                "Cite source URLs. Prefer 2025 sources. End with signal.",
                model=MODEL_PRO, max_tokens=550, temperature=0.3,
            )

        s = ""
        if text or sources:
            s = _do_q_sum(query, text, sources, notes)

        if _has_findings(s):
            _res_c2["findings"] += 1
            return s

        # Semantic retry
        _res_c2["retries"] += 1
        try:
            alt_q_raw = _llm(
                RESEARCHER_SYSTEM,
                f'Search for "{query}" found no citable data.\n'
                f'Suggest ONE alternative search query (keywords only, max 12 words) '
                f'to find this from a completely different angle.',
                model=MODEL_PRO, max_tokens=60, temperature=0.3,
            )
            alt_q = (alt_q_raw.strip().split('\n')[0].lstrip("•-0123456789.) ").strip()[:200])
        except Exception:
            alt_q = ""

        if alt_q:
            alt_sr = _search_with_retry(alt_q)
            if alt_sr.get("text"):
                alt_src = _fmt_sources(alt_sr.get("sources", []))
                s2 = _do_q_sum(alt_q, alt_sr["text"], alt_src)
                if _has_findings(s2):
                    _res_c2["findings"] += 1
                    return s2

        _res_c2["unavail"] += 1
        return (f"## 🔍 — {query}\n\n**Found:** Nothing after retry\n\n"
                f"**Not found:** {query}\n\n**Plausibility:** N/A\n\n"
                f"**Summary:** No citable data found after two search attempts.\n\n"
                f"[RESEARCH: UNAVAILABLE | data: {query}]")

    query_summaries = []
    for i in range(0, len(all_sr), 4):
        batch_sums = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(_summarize_query, sr): idx
                    for idx, sr in enumerate(all_sr[i:i+4])}
            for f in as_completed(futs):
                idx = futs[f]
                try:
                    batch_sums[idx] = f.result()
                except Exception as e:
                    batch_sums[idx] = f"[ERROR summarizing query: {e}]"
        for idx in range(min(4, len(all_sr) - i)):
            query_summaries.append(batch_sums.get(idx, ""))

    all_outputs = []
    for s in query_summaries:
        m7 = _make_msg("researcher", s, "research")
        emit(q, "researcher", s, "research")
        msgs_out.append(m7)
        all_outputs.append(s)
        sm = re.search(r'\[RESEARCH[^\]]*\]', s, re.IGNORECASE)
        if sm:
            sig = sm.group(0)

    output = "\n\n".join(all_outputs)
    if not output.strip():
        output = (f"No data found for directive: {directive}. "
                  "[RESEARCH: UNAVAILABLE | data: all queries returned empty]")

    new_entries = [{"round": rnd, "agent": "researcher",
                    "task": directive, "output": output, "signal": sig}]

    return {
        "researcher_calls": new_researcher_calls,
        "post_analyst_calls": new_post_analyst,
        "workspace": new_entries,
        "team_messages": msgs_out,
    }


# ── Analyst node ──────────────────────────────────────────────────────────────

def analyst_node(state: State, config: RunnableConfig) -> dict:
    q = get_queue(config)
    workspace      = state.get("workspace", [])
    param          = state.get("next_param", "")
    rnd            = state.get("rnd", 1)
    analyst_called = state.get("analyst_called", False)
    revision_count = state.get("revision_count", 0)
    stale_report   = state.get("stale_report", "")
    stale_age      = state.get("stale_age", 0)
    researcher_calls = state.get("researcher_calls", 0)

    msgs_out: List[TeamMsg] = []

    ctx = _build_ctx_for("analyst", workspace)

    extra = ""
    if stale_report and researcher_calls <= 2:
        extra += (f"\n\nHISTORICAL CONTEXT (prev report, {stale_age}d old):\n"
                  f"{stale_report[:1500]}")

    # Add RAG context if available
    rag_context = ""
    try:
        from rag.manager import query_rag
        rag_context = query_rag(state["topic"], n_results=4)
    except Exception:
        pass
    if rag_context:
        extra += f"\n\nKNOWLEDGE BASE:\n{rag_context[:800]}"

    result = _llm_ctx(
        ANALYST_SYSTEM,
        f"Supervisor instruction: {param}{extra}",
        ctx,
        model=MODEL_V3, max_tokens=1800, temperature=0.5,
    )

    emit(q, "analyst", result, "analysis")
    msgs_out.append(_make_msg("analyst", result, "analysis"))

    am = re.search(r'\[ANALYSIS[^\]]*\]', result, re.IGNORECASE)
    sig = am.group(0) if am else ""

    new_revision_count = revision_count
    if analyst_called:
        last_agent = next(
            (w["agent"] for w in reversed(workspace) if w["agent"] != "researcher"), None)
        if last_agent == "critic":
            new_revision_count += 1

    new_entry = [{"round": rnd, "agent": "analyst",
                  "task": param, "output": result, "signal": sig}]

    return {
        "analyst_called": True,
        "revision_count": new_revision_count,
        "workspace": new_entry,
        "team_messages": msgs_out,
    }


# ── Validator (Data Synthesizer) node ────────────────────────────────────────

def validator_node(state: State, config: RunnableConfig) -> dict:
    q = get_queue(config)
    workspace       = state.get("workspace", [])
    user_rag_chunks = state.get("user_rag_chunks", [])
    rnd             = state.get("rnd", 1)
    topic           = state["topic"]

    msgs_out: List[TeamMsg] = []

    all_findings = _extract_finding_texts(workspace)
    rag_chunks   = user_rag_chunks

    if all_findings and rag_chunks:
        matched, supplements = _embed_pair_match(all_findings, rag_chunks)

        pairs_block = ""
        for p in matched:
            pairs_block += (
                f"\n[OVERLAP — sim={p['sim']}]\n"
                f"  Web finding : {p['best_finding'][:220]}\n"
                f"  RAG chunk   : [{p['filename']}] {p['text'][:220]}\n"
            )

        supps_block = ""
        for c in supplements[:15]:
            supps_block += f"\n[RAG-ONLY — {c['filename']}]\n  {c['text'][:200]}\n"

        status_msg = (
            f"🔄 **Data Reconciliation** — {len(matched)} overlap pairs, "
            f"{len(supplements)} RAG supplements\n"
            f"Pair-match threshold: {PAIR_MATCH_THRESHOLD} | Findings: {len(all_findings)} | "
            f"RAG chunks: {len(rag_chunks)}"
        )
        emit(q, "validator", status_msg, "validation")
        msgs_out.append(_make_msg("validator", status_msg, "validation"))

        synth_input = (
            f"Research topic: {topic}\n\n"
            f"OVERLAP PAIRS (embedding similarity ≥ {PAIR_MATCH_THRESHOLD}):\n"
            f"{pairs_block or '(none — web and RAG cover different metrics)'}\n\n"
            f"RAG SUPPLEMENTS (imported data not covered by web search):\n"
            f"{supps_block or '(none)'}"
        )

        result = _llm(
            DATA_SYNTHESIZER_SYSTEM,
            synth_input,
            model=MODEL_V3, max_tokens=800, temperature=0.2,
        )
    else:
        result = (
            f"[SYNTHESIS: SKIPPED — "
            f"{'no researcher findings yet' if not all_findings else 'no user RAG data'}]"
        )

    emit(q, "validator", result, "validation")
    msgs_out.append(_make_msg("validator", result, "validation"))

    sig = "[SYNTHESIS: COMPLETE]" if "SYNTHESIS: COMPLETE" in result else ""
    new_entry = [{"round": rnd, "agent": "validator",
                  "task": "reconcile web vs RAG", "output": result, "signal": sig}]

    return {
        "validator_called": True,
        "workspace": new_entry,
        "team_messages": msgs_out,
    }


# ── Critic node ───────────────────────────────────────────────────────────────

def critic_node(state: State, config: RunnableConfig) -> dict:
    q = get_queue(config)
    workspace = state.get("workspace", [])
    param     = state.get("next_param", "")
    rnd       = state.get("rnd", 1)

    msgs_out: List[TeamMsg] = []
    ctx = _build_ctx_for("critic", workspace)

    result = _llm_ctx(
        CRITIC_SYSTEM,
        f"Supervisor instruction: {param}",
        ctx,
        model=MODEL_V3, max_tokens=900, temperature=0.6,
    )

    emit(q, "critic", result, "critique")
    msgs_out.append(_make_msg("critic", result, "critique"))

    cm = re.search(r'\[VERDICT[^\]]*\]', result, re.IGNORECASE)
    sig = cm.group(0) if cm else ""

    new_entry = [{"round": rnd, "agent": "critic",
                  "task": param, "output": result, "signal": sig}]

    return {
        "workspace": new_entry,
        "team_messages": msgs_out,
    }


# ── Writer node ───────────────────────────────────────────────────────────────

def writer_node(state: State, config: RunnableConfig) -> dict:
    q = get_queue(config)
    workspace = state.get("workspace", [])
    param     = state.get("next_param", "write final report")
    rnd       = state.get("rnd", 1)
    topic     = state["topic"]

    msgs_out: List[TeamMsg] = []

    writer_ctx = []
    for r in (w for w in workspace if w["agent"] == "researcher"):
        writer_ctx.append({"role": "user",
                           "content": f"[RESEARCH Round {r['round']}]\n{r['output']}"})
    for w in workspace:
        if w["agent"] == "analyst":
            writer_ctx.append({"role": "user", "content": f"[ANALYST]\n{w['output']}"})
        elif w["agent"] == "critic":
            writer_ctx.append({"role": "user", "content": f"[CRITIC]\n{w['output']}"})

    rag_note = ""
    try:
        from rag.manager import query_rag
        rag_ctx = query_rag(topic, n_results=4)
        if rag_ctx:
            rag_note = f"\n\nKNOWLEDGE BASE:\n{rag_ctx[:800]}"
    except Exception:
        pass

    result = _llm_ctx(
        WRITER_SYSTEM,
        f"Supervisor instruction: {param}{rag_note}",
        writer_ctx,
        model=MODEL_V3, max_tokens=3000, temperature=0.4,
    )

    emit(q, "writer", result, "writing")
    msgs_out.append(_make_msg("writer", result, "writing"))

    sig = "[REPORT: COMPLETE]" if re.search(r'\[REPORT', result, re.IGNORECASE) else ""
    new_entry = [{"round": rnd, "agent": "writer",
                  "task": param, "output": result, "signal": sig}]

    return {
        "report_content": result,
        "workspace": new_entry,
        "team_messages": msgs_out,
    }


# ── Finalize node ─────────────────────────────────────────────────────────────

def finalize_node(state: State, config: RunnableConfig) -> dict:
    q = get_queue(config)
    topic        = state.get("topic", "the requested topic")
    report       = state.get("report_content", "")
    workspace    = state.get("workspace", [])
    stale_id     = state.get("stale_entry_id", None) or None

    # Emit completion
    counts: dict = {}
    for w in workspace:
        counts[w["agent"]] = counts.get(w["agent"], 0) + 1
    stats = " · ".join(f"{a} ×{n}" for a, n in counts.items())
    words = len(report.split())

    complete_msg = (
        f"🎉 **Complete — {topic}**\n\n"
        f"{len(workspace)} rounds · {stats}\n\n"
        f"Report: ~{words} words · available in the Report tab."
    )
    emit(q, "supervisor", complete_msg, "complete")

    # Save to file
    safe   = "".join(c if c.isalnum() or c in "-_ " else "_" for c in topic)[:40]
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if report:
        (REPORTS_DIR / f"gtm_{safe}_{ts_str}.md").write_text(report, encoding="utf-8")

    # Store in semantic cache — DISABLED for testing
    # if report and report != "Report generation incomplete.":
    #     _cache_store(topic, report, stale_id)

    supervisor_resp = (
        f"Your GTM Intelligence Report on **{topic}** is ready! "
        f"~{words} words. Check the **Report** tab."
    )

    return {
        "supervisor_response": supervisor_resp,
        "team_messages": [_make_msg("supervisor", complete_msg, "complete")],
    }


# ── Doc analyst node ──────────────────────────────────────────────────────────

def doc_analyst_node(state: State, config: RunnableConfig) -> dict:
    q = get_queue(config)
    doc = state.get("doc_content", "")[:8000]
    msgs_out: List[TeamMsg] = []

    emit(q, "analyst", "📄 Starting document analysis...", "analysis")
    msgs_out.append(_make_msg("analyst", "📄 Starting document analysis...", "analysis"))

    rag_note = ""
    try:
        from rag.manager import query_rag
        rag_ctx = query_rag(doc[:500], n_results=3)
        if rag_ctx:
            rag_note = f"\n\nKnowledge Base:\n{rag_ctx[:800]}"
    except Exception:
        pass

    analysis = _llm(
        DOCUMENT_ANALYSIS_SYSTEM,
        f"Analyze this document:\n\n{doc}{rag_note}",
        model=MODEL_V3, max_tokens=1500, temperature=0.5,
    )

    emit(q, "analyst", analysis, "analysis")
    msgs_out.append(_make_msg("analyst", analysis, "analysis"))

    # Critic pass
    critique = _llm(
        CRITIC_SYSTEM,
        f"Jamie analysed a document:\n{analysis}\n\nIdentify 2 key gaps. Under 200 words.",
        model=MODEL_V3, max_tokens=300, temperature=0.6,
    )

    emit(q, "critic", critique, "critique")
    msgs_out.append(_make_msg("critic", critique, "critique"))

    emit(q, "supervisor", "✅ Document analysis complete.", "complete")
    msgs_out.append(_make_msg("supervisor", "✅ Document analysis complete.", "complete"))

    combined = f"{analysis}\n\n---\n\n**Quality Review (Morgan):**\n{critique}"

    return {
        "report_content": combined,
        "supervisor_response": "Document analysis complete. Check the **Report** tab.",
        "team_messages": msgs_out,
    }


# ── Routing functions ─────────────────────────────────────────────────────────

def route_init(state: State) -> Literal["general", "research", "doc_analysis"]:
    # If cache hit returned a report immediately, we're done
    if state.get("route") == "general" and state.get("report_content"):
        return "general"
    return state.get("route", "general")


def route_decide(state: State) -> str:
    action = state.get("next_action", "DONE")
    mapping = {
        "CALL_RESEARCHER": "researcher",
        "CALL_ANALYST":    "analyst",
        "CALL_VALIDATOR":  "validator",
        "CALL_CRITIC":     "critic",
        "CALL_WRITER":     "writer",
    }
    return mapping.get(action, "writer")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILD
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(State)

    g.add_node("supervisor_init",   supervisor_init_node)
    g.add_node("direct_response",   direct_response_node)
    g.add_node("supervisor_decide", supervisor_decide_node)
    g.add_node("researcher",        researcher_node)
    g.add_node("analyst",           analyst_node)
    g.add_node("validator",         validator_node)
    g.add_node("critic",            critic_node)
    g.add_node("writer",            writer_node)
    g.add_node("finalize",          finalize_node)
    g.add_node("doc_analyst",       doc_analyst_node)

    g.add_edge(START, "supervisor_init")

    g.add_conditional_edges(
        "supervisor_init",
        route_init,
        {
            "general":      "direct_response",
            "research":     "supervisor_decide",
            "doc_analysis": "doc_analyst",
        },
    )

    g.add_edge("direct_response", END)
    g.add_edge("doc_analyst", END)

    # ReAct loop: supervisor_decide → agent → supervisor_decide
    g.add_conditional_edges(
        "supervisor_decide",
        route_decide,
        {
            "researcher": "researcher",
            "analyst":    "analyst",
            "validator":  "validator",
            "critic":     "critic",
            "writer":     "writer",
        },
    )

    g.add_edge("researcher", "supervisor_decide")
    g.add_edge("analyst",    "supervisor_decide")
    g.add_edge("validator",  "supervisor_decide")
    g.add_edge("critic",     "supervisor_decide")

    g.add_edge("writer",   "finalize")
    g.add_edge("finalize", END)

    return g.compile()


GRAPH = build_graph()
