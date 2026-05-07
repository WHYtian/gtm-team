"""
Performance telemetry for the GTM research pipeline.

Collects per-call timing, character-based token estimates, researcher stats
(queries, retries, RAG chunks), synthesizer pair counts, and critic verdicts.
Outputs a clean Markdown report at the end of each run.

Token estimation: 1 token ≈ 4 characters (reasonable for English + some CJK).
"""
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

TELEMETRY_DIR = Path("/home/admin/gtm-team/telemetry")
TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

_CPT = 4  # chars per token


def _tok(chars: int) -> int:
    return max(0, chars // _CPT)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    seq: int
    rnd: int
    agent_id: str
    model: str
    duration_s: float
    input_tok: int
    output_tok: int

    @property
    def total_tok(self) -> int:
        return self.input_tok + self.output_tok


@dataclass
class ResearcherRound:
    rnd: int
    call_type: str            # "initial (4-dim)" | "follow-up #N"
    duration_s: float = 0.0
    queries_generated: int = 0
    queries_with_findings: int = 0
    queries_unavailable: int = 0
    semantic_retries: int = 0
    rag_chunks_found: int = 0
    input_tok: int = 0
    output_tok: int = 0

    @property
    def total_tok(self) -> int:
        return self.input_tok + self.output_tok


@dataclass
class SynthStats:
    web_findings: int = 0
    rag_chunks: int = 0
    matched_pairs: int = 0
    supplements: int = 0
    confirmed: int = 0
    conflicts: int = 0
    scope_mismatches: int = 0
    rag_supplements_llm: int = 0
    duration_s: float = 0.0
    input_tok: int = 0
    output_tok: int = 0


@dataclass
class CriticRecord:
    rnd: int
    verdict: str
    reason: str


# ── Collector ────────────────────────────────────────────────────────────────

class Telemetry:

    def __init__(self, topic: str):
        self.topic = topic
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._start = time.time()

        self.calls: list[CallRecord] = []
        self.researcher_rounds: list[ResearcherRound] = []
        self.synth = SynthStats()
        self.critic_records: list[CriticRecord] = []

        # Pipeline-level counters (filled at end of run)
        self.cache_hit = False
        self.total_rounds = 0
        self.researcher_calls_pre = 0
        self.researcher_calls_post = 0
        self.analyst_revisions = 0

        # Mutable current researcher round — updated by inner closures
        self._cur_res: Optional[ResearcherRound] = None
        self._cur_res_t0: float = 0.0
        self._seq = 0

    # ── Agent call recording ─────────────────────────────────────────────────

    def record_call(self, rnd: int, agent_id: str, model: str,
                    in_chars: int, out_chars: int, duration_s: float):
        self._seq += 1
        self.calls.append(CallRecord(
            seq=self._seq, rnd=rnd, agent_id=agent_id, model=model,
            duration_s=round(duration_s, 2),
            input_tok=_tok(in_chars), output_tok=_tok(out_chars),
        ))

    # ── Researcher round ─────────────────────────────────────────────────────

    def start_researcher_round(self, rnd: int, call_type: str):
        self._cur_res = ResearcherRound(rnd=rnd, call_type=call_type)
        self._cur_res_t0 = time.time()

    def finish_researcher_round(self):
        if self._cur_res:
            self._cur_res.duration_s = round(time.time() - self._cur_res_t0, 2)
            self.researcher_rounds.append(self._cur_res)
            self._cur_res = None

    def add_res_tokens(self, in_chars: int, out_chars: int):
        if self._cur_res:
            self._cur_res.input_tok  += _tok(in_chars)
            self._cur_res.output_tok += _tok(out_chars)

    # ── Synthesizer ──────────────────────────────────────────────────────────

    def record_synth(self, output: str, in_chars: int, duration_s: float):
        self.synth.duration_s  = round(duration_s, 2)
        self.synth.input_tok   = _tok(in_chars)
        self.synth.output_tok  = _tok(len(output))

        def _bullets(section: str) -> int:
            return len(re.findall(r'^\s*-\s+\*\*', section, re.MULTILINE))

        self.synth.confirmed          = _bullets(_after(output, r'✅\s*CONFIRMED|###.*CONFIRMED'))
        self.synth.conflicts          = _bullets(_after(output, r'⚠️\s*CONFLICT|###.*CONFLICT'))
        self.synth.scope_mismatches   = _bullets(_after(output, r'➕\s*SCOPE|###.*SCOPE'))
        self.synth.rag_supplements_llm= _bullets(_after(output, r'📚\s*RAG SUPPLEMENT|###.*SUPPLEMENT'))

    # ── Critic ───────────────────────────────────────────────────────────────

    def record_critic(self, rnd: int, output: str):
        vm = re.search(r'\[VERDICT:\s*(APPROVED|NEEDS_REVISION|REJECT_DATA)', output, re.IGNORECASE)
        verdict = vm.group(1).upper() if vm else "UNKNOWN"
        rm = re.search(r'reason:\s*(\w+)', output, re.IGNORECASE)
        reason = rm.group(1).lower() if rm else ("-" if verdict != "REJECT_DATA" else "reject_data")
        if verdict == "REJECT_DATA":
            reason = "reject_data"
        self.critic_records.append(CriticRecord(rnd=rnd, verdict=verdict, reason=reason))

    # ── Report ───────────────────────────────────────────────────────────────

    @property
    def elapsed_s(self) -> float:
        return time.time() - self._start

    def save_markdown(self) -> Path:
        slug = re.sub(r'[^a-zA-Z0-9 \-]', '_', self.topic)[:35].strip().replace(' ', '_')
        path = TELEMETRY_DIR / f"perf_{self.session_id}_{slug}.md"
        path.write_text(self._render(), encoding="utf-8")
        return path

    # ── Markdown renderer ────────────────────────────────────────────────────

    def _render(self) -> str:
        total_s = self.elapsed_s
        mm, ss  = divmod(int(total_s), 60)
        ts_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        n_logic  = sum(1 for c in self.critic_records if c.reason == "logic_error")
        n_reject = sum(1 for c in self.critic_records if c.reason == "reject_data")
        n_miss   = sum(1 for c in self.critic_records if c.reason == "missing_data")
        n_ok     = sum(1 for c in self.critic_records if c.verdict == "APPROVED")

        L = []   # output lines

        # ── Header ───────────────────────────────────────────────────────────
        L += [
            "# GTM Research — Performance Report",
            "",
            f"**Topic:** {self.topic}  ",
            f"**Date:** {ts_str}  ",
            f"**Session:** {self.session_id}  ",
            f"**Total Duration:** {mm}m {ss}s  ",
            f"**Cache Hit:** {'Yes' if self.cache_hit else 'No'}  ",
            "",
            "---",
            "",
        ]

        # ── 1. Pipeline Summary ───────────────────────────────────────────────
        L += [
            "## 1. Pipeline Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Rounds | {self.total_rounds} |",
            f"| Researcher Calls — Pre-Analyst | {self.researcher_calls_pre} |",
            f"| Researcher Calls — Post-Analyst | {self.researcher_calls_post} |",
            f"| Analyst Revisions | {self.analyst_revisions} |",
            f"| Critic: Approved | {n_ok} |",
            f"| Critic: logic_error | {n_logic} |",
            f"| Critic: REJECT_DATA | {n_reject} |",
            f"| Critic: missing_data | {n_miss} |",
            "",
            "---",
            "",
        ]

        # ── 2. Agent Call Log (supervisor / analyst / critic / writer / synth) ─
        tot_in  = sum(c.input_tok  for c in self.calls)
        tot_out = sum(c.output_tok for c in self.calls)
        L += [
            "## 2. Agent Call Log",
            "",
            "| # | Rnd | Agent | Model | Duration (s) | Input Tok | Output Tok | Total |",
            "|---|-----|-------|-------|-------------|-----------|-----------|-------|",
        ]
        for c in self.calls:
            L.append(
                f"| {c.seq} | {c.rnd} | {c.agent_id} | {c.model[:30]} "
                f"| {c.duration_s:.1f} | {c.input_tok:,} | {c.output_tok:,} | {c.total_tok:,} |"
            )
        L += [
            "",
            f"**Sub-total (excl. researcher):** Input {tot_in:,} | Output {tot_out:,} | Total {tot_in+tot_out:,}",
            "",
            "---",
            "",
        ]

        # ── 3. Per-Agent Aggregation ──────────────────────────────────────────
        by: dict[str, dict] = {}
        for c in self.calls:
            b = by.setdefault(c.agent_id, {"n": 0, "in": 0, "out": 0, "dur": 0.0})
            b["n"] += 1; b["in"] += c.input_tok; b["out"] += c.output_tok; b["dur"] += c.duration_s
        L += [
            "## 3. Per-Agent Totals",
            "",
            "| Agent | Calls | Total Input | Total Output | Total Tokens | Total Duration (s) | Avg Duration (s) |",
            "|-------|-------|-------------|-------------|-------------|-------------------|-----------------|",
        ]
        for aid, b in by.items():
            avg = b["dur"] / b["n"] if b["n"] else 0
            L.append(
                f"| {aid} | {b['n']} | {b['in']:,} | {b['out']:,} | {b['in']+b['out']:,} "
                f"| {b['dur']:.1f} | {avg:.1f} |"
            )
        L += ["", "---", ""]

        # ── 4. Researcher Stats ───────────────────────────────────────────────
        res_in  = sum(r.input_tok  for r in self.researcher_rounds)
        res_out = sum(r.output_tok for r in self.researcher_rounds)
        L += [
            "## 4. Researcher Stats",
            "",
            "| Rnd | Type | Duration (s) | Queries | w/ Findings | Unavailable | Retries | RAG Chunks | Input Tok | Output Tok |",
            "|-----|------|-------------|---------|-------------|-------------|---------|------------|-----------|-----------|",
        ]
        for r in self.researcher_rounds:
            L.append(
                f"| {r.rnd} | {r.call_type} | {r.duration_s:.1f} | {r.queries_generated} "
                f"| {r.queries_with_findings} | {r.queries_unavailable} | {r.semantic_retries} "
                f"| {r.rag_chunks_found} | {r.input_tok:,} | {r.output_tok:,} |"
            )
        L += [
            "",
            f"**Researcher sub-total:** Input {res_in:,} | Output {res_out:,} | Total {res_in+res_out:,}",
            "",
            "---",
            "",
        ]

        # ── 5. Data Synthesizer ───────────────────────────────────────────────
        s = self.synth
        L += ["## 5. Data Synthesizer", ""]
        if s.web_findings or s.rag_chunks:
            matrix = f"{s.web_findings} × {s.rag_chunks} = {s.web_findings * s.rag_chunks}"
            L += [
                "| Metric | Value |",
                "|--------|-------|",
                f"| Web Findings Extracted | {s.web_findings} |",
                f"| User RAG Chunks (topic-relevant) | {s.rag_chunks} |",
                f"| Embedding Matrix (findings × chunks) | {matrix} |",
                f"| Matched Pairs (cosine ≥ 0.42) | {s.matched_pairs} |",
                f"| RAG Supplements | {s.supplements} |",
                f"| LLM — Confirmed | {s.confirmed} |",
                f"| LLM — Conflicts | {s.conflicts} |",
                f"| LLM — Scope Mismatches | {s.scope_mismatches} |",
                f"| LLM — RAG Supplements reported | {s.rag_supplements_llm} |",
                f"| Call Duration (s) | {s.duration_s:.1f} |",
                f"| Input Tok (est) | {s.input_tok:,} |",
                f"| Output Tok (est) | {s.output_tok:,} |",
            ]
        else:
            L.append("*Skipped — no user-uploaded documents in this run.*")
        L += ["", "---", ""]

        # ── 6. Critic Verdicts ────────────────────────────────────────────────
        L += [
            "## 6. Critic Verdicts",
            "",
            "| Rnd | Verdict | Reason |",
            "|-----|---------|--------|",
        ]
        if self.critic_records:
            for cr in self.critic_records:
                L.append(f"| {cr.rnd} | {cr.verdict} | {cr.reason} |")
        else:
            L.append("| — | — | No critic calls recorded |")
        L += ["", "---", ""]

        # ── 7. Overall Token Summary ──────────────────────────────────────────
        all_in  = tot_in  + res_in  + s.input_tok
        all_out = tot_out + res_out + s.output_tok
        L += [
            "## 7. Overall Token Summary",
            "",
            "| Scope | Input (est) | Output (est) | Total |",
            "|-------|-------------|-------------|-------|",
            f"| Supervisor + Analyst + Critic + Writer + Synth | {tot_in:,} | {tot_out:,} | {tot_in+tot_out:,} |",
            f"| Researcher (all rounds) | {res_in:,} | {res_out:,} | {res_in+res_out:,} |",
            f"| **Grand Total** | **{all_in:,}** | **{all_out:,}** | **{all_in+all_out:,}** |",
            "",
            "> Token counts are estimates: 1 token ≈ 4 characters.",
            "",
        ]

        return "\n".join(L)


# ── Helper ────────────────────────────────────────────────────────────────────

def _after(text: str, pattern: str) -> str:
    """Text content after the first matching header until the next ### header."""
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r'\n###', text[start:])
    return text[start: start + nxt.start()] if nxt else text[start:]
