#!/usr/bin/env python3
"""GTM Pipeline - delegates to openclaw gtm-pipeline skill, emits SSE events."""
import sys
import json
import subprocess
import os
import re
import queue
import threading
import time
from pathlib import Path

SKILL_SCRIPT = Path.home() / ".openclaw/workspace/skills/gtm-pipeline/scripts/gtm_pipeline.py"
PYTHON = sys.executable


def parse_log_line(line: str, known_dims: list) -> dict | None:
    """Parse a gtm_pipeline.py log line into an SSE event."""
    line = line.strip()
    if not line:
        return None

    # [HH:MM:SS] [AGENT:dim] 搜索: xxx
    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[AGENT:(\w+)\]\s*搜索:\s*(.+)$', line)
    if m:
        dim = m.group(1)
        return {"type": "agent_start", "dimension": dim, "query": m.group(2)}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[AGENT:(\w+)\]\s*找到\s*(\d+)\s*条结果$', line)
    if m:
        return {"type": "agent_search_done", "dimension": m.group(1), "count": int(m.group(2))}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[AGENT:(\w+)\]\s*抓取\s*(\d+)\s*字符$', line)
    if m:
        return {"type": "agent_scrape_done", "dimension": m.group(1),
                "chars": int(m.group(2)), "success": True}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[AGENT:(\w+)\]\s*超时', line)
    if m:
        return {"type": "agent_scrape_done", "dimension": m.group(1),
                "chars": 0, "success": False, "error": "timeout"}

    # [HH:MM:SS] [ORCH] Agent [dim] 完成，收集到 N 篇文档
    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[ORCH\].*Agent\s*\[(.+?)\]\s*完成.*?(\d+)\s*篇文档', line)
    if m:
        return {"type": "agent_done", "dimension": m.group(1), "docs_count": int(m.group(2))}

    # Phase 2
    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[INGEST\].*存入\s*(\d+)\s*篇文档', line)
    if m:
        return None  # ingest completion, we use ORCH line for phase transition

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[CRITIC\]\s*交叉验证', line)
    if m:
        return None  # will be handled by the ORCH phase log

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[CRITIC\]\s*并行运行\s*(\d+)\s*对', line)
    if m:
        return None

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[CRITIC\]\s*\[(\w+)\]\s*可靠性:\s*(\w+)', line)
    if m:
        return {"type": "critic_done", "pair": m.group(1), "reliability": m.group(2).lower()}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[RAG\]\s*检索到\s*(\d+)\s*个', line)
    if m:
        return {"type": "rag_done", "chunks": int(m.group(1))}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[REPORT\].*生成', line)
    if m:
        return {"type": "report_start"}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[REPORT\].*报告已保存.*?(\d+)\s*字符', line)
    if m:
        return {"type": "report_done", "chars": int(m.group(1))}

    # ORCH phase transitions
    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[ORCH\]\s*Phase 1 完成.*?(\d+)\s*篇文档', line)
    if m:
        return {"type": "phase2_start", "total_docs": int(m.group(1))}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[ORCH\]\s*Phase 2: 并行', line)
    if m:
        return None  # already emitted via Phase 1 完成

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[ORCH\]\s*Phase 2 完成', line)
    if m:
        return {"type": "ingest_done"}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[ORCH\]\s*全流程完成.*?(\d+\.?\d*)s', line)
    if m:
        return {"type": "pipeline_done_internal"}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[ORCH\].*报告路径:\s*(.+)$', line)
    if m:
        return {"type": "report_path", "path": m.group(1).strip()}

    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[ORCH\]\s*研究主题:\s*(.+)$', line)
    if m:
        return None

    # Launching agents
    m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*\[ORCH\]\s*启动\s*(\d+)\s*个', line)
    if m:
        return None

    return None


def run_pipeline_with_events(topic: str, eq: queue.Queue, max_search: int = 3) -> dict:
    """Run the openclaw gtm-pipeline skill and emit SSE events."""
    from datetime import datetime
    t0 = time.time()

    eq.put({
        "type": "pipeline_start", "topic": topic,
        "dimensions": [
            {"dim": "market_overview", "label": "Market Overview", "color": "#00d4aa", "query": f"{topic} market size"},
            {"dim": "competitive_landscape", "label": "Competitive Landscape", "color": "#f472b6", "query": f"{topic} competitors"},
            {"dim": "regulatory_env", "label": "Regulatory Environment", "color": "#fbbf24", "query": f"{topic} regulations"},
            {"dim": "technology_trends", "label": "Technology Trends", "color": "#a78bfa", "query": f"{topic} technology"},
        ]
    })

    proc = subprocess.Popen(
        [PYTHON, str(SKILL_SCRIPT), topic, "--max-search", str(max_search)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    report_path = None

    for line in proc.stdout or []:
        event = parse_log_line(line, [])
        if event is None:
            continue

        ev_type = event["type"]

        if ev_type == "phase2_start":
            eq.put({"type": "phase2_start", "total_docs": event.get("total_docs", 0)})

        elif ev_type == "ingest_done":
            eq.put({"type": "ingest_done", "total": 0})

        elif ev_type == "critic_done":
            eq.put(event)

        elif ev_type == "rag_done":
            eq.put(event)

        elif ev_type == "report_start":
            eq.put(event)

        elif ev_type == "report_done":
            eq.put(event)

        elif ev_type == "report_path":
            report_path = event["path"]

        elif ev_type == "pipeline_done_internal":
            pass  # will be handled after process exit

        elif ev_type == "agent_start":
            eq.put(event)
        elif ev_type == "agent_search_done":
            eq.put(event)
        elif ev_type == "agent_scrape_done":
            eq.put(event)
        elif ev_type == "agent_done":
            eq.put(event)

    proc.wait()

    duration = round(time.time() - t0, 1)

    eq.put({
        "type": "pipeline_done",
        "duration": duration,
        "stats": {"report_path": report_path or ""}
    })

    return {"report_path": report_path, "duration_s": duration}
