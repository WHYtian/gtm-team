#!/usr/bin/env python3
"""Performance test — run 3 research topics and collect telemetry reports."""
import asyncio, time
from team.orchestrator import run_research

TOPICS = [
    ("T1", "Global SaaS CRM Market 2025",      "Baseline — hot market, data-rich"),
    ("T2", "Enterprise Quantum Computing Services Market 2025", "Stress — scarce data, retries"),
    ("T3", "HR SaaS Market 2025",              "Synthesizer — user RAG docs exist"),
]

async def run_one(label: str, topic: str, desc: str):
    print(f"\n{'='*60}")
    print(f"[{label}] {topic}")
    print(f"Purpose: {desc}")
    print(f"{'='*60}")

    q = asyncio.Queue()
    t0 = time.time()

    async def drain():
        while True:
            msg = await q.get()
            if msg.get("type") == "team_chat":
                c = msg.get("msg", {})
                agent = c.get("agent", "?")
                phase = c.get("phase", "")
                content = c.get("content", "")
                is_think = c.get("is_think", False)
                prefix = "🧠" if is_think else {"routing": "▶", "research": "🔍",
                    "analysis": "📊", "critique": "🛡", "writing": "✍",
                    "validation": "🔄", "complete": "🏁", "error": "⚠"}.get(phase, "·")
                short = content[:120].replace("\n", " ")
                print(f"  {prefix} [{agent:12}] {short}")

    drain_task = asyncio.create_task(drain())
    try:
        result = await asyncio.wait_for(run_research(topic, q), timeout=900)
    except asyncio.TimeoutError:
        print(f"  ⚠ TIMEOUT after 15 min")
        result = {}
    drain_task.cancel()

    elapsed = time.time() - t0
    mm, ss = divmod(int(elapsed), 60)
    print(f"\n  Done in {mm}m {ss}s")
    if result.get("report"):
        print(f"  Report length: {len(result['report'])} chars")
    return result

async def main():
    for label, topic, desc in TOPICS:
        await run_one(label, topic, desc)
        print("\n  Waiting 10s before next topic...")
        await asyncio.sleep(10)

    print("\n\nTelemetry reports saved to /home/admin/gtm-team/telemetry/")

if __name__ == "__main__":
    asyncio.run(main())
