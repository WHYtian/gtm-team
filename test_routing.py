#!/usr/bin/env python3
"""
Routing accuracy test — GTM vs direct-answer classification.
Tests the SUPERVISOR agent against a labelled set of 16 cases.
Usage: python3 test_routing.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from team.agent import Agent
from team.personas import SUPERVISOR

# (message, expected_route)
# expected_route: "research" | "direct"
CASES = [
    # ── Should trigger GTM (industry research) ────────────────────────────────
    ("云计算市场分析",                                   "research"),
    ("SaaS CRM competitive landscape",                  "research"),
    ("帮我研究一下全球HR软件行业",                          "research"),
    ("EV battery supply chain market size 2025",        "research"),
    ("东南亚外卖行业调研",                                 "research"),
    ("China fintech industry GTM analysis",             "research"),
    ("新能源汽车市场竞争格局",                              "research"),
    ("全球半导体行业分析",                                 "research"),

    # ── Should NOT trigger GTM (general / conceptual) ─────────────────────────
    ("什么是PESTEL分析",                                  "direct"),
    ("帮我写一封英文邮件",                                 "direct"),
    ("How do I calculate TAM?",                         "direct"),
    ("What is Porter's Five Forces?",                   "direct"),
    ("今天天气怎么样",                                    "direct"),
    ("如何制定GTM策略",                                   "direct"),
    ("Python如何读取CSV文件",                             "direct"),
    ("What is a competitive moat?",                     "direct"),
]


async def main():
    supervisor = Agent(**SUPERVISOR)
    results = []

    print("=" * 60)
    print("GTM Routing Accuracy Test")
    print("=" * 60)

    for msg, expected in CASES:
        try:
            resp = await asyncio.wait_for(
                supervisor.speak(msg, max_tokens=150, remember=False),
                timeout=30,
            )
            actual = "research" if resp.strip().startswith("TASK:RESEARCH") else "direct"
            ok = actual == expected
            results.append((msg, expected, actual, ok, resp[:100].replace("\n", " ")))
            symbol = "✓" if ok else "✗"
            label = f"[{expected:8s} → {actual:8s}]"
            print(f"  {symbol} {label}  {msg[:45]}")
        except Exception as e:
            results.append((msg, expected, "error", False, str(e)[:80]))
            print(f"  ✗ [ERROR          ]  {msg[:45]} — {e}")

    total   = len(results)
    correct = sum(1 for r in results if r[3])
    rate    = correct / total * 100

    print()
    print("=" * 60)
    print(f"Success rate: {correct}/{total} = {rate:.0f}%")
    print("=" * 60)

    failed = [r for r in results if not r[3]]
    if failed:
        print("\nFailed cases:")
        for msg, expected, actual, _, preview in failed:
            print(f"  [{expected} → {actual}]  {msg}")
            print(f"    Response preview: {preview}")

    return rate


if __name__ == "__main__":
    asyncio.run(main())
