"""
OpenClaw skill wrappers — calls scripts from ~/.openclaw/workspace/skills/.
These are the same skills the OpenClaw main agent uses.
"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

SKILLS = Path.home() / ".openclaw/workspace/skills"
PYTHON = sys.executable


def _run(script_rel: str, args: list, timeout: int = 40) -> dict:
    script = str(SKILLS / script_rel)
    if not Path(script).exists():
        return {"error": f"skill not found: {script_rel}"}
    try:
        r = subprocess.run(
            [PYTHON, script] + [str(a) for a in args],
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


async def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Async wrapper: run DuckDuckGo search via OpenClaw web-search skill."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: _run("web-search/scripts/web_search.py", [query, max_results])
    )
    if "error" in result:
        return []
    if isinstance(result, list):
        return result
    return result.get("results", [])


async def web_scrape(url: str) -> str:
    """Async wrapper: scrape URL via OpenClaw web-scrape skill."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: _run("web-scrape/scripts/web_scrape.py", [url])
    )
    return result.get("text") or result.get("raw", "")[:2000]


async def gather_dimension(topic: str, dimension: str, query_suffix: str) -> dict:
    """
    Search + scrape for one research dimension. Returns dict with dim + text.
    Fully async: search and top-2 scrapes run concurrently.
    """
    query = f"{topic} {query_suffix}"
    results = await web_search(query, max_results=3)

    if not results:
        return {"dimension": dimension, "text": f"No results found for {query}"}

    # Scrape top 2 URLs concurrently
    urls = [r.get("url") or r.get("href", "") for r in results[:2] if r.get("url") or r.get("href")]
    snippets = [r.get("body") or r.get("snippet", "") for r in results[:3]]

    if urls:
        scraped = await asyncio.gather(*[web_scrape(u) for u in urls])
        texts = [t for t in scraped if t and len(t) > 100]
    else:
        texts = []

    combined = "\n\n---\n\n".join(texts) if texts else "\n".join(snippets[:3])
    return {"dimension": dimension, "text": combined[:4000]}
