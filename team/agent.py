"""
Base Agent class — powered by OpenClaw's _llm.py config reader.
Supports custom API endpoints (e.g. DeepSeek) via api_base / api_key.
Each Agent instance maintains its own conversation history and persona.
"""
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import openai as _openai

sys.path.insert(0, str(Path.home() / ".openclaw/workspace/skills"))
from _llm import get_model_client

AGENT_CALL_TIMEOUT = 90   # seconds per LLM call before we abort
MAX_RETRIES        = 1    # one retry on transient network / rate-limit errors


class AgentCallError(RuntimeError):
    """Raised when an LLM call fails after retries or hits a permanent error."""


class Agent:
    """
    An LLM-powered conversational agent.
    - Default credentials: read from ~/.openclaw/openclaw.json via _llm.py
    - Custom endpoint: pass api_base + api_key (e.g. for DeepSeek API)
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        color: str,
        avatar: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.agent_id      = agent_id
        self.name          = name
        self.system_prompt = system_prompt
        self.color         = color
        self.avatar        = avatar
        self.temperature   = temperature
        self.history: list[dict] = []

        if api_base and api_key:
            self._client = _openai.OpenAI(base_url=api_base, api_key=api_key)
            self._model  = model  # must be explicitly provided for custom endpoints
        else:
            self._client, default_model = get_model_client()
            self._model = model or default_model

    # ── Synchronous LLM call (runs inside thread executor) ────────────────────

    def _call_llm(self, messages: list[dict], max_tokens: int = 800) -> str:
        last_err = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
                msg     = resp.choices[0].message
                content = (msg.content or "").strip()
                # DeepSeek reasoning models expose thinking via reasoning_content;
                # if content is empty (token budget too tight) fall back to it
                if not content:
                    reasoning = getattr(msg, "reasoning_content", None) or ""
                    if reasoning.strip():
                        return reasoning.strip()
                    finish = resp.choices[0].finish_reason
                    if finish == "length":
                        raise AgentCallError(
                            f"Model '{self._model}' hit max_tokens={max_tokens} before "
                            "producing output — increase max_tokens for this agent"
                        )
                return content

            except _openai.NotFoundError:
                raise AgentCallError(
                    f"Model '{self._model}' not found — check model ID and API endpoint"
                )
            except _openai.AuthenticationError:
                raise AgentCallError(
                    f"Authentication failed for '{self._model}' — check API key"
                )
            except _openai.RateLimitError as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(3)
                else:
                    raise AgentCallError(
                        f"Rate limit exceeded for '{self._model}': {e}"
                    )
            except _openai.APIConnectionError as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(2)
                else:
                    raise AgentCallError(
                        f"API connection error for '{self._model}': {e}"
                    )
            except _openai.APIStatusError as e:
                raise AgentCallError(
                    f"API error {e.status_code} for '{self._model}': {e.message}"
                )
            except AgentCallError:
                raise
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(2)
                else:
                    raise AgentCallError(
                        f"Unexpected error from '{self._model}': {e}"
                    )

        raise AgentCallError(
            f"Agent '{self.name}' failed after {MAX_RETRIES + 1} attempts: {last_err}"
        )

    # ── Async speak ───────────────────────────────────────────────────────────

    async def speak(
        self,
        prompt: str,
        extra_context: Optional[list[dict]] = None,
        max_tokens: int = 800,
        remember: bool = True,
    ) -> str:
        """
        Generate a response. Runs the blocking LLM call in a thread executor.
        Raises AgentCallError on model errors or timeout.

        After each call, self.last_call_stats is populated with:
          {duration_s, in_chars, out_chars}
        """
        self.last_call_stats = {"duration_s": 0.0, "in_chars": 0, "out_chars": 0}

        msgs: list[dict] = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(self.history)
        if extra_context:
            msgs.extend(extra_context)
        msgs.append({"role": "user", "content": prompt})

        in_chars = sum(len(m.get("content", "")) for m in msgs)

        t0 = time.time()
        loop = asyncio.get_event_loop()
        try:
            reply = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._call_llm(msgs, max_tokens)),
                timeout=AGENT_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            self.last_call_stats.update(duration_s=round(time.time() - t0, 2), in_chars=in_chars)
            raise AgentCallError(
                f"Agent '{self.name}' timed out after {AGENT_CALL_TIMEOUT}s — "
                "model may be overloaded"
            )

        self.last_call_stats.update(
            duration_s=round(time.time() - t0, 2),
            in_chars=in_chars,
            out_chars=len(reply),
        )

        if remember:
            self.history.append({"role": "user",      "content": prompt})
            self.history.append({"role": "assistant",  "content": reply})

        return reply

    @property
    def model(self) -> str:
        return self._model

    def reset(self):
        self.history = []
