from openai import OpenAI
from typing import Optional, List

DOUBAO_API_KEY = "ark-f1d21ed1-6d5d-4188-9310-bf90426a5229-893c2"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

MODEL_PRO  = "doubao-1-5-pro-32k-250115"
MODEL_V3   = "deepseek-v3-2-251201"
MODEL_SUPV = "doubao-seed-2-0-pro-260215"

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=DOUBAO_API_KEY, base_url=DOUBAO_BASE_URL)
    return _client


def chat(
    messages: List[dict],
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)
    resp = get_client().chat.completions.create(
        model=model or MODEL_PRO,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
        content = reasoning.strip()
    return content
