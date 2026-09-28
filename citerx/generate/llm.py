"""Minimal client for any OpenAI-compatible chat endpoint (Ollama, Groq, vLLM, OpenRouter ...).

Uses only the standard library so the LLM backend adds no dependency.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict], base_url: str, model: str, api_key: str = "", timeout: int = 120) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.0}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'none'}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise LLMError(f"LLM returned HTTP {e.code}: {e.read()[:300]!r}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMError(f"Could not reach LLM at {base_url}: {e}") from e

    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected LLM response: {str(data)[:300]}") from e
    # Reasoning models (qwen3, deepseek-r1) may emit a thinking block first.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
