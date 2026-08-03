"""Tokenizer and OpenAI-compatible model client for prompt agents."""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, provider: str, model_name: str) -> None:
        if provider != "huggingface":
            raise ValueError(f"unsupported tokenizer provider: {provider}")
        local_only = os.path.isdir(model_name) or os.getenv(
            "MINI_WEB_ARENA_TOKENIZER_LOCAL_ONLY", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_only,
        )
        self.tokenizer.add_special_tokens = False
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)


def _chat_endpoint(configured: str | None, port: int) -> str:
    base_url = configured or os.getenv(
        "MINI_WEB_ARENA_LLM_URL", f"http://127.0.0.1:{port}/v1"
    )
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def call_llm(lm_config: Any, prompt: str, port: int = 8000) -> str:
    if lm_config.provider != "huggingface":
        raise ValueError(f"unsupported model provider: {lm_config.provider}")

    generation = lm_config.gen_config
    payload = {
        "model": lm_config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": generation.get("temperature", 0.0),
        "max_tokens": generation.get("max_new_tokens", 4096),
    }
    if "top_p" in generation:
        payload["top_p"] = generation["top_p"]

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = _chat_endpoint(generation.get("model_endpoint"), port)
    retries = max(1, int(generation.get("max_retry", 1)))
    timeout = float(generation.get("request_timeout", 120))
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except (requests.RequestException, KeyError, IndexError, TypeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"model request failed after {retries} attempts") from last_error
