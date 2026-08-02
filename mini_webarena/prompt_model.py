import os
from pathlib import Path


HF_PROMPT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
LOCAL_PROMPT_MODEL = (
    Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-14B-Instruct"
)


def resolve_prompt_model() -> str:
    configured_model = os.getenv("MINI_WEB_ARENA_PROMPT_MODEL")
    if configured_model:
        return configured_model
    if (LOCAL_PROMPT_MODEL / "tokenizer.json").is_file():
        return str(LOCAL_PROMPT_MODEL)
    return HF_PROMPT_MODEL


DEFAULT_PROMPT_MODEL = resolve_prompt_model()
