#!/usr/bin/env python3
"""Emit bash assignments from the RL YAML config.

The config intentionally maps to the existing *_OVERRIDE environment variable
names used by the bash scripts. Environment variables already set by the caller
win over YAML values, so ad-hoc one-off overrides still work.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path


def _parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    if value in {"true", "True"}:
        return "True"
    if value in {"false", "False"}:
        return "False"
    if value in {"null", "Null", "NULL", "~"}:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    return value


def _fallback_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict | list]] = [(-1, root)]

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        item = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if item.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item has non-list parent: {raw}")
            parent.append(_parse_scalar(item[2:]))
            continue

        key, sep, value = item.partition(":")
        if not sep:
            raise ValueError(f"Expected key: value line: {raw}")
        key = key.strip()
        value = value.strip()

        if value == "":
            next_container: dict | list = {}
            if isinstance(parent, dict):
                parent[key] = next_container
            else:
                raise ValueError(f"Nested mapping has non-dict parent: {raw}")
            stack.append((indent, next_container))
        else:
            if isinstance(parent, dict):
                parent[key] = _parse_scalar(value)
            else:
                raise ValueError(f"Scalar has non-dict parent: {raw}")

    return root


def load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except ModuleNotFoundError:
        return _fallback_yaml(text)


def merge_env(config: dict, mode: str, preset: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    env: dict[str, str] = {}
    arrays: dict[str, list[str]] = {}

    def add_env(mapping):
        for key, value in (mapping or {}).items():
            if value is None:
                continue
            env[str(key)] = str(value)

    add_env(config.get("common", {}).get("env"))

    if mode == "train":
        add_env(config.get("train", {}).get("env"))
        add_env(config.get("presets", {}).get(preset, {}).get("env"))
    elif mode == "auto":
        add_env(config.get("auto", {}).get("env"))
        grid = config.get("auto", {}).get("grid", {})
        for key, value in grid.items():
            if isinstance(value, list):
                arrays[str(key)] = [str(item) for item in value]
            elif value is not None:
                env[str(key)] = str(value)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return env, arrays


def emit(env: dict[str, str], arrays: dict[str, list[str]]) -> None:
    for key, value in env.items():
        if key in os.environ:
            continue
        print(f"export {key}={shlex.quote(value)}")
    for key, values in arrays.items():
        quoted = " ".join(shlex.quote(value) for value in values)
        print(f"{key}=({quoted})")


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: yaml_env.py <train|auto> <config.yaml> [preset]", file=sys.stderr)
        return 2
    mode = sys.argv[1]
    path = Path(sys.argv[2])
    preset = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("TRAIN_PRESET", "mt_grpo")
    config = load_yaml(path)
    env, arrays = merge_env(config, mode, preset)
    emit(env, arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
