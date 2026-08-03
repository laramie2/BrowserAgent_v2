#!/usr/bin/env python3
"""Build RL train and validation data from BrowserAgent SeedData."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RL_DIR = PROJECT_ROOT / "RL"
DEFAULT_SOURCE_ROOT = RL_DIR / "dataset" / "BrowserAgent-SeedData"
DEFAULT_OUTPUT_ROOT = RL_DIR / "dataset"
DEFAULT_EXCLUSION_FILE = RL_DIR / "configs" / "excluded_sft_questions.json"
DEFAULT_SYSTEM_PROMPT = (
    PROJECT_ROOT / "prompt" / "system_prompt_with_history_info_enhance.txt"
)
TRAIN_SPLIT = "train-00000-of-00001.parquet"
VALIDATION_SPLITS = {
    "nq": "test-00000-of-00001.parquet",
    "hotpot": "validation-00000-of-00001.parquet",
}
DEFAULT_BROWSER_URL = (
    "http://localhost:22015/wikipedia_en_all_maxi_2022-05/"
    "A/User:The_other_Kiwix_guy/Landing/"
)


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value)


def decode_nested(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    for decoder in (json.loads, ast.literal_eval):
        try:
            return decoder(stripped)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return value


def as_answers(value: Any) -> list[Any]:
    value = decode_nested(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or value == "":
        return []
    return [value]


def question_from_row(row: pd.Series) -> str:
    extra = decode_nested(row.get("extra_info", {}))
    if isinstance(extra, dict):
        return str(extra.get("question", "")).strip()
    return ""


def load_excluded_questions(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"exclusion file must contain a JSON list: {path}")
    questions: set[str] = set()
    for item in data:
        if isinstance(item, str):
            questions.add(item.strip())
        elif isinstance(item, dict):
            question = item.get("question") or item.get("Objective")
            if question:
                questions.add(str(question).strip())
    return questions


def adapt_row(row: pd.Series, source: str, system_prompt: str) -> dict[str, Any]:
    record = row.to_dict()
    extra = decode_nested(record.get("extra_info", {}))
    reward_model = decode_nested(record.get("reward_model", {}))
    if not isinstance(extra, dict):
        extra = {}
    if not isinstance(reward_model, dict):
        reward_model = {}

    extra = dict(extra)
    reward_model = dict(reward_model)
    question = str(extra.get("question", "")).strip()
    answers = as_answers(extra.get("golden_answers", []))
    selected_answer = extra.get("selected_answer")
    if is_missing(extra.get("gt")):
        extra["gt"] = selected_answer if not is_missing(selected_answer) else (
            answers[0] if answers else ""
        )
    extra.setdefault("id", extra.get("index", 0))
    extra["url"] = DEFAULT_BROWSER_URL
    extra["data_source"] = source
    reward_model.setdefault("style", "rule")
    if is_missing(reward_model.get("ground_truth")):
        reward_model["ground_truth"] = answers

    user_prompt = (
        f"Objective: {question}\n"
        f"URL: {DEFAULT_BROWSER_URL}\n"
        "Observation: None\n"
        "Parsed Previous Action: None"
    )
    record.update(
        {
            "data_source": source,
            "extra_info": extra,
            "reward_model": reward_model,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    )
    return record


def sample_split(
    path: Path,
    source: str,
    count: int,
    seed: int,
    excluded: set[str],
    system_prompt: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"BrowserAgent SeedData split not found: {path}")
    frame = pd.read_parquet(path)
    if excluded:
        questions = frame.apply(question_from_row, axis=1)
        frame = frame.loc[~questions.isin(excluded)]
    if frame.empty:
        raise ValueError(f"no usable rows remain in {path}")
    sample_count = min(count, len(frame))
    sampled = frame.sample(n=sample_count, random_state=seed).reset_index(drop=True)
    return pd.DataFrame(
        [adapt_row(row, source, system_prompt) for _, row in sampled.iterrows()]
    )


def write_dataset(frame: pd.DataFrame, directory: Path, basename: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    parquet = directory / f"{basename}.parquet"
    jsonl = directory / f"{basename}.jsonl"
    frame.to_parquet(parquet, index=False)
    frame.to_json(jsonl, orient="records", lines=True, force_ascii=False)
    print(f"Wrote {len(frame)} rows to {parquet}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("nq", "hotpot"),
        default=["nq", "hotpot"],
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output-prefix")
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--exclude-json", type=Path, default=DEFAULT_EXCLUSION_FILE)
    parser.add_argument("--no-exclude", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_samples < 1 or args.validation_samples < 1:
        raise ValueError("sample counts must be positive")
    if not args.system_prompt.is_file():
        raise FileNotFoundError(f"system prompt not found: {args.system_prompt}")

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    exclusion_file = None if args.no_exclude else args.exclude_json
    excluded = load_excluded_questions(exclusion_file)
    if excluded:
        print(f"Loaded {len(excluded)} SFT questions to exclude")

    output_prefix = args.output_prefix or f"train_{args.num_samples}_labelled"
    for offset, source in enumerate(args.datasets):
        frame = sample_split(
            args.source_root / source / TRAIN_SPLIT,
            source,
            args.num_samples,
            args.seed + offset,
            excluded,
            system_prompt,
        )
        write_dataset(frame, args.output_root / source, output_prefix)

    if not args.skip_validation:
        base, remainder = divmod(args.validation_samples, len(args.datasets))
        validation_frames = []
        for offset, source in enumerate(args.datasets):
            count = base + (1 if offset < remainder else 0)
            validation_frames.append(
                sample_split(
                    args.source_root / source / VALIDATION_SPLITS[source],
                    source,
                    count,
                    args.seed + 1000 + offset,
                    excluded,
                    system_prompt,
                )
            )
        validation = pd.concat(validation_frames, ignore_index=True)
        validation = validation.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        write_dataset(
            validation,
            args.output_root / f"validation_{args.validation_samples}",
            "data",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
