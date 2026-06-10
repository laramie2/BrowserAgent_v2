#!/usr/bin/env python3
"""Build fixed-size curriculum stage parquet datasets from rollout scores.

The script samples from the scored NQ/Hotpot 5000+5000 pools and writes one
trainable parquet directory per curriculum stage:

    <output_dir>/<run_name>/stage_1_easy_heavy/data.parquet

Two sampling modes are supported:
  - disjoint: a source row can appear in at most one stage.
  - random: stages are sampled independently, so rows may repeat across stages.

Default stages use the fine-grained k=8 pass-count buckets emitted by
RL/data_filter.py:

trivial:      pass_count = 8/8
easy_high:    pass_count = 6-7/8
medium_high:  pass_count = 5/8
medium_mid:   pass_count = 3-4/8
medium_low:   pass_count = 2/8
hard:         pass_count = 1/8 or mean_reward > 0.05
unsolved:     pass_count = 0/8 and mean_reward <= 0.05

"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RL_DIR = Path(__file__).resolve().parents[0]

DEFAULT_SCORE_FILES = [
    RL_DIR / "filter_results" / "sft_rollout_k8_nq" / "sample_scores.jsonl",
    RL_DIR / "filter_results" / "sft_rollout_k8_hotpot" / "sample_scores.jsonl",
]
DEFAULT_SOURCE_PARQUETS = {
    "nq": RL_DIR / "dataset" / "nq" / "train_5000_labelled.parquet",
    "hotpot": RL_DIR / "dataset" / "hotpot" / "train_5000_labelled.parquet",
}

STAGE_DEFS = [
    {
        "name": "stage_1_medium_warmup",
        "ratios": {
            "trivial": 0.05,
            "easy_high": 0.17,
            "medium_high": 0.23,
            "medium_mid": 0.37,
            "medium_low": 0.13,
            "hard": 0.05,
        },
    },
    {
        "name": "stage_2_core_medium",
        "ratios": {
            "trivial": 0.03,
            "easy_high": 0.12,
            "medium_high": 0.25,
            "medium_mid": 0.42,
            "medium_low": 0.13,
            "hard": 0.05,
        },
    },
    {
        "name": "stage_3_medium_low_hard",
        "ratios": {
            "trivial": 0.03,
            "easy_high": 0.08,
            "medium_high": 0.17,
            "medium_mid": 0.39,
            "medium_low": 0.23,
            "hard": 0.10,
        },
    },
    {
        "name": "stage_4_mixed_final",
        "ratios": {
            "trivial": 0.05,
            "easy_high": 0.17,
            "medium_high": 0.22,
            "medium_mid": 0.37,
            "medium_low": 0.13,
            "hard": 0.06,
        },
    },
]



class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_source_parquets(items: list[str]) -> dict[str, Path]:
    paths = dict(DEFAULT_SOURCE_PARQUETS)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--source_parquet entries must be source=path, got: {item}")
        source, path = item.split("=", 1)
        paths[source.strip()] = Path(path).expanduser()
    return paths


def exact_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {label: ratios[label] * total for label in ratios}
    counts = {label: int(np.floor(value)) for label, value in raw.items()}
    remaining = total - sum(counts.values())
    by_remainder = sorted(ratios, key=lambda label: raw[label] - counts[label], reverse=True)
    for label in by_remainder[:remaining]:
        counts[label] += 1
    return counts


def split_by_source(count: int, sources: list[str]) -> dict[str, int]:
    base = count // len(sources)
    remainder = count - base * len(sources)
    return {source: base + (1 if i < remainder else 0) for i, source in enumerate(sources)}


def row_by_source_index(dataframes: dict[str, pd.DataFrame], source: str, row_index: int) -> dict[str, Any]:
    df = dataframes[source]
    if row_index in df.index:
        row = df.loc[row_index]
    else:
        row = df.iloc[row_index]
    return row.to_dict()


def augment_row(row: dict[str, Any], score: dict[str, Any], stage_name: str, stage_id: int) -> dict[str, Any]:
    row = dict(row)
    extra = row.get("extra_info")
    if not isinstance(extra, dict):
        extra = {}
    else:
        extra = dict(extra)

    extra.update(
        {
            "curriculum_stage": stage_name,
            "curriculum_stage_id": stage_id,
            "curriculum_difficulty": score["difficulty"],
            "filter_uid": score["uid"],
            "rollout_mean_reward": float(score.get("mean_reward", 0.0)),
            "rollout_solve_rate": float(score.get("solve_rate", 0.0)),
            "rollout_success_count": int(score.get("success_count", 0)),
            "rollout_num_trials": int(score.get("num_rollouts", 0)),
        }
    )
    row["extra_info"] = extra
    return row


def sample_pool(
    pool: list[dict[str, Any]],
    count: int,
    rng: np.random.Generator,
    mode: str,
    used: set[str],
) -> list[dict[str, Any]]:
    candidates = pool if mode == "random" else [item for item in pool if item["uid"] not in used]
    if len(candidates) < count:
        raise ValueError(
            f"Not enough samples for count={count}; available={len(candidates)} "
            f"(mode={mode}, difficulty/source pool may be too small)."
        )
    chosen_indices = rng.choice(len(candidates), size=count, replace=False)
    chosen = [candidates[int(i)] for i in chosen_indices]
    if mode == "disjoint":
        used.update(item["uid"] for item in chosen)
    return chosen


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, cls=NumpyJSONEncoder) + "\n")


def write_bucket_csv(path: Path, rows: list[dict[str, Any]], bucket_field: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", "bucket", "source_name", "uid"])
        writer.writeheader()
        for row in rows:
            extra = row.get("extra_info", {})
            writer.writerow(
                {
                    "sample_index": extra.get("index"),
                    "bucket": extra[bucket_field],
                    "source_name": extra.get("data_source") or row.get("data_source"),
                    "uid": extra["filter_uid"],
                }
            )


def build(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    source_parquets = parse_source_parquets(args.source_parquet)
    dataframes = {source: pd.read_parquet(path) for source, path in source_parquets.items()}

    scores = []
    for path in args.score_file:
        scores.extend(iter_jsonl(Path(path)))
    scores = [row for row in scores if row.get("difficulty") not in set(args.exclude_difficulty)]

    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        source = str(row["source_name"])
        difficulty = str(row["difficulty"])
        if source in dataframes:
            pools[(source, difficulty)].append(row)

    for pool in pools.values():
        rng.shuffle(pool)

    sources = sorted(dataframes)
    output_root = Path(args.output_dir)
    run_name = args.run_name or f"curriculum_medium_{args.mode}_n{args.stage_size}_seed{args.seed}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    global_manifest: dict[str, Any] = {
        "mode": args.mode,
        "seed": args.seed,
        "stage_size": args.stage_size,
        "balance_sources": args.balance_sources,
        "excluded_difficulties": list(args.exclude_difficulty),
        "source_parquets": {source: str(path) for source, path in source_parquets.items()},
        "score_files": [str(path) for path in args.score_file],
        "stages": [],
    }

    for stage_id, stage_def in enumerate(STAGE_DEFS, start=1):
        stage_name = stage_def["name"]
        difficulty_counts = exact_counts(args.stage_size, stage_def["ratios"])
        selected_scores: list[dict[str, Any]] = []

        for difficulty, count in difficulty_counts.items():
            if args.balance_sources:
                per_source = split_by_source(count, sources)
                for source, source_count in per_source.items():
                    selected_scores.extend(
                        sample_pool(pools[(source, difficulty)], source_count, rng, args.mode, used)
                    )
            else:
                merged = []
                for source in sources:
                    merged.extend(pools[(source, difficulty)])
                selected_scores.extend(sample_pool(merged, count, rng, args.mode, used))

        rng.shuffle(selected_scores)
        stage_rows = [
            augment_row(
                row_by_source_index(dataframes, score["source_name"], int(score["source_row_index"])),
                score,
                stage_name,
                stage_id,
            )
            for score in selected_scores
        ]

        stage_dir = run_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(stage_rows).to_parquet(stage_dir / "data.parquet", index=False)
        write_jsonl(stage_dir / "selected_scores.jsonl", selected_scores)
        write_bucket_csv(stage_dir / "difficulty_buckets.csv", stage_rows, "curriculum_difficulty")

        difficulty_counter = Counter(score["difficulty"] for score in selected_scores)
        source_counter = Counter(score["source_name"] for score in selected_scores)
        crossed_counter = Counter((score["source_name"], score["difficulty"]) for score in selected_scores)
        stage_manifest = {
            "stage_id": stage_id,
            "name": stage_name,
            "size": len(stage_rows),
            "difficulty_counts": dict(sorted(difficulty_counter.items())),
            "source_counts": dict(sorted(source_counter.items())),
            "source_difficulty_counts": {
                f"{source}:{difficulty}": count
                for (source, difficulty), count in sorted(crossed_counter.items())
            },
            "data_parquet": str(stage_dir / "data.parquet"),
            "selected_scores_jsonl": str(stage_dir / "selected_scores.jsonl"),
            "difficulty_buckets_csv": str(stage_dir / "difficulty_buckets.csv"),
        }
        with (stage_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(stage_manifest, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        global_manifest["stages"].append(stage_manifest)

    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(global_manifest, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)

    print(f"Wrote curriculum stages to {run_dir}")
    for stage in global_manifest["stages"]:
        print(
            f"{stage['name']}: size={stage['size']} "
            f"difficulty={stage['difficulty_counts']} source={stage['source_counts']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["disjoint", "random"], default="disjoint")
    parser.add_argument("--stage_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=RL_DIR / "filter_results")
    parser.add_argument("--run_name", type=str, default=None, help="Optional output subdirectory name.")
    parser.add_argument(
        "--score_file",
        action="append",
        default=[str(path) for path in DEFAULT_SCORE_FILES],
        help="Path to a sample_scores.jsonl file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--source_parquet",
        action="append",
        default=[],
        help="Override source parquet as source=path, e.g. nq=/path/train.parquet.",
    )
    parser.add_argument(
        "--exclude_difficulty",
        action="append",
        default=["unsolved"],
        help="Difficulty label to exclude. Defaults to unsolved. Can be passed multiple times.",
    )
    parser.add_argument(
        "--balance_sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep each difficulty quota balanced across NQ and Hotpot when possible.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
