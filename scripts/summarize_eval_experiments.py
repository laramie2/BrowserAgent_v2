#!/usr/bin/env python3
"""Build CSV and Markdown experiment tables from evaluation JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ["nq", "triviaqa", "popqa", "hotpot", "2wiki", "musique", "bamboogle"]
BENCHMARK_LABELS = {
    "nq": "NQ",
    "triviaqa": "TriviaQA",
    "popqa": "PopQA",
    "hotpot": "HotpotQA",
    "2wiki": "2Wiki",
    "musique": "MuSiQue",
    "bamboogle": "Bamboogle",
}
UNRESOLVED_ENV = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "eval_matrix.json")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "gen_seq" / "results" / "experiments",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "reports")
    parser.add_argument("--metric", default="success_substring")
    return parser.parse_args()


def expanded_path(value: str) -> Path | None:
    expanded = os.path.expanduser(os.path.expandvars(value))
    if not expanded or UNRESOLVED_ENV.search(expanded):
        return None
    return Path(expanded)


def score_file(path: Path, metric: str) -> tuple[float | None, int]:
    if not path.is_file():
        return None, 0
    successes = 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed JSON at {path}:{line_no}: {exc}", file=sys.stderr)
                continue
            if metric not in record:
                continue
            count += 1
            successes += int(bool(record[metric]))
    return ((100.0 * successes / count) if count else None), count


def token_stats(result_dir: Path | None) -> dict[str, float | None]:
    empty = {
        "raw_avg": None,
        "raw_max": None,
        "compressed_avg": None,
        "compressed_max": None,
        "avg_saved": None,
    }
    if result_dir is None:
        return empty
    path = result_dir / "token_usage_summary.json"
    if not path.is_file():
        return empty
    overall = json.loads(path.read_text(encoding="utf-8")).get("overall", {})
    raw = overall.get("no_image_compression", {})
    compressed = overall.get("image_compression", {})
    delta = overall.get("delta", {})
    return {
        "raw_avg": raw.get("avg_step_tokens"),
        "raw_max": raw.get("max_step_tokens"),
        "compressed_avg": compressed.get("avg_step_tokens"),
        "compressed_max": compressed.get("max_step_tokens"),
        "avg_saved": delta.get("avg_tokens_saved_per_step"),
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def result_dir_for(
    group_id: str,
    experiment: dict[str, Any],
    results_root: Path,
) -> Path | None:
    if experiment.get("existing_result_dir"):
        return expanded_path(str(experiment["existing_result_dir"]))
    if experiment.get("model_path"):
        return results_root / group_id / experiment["id"]
    return None


def build_rows(config: dict[str, Any], results_root: Path, metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_id, group in config.get("groups", {}).items():
        for experiment in group.get("experiments", []):
            result_dir = result_dir_for(group_id, experiment, results_root)
            fixed = experiment.get("fixed_scores", {})
            scores: dict[str, float | None] = {}
            counts: dict[str, int] = {}
            for benchmark in BENCHMARKS:
                if benchmark in fixed:
                    scores[benchmark] = float(fixed[benchmark])
                    counts[benchmark] = 0
                elif result_dir is not None:
                    scores[benchmark], counts[benchmark] = score_file(
                        result_dir / f"{benchmark}_test_results.jsonl",
                        metric,
                    )
                else:
                    scores[benchmark], counts[benchmark] = None, 0
            present = [score for score in scores.values() if score is not None]
            row = {
                "group": group_id,
                "id": experiment["id"],
                "label": experiment.get("label", experiment["id"]),
                "result_dir": str(result_dir) if result_dir else "",
                **scores,
                "avg": sum(present) / len(BENCHMARKS) if len(present) == len(BENCHMARKS) else None,
                "evaluated_records": sum(counts.values()),
                **token_stats(result_dir),
            }
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "group", "id", "label", *BENCHMARKS, "avg",
        "compressed_avg", "compressed_max", "raw_avg", "raw_max", "avg_saved",
        "evaluated_records", "result_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def markdown_table(group_id: str, rows: list[dict[str, Any]]) -> str:
    title = {
        "main": "Main experiments",
        "rl_ablation": "RL reward ablation (1000 samples/benchmark)",
        "compression_ablation": "Compression ratio ablation (1000 samples/benchmark)",
    }.get(group_id, group_id)
    headers = ["method", *(BENCHMARK_LABELS[b] for b in BENCHMARKS), "Avg", "Tokens/Step Avg", "Max"]
    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            str(row["label"]),
            *(fmt(row[benchmark]) for benchmark in BENCHMARKS),
            fmt(row["avg"]),
            fmt(row["compressed_avg"]),
            fmt(row["compressed_max"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def token_detail_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Token compression details",
        "",
        "| experiment | raw avg | raw max | compressed avg | compressed max | avg saved/step |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join([
                f"{row['group']}/{row['label']}",
                fmt(row["raw_avg"]),
                fmt(row["raw_max"]),
                fmt(row["compressed_avg"]),
                fmt(row["compressed_max"]),
                fmt(row["avg_saved"]),
            ]) + " |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = build_rows(config, args.results_root, args.metric)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "evaluation_tables.csv"
    markdown_path = args.output_dir / "evaluation_tables.md"
    write_csv(csv_path, rows)

    sections = []
    for group_id in config.get("groups", {}):
        sections.append(markdown_table(group_id, [row for row in rows if row["group"] == group_id]))
    sections.append(token_detail_table(rows))
    markdown_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
