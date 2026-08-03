#!/usr/bin/env python3
"""Run the configured evaluation matrix through run.sh."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "evaluate" / "configs" / "eval_matrix.json"
UNRESOLVED_ENV = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--group", action="append", help="Group to run; repeat or use commas.")
    parser.add_argument("--experiment", action="append", help="Experiment id to run; repeat or use commas.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "evaluate" / "results" / "experiments",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="List configured experiments and exit.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true", help="Use an existing vLLM service.")
    parser.add_argument("--skip-tool-server", action="store_true", help="Use an existing tool server.")
    parser.add_argument("--no-token-stats", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra run.sh arguments after --.",
    )
    return parser.parse_args()


def selected_values(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {part.strip() for value in values for part in value.split(",") if part.strip()}


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def merge_settings(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged


def require_resolved(name: str, value: str) -> str:
    unresolved = UNRESOLVED_ENV.search(value)
    if unresolved:
        variable = unresolved.group(0)
        raise ValueError(f"{name} contains unset environment variable {variable}")
    if not value:
        raise ValueError(f"{name} is empty")
    return value


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    defaults = config.get("defaults", {})
    groups = config.get("groups", {})
    wanted_groups = selected_values(args.group)
    wanted_experiments = selected_values(args.experiment)

    if wanted_groups:
        unknown = wanted_groups - groups.keys()
        if unknown:
            raise ValueError(f"Unknown groups: {', '.join(sorted(unknown))}")

    jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for group_id, group in groups.items():
        if wanted_groups and group_id not in wanted_groups:
            continue
        group_defaults = {key: value for key, value in group.items() if key != "experiments"}
        for experiment in group.get("experiments", []):
            if wanted_experiments and experiment["id"] not in wanted_experiments:
                continue
            jobs.append((group_id, experiment, merge_settings(defaults, group_defaults, experiment)))

    if wanted_experiments:
        found = {experiment["id"] for _, experiment, _ in jobs}
        missing = wanted_experiments - found
        if missing:
            raise ValueError(f"Unknown or filtered experiments: {', '.join(sorted(missing))}")

    if args.list:
        for group_id, experiment, settings in jobs:
            run_kind = "run" if settings.get("model_path") else "existing"
            print(f"{group_id:22s} {experiment['id']:24s} {run_kind:8s} {experiment.get('label', '')}")
        return 0

    failures: list[str] = []
    for group_id, experiment, raw_settings in jobs:
        settings = expand(raw_settings)
        experiment_id = experiment["id"]
        if settings.get("fixed_scores") or settings.get("existing_result_dir"):
            print(f"SKIP {group_id}/{experiment_id}: existing/fixed result entry")
            continue

        try:
            model_path = require_resolved("model_path", str(settings.get("model_path", "")))
            output_dir = args.results_root / group_id / experiment_id
            benchmarks = settings.get("benchmarks", defaults.get("benchmarks", []))
            command = [
                str(ROOT / "evaluate" / "run.sh"),
                "--benchmarks", ",".join(benchmarks),
                "--model-path", model_path,
                "--output-dir", str(output_dir),
                "--max-samples", str(settings["max_samples"]),
                "--sample-seed", str(settings["sample_seed"]),
                "--num-trials", str(settings["num_trials"]),
                "--num-workers", str(settings["num_workers"]),
                "--compression-factor", str(settings["compression_factor"]),
            ]
            if not settings.get("resume", True):
                command.append("--no-resume")
            if args.skip_vllm:
                command.append("--skip-vllm")
            if args.skip_tool_server:
                command.append("--skip-tool-server")
            if args.no_token_stats:
                command.append("--no-token-stats")
            if args.dry_run:
                command.append("--dry-run")
            extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
            command.extend(extra_args)

            print(f"\n=== {group_id}/{experiment_id}: {experiment.get('label', experiment_id)} ===")
            print(shlex.join(command))
            if args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            resolved = {
                "group": group_id,
                "experiment": experiment_id,
                "label": experiment.get("label", experiment_id),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "command": command,
            }
            (output_dir / "experiment_config.resolved.json").write_text(
                json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            run_env = os.environ.copy()
            run_env["RUN_ID"] = f"{group_id}_{experiment_id}_{datetime.now():%Y%m%d_%H%M%S}"
            subprocess.run(command, cwd=ROOT, env=run_env, check=True)
        except (ValueError, subprocess.CalledProcessError) as exc:
            failures.append(f"{group_id}/{experiment_id}: {exc}")
            print(f"ERROR: {failures[-1]}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailed experiments:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
