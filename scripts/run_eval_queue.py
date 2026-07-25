#!/usr/bin/env python3
"""Run an ordered queue of evaluation experiments with persistent queue state."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "experiments" / "eval_matrix.json"
DEFAULT_RESULTS_ROOT = ROOT / "gen_seq" / "results" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def merge(*values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        result.update(value)
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def string_list(value: Any, location: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, (str, int, float)) for item in value
    ):
        raise ValueError(f"{location} must be a JSON list of strings/numbers")
    return [str(item) for item in value]


def matrix_index(matrix: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    defaults = matrix.get("defaults", {})
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for group_name, group in matrix.get("groups", {}).items():
        group_defaults = {
            key: value for key, value in group.items() if key != "experiments"
        }
        for experiment in group.get("experiments", []):
            index[(group_name, experiment["id"])] = merge(
                defaults,
                group_defaults,
                experiment,
            )
    return index


def load_jobs(
    queue_path: Path,
    matrix_path: Path,
    results_root: Path,
    dry_run: bool,
) -> tuple[str, list[dict[str, Any]]]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    queue_name = str(queue.get("name") or queue_path.stem)
    queue_defaults = queue.get("defaults", {})
    entries = queue.get("jobs")
    if not isinstance(queue_defaults, dict):
        raise ValueError("queue.defaults must be a JSON object")
    if not isinstance(entries, list) or not entries:
        raise ValueError("queue.jobs must be a non-empty JSON list")

    index = matrix_index(matrix)
    jobs: list[dict[str, Any]] = []
    job_ids: set[str] = set()
    output_dirs: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"jobs[{position}] must be a JSON object")
        group_name = str(entry.get("group", ""))
        experiment_id = str(entry.get("experiment", ""))
        base = index.get((group_name, experiment_id))
        if base is None:
            raise ValueError(
                f"jobs[{position}] references unknown {group_name}/{experiment_id}"
            )
        job_id = str(entry.get("job_id") or f"{group_name}_{experiment_id}")
        if job_id in job_ids:
            raise ValueError(f"Duplicate job_id: {job_id}")
        job_ids.add(job_id)

        settings = expand(merge(base, queue_defaults, entry))
        model_path = str(settings.get("model_path", ""))
        if not model_path or "$" in model_path:
            raise ValueError(f"{job_id}: model_path is empty or has an unset variable")
        if not dry_run and not Path(model_path).is_dir():
            raise ValueError(f"{job_id}: model_path is not a directory: {model_path}")
        benchmarks = settings.get("benchmarks")
        if not isinstance(benchmarks, list) or not benchmarks:
            raise ValueError(f"{job_id}: benchmarks must be a non-empty list")

        extra_args = [
            *string_list(base.get("extra_args"), f"{job_id} matrix extra_args"),
            *string_list(queue_defaults.get("extra_args"), "queue defaults extra_args"),
            *string_list(entry.get("extra_args"), f"{job_id} extra_args"),
        ]
        environments = merge(
            base.get("env", {}),
            queue_defaults.get("env", {}),
            entry.get("env", {}),
        )
        if not all(isinstance(key, str) for key in environments):
            raise ValueError(f"{job_id}: env keys must be strings")
        environments = expand(environments)
        for key, value in environments.items():
            if "$" in str(value):
                raise ValueError(f"{job_id}: env.{key} has an unset variable")

        output_value = entry.get("output_dir")
        if output_value:
            output_dir = Path(expand(str(output_value)))
            if not output_dir.is_absolute():
                output_dir = ROOT / output_dir
        else:
            output_dir = results_root / group_name / experiment_id
        output_text = str(output_dir)
        if "$" in output_text:
            raise ValueError(f"{job_id}: output_dir has an unset variable")
        if output_text in output_dirs:
            raise ValueError(f"Duplicate output_dir: {output_dir}")
        output_dirs.add(output_text)

        jobs.append({
            "position": position,
            "job_id": job_id,
            "group": group_name,
            "experiment": experiment_id,
            "label": str(entry.get("label") or base.get("label") or experiment_id),
            "model_path": model_path,
            "output_dir": output_dir,
            "settings": settings,
            "extra_args": extra_args,
            "env": environments,
        })
    return queue_name, jobs


def signature(job: dict[str, Any]) -> str:
    value = {
        "job_id": job["job_id"],
        "group": job["group"],
        "experiment": job["experiment"],
        "model_path": job["model_path"],
        "output_dir": str(job["output_dir"]),
        "settings": job["settings"],
        "extra_args": job["extra_args"],
        "env": job["env"],
    }
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_state(path: Path, queue_name: str, queue_path: Path) -> dict[str, Any]:
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value.get("jobs", {}), dict):
            raise ValueError(f"Malformed queue state: {path}")
        return value
    return {
        "version": 1,
        "queue_name": queue_name,
        "queue_file": str(queue_path.resolve()),
        "created_at": now(),
        "updated_at": now(),
        "jobs": {},
    }


def update_state(
    path: Path,
    state: dict[str, Any],
    job: dict[str, Any],
    status: str,
    **fields: Any,
) -> None:
    previous = state.setdefault("jobs", {}).get(job["job_id"], {})
    value = {
        **previous,
        "position": job["position"],
        "group": job["group"],
        "experiment": job["experiment"],
        "label": job["label"],
        "model_path": job["model_path"],
        "output_dir": str(job["output_dir"]),
        "signature": signature(job),
        "status": status,
        "updated_at": now(),
        **fields,
    }
    state["jobs"][job["job_id"]] = value
    state["updated_at"] = now()
    atomic_json(path, state)


def command_for(job: dict[str, Any], dry_run: bool, cli_args: list[str]) -> list[str]:
    settings = job["settings"]
    command = [
        str(ROOT / "run_eval_all.sh"),
        "--benchmarks", ",".join(settings["benchmarks"]),
        "--model-path", job["model_path"],
        "--output-dir", str(job["output_dir"]),
        "--max-samples", str(settings["max_samples"]),
        "--sample-seed", str(settings["sample_seed"]),
        "--num-trials", str(settings["num_trials"]),
        "--num-workers", str(settings["num_workers"]),
        "--compression-factor", str(settings["compression_factor"]),
    ]
    if not settings.get("resume", True):
        command.append("--no-resume")
    if dry_run:
        command.append("--dry-run")
    command.extend(job["extra_args"])
    command.extend(cli_args)
    return command


def main() -> int:
    args = parse_args()
    queue_name, jobs = load_jobs(
        args.queue,
        args.matrix,
        args.results_root,
        args.dry_run or args.list,
    )
    cli_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    state_path = args.state or args.results_root / "_queues" / f"{queue_name}.json"
    lock_path = args.lock_file or args.results_root / "_queues" / "evaluation.lock"

    if args.list:
        for job in jobs:
            print(
                f"{job['position']:02d} {job['job_id']:28s} "
                f"{job['group']}/{job['experiment']} "
                f"compression={job['settings']['compression_factor']} "
                f"model={job['model_path']}"
            )
        return 0

    lock_handle = None
    if not args.dry_run:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another evaluation queue holds lock: {lock_path}") from exc
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(f"pid={os.getpid()} queue={queue_name} started_at={now()}\n")
        lock_handle.flush()
        print(f"Queue lock: {lock_path}")

    state = None if args.dry_run else load_state(state_path, queue_name, args.queue)
    if state is not None:
        atomic_json(state_path, state)
        print(f"Queue state: {state_path}")

    failures: list[str] = []
    completed = 0
    skipped = 0
    for job in jobs:
        previous = (state or {}).get("jobs", {}).get(job["job_id"], {})
        if (
            state is not None
            and not args.rerun_completed
            and previous.get("status") == "completed"
            and previous.get("signature") == signature(job)
            and job["output_dir"].is_dir()
        ):
            print(f"SKIP {job['job_id']}: completed with matching state")
            skipped += 1
            continue

        command = command_for(job, args.dry_run, cli_args)
        print(
            f"\n=== [{job['position']}/{len(jobs)}] "
            f"{job['job_id']}: {job['label']} ==="
        )
        print(shlex.join(command))
        run_env = os.environ.copy()
        run_env.update({key: str(value) for key, value in job["env"].items()})
        run_env["RUN_ID"] = f"{job['job_id']}_{datetime.now():%Y%m%d_%H%M%S}"
        if args.dry_run:
            subprocess.run(command, cwd=ROOT, env=run_env, check=True)
            continue

        job["output_dir"].mkdir(parents=True, exist_ok=True)
        atomic_json(job["output_dir"] / "experiment_config.resolved.json", {
            "queue": queue_name,
            "job_id": job["job_id"],
            "group": job["group"],
            "experiment": job["experiment"],
            "label": job["label"],
            "started_at": now(),
            "model_path": job["model_path"],
            "settings": job["settings"],
            "extra_args": job["extra_args"],
            "env": job["env"],
            "command": command,
        })
        attempts = int(previous.get("attempts", 0)) + 1
        update_state(
            state_path,
            state,
            job,
            "running",
            attempts=attempts,
            started_at=now(),
            command=command,
        )
        try:
            subprocess.run(command, cwd=ROOT, env=run_env, check=True)
            completed += 1
            update_state(
                state_path,
                state,
                job,
                "completed",
                completed_at=now(),
                exit_code=0,
            )
        except subprocess.CalledProcessError as exc:
            message = f"{job['job_id']}: exit code {exc.returncode}"
            failures.append(message)
            update_state(
                state_path,
                state,
                job,
                "failed",
                failed_at=now(),
                exit_code=exc.returncode,
                error=message,
            )
            print(f"ERROR: {message}", file=sys.stderr)
            if not args.continue_on_error:
                break
        except KeyboardInterrupt:
            update_state(
                state_path,
                state,
                job,
                "interrupted",
                interrupted_at=now(),
                exit_code=130,
            )
            print(f"Interrupted while running {job['job_id']}", file=sys.stderr)
            return 130

    print(
        f"Queue summary: completed_now={completed}, skipped={skipped}, "
        f"failed={len(failures)}, total={len(jobs)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
