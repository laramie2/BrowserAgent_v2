#!/usr/bin/env python3
"""Print success counts for one or more BrowserAgent evaluation JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def count_results(path: Path, metric: str) -> dict[str, float | int]:
    total = 0
    successful = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            total += 1
            successful += int(bool(record.get(metric, False)))
    return {
        "total": total,
        "success": successful,
        "failed": total - successful,
        "success_rate": (100.0 * successful / total) if total else 0.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--metric",
        default="success",
        help="boolean result field to count (default: success)",
    )
    args = parser.parse_args(argv)

    status = 0
    for path in args.files:
        try:
            stats = count_results(path, args.metric)
        except (OSError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            status = 2
            continue
        print(
            f"{path}: total={stats['total']} success={stats['success']} "
            f"failed={stats['failed']} rate={stats['success_rate']:.2f}%"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
