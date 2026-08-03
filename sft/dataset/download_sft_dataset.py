#!/usr/bin/env python3
"""Download teacher-generated SFT trajectories from Hugging Face Hub.

No repository is baked into this script. Pass it explicitly with ``--repo-id``
or set ``SFT_DATASET_REPO_ID`` when the public dataset is available.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


DATASET_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download BrowserAgent SFT teacher trajectories from Hugging Face"
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("SFT_DATASET_REPO_ID"),
        help="Hugging Face dataset repository ID (or set SFT_DATASET_REPO_ID)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_ROOT / "raw",
        help="local destination (default: sft/dataset/raw)",
    )
    parser.add_argument("--revision", help="optional branch, tag, or commit")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="file glob to include; repeat for multiple patterns",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="file glob to exclude; repeat for multiple patterns",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="download files again even when they are already cached",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved download without contacting Hugging Face",
    )
    return parser


def download_dataset(args: argparse.Namespace) -> Path:
    if not args.repo_id:
        raise ValueError(
            "A dataset repository is required. Pass --repo-id or set "
            "SFT_DATASET_REPO_ID."
        )

    output_dir = args.output_dir.expanduser().resolve()
    print(f"Dataset repository: {args.repo_id}")
    print(f"Destination: {output_dir}")
    if args.dry_run:
        return output_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required. Install the project environment first."
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=args.include or None,
        ignore_patterns=args.exclude or None,
        local_dir=output_dir,
        force_download=args.force_download,
        token=os.environ.get("HF_TOKEN"),
    )
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_dir = download_dataset(args)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"SFT teacher data is available at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
