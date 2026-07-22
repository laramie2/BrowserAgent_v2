#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    project_root_for_import = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root_for_import))

from scripts.training_setup import (  # noqa: E402
    CommandRunner,
    ProjectPaths,
    SetupError,
    prepare_resources,
    prepare_sft,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare BrowserAgent RL training resources"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Download base resources and prepare Wiki"
    )
    prepare.add_argument(
        "--wiki-copies",
        type=int,
        default=1,
        help="number of compatible Wiki ZIM paths to expose (default: 1)",
    )
    prepare.add_argument(
        "--dry-run", action="store_true", help="print operations without running them"
    )

    sft = subparsers.add_parser(
        "prepare-sft", help="Download and merge one SFT LoRA dataset"
    )
    sft.add_argument(
        "dataset_id",
        help="unique dataset fragment, for example hotpot6500-nq6300-cr1_2-2048",
    )
    sft.add_argument(
        "--force",
        action="store_true",
        help="replace only the selected incomplete merged model directory",
    )
    sft.add_argument(
        "--dry-run", action="store_true", help="print operations without running them"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    project_root: Path | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    root = project_root or Path(__file__).resolve().parents[1]
    paths = ProjectPaths.from_root(root)
    runner = CommandRunner(dry_run=args.dry_run)
    try:
        if args.command == "prepare":
            prepare_resources(
                paths=paths,
                wiki_copies=args.wiki_copies,
                runner=runner,
                dry_run=args.dry_run,
            )
            print("Resource preparation complete")
        else:
            result = prepare_sft(
                paths=paths,
                dataset_id=args.dataset_id,
                runner=runner,
                force=args.force,
            )
            print(f"SFT directory: {result.top_level}")
            print(f"Checkpoint: {result.checkpoint}")
            print(f"Merged model: {result.merged_output}")
            print(f"SFT_MODEL_NAME_OVERRIDE: {result.model_name}")
        return 0
    except (SetupError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
