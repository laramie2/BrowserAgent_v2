#!/usr/bin/env python3
"""Download BrowserAgent base model, SeedData, and Wiki resources."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    project_root_for_import = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root_for_import))

from env.resource_setup import (  # noqa: E402
    CommandRunner,
    ProjectPaths,
    SetupError,
    prepare_resources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="download public resources and prepare the local Wiki"
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
        prepare_resources(
            paths=paths,
            wiki_copies=args.wiki_copies,
            runner=runner,
            dry_run=args.dry_run,
        )
        print("Resource preparation complete")
        return 0
    except (SetupError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
