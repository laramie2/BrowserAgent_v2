#!/usr/bin/env python3
"""Download only the files needed by mini_webarena's prompt tokenizer."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "Qwen2.5-14B-Instruct",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.output_dir,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        ],
    )
    tokenizer_file = args.output_dir / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise SystemExit(f"Download finished but {tokenizer_file} is missing")
    print(f"Prompt tokenizer ready: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
