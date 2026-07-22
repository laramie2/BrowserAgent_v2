from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

BASE_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
VERSION_PATTERN = re.compile(r"^v[^/]*-(\d{8})-(\d{6})$")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


class SetupError(RuntimeError):
    """Raised when training resources cannot be prepared safely."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    base_model: Path
    benchmark: Path
    sft_output: Path
    rl_models: Path
    wiki_root: Path
    kiwix_archive: Path
    kiwix_directory: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve()
        tools = root / "wiki_cluster/tools"
        return cls(
            root=root,
            base_model=root / f"models/{BASE_MODEL_NAME}",
            benchmark=root / "benchmark",
            sft_output=root / "sft/output",
            rl_models=root / "RL/models",
            wiki_root=root / "webarena/webarena_zim",
            kiwix_archive=tools / "kiwix-tools_linux-x86_64-3.3.0.tar.gz",
            kiwix_directory=tools / "kiwix-tools_linux-x86_64-3.3.0",
        )


def _top_level_directories(repo_files: Iterable[str]) -> list[str]:
    return sorted({path.split("/", 1)[0] for path in repo_files if "/" in path})


def match_sft_directory(repo_files: Iterable[str], dataset_id: str) -> str:
    if not dataset_id or "/" in dataset_id:
        raise SetupError(
            "Dataset identifier must be a non-empty directory-name fragment"
        )
    directories = _top_level_directories(repo_files)
    candidates = [name for name in directories if dataset_id in name]
    if not candidates:
        available = ", ".join(directories)
        raise SetupError(
            f"No SFT directory matches {dataset_id!r}. Available: {available}"
        )
    if len(candidates) != 1:
        raise SetupError(
            f"Multiple SFT directories match {dataset_id!r}: "
            f"{', '.join(candidates)}"
        )
    return candidates[0]


def select_checkpoint(sft_directory: Path) -> Path:
    versions: list[tuple[datetime, Path]] = []
    children = sft_directory.iterdir() if sft_directory.is_dir() else ()
    for child in children:
        match = VERSION_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
            versions.append((stamp, child))
    if not versions:
        raise SetupError(
            f"No timestamped version directory found in {sft_directory}"
        )
    version = max(versions, key=lambda item: item[0])[1]
    checkpoints: list[tuple[int, Path]] = []
    for child in version.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            checkpoints.append((int(match.group(1)), child))
    if not checkpoints:
        raise SetupError(f"No checkpoint-N directory found in {version}")
    return max(checkpoints, key=lambda item: item[0])[1]


def sft_model_name(top_level: str) -> str:
    prefix = f"{BASE_MODEL_NAME}-"
    if not top_level.startswith(prefix):
        raise SetupError(f"SFT directory must start with {prefix}")
    return top_level[len(prefix) :]
