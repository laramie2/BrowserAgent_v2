"""Download and validate public model, SeedData, Kiwix, and Wiki resources."""

from __future__ import annotations

import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


BASE_MODEL_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
SEED_DATASET_REPO = "TIGER-Lab/BrowserAgent-SeedData"
WIKI_DATASET_REPO = "cogito233/WikiEnv"
BASE_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
KIWIX_SHA256 = "cdea8226b479515c9495868dec196de9286cba57bc024df7cd15a83690dfbafc"
ZIM_NAME = "wikipedia_en_all_maxi_2022-05.zim"


class SetupError(RuntimeError):
    """Raised when public training resources cannot be prepared safely."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    base_model: Path
    seed_dataset: Path
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
            seed_dataset=root / "RL/dataset/BrowserAgent-SeedData",
            wiki_root=root / "webarena/webarena_zim",
            kiwix_archive=tools / "kiwix-tools_linux-x86_64-3.3.0.tar.gz",
            kiwix_directory=tools / "kiwix-tools_linux-x86_64-3.3.0",
        )


@dataclass
class CommandRunner:
    dry_run: bool = False

    def run(self, args: Sequence[str], *, cwd: Path | None = None) -> None:
        print("+", shlex.join(str(value) for value in args), flush=True)
        if not self.dry_run:
            subprocess.run([str(value) for value in args], cwd=cwd, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract_kiwix(archive: Path, destination: Path) -> Path:
    if sha256_file(archive) != KIWIX_SHA256:
        raise SetupError(f"Kiwix archive checksum mismatch: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            member_path = (resolved_destination / member.name).resolve()
            if not member_path.is_relative_to(resolved_destination):
                raise SetupError(f"Kiwix archive contains unsafe path: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise SetupError(f"Kiwix archive contains unsafe member: {member.name}")
        handle.extractall(resolved_destination)
    binary = resolved_destination / "kiwix-tools_linux-x86_64-3.3.0/kiwix-serve"
    if not binary.is_file():
        raise SetupError(f"Kiwix archive did not contain {binary}")
    binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def combine_zim_parts(parts: Sequence[Path], output: Path) -> Path:
    ordered = sorted(parts, key=lambda path: path.name)
    if not ordered:
        raise SetupError("No ZIM part files were found")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    expected_size = sum(path.stat().st_size for path in ordered)
    try:
        with temporary.open("wb") as target:
            for part in ordered:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary.stat().st_size != expected_size:
            raise SetupError("Combined ZIM size does not equal the sum of its parts")
        os.replace(temporary, output)
        for part in ordered:
            part.unlink()
        return output
    finally:
        temporary.unlink(missing_ok=True)


def ensure_wiki_copies(wiki_root: Path, copies: int) -> list[Path]:
    if copies < 1:
        raise SetupError("--wiki-copies must be at least 1")
    source = wiki_root / f"1/{ZIM_NAME}"
    if not source.is_file():
        raise SetupError(f"Wiki ZIM does not exist: {source}")
    results = [source]
    for index in range(2, copies + 1):
        link = wiki_root / str(index) / ZIM_NAME
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == source.resolve():
                results.append(link)
                continue
            raise SetupError(f"Refusing to replace existing Wiki copy: {link}")
        link.symlink_to(Path("../1") / ZIM_NAME)
        results.append(link)
    return results


def resource_download_commands(paths: ProjectPaths) -> list[list[str]]:
    return [
        ["hf", "download", BASE_MODEL_REPO, "--local-dir", str(paths.base_model)],
        [
            "hf",
            "download",
            SEED_DATASET_REPO,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(paths.seed_dataset),
        ],
        [
            "hf",
            "download",
            WIKI_DATASET_REPO,
            "--repo-type",
            "dataset",
            "--include",
            f"{ZIM_NAME}.part-*",
            "--local-dir",
            str(paths.wiki_root / "1"),
        ],
    ]


def prepare_resources(
    paths: ProjectPaths,
    wiki_copies: int,
    runner: CommandRunner,
    dry_run: bool,
) -> None:
    commands = resource_download_commands(paths)
    if dry_run:
        for command in commands:
            runner.run(command)
        return
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise SetupError("Bundled Kiwix supports Linux x86_64 only")
    if shutil.which("hf") is None:
        raise SetupError("Cannot find hf. Install huggingface_hub and run hf auth login")
    if not paths.root.is_dir() or not os.access(paths.root, os.W_OK):
        raise SetupError(f"Project root is not writable: {paths.root}")
    if not paths.kiwix_archive.is_file():
        raise SetupError(f"Bundled Kiwix archive is missing: {paths.kiwix_archive}")
    if sha256_file(paths.kiwix_archive) != KIWIX_SHA256:
        raise SetupError(f"Kiwix archive checksum mismatch: {paths.kiwix_archive}")

    binary = paths.kiwix_directory / "kiwix-serve"
    if not binary.is_file():
        safe_extract_kiwix(paths.kiwix_archive, paths.kiwix_archive.parent)

    runner.run(commands[0])
    runner.run(commands[1])
    zim = paths.wiki_root / f"1/{ZIM_NAME}"
    if not zim.is_file() or zim.stat().st_size == 0:
        runner.run(commands[2])
        combine_zim_parts(list(zim.parent.glob(f"{ZIM_NAME}.part-*")), zim)
    ensure_wiki_copies(paths.wiki_root, wiki_copies)
