from __future__ import annotations

import hashlib
import os
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

BASE_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
VERSION_PATTERN = re.compile(r"^v[^/]*-(\d{8})-(\d{6})$")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
KIWIX_SHA256 = "cdea8226b479515c9495868dec196de9286cba57bc024df7cd15a83690dfbafc"
ZIM_NAME = "wikipedia_en_all_maxi_2022-05.zim"
REQUIRED_RL_DATA = (
    "hotpot/test_50.parquet",
    "nq/test_50.parquet",
    "test_100/data.parquet",
    "t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/data.parquet",
    "t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/data.parquet",
    "t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/data.parquet",
)
SFT_REPO_ID = "Laramie2/browseragent-sft-lora"


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


@dataclass(frozen=True)
class SftResult:
    top_level: str
    checkpoint: Path
    merged_output: Path
    model_name: str


@dataclass
class CommandRunner:
    dry_run: bool = False

    def run(
        self, args: Sequence[str], *, cwd: Path | None = None
    ) -> None:
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
                raise SetupError(
                    f"Kiwix archive contains unsafe path: {member.name}"
                )
            if member.issym() or member.islnk() or member.isdev():
                raise SetupError(
                    f"Kiwix archive contains unsafe member: {member.name}"
                )
        handle.extractall(resolved_destination)
    binary = (
        resolved_destination
        / "kiwix-tools_linux-x86_64-3.3.0/kiwix-serve"
    )
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
            raise SetupError(
                "Combined ZIM size does not equal the sum of its parts"
            )
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


def _resource_download_commands(paths: ProjectPaths) -> list[list[str]]:
    return [
        [
            "hf",
            "download",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--local-dir",
            str(paths.base_model),
        ],
        [
            "hf",
            "download",
            "TIGER-Lab/BrowserAgent-SeedData",
            "--repo-type",
            "dataset",
            "--local-dir",
            str(paths.benchmark),
        ],
        [
            "hf",
            "download",
            "cogito233/WikiEnv",
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
    commands = _resource_download_commands(paths)
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
        raise SetupError(
            "Cannot find hf. Install huggingface_hub and run hf auth login"
        )
    if not paths.root.is_dir() or not os.access(paths.root, os.W_OK):
        raise SetupError(f"Project root is not writable: {paths.root}")
    if not paths.kiwix_archive.is_file():
        raise SetupError(
            f"Bundled Kiwix archive is missing: {paths.kiwix_archive}"
        )
    if sha256_file(paths.kiwix_archive) != KIWIX_SHA256:
        raise SetupError(
            f"Kiwix archive checksum mismatch: {paths.kiwix_archive}"
        )
    dataset_root = paths.root / "RL/dataset"
    missing_rl_data = [
        str(dataset_root / relative)
        for relative in REQUIRED_RL_DATA
        if not (dataset_root / relative).is_file()
    ]
    if missing_rl_data:
        raise SetupError(
            "Clone is missing bundled RL data: " + ", ".join(missing_rl_data)
        )
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


def swift_command(environment: str = "swift-sft") -> list[str]:
    swift = shutil.which("swift")
    if swift:
        return [swift]
    conda = shutil.which("conda")
    if conda:
        return [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            environment,
            "swift",
        ]
    raise SetupError(
        "Cannot find swift or conda; install the swift-sft environment first"
    )


def is_complete_merged_model(path: Path) -> bool:
    has_config = (path / "config.json").is_file()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_config and has_weights


def _repair_merged_metadata(base_model: Path, output: Path) -> None:
    for source in base_model.iterdir():
        if not source.is_file():
            continue
        if source.suffix in {".safetensors", ".bin"}:
            continue
        if source.name.endswith("index.json"):
            continue
        shutil.copy2(source, output / source.name)
    for name in ("processor_config.json", "chat_template.jinja"):
        (output / name).unlink(missing_ok=True)


def merge_lora(
    paths: ProjectPaths,
    checkpoint: Path,
    top_level: str,
    runner: CommandRunner,
    force: bool,
) -> Path:
    if Path(top_level).name != top_level:
        raise SetupError(f"Invalid SFT directory name: {top_level}")
    sft_model_name(top_level)
    output = paths.rl_models / f"{top_level}-merged"
    if output.is_symlink():
        raise SetupError(f"Merged output must not be a symbolic link: {output}")
    if is_complete_merged_model(output):
        return output
    if output.exists():
        if output.resolve().parent != paths.rl_models.resolve():
            raise SetupError(
                f"Refusing to remove output outside RL/models: {output}"
            )
        if not force:
            raise SetupError(
                f"Merged output exists but is incomplete: {output}; use --force"
            )
    if not paths.base_model.is_dir():
        raise SetupError(
            f"Base model is missing: {paths.base_model}; run prepare first"
        )
    if not checkpoint.is_dir():
        raise SetupError(f"SFT checkpoint is missing: {checkpoint}")
    export_command = swift_command() + [
        "export",
        "--model",
        str(paths.base_model),
        "--adapters",
        str(checkpoint),
        "--merge_lora",
        "true",
        "--output_dir",
        str(output),
    ]
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    paths.rl_models.mkdir(parents=True, exist_ok=True)
    runner.run(export_command)
    if not is_complete_merged_model(output):
        raise SetupError(
            f"Swift completed without a valid merged model: {output}"
        )
    _repair_merged_metadata(paths.base_model, output)
    return output


def list_sft_repo_files() -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SetupError(
            "huggingface_hub is required; install it to provide the hf CLI"
        ) from error
    try:
        return list(
            HfApi().list_repo_files(SFT_REPO_ID, repo_type="model")
        )
    except Exception as error:
        raise SetupError(
            f"Cannot list SFT repository {SFT_REPO_ID}: {error}"
        ) from error


def select_repo_checkpoint(
    repo_files: Iterable[str], top_level: str
) -> Path:
    candidates: list[tuple[datetime, int, Path]] = []
    for repo_file in repo_files:
        parts = Path(repo_file).parts
        if len(parts) < 4 or parts[0] != top_level:
            continue
        version_match = VERSION_PATTERN.fullmatch(parts[1])
        checkpoint_match = CHECKPOINT_PATTERN.fullmatch(parts[2])
        if not version_match or not checkpoint_match:
            continue
        stamp = datetime.strptime(
            "".join(version_match.groups()), "%Y%m%d%H%M%S"
        )
        candidates.append(
            (
                stamp,
                int(checkpoint_match.group(1)),
                Path(parts[1]) / parts[2],
            )
        )
    if not candidates:
        raise SetupError(f"No timestamped checkpoint found for {top_level}")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def prepare_sft(
    paths: ProjectPaths,
    dataset_id: str,
    runner: CommandRunner,
    force: bool,
    repo_files: Iterable[str] | None = None,
) -> SftResult:
    files = list(repo_files) if repo_files is not None else list_sft_repo_files()
    top_level = match_sft_directory(files, dataset_id)
    runner.run(
        [
            "hf",
            "download",
            SFT_REPO_ID,
            "--repo-type",
            "model",
            "--include",
            f"{top_level}/*",
            "--local-dir",
            str(paths.sft_output),
        ]
    )
    if runner.dry_run:
        relative_checkpoint = select_repo_checkpoint(files, top_level)
        checkpoint = paths.sft_output / top_level / relative_checkpoint
        merged_output = paths.rl_models / f"{top_level}-merged"
        runner.run(
            [
                "swift",
                "export",
                "--model",
                str(paths.base_model),
                "--adapters",
                str(checkpoint),
                "--merge_lora",
                "true",
                "--output_dir",
                str(merged_output),
            ]
        )
        return SftResult(
            top_level=top_level,
            checkpoint=checkpoint,
            merged_output=merged_output,
            model_name=sft_model_name(top_level),
        )
    checkpoint = select_checkpoint(paths.sft_output / top_level)
    merged_output = merge_lora(paths, checkpoint, top_level, runner, force)
    return SftResult(
        top_level=top_level,
        checkpoint=checkpoint,
        merged_output=merged_output,
        model_name=sft_model_name(top_level),
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
