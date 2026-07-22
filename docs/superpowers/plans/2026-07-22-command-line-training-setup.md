# Command-Line Training Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an idempotent terminal CLI that prepares BrowserAgent models, benchmark, Wiki resources, and SFT LoRA merges while keeping RL launch configuration and scripts unchanged.

**Architecture:** A thin `argparse` entry point delegates to a standard-library-focused `training_setup` module. Filesystem and subprocess behavior is injected at module boundaries so network-, GPU-, and conda-independent tests can exercise matching, checkpoint selection, archive extraction, ZIM assembly, downloads, and merge orchestration.

**Tech Stack:** Python 3.10, standard library (`argparse`, `dataclasses`, `hashlib`, `pathlib`, `subprocess`, `tarfile`, `unittest`), Hugging Face `hf` CLI and `huggingface_hub`, Swift CLI, Git.

## Global Constraints

- Target branch is `feat/command-line-training-setup` based on commit `f20929b` from `slim-server`.
- Do not create, update, or delete conda environments.
- Do not modify RL algorithms, hyperparameters, rewards, samplers, `RL/scripts/train.sh`, or `RL/scripts/auto_train.sh`.
- Derive all runtime paths from the CLI file location; production code must contain no machine-specific project path.
- Read Hugging Face authentication only from `HF_TOKEN` or the existing `hf auth login` state; never accept or log a token argument.
- Keep all downloads and model merges in the foreground.
- Preserve the user's existing modified parquet files and the modified `verl-tool` submodule; never stage them.
- Use only the unique SFT top-level directory containing the dataset identifier; zero or multiple matches are errors.
- Choose the newest parsed `v*-YYYYMMDD-HHMMSS` directory, then its numerically largest `checkpoint-N`.
- Bundle Kiwix 3.3.0 as `wiki_cluster/tools/kiwix-tools_linux-x86_64-3.3.0.tar.gz` with SHA-256 `cdea8226b479515c9495868dec196de9286cba57bc024df7cd15a83690dfbafc`.
- Keep one physical ZIM by default; optional extra Wiki paths are relative symlinks.
- `--dry-run` must produce no filesystem changes and launch no download or merge command.

---

### Task 1: Core path, SFT matching, and checkpoint selection model

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/training_setup.py`
- Create: `tests/__init__.py`
- Create: `tests/test_training_setup.py`

**Interfaces:**
- Produces: `SetupError`, `ProjectPaths.from_root(root: Path)`, `match_sft_directory(repo_files: Iterable[str], dataset_id: str) -> str`, `select_checkpoint(sft_directory: Path) -> Path`, and `sft_model_name(top_level: str) -> str`.
- Consumes: no project code; all inputs are strings or `pathlib.Path` values.

- [ ] **Step 1: Write failing tests for path derivation and SFT selection**

Create package marker files and write `tests/test_training_setup.py` with these initial cases:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.training_setup import (
    ProjectPaths,
    SetupError,
    match_sft_directory,
    select_checkpoint,
    sft_model_name,
)


class ProjectPathsTest(unittest.TestCase):
    def test_from_root_builds_only_project_relative_paths(self) -> None:
        root = Path("/tmp/project with spaces")
        paths = ProjectPaths.from_root(root)
        self.assertEqual(paths.base_model, root / "models/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(paths.benchmark, root / "benchmark")
        self.assertEqual(paths.sft_output, root / "sft/output")
        self.assertEqual(paths.rl_models, root / "RL/models")
        self.assertEqual(paths.wiki_root, root / "webarena/webarena_zim")


class SftSelectionTest(unittest.TestCase):
    REPO_FILES = [
        "Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false/v5-20260705-013607/checkpoint-854/adapter_config.json",
        "Qwen2.5-VL-7B-Instruct-task-opsrc-other-data-sft-5e-5lr-freeze_false/v1-20260701-010101/checkpoint-10/adapter_config.json",
    ]

    def test_match_sft_directory_returns_unique_top_level_directory(self) -> None:
        result = match_sft_directory(self.REPO_FILES, "hotpot6500-nq6300-cr1_2-2048")
        self.assertEqual(
            result,
            "Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false",
        )

    def test_match_sft_directory_rejects_no_match(self) -> None:
        with self.assertRaisesRegex(SetupError, "No SFT directory"):
            match_sft_directory(self.REPO_FILES, "missing-dataset")

    def test_match_sft_directory_rejects_multiple_matches(self) -> None:
        files = self.REPO_FILES + [
            "Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-3e-1e-5lr-freeze_true/v1-20260706-010101/checkpoint-1/adapter_config.json"
        ]
        with self.assertRaisesRegex(SetupError, "Multiple SFT directories"):
            match_sft_directory(files, "hotpot6500-nq6300-cr1_2-2048")

    def test_sft_model_name_removes_only_base_model_prefix(self) -> None:
        value = sft_model_name(
            "Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false"
        )
        self.assertEqual(
            value,
            "task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false",
        )


class CheckpointSelectionTest(unittest.TestCase):
    def test_selects_latest_timestamp_then_largest_numeric_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sft_dir = Path(directory)
            for relative in (
                "v9-20260704-235959/checkpoint-9999",
                "v1-20260705-013607/checkpoint-99",
                "v1-20260705-013607/checkpoint-854",
            ):
                (sft_dir / relative).mkdir(parents=True)
            self.assertEqual(
                select_checkpoint(sft_dir),
                sft_dir / "v1-20260705-013607/checkpoint-854",
            )

    def test_rejects_directory_without_valid_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "latest/checkpoint-2").mkdir(parents=True)
            with self.assertRaisesRegex(SetupError, "timestamped version"):
                select_checkpoint(Path(directory))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the feature is absent**

Run:

```bash
python3 -m unittest tests.test_training_setup -v
```

Expected: non-zero exit because `scripts.training_setup` does not yet provide the imported interfaces.

- [ ] **Step 3: Implement the minimal core model**

Create `scripts/training_setup.py` with constants, the path dataclass, and deterministic matching:

```python
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
    pass


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
        raise SetupError("Dataset identifier must be a non-empty directory-name fragment")
    candidates = [name for name in _top_level_directories(repo_files) if dataset_id in name]
    if not candidates:
        available = ", ".join(_top_level_directories(repo_files))
        raise SetupError(f"No SFT directory matches {dataset_id!r}. Available: {available}")
    if len(candidates) != 1:
        raise SetupError(
            f"Multiple SFT directories match {dataset_id!r}: {', '.join(candidates)}"
        )
    return candidates[0]


def select_checkpoint(sft_directory: Path) -> Path:
    versions: list[tuple[datetime, Path]] = []
    for child in sft_directory.iterdir() if sft_directory.is_dir() else ():
        match = VERSION_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
            versions.append((stamp, child))
    if not versions:
        raise SetupError(f"No timestamped version directory found in {sft_directory}")
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
    return top_level[len(prefix):]
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
python3 -m unittest tests.test_training_setup -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit only the core model and tests**

```bash
git add scripts/__init__.py scripts/training_setup.py tests/__init__.py tests/test_training_setup.py
git diff --cached --check
git commit -m "feat: add training setup selection core"
```

### Task 2: Kiwix, ZIM, Wiki links, and resource download preparation

**Files:**
- Modify: `scripts/training_setup.py`
- Modify: `tests/test_training_setup.py`

**Interfaces:**
- Consumes: `ProjectPaths` and `SetupError` from Task 1.
- Produces: `CommandRunner.run(args: Sequence[str])`, `sha256_file(path: Path) -> str`, `safe_extract_kiwix(archive: Path, destination: Path) -> Path`, `combine_zim_parts(parts: Sequence[Path], output: Path) -> Path`, `ensure_wiki_copies(wiki_root: Path, copies: int) -> list[Path]`, and `prepare_resources(paths: ProjectPaths, wiki_copies: int, runner: CommandRunner, dry_run: bool) -> None`.

- [ ] **Step 1: Add failing resource-operation tests**

Append tests that create only temporary files and a recording runner:

```python
import hashlib
import io
import os
import tarfile

from scripts.training_setup import (
    CommandRunner,
    combine_zim_parts,
    ensure_wiki_copies,
    safe_extract_kiwix,
    sha256_file,
)


class ResourceOperationTest(unittest.TestCase):
    def test_sha256_file_streams_expected_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.tar.gz"
            path.write_bytes(b"kiwix")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"kiwix").hexdigest())

    def test_safe_extract_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                handle.addfile(info, io.BytesIO(b"x"))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with patch("scripts.training_setup.KIWIX_SHA256", digest):
                with self.assertRaisesRegex(SetupError, "unsafe path"):
                    safe_extract_kiwix(archive, Path(directory) / "tools")

    def test_combine_zim_parts_uses_filename_order_and_removes_parts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            second = root / "wikipedia.zim.part-02"
            first = root / "wikipedia.zim.part-01"
            second.write_bytes(b"B")
            first.write_bytes(b"A")
            output = root / "1/wikipedia.zim"
            combine_zim_parts([second, first], output)
            self.assertEqual(output.read_bytes(), b"AB")
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())

    def test_ensure_wiki_copies_defaults_to_one_physical_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "1/wikipedia_en_all_maxi_2022-05.zim"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"zim")
            self.assertEqual(ensure_wiki_copies(root, 1), [source])
            self.assertFalse((root / "2").exists())

    def test_ensure_wiki_copies_creates_relative_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "1/wikipedia_en_all_maxi_2022-05.zim"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"zim")
            paths = ensure_wiki_copies(root, 3)
            self.assertEqual(len(paths), 3)
            self.assertTrue(paths[1].is_symlink())
            self.assertEqual(os.readlink(paths[1]), "../1/wikipedia_en_all_maxi_2022-05.zim")
            self.assertEqual(paths[2].read_bytes(), b"zim")
```

- [ ] **Step 2: Run focused tests and observe missing interfaces**

Run:

```bash
python3 -m unittest tests.test_training_setup.ResourceOperationTest -v
```

Expected: non-zero exit because the resource functions are not implemented.

- [ ] **Step 3: Implement safe local resource operations**

Add these behaviors to `scripts/training_setup.py`:

```python
import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
from collections.abc import Sequence

KIWIX_SHA256 = "cdea8226b479515c9495868dec196de9286cba57bc024df7cd15a83690dfbafc"
ZIM_NAME = "wikipedia_en_all_maxi_2022-05.zim"


@dataclass
class CommandRunner:
    dry_run: bool = False

    def run(self, args: Sequence[str], *, cwd: Path | None = None) -> None:
        print("+", " ".join(str(value) for value in args), flush=True)
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
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination):
                raise SetupError(f"Kiwix archive contains unsafe path: {member.name}")
        handle.extractall(destination)
    binary = destination / "kiwix-tools_linux-x86_64-3.3.0/kiwix-serve"
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
```

- [ ] **Step 4: Add the failing prepare orchestration test**

Add a `RecordingRunner` test double and assert exact HF destinations:

```python
from unittest.mock import patch

from scripts.training_setup import prepare_resources


class RecordingRunner(CommandRunner):
    def __init__(self, dry_run: bool = False) -> None:
        super().__init__(dry_run=dry_run)
        self.commands: list[list[str]] = []

    def run(self, args, *, cwd=None) -> None:
        self.commands.append([str(value) for value in args])


class PrepareResourcesTest(unittest.TestCase):
    def test_dry_run_emits_three_hf_downloads_without_creating_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            paths = ProjectPaths.from_root(root)
            runner = RecordingRunner(dry_run=True)
            prepare_resources(paths, wiki_copies=1, runner=runner, dry_run=True)
            self.assertEqual([command[:3] for command in runner.commands], [
                ["hf", "download", "Qwen/Qwen2.5-VL-7B-Instruct"],
                ["hf", "download", "TIGER-Lab/BrowserAgent-SeedData"],
                ["hf", "download", "cogito233/WikiEnv"],
            ])
            self.assertFalse(root.exists())
```

- [ ] **Step 5: Implement prepare orchestration and platform checks**

Implement `prepare_resources` so dry-run only records commands. A real run must check `platform.system() == "Linux"`, normalize `platform.machine()` to x86_64/amd64, verify `shutil.which("hf")`, extract the bundled archive if needed, run these commands in this order, combine `*.zim.part-*`, and create links:

```python
def prepare_resources(
    paths: ProjectPaths,
    wiki_copies: int,
    runner: CommandRunner,
    dry_run: bool,
) -> None:
    commands = [
        ["hf", "download", "Qwen/Qwen2.5-VL-7B-Instruct", "--local-dir", str(paths.base_model)],
        ["hf", "download", "TIGER-Lab/BrowserAgent-SeedData", "--repo-type", "dataset", "--local-dir", str(paths.benchmark)],
        [
            "hf", "download", "cogito233/WikiEnv", "--repo-type", "dataset",
            "--include", f"{ZIM_NAME}.part-*", "--local-dir", str(paths.wiki_root / "1"),
        ],
    ]
    if dry_run:
        for command in commands:
            runner.run(command)
        return
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SetupError("Bundled Kiwix supports Linux x86_64 only")
    if shutil.which("hf") is None:
        raise SetupError("Cannot find hf. Install huggingface_hub and run hf auth login")
    if not paths.root.is_dir() or not os.access(paths.root, os.W_OK):
        raise SetupError(f"Project root is not writable: {paths.root}")
    if not paths.kiwix_archive.is_file():
        raise SetupError(f"Bundled Kiwix archive is missing: {paths.kiwix_archive}")
    if sha256_file(paths.kiwix_archive) != KIWIX_SHA256:
        raise SetupError(f"Kiwix archive checksum mismatch: {paths.kiwix_archive}")
    required_rl_data = (
        paths.root / "RL/dataset/hotpot/test_50.parquet",
        paths.root / "RL/dataset/nq/test_50.parquet",
        paths.root / "RL/dataset/test_100/data.parquet",
        paths.root / "RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/data.parquet",
        paths.root / "RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/data.parquet",
        paths.root / "RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/data.parquet",
    )
    missing_rl_data = [str(path) for path in required_rl_data if not path.is_file()]
    if missing_rl_data:
        raise SetupError("Clone is missing bundled RL data: " + ", ".join(missing_rl_data))
    binary = paths.kiwix_directory / "kiwix-serve"
    if not binary.is_file():
        safe_extract_kiwix(paths.kiwix_archive, paths.kiwix_archive.parent)
    for command in commands:
        runner.run(command)
    zim = paths.wiki_root / f"1/{ZIM_NAME}"
    if not zim.is_file() or zim.stat().st_size == 0:
        combine_zim_parts(list(zim.parent.glob(f"{ZIM_NAME}.part-*")), zim)
    ensure_wiki_copies(paths.wiki_root, wiki_copies)
```

- [ ] **Step 6: Run all core tests and commit**

Run:

```bash
python3 -m unittest tests.test_training_setup -v
```

Expected: all tests pass.

Commit:

```bash
git add scripts/training_setup.py tests/test_training_setup.py
git diff --cached --check
git commit -m "feat: prepare wiki and model resources"
```

### Task 3: SFT repository discovery, download, and foreground LoRA merge

**Files:**
- Modify: `scripts/training_setup.py`
- Modify: `tests/test_training_setup.py`

**Interfaces:**
- Consumes: selection functions, `ProjectPaths`, and `CommandRunner` from Tasks 1–2.
- Produces: `list_sft_repo_files() -> list[str]`, `select_repo_checkpoint(repo_files: Iterable[str], top_level: str) -> Path`, `swift_command(environment: str = "swift-sft") -> list[str]`, `is_complete_merged_model(path: Path) -> bool`, `merge_lora(paths: ProjectPaths, checkpoint: Path, top_level: str, runner: CommandRunner, force: bool) -> Path`, `prepare_sft(paths: ProjectPaths, dataset_id: str, runner: CommandRunner, force: bool, repo_files: Iterable[str] | None = None) -> SftResult`, and immutable `SftResult(top_level, checkpoint, merged_output, model_name)`.

- [ ] **Step 1: Write failing Swift command and merge metadata tests**

Add tests using temporary base/output directories and `unittest.mock.patch`:

```python
from scripts.training_setup import (
    SftResult,
    is_complete_merged_model,
    merge_lora,
    swift_command,
)


class MergeTest(unittest.TestCase):
    def test_swift_command_prefers_current_path(self) -> None:
        with patch("scripts.training_setup.shutil.which", side_effect=lambda name: "/env/bin/swift" if name == "swift" else None):
            self.assertEqual(swift_command(), ["/env/bin/swift"])

    def test_swift_command_falls_back_to_conda_environment(self) -> None:
        with patch("scripts.training_setup.shutil.which", side_effect=lambda name: "/conda/bin/conda" if name == "conda" else None):
            self.assertEqual(
                swift_command(),
                ["/conda/bin/conda", "run", "--no-capture-output", "-n", "swift-sft", "swift"],
            )

    def test_complete_merged_model_requires_config_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(is_complete_merged_model(output))
            (output / "model.safetensors").write_bytes(b"weights")
            self.assertTrue(is_complete_merged_model(output))

    def test_merge_runs_foreground_export_and_repairs_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ProjectPaths.from_root(root)
            paths.base_model.mkdir(parents=True)
            (paths.base_model / "config.json").write_text('{"base": true}', encoding="utf-8")
            (paths.base_model / "tokenizer.json").write_text("{}", encoding="utf-8")
            (paths.base_model / "model.safetensors").write_bytes(b"base")
            checkpoint = root / "checkpoint-854"
            checkpoint.mkdir()
            runner = RecordingRunner()
            output = paths.rl_models / "Qwen2.5-VL-7B-Instruct-example-merged"

            def create_fake_output(args, *, cwd=None):
                runner.commands.append([str(value) for value in args])
                output.mkdir(parents=True)
                (output / "model.safetensors").write_bytes(b"merged")
                (output / "config.json").write_text("{}", encoding="utf-8")
                (output / "processor_config.json").write_text("{}", encoding="utf-8")
                (output / "chat_template.jinja").write_text("template", encoding="utf-8")

            runner.run = create_fake_output
            result = merge_lora(paths, checkpoint, "Qwen2.5-VL-7B-Instruct-example", runner, force=False)
            self.assertEqual(result, output)
            self.assertIn("--adapters", runner.commands[0])
            self.assertEqual((output / "tokenizer.json").read_text(encoding="utf-8"), "{}")
            self.assertFalse((output / "processor_config.json").exists())
            self.assertFalse((output / "chat_template.jinja").exists())
```

- [ ] **Step 2: Run merge tests and confirm missing behavior**

Run:

```bash
python3 -m unittest tests.test_training_setup.MergeTest -v
```

Expected: non-zero exit because merge interfaces are absent.

- [ ] **Step 3: Implement foreground merge and bounded force deletion**

Implement these rules:

```python
@dataclass(frozen=True)
class SftResult:
    top_level: str
    checkpoint: Path
    merged_output: Path
    model_name: str


def swift_command(environment: str = "swift-sft") -> list[str]:
    swift = shutil.which("swift")
    if swift:
        return [swift]
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "--no-capture-output", "-n", environment, "swift"]
    raise SetupError("Cannot find swift or conda; install the swift-sft environment first")


def is_complete_merged_model(path: Path) -> bool:
    has_config = (path / "config.json").is_file()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_config and has_weights


def _repair_merged_metadata(base_model: Path, output: Path) -> None:
    for source in base_model.iterdir():
        if not source.is_file():
            continue
        if source.suffix in {".safetensors", ".bin"} or source.name.endswith("index.json"):
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
    output = paths.rl_models / f"{top_level}-merged"
    if is_complete_merged_model(output):
        return output
    if output.exists():
        resolved_parent = output.resolve().parent
        if resolved_parent != paths.rl_models.resolve():
            raise SetupError(f"Refusing to remove output outside RL/models: {output}")
        if not force:
            raise SetupError(f"Merged output exists but is incomplete: {output}; use --force")
        shutil.rmtree(output)
    if not paths.base_model.is_dir():
        raise SetupError(f"Base model is missing: {paths.base_model}; run prepare first")
    if not checkpoint.is_dir():
        raise SetupError(f"SFT checkpoint is missing: {checkpoint}")
    paths.rl_models.mkdir(parents=True, exist_ok=True)
    runner.run(swift_command() + [
        "export", "--model", str(paths.base_model), "--adapters", str(checkpoint),
        "--merge_lora", "true", "--output_dir", str(output),
    ])
    if not is_complete_merged_model(output):
        raise SetupError(f"Swift completed without a valid merged model: {output}")
    _repair_merged_metadata(paths.base_model, output)
    return output
```

- [ ] **Step 4: Write a failing end-to-end SFT orchestration test**

Inject repository files rather than using the network. Assert the download include pattern, checkpoint, output, and YAML value:

```python
from scripts.training_setup import prepare_sft


class PrepareSftTest(unittest.TestCase):
    def test_prepare_sft_downloads_unique_directory_and_returns_training_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ProjectPaths.from_root(root)
            top_level = "Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false"
            repo_files = [f"{top_level}/v5-20260705-013607/checkpoint-854/adapter_config.json"]
            runner = RecordingRunner()

            def fake_run(args, *, cwd=None):
                runner.commands.append([str(value) for value in args])
                if args[:2] == ["hf", "download"]:
                    checkpoint = paths.sft_output / top_level / "v5-20260705-013607/checkpoint-854"
                    checkpoint.mkdir(parents=True)

            runner.run = fake_run
            with patch("scripts.training_setup.merge_lora", return_value=paths.rl_models / f"{top_level}-merged"):
                result = prepare_sft(
                    paths,
                    "hotpot6500-nq6300-cr1_2-2048",
                    runner,
                    force=False,
                    repo_files=repo_files,
                )
            self.assertEqual(result.top_level, top_level)
            self.assertEqual(result.checkpoint.name, "checkpoint-854")
            self.assertEqual(
                result.model_name,
                "task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false",
            )
            self.assertIn(f"{top_level}/*", runner.commands[0])

    def test_prepare_sft_dry_run_uses_remote_checkpoint_without_filesystem_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            paths = ProjectPaths.from_root(root)
            top_level = "Qwen2.5-VL-7B-Instruct-task-opsrc-example-sft-5e-5lr-freeze_false"
            repo_files = [
                f"{top_level}/v1-20260704-235959/checkpoint-999/adapter_config.json",
                f"{top_level}/v2-20260705-013607/checkpoint-854/adapter_config.json",
            ]
            runner = RecordingRunner(dry_run=True)
            result = prepare_sft(paths, "example", runner, force=False, repo_files=repo_files)
            self.assertEqual(result.checkpoint.name, "checkpoint-854")
            self.assertEqual(runner.commands[0][:2], ["hf", "download"])
            self.assertEqual(runner.commands[1][0:2], ["swift", "export"])
            self.assertFalse(root.exists())
```

- [ ] **Step 5: Implement Hub discovery and SFT orchestration**

Use the already-required `huggingface_hub` package installed with `hf`:

```python
SFT_REPO_ID = "Laramie2/browseragent-sft-lora"


def list_sft_repo_files() -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SetupError("huggingface_hub is required; install it to provide the hf CLI") from error
    try:
        return list(HfApi().list_repo_files(SFT_REPO_ID, repo_type="model"))
    except Exception as error:
        raise SetupError(f"Cannot list SFT repository {SFT_REPO_ID}: {error}") from error


def select_repo_checkpoint(repo_files: Iterable[str], top_level: str) -> Path:
    candidates: list[tuple[datetime, int, Path]] = []
    for repo_file in repo_files:
        parts = Path(repo_file).parts
        if len(parts) < 4 or parts[0] != top_level:
            continue
        version_match = VERSION_PATTERN.fullmatch(parts[1])
        checkpoint_match = CHECKPOINT_PATTERN.fullmatch(parts[2])
        if not version_match or not checkpoint_match:
            continue
        stamp = datetime.strptime("".join(version_match.groups()), "%Y%m%d%H%M%S")
        candidates.append((stamp, int(checkpoint_match.group(1)), Path(parts[1]) / parts[2]))
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
    runner.run([
        "hf", "download", SFT_REPO_ID, "--repo-type", "model",
        "--include", f"{top_level}/*", "--local-dir", str(paths.sft_output),
    ])
    if runner.dry_run:
        relative_checkpoint = select_repo_checkpoint(files, top_level)
        checkpoint = paths.sft_output / top_level / relative_checkpoint
        merged_output = paths.rl_models / f"{top_level}-merged"
        runner.run([
            "swift", "export", "--model", str(paths.base_model),
            "--adapters", str(checkpoint), "--merge_lora", "true",
            "--output_dir", str(merged_output),
        ])
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
```

For CLI dry-run, remote listing is allowed but `hf download`, directory creation, deletion, and Swift remain disabled. If a real repo listing cannot be obtained, return a clear `SetupError` without printing credentials.

- [ ] **Step 6: Run tests and commit SFT support**

Run:

```bash
python3 -m unittest tests.test_training_setup -v
```

Expected: all tests pass.

Commit:

```bash
git add scripts/training_setup.py tests/test_training_setup.py
git diff --cached --check
git commit -m "feat: download and merge SFT checkpoints"
```

### Task 4: CLI entry point and deployment documentation

**Files:**
- Create: `scripts/prepare_training.py`
- Create: `tests/test_prepare_training_cli.py`
- Create: `docs/command-line-training.md`
- Modify: `wiki_cluster/README.md`
- Modify: `wiki_cluster/start.sh:11`

**Interfaces:**
- Consumes: `ProjectPaths`, `CommandRunner`, `prepare_resources`, and `prepare_sft`.
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv: Sequence[str] | None = None, project_root: Path | None = None) -> int`, and the documented commands used by operators.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_prepare_training_cli.py`:

```python
from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_training import main
from scripts.training_setup import SftResult, SetupError


class PrepareTrainingCliTest(unittest.TestCase):
    def test_prepare_passes_wiki_copies_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("scripts.prepare_training.prepare_resources") as prepare:
                status = main(
                    ["prepare", "--wiki-copies", "4", "--dry-run"],
                    project_root=Path(directory),
                )
            self.assertEqual(status, 0)
            self.assertEqual(prepare.call_args.kwargs["wiki_copies"], 4)
            self.assertTrue(prepare.call_args.kwargs["dry_run"])

    def test_prepare_sft_prints_yaml_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = SftResult(
                top_level="Qwen2.5-VL-7B-Instruct-example",
                checkpoint=Path(directory) / "checkpoint-854",
                merged_output=Path(directory) / "RL/models/Qwen2.5-VL-7B-Instruct-example-merged",
                model_name="example",
            )
            output = io.StringIO()
            with patch("scripts.prepare_training.prepare_sft", return_value=result):
                with redirect_stdout(output):
                    status = main(["prepare-sft", "example"], project_root=Path(directory))
            self.assertEqual(status, 0)
            self.assertIn("SFT_MODEL_NAME_OVERRIDE: example", output.getvalue())

    def test_setup_error_returns_actionable_nonzero_status(self) -> None:
        error = io.StringIO()
        with patch("scripts.prepare_training.prepare_resources", side_effect=SetupError("missing hf")):
            with redirect_stderr(error):
                status = main(["prepare"], project_root=Path("/tmp/project"))
        self.assertEqual(status, 2)
        self.assertIn("missing hf", error.getvalue())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run CLI tests and observe missing entry point**

Run:

```bash
python3 -m unittest tests.test_prepare_training_cli -v
```

Expected: non-zero exit because `scripts.prepare_training` is absent.

- [ ] **Step 3: Implement the thin argparse entry point**

Create `scripts/prepare_training.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts.training_setup import (
    CommandRunner,
    ProjectPaths,
    SetupError,
    prepare_resources,
    prepare_sft,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare BrowserAgent RL training resources")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Download base resources and prepare Wiki")
    prepare.add_argument("--wiki-copies", type=int, default=1)
    prepare.add_argument("--dry-run", action="store_true")
    sft = subparsers.add_parser("prepare-sft", help="Download and merge one SFT LoRA dataset")
    sft.add_argument("dataset_id")
    sft.add_argument("--force", action="store_true")
    sft.add_argument("--dry-run", action="store_true")
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
```

When invoked as `python3 scripts/prepare_training.py`, place this exact bootstrap after standard-library imports and before importing `scripts.training_setup`:

```python
if __package__ in {None, ""}:
    project_root_for_import = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root_for_import))
```

Normal execution still computes `root = project_root or Path(__file__).resolve().parents[1]`, so the import root and runtime project root remain identical.

- [ ] **Step 4: Add operator documentation**

Write `docs/command-line-training.md` with exact prerequisites and commands:

```markdown
# BrowserAgent 命令行训练准备

前置条件：已按 `env/README.md` 创建 `browseragent-v2` 与 `swift-sft` 环境，并完成 `hf auth login` 或设置 `HF_TOKEN`。CLI 不安装或更新环境。

## 1. 准备公共资源

python3 scripts/prepare_training.py prepare --dry-run
python3 scripts/prepare_training.py prepare

默认只保留一份 Wiki ZIM。需要兼容四路径布局时运行：

python3 scripts/prepare_training.py prepare --wiki-copies 4

## 2. 下载并合并 SFT

python3 scripts/prepare_training.py prepare-sft hotpot6500-nq6300-cr1_2-2048

命令会前台显示 Swift 日志，并打印 `SFT_MODEL_NAME_OVERRIDE`。将该值写入 `RL/configs/train.yaml` 的 `common.env`。

## 3. 启动 RL

bash RL/scripts/train.sh
# 或
bash RL/scripts/auto_train.sh
```

Also update `wiki_cluster/README.md` to state that `prepare` extracts the bundled Kiwix archive and that plain `./start.sh` uses the one-copy default.

Append this operator-facing section to `wiki_cluster/README.md`:

```markdown
## Prepare Wiki Assets

From the project root, the training preparation CLI verifies and extracts the bundled Kiwix 3.3.0 archive, downloads the Wiki ZIM parts, and assembles the ZIM locally:

python3 scripts/prepare_training.py prepare

The default layout stores one physical ZIM and `./wiki_cluster/start.sh` uses one ZIM path. To expose four compatible paths without duplicating the ZIM bytes:

python3 scripts/prepare_training.py prepare --wiki-copies 4
ZIM_COPIES=4 ./wiki_cluster/start.sh
```

Change `wiki_cluster/start.sh` line 11 so a plain start matches the one-ZIM preparation default:

```bash
ZIM_COPIES="${ZIM_COPIES:-1}"
```

Document `python3 scripts/prepare_training.py prepare --wiki-copies 4` plus `ZIM_COPIES=4 ./wiki_cluster/start.sh` for operators who explicitly want the four-path layout.

- [ ] **Step 5: Run CLI tests and smoke checks**

Run:

```bash
python3 -m unittest tests.test_prepare_training_cli -v
python3 scripts/prepare_training.py --help
python3 scripts/prepare_training.py prepare --help
python3 scripts/prepare_training.py prepare-sft --help
bash -n wiki_cluster/start.sh
```

Expected: tests pass and all commands exit 0 with usage text.

- [ ] **Step 6: Commit the CLI and documentation**

```bash
git add scripts/prepare_training.py tests/test_prepare_training_cli.py docs/command-line-training.md wiki_cluster/README.md wiki_cluster/start.sh
git diff --cached --check
git commit -m "feat: expose terminal training preparation CLI"
```

### Task 5: Bundle offline Kiwix and missing RL datasets, then verify the branch

**Files:**
- Modify: `.gitignore`
- Create: `wiki_cluster/tools/kiwix-tools_linux-x86_64-3.3.0.tar.gz`
- Create: `RL/dataset/hotpot/test_50.jsonl`
- Create: `RL/dataset/hotpot/test_50.parquet`
- Create: `RL/dataset/nq/test_50.jsonl`
- Create: `RL/dataset/nq/test_50.parquet`
- Create: `RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/data.parquet`
- Create: `RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/difficulty_buckets.csv`
- Create: `RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/manifest.json`
- Create: `RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/data.parquet`
- Create: `RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/difficulty_buckets.csv`
- Create: `RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/manifest.json`
- Create: `RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/data.parquet`
- Create: `RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/difficulty_buckets.csv`
- Create: `RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/manifest.json`
- Create: `RL/dataset/test_100/data.jsonl`
- Create: `RL/dataset/test_100/data.parquet`

**Interfaces:**
- Consumes: fixed archive path/checksum and RL dataset checks used by `prepare_resources`.
- Produces: a clone-complete offline Kiwix archive and all RL train/validation data currently present in the old repository.

- [ ] **Step 1: Add a failing repository asset test**

Append a test that operates on the real repository paths without loading parquet contents:

```python
class RepositoryAssetTest(unittest.TestCase):
    def test_bundled_kiwix_and_required_rl_datasets_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = ProjectPaths.from_root(root)
        self.assertTrue(paths.kiwix_archive.is_file())
        self.assertEqual(sha256_file(paths.kiwix_archive), KIWIX_SHA256)
        required = (
            "hotpot/test_50.parquet",
            "nq/test_50.parquet",
            "test_100/data.parquet",
            "t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/data.parquet",
            "t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/data.parquet",
            "t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/data.parquet",
        )
        for relative in required:
            with self.subTest(relative=relative):
                self.assertTrue((root / "RL/dataset" / relative).is_file())
```

- [ ] **Step 2: Run the asset test and confirm missing files**

Run:

```bash
python3 -m unittest tests.test_training_setup.RepositoryAssetTest -v
```

Expected: fail because the archive and 15 RL files have not yet been copied.

- [ ] **Step 3: Narrow `.gitignore` exceptions**

Replace the broad `wiki_cluster/tools/` rule with `wiki_cluster/tools/*`. Append the following exception block at the end of `.gitignore`, after the global `*.tar.gz`, `*.jsonl`, and `dataset/` patterns, so later rules do not re-ignore approved assets:

```gitignore
!wiki_cluster/tools/kiwix-tools_linux-x86_64-3.3.0.tar.gz

!RL/dataset/
!RL/dataset/hotpot/
!RL/dataset/hotpot/test_50.jsonl
!RL/dataset/hotpot/test_50.parquet
!RL/dataset/nq/
!RL/dataset/nq/test_50.jsonl
!RL/dataset/nq/test_50.parquet
!RL/dataset/test_100/
!RL/dataset/test_100/data.jsonl
!RL/dataset/test_100/data.parquet
!RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/
!RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/*
!RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/
!RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/*
!RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/
!RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/*
```

Keep the global model, benchmark, ZIM, SFT output, RL output, logs, checkpoints, and extracted Kiwix paths ignored.

- [ ] **Step 4: Copy only the approved source assets**

Use explicit source and target roots, creating parent directories first. Copy the fixed Kiwix archive and the exact 15 paths listed in this task from:

```text
/home/nvidia/yutao/lzt/BrowserAgent_v2/wiki_cluster/tools/
/home/nvidia/yutao/lzt/BrowserAgent_v2/RL/dataset/
```

Do not synchronize whole directories and do not use overwrite flags. Before each copy, assert the target does not exist. After copying, run:

```bash
kiwix_source=/home/nvidia/yutao/lzt/BrowserAgent_v2/wiki_cluster/tools/kiwix-tools_linux-x86_64-3.3.0.tar.gz
kiwix_target=wiki_cluster/tools/kiwix-tools_linux-x86_64-3.3.0.tar.gz
test ! -e "$kiwix_target"
mkdir -p wiki_cluster/tools
cp "$kiwix_source" "$kiwix_target"

dataset_source=/home/nvidia/yutao/lzt/BrowserAgent_v2/RL/dataset
for dataset_relative in \
  hotpot/test_50.jsonl \
  hotpot/test_50.parquet \
  nq/test_50.jsonl \
  nq/test_50.parquet \
  t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/data.parquet \
  t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/difficulty_buckets.csv \
  t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/manifest.json \
  t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/data.parquet \
  t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/difficulty_buckets.csv \
  t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/manifest.json \
  t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/data.parquet \
  t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/difficulty_buckets.csv \
  t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/manifest.json \
  test_100/data.jsonl \
  test_100/data.parquet
do
  dataset_target="RL/dataset/$dataset_relative"
  test ! -e "$dataset_target"
  mkdir -p "$(dirname "$dataset_target")"
  cp "$dataset_source/$dataset_relative" "$dataset_target"
done
```

After copying, run:

```bash
sha256sum wiki_cluster/tools/kiwix-tools_linux-x86_64-3.3.0.tar.gz
comm -23 \
  <(cd /home/nvidia/yutao/lzt/BrowserAgent_v2/RL/dataset && find . -type f -printf '%P\n' | sort) \
  <(cd RL/dataset && find . -type f -printf '%P\n' | sort)
```

Expected checksum: `cdea8226b479515c9495868dec196de9286cba57bc024df7cd15a83690dfbafc`. Expected `comm` output: empty.

- [ ] **Step 5: Run the asset test and full verification**

Run:

```bash
python3 -m unittest tests.test_training_setup.RepositoryAssetTest -v
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/training_setup.py scripts/prepare_training.py
python3 scripts/prepare_training.py --help
python3 scripts/prepare_training.py prepare --dry-run
git diff --check
git status --short
```

Expected: all tests pass; compilation and help exit 0; dry-run prints three HF download commands and performs no filesystem mutations; `git diff --check` is silent. `git status` must still show the pre-existing modified parquet files and `verl-tool`, but they must remain unstaged.

- [ ] **Step 6: Stage an explicit allowlist and inspect it before commit**

Stage only `.gitignore`, the Kiwix archive, the 15 named dataset files, and the asset-test modification. Then run:

```bash
git add .gitignore tests/test_training_setup.py
git add -f wiki_cluster/tools/kiwix-tools_linux-x86_64-3.3.0.tar.gz
git add -f \
  RL/dataset/hotpot/test_50.jsonl \
  RL/dataset/hotpot/test_50.parquet \
  RL/dataset/nq/test_50.jsonl \
  RL/dataset/nq/test_50.parquet \
  RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/data.parquet \
  RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/difficulty_buckets.csv \
  RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-1000/manifest.json \
  RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/data.parquet \
  RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/difficulty_buckets.csv \
  RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-2000/manifest.json \
  RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/data.parquet \
  RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/difficulty_buckets.csv \
  RL/dataset/t0-e0.25-mh0.25-mm0.25-ml0.15-h0.10-3000/manifest.json \
  RL/dataset/test_100/data.jsonl \
  RL/dataset/test_100/data.parquet
git diff --cached --name-status
git diff --cached --check
git status --short
```

Expected: no pre-existing modified `RL/dataset/BrowserAgent-SeedData/*.parquet` path and no `verl-tool` entry in the staged diff.

- [ ] **Step 7: Commit assets**

```bash
git commit -m "chore: bundle offline training resources"
```

- [ ] **Step 8: Fresh final verification and requirement audit**

Run:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/training_setup.py scripts/prepare_training.py
python3 scripts/prepare_training.py --help
python3 scripts/prepare_training.py prepare --dry-run
git diff slim-server...HEAD --check
git diff --stat slim-server...HEAD
git status --short --branch
```

Audit the design requirements one-by-one: two CLI stages, no environment installation, fixed download destinations, bundled Kiwix checksum, single physical ZIM default, SFT ambiguity error, newest timestamp plus maximum checkpoint, foreground Swift logs, merged model handoff value, unchanged RL launch scripts, bundled RL data, no staged user changes, and operator documentation.
