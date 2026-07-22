from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.training_setup import (
    CommandRunner,
    ProjectPaths,
    SetupError,
    combine_zim_parts,
    ensure_wiki_copies,
    match_sft_directory,
    prepare_resources,
    safe_extract_kiwix,
    select_checkpoint,
    sha256_file,
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
        result = match_sft_directory(
            self.REPO_FILES, "hotpot6500-nq6300-cr1_2-2048"
        )
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


class RecordingRunner(CommandRunner):
    def __init__(self, dry_run: bool = False) -> None:
        super().__init__(dry_run=dry_run)
        self.commands: list[list[str]] = []

    def run(self, args, *, cwd=None) -> None:
        self.commands.append([str(value) for value in args])


class ResourceOperationTest(unittest.TestCase):
    def test_sha256_file_streams_expected_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.tar.gz"
            path.write_bytes(b"kiwix")
            self.assertEqual(
                sha256_file(path), hashlib.sha256(b"kiwix").hexdigest()
            )

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

    def test_safe_extract_returns_executable_kiwix_serve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "kiwix.tar.gz"
            member_name = "kiwix-tools_linux-x86_64-3.3.0/kiwix-serve"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo(member_name)
                info.size = 6
                handle.addfile(info, io.BytesIO(b"binary"))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with patch("scripts.training_setup.KIWIX_SHA256", digest):
                binary = safe_extract_kiwix(archive, root / "tools")
            self.assertEqual(binary.read_bytes(), b"binary")
            self.assertTrue(os.access(binary, os.X_OK))

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
            self.assertEqual(
                os.readlink(paths[1]), "../1/wikipedia_en_all_maxi_2022-05.zim"
            )
            self.assertEqual(paths[2].read_bytes(), b"zim")


class PrepareResourcesTest(unittest.TestCase):
    def test_dry_run_emits_three_hf_downloads_without_creating_directories(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            paths = ProjectPaths.from_root(root)
            runner = RecordingRunner(dry_run=True)
            prepare_resources(paths, wiki_copies=1, runner=runner, dry_run=True)
            self.assertEqual(
                [command[:3] for command in runner.commands],
                [
                    ["hf", "download", "Qwen/Qwen2.5-VL-7B-Instruct"],
                    ["hf", "download", "TIGER-Lab/BrowserAgent-SeedData"],
                    ["hf", "download", "cogito233/WikiEnv"],
                ],
            )
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
