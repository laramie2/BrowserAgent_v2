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


if __name__ == "__main__":
    unittest.main()
