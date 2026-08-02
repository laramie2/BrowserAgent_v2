from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.training_setup import (
    CommandRunner,
    KIWIX_SHA256,
    ProjectPaths,
    SftResult,
    SetupError,
    combine_zim_parts,
    ensure_wiki_copies,
    is_complete_merged_model,
    match_sft_directory,
    merge_lora,
    prepare_resources,
    prepare_sft,
    safe_extract_kiwix,
    select_checkpoint,
    sha256_file,
    sft_model_name,
    swift_command,
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


class MergeTest(unittest.TestCase):
    def test_swift_command_prefers_current_path(self) -> None:
        with patch(
            "scripts.training_setup.shutil.which",
            side_effect=lambda name: "/env/bin/swift" if name == "swift" else None,
        ):
            self.assertEqual(swift_command(), ["/env/bin/swift"])

    def test_swift_command_falls_back_to_conda_environment(self) -> None:
        with patch(
            "scripts.training_setup.shutil.which",
            side_effect=lambda name: "/conda/bin/conda" if name == "conda" else None,
        ):
            self.assertEqual(
                swift_command(),
                [
                    "/conda/bin/conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "swift-sft",
                    "swift",
                ],
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
            (paths.base_model / "config.json").write_text(
                '{"base": true}', encoding="utf-8"
            )
            (paths.base_model / "tokenizer.json").write_text("{}", encoding="utf-8")
            (paths.base_model / "model.safetensors").write_bytes(b"base")
            checkpoint = root / "checkpoint-854"
            checkpoint.mkdir()
            runner = RecordingRunner()
            output = (
                paths.rl_models / "Qwen2.5-VL-7B-Instruct-example-merged"
            )

            def create_fake_output(args, *, cwd=None):
                runner.commands.append([str(value) for value in args])
                output.mkdir(parents=True)
                (output / "model.safetensors").write_bytes(b"merged")
                (output / "config.json").write_text("{}", encoding="utf-8")
                (output / "processor_config.json").write_text(
                    "{}", encoding="utf-8"
                )
                (output / "chat_template.jinja").write_text(
                    "template", encoding="utf-8"
                )

            runner.run = create_fake_output
            result = merge_lora(
                paths,
                checkpoint,
                "Qwen2.5-VL-7B-Instruct-example",
                runner,
                force=False,
            )
            self.assertEqual(result, output)
            self.assertIn("--adapters", runner.commands[0])
            self.assertEqual(
                (output / "tokenizer.json").read_text(encoding="utf-8"), "{}"
            )
            self.assertFalse((output / "processor_config.json").exists())
            self.assertFalse((output / "chat_template.jinja").exists())

    def test_merge_rejects_incomplete_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ProjectPaths.from_root(Path(directory))
            output = paths.rl_models / "Qwen2.5-VL-7B-Instruct-example-merged"
            output.mkdir(parents=True)
            runner = RecordingRunner()
            with self.assertRaisesRegex(SetupError, "incomplete"):
                merge_lora(
                    paths,
                    Path(directory) / "checkpoint-1",
                    "Qwen2.5-VL-7B-Instruct-example",
                    runner,
                    force=False,
                )
            self.assertEqual(runner.commands, [])

    def test_merge_rejects_symlink_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ProjectPaths.from_root(root / "project")
            outside = root / "outside"
            outside.mkdir()
            (outside / "config.json").write_text("{}", encoding="utf-8")
            (outside / "model.safetensors").write_bytes(b"weights")
            paths.rl_models.mkdir(parents=True)
            output = paths.rl_models / "Qwen2.5-VL-7B-Instruct-example-merged"
            output.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(SetupError, "symbolic link"):
                merge_lora(
                    paths,
                    root / "checkpoint-1",
                    "Qwen2.5-VL-7B-Instruct-example",
                    RecordingRunner(),
                    force=True,
                )

    def test_force_keeps_incomplete_output_when_swift_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ProjectPaths.from_root(root)
            paths.base_model.mkdir(parents=True)
            checkpoint = root / "checkpoint-1"
            checkpoint.mkdir()
            output = paths.rl_models / "Qwen2.5-VL-7B-Instruct-example-merged"
            output.mkdir(parents=True)
            marker = output / "partial.marker"
            marker.write_text("keep", encoding="utf-8")
            with patch("scripts.training_setup.shutil.which", return_value=None):
                with self.assertRaisesRegex(SetupError, "Cannot find swift"):
                    merge_lora(
                        paths,
                        checkpoint,
                        "Qwen2.5-VL-7B-Instruct-example",
                        RecordingRunner(),
                        force=True,
                    )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


class PrepareSftTest(unittest.TestCase):
    def test_prepare_sft_downloads_unique_directory_and_returns_training_value(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ProjectPaths.from_root(root)
            top_level = "Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false"
            repo_files = [
                f"{top_level}/v5-20260705-013607/checkpoint-854/adapter_config.json"
            ]
            runner = RecordingRunner()

            def fake_run(args, *, cwd=None):
                runner.commands.append([str(value) for value in args])
                if list(args[:2]) == ["hf", "download"]:
                    checkpoint = (
                        paths.sft_output
                        / top_level
                        / "v5-20260705-013607/checkpoint-854"
                    )
                    checkpoint.mkdir(parents=True)

            runner.run = fake_run
            merged = paths.rl_models / f"{top_level}-merged"
            with patch("scripts.training_setup.merge_lora", return_value=merged):
                result = prepare_sft(
                    paths,
                    "hotpot6500-nq6300-cr1_2-2048",
                    runner,
                    force=False,
                    repo_files=repo_files,
                )
            self.assertIsInstance(result, SftResult)
            self.assertEqual(result.top_level, top_level)
            self.assertEqual(result.checkpoint.name, "checkpoint-854")
            self.assertEqual(
                result.model_name,
                "task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false",
            )
            self.assertIn(f"{top_level}/*", runner.commands[0])

    def test_prepare_sft_dry_run_uses_remote_checkpoint_without_filesystem_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            paths = ProjectPaths.from_root(root)
            top_level = "Qwen2.5-VL-7B-Instruct-task-opsrc-example-sft-5e-5lr-freeze_false"
            repo_files = [
                f"{top_level}/v1-20260704-235959/checkpoint-999/adapter_config.json",
                f"{top_level}/v2-20260705-013607/checkpoint-99/adapter_config.json",
                f"{top_level}/v2-20260705-013607/checkpoint-854/adapter_config.json",
            ]
            runner = RecordingRunner(dry_run=True)
            result = prepare_sft(
                paths, "example", runner, force=False, repo_files=repo_files
            )
            self.assertEqual(result.checkpoint.name, "checkpoint-854")
            self.assertEqual(runner.commands[0][:2], ["hf", "download"])
            self.assertEqual(runner.commands[1][0:2], ["swift", "export"])
            self.assertFalse(root.exists())


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

    def test_unapproved_rl_dataset_paths_remain_ignored(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                "git",
                "check-ignore",
                "--no-index",
                "--quiet",
                "RL/dataset/unapproved/data.parquet",
            ],
            cwd=root,
            check=False,
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
