from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from env.resource_setup import (
    CommandRunner,
    ProjectPaths,
    SetupError,
    combine_zim_parts,
    ensure_wiki_copies,
    prepare_resources,
    resource_download_commands,
    safe_extract_kiwix,
    sha256_file,
)


class RecordingRunner(CommandRunner):
    def __init__(self, dry_run: bool = False) -> None:
        super().__init__(dry_run=dry_run)
        self.commands: list[list[str]] = []

    def run(self, args, *, cwd=None) -> None:
        self.commands.append([str(value) for value in args])


class ProjectPathsTest(unittest.TestCase):
    def test_paths_are_project_relative(self) -> None:
        root = Path("/tmp/project with spaces")
        paths = ProjectPaths.from_root(root)
        self.assertEqual(paths.base_model, root / "models/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(
            paths.seed_dataset, root / "RL/dataset/BrowserAgent-SeedData"
        )
        self.assertEqual(paths.wiki_root, root / "webarena/webarena_zim")

    def test_seed_data_download_uses_single_canonical_directory(self) -> None:
        paths = ProjectPaths.from_root(Path("/tmp/project"))
        commands = resource_download_commands(paths)
        seed_command = commands[1]
        self.assertEqual(seed_command[:3], ["hf", "download", "TIGER-Lab/BrowserAgent-SeedData"])
        self.assertEqual(seed_command[-1], str(paths.seed_dataset))


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
            with patch("env.resource_setup.KIWIX_SHA256", digest):
                with self.assertRaisesRegex(SetupError, "unsafe path"):
                    safe_extract_kiwix(archive, Path(directory) / "tools")

    def test_safe_extract_returns_executable_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "kiwix.tar.gz"
            member_name = "kiwix-tools_linux-x86_64-3.3.0/kiwix-serve"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo(member_name)
                info.size = 6
                handle.addfile(info, io.BytesIO(b"binary"))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with patch("env.resource_setup.KIWIX_SHA256", digest):
                binary = safe_extract_kiwix(archive, root / "tools")
            self.assertEqual(binary.read_bytes(), b"binary")
            self.assertTrue(os.access(binary, os.X_OK))

    def test_combine_zim_parts_orders_and_removes_parts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            second = root / "wiki.part-02"
            first = root / "wiki.part-01"
            second.write_bytes(b"B")
            first.write_bytes(b"A")
            output = root / "1/wiki.zim"
            combine_zim_parts([second, first], output)
            self.assertEqual(output.read_bytes(), b"AB")
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())

    def test_wiki_copies_use_relative_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "1/wikipedia_en_all_maxi_2022-05.zim"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"zim")
            paths = ensure_wiki_copies(root, 3)
            self.assertEqual(len(paths), 3)
            self.assertEqual(
                os.readlink(paths[1]), "../1/wikipedia_en_all_maxi_2022-05.zim"
            )


class PrepareResourcesTest(unittest.TestCase):
    def test_dry_run_emits_three_hf_downloads_without_writes(self) -> None:
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
