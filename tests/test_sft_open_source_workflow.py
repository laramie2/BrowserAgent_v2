from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SftDatasetDownloadTest(unittest.TestCase):
    def test_download_requires_repository_id(self) -> None:
        result = subprocess.run(
            [sys.executable, "sft/dataset/download_sft_dataset.py", "--dry-run"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--repo-id", result.stderr)

    def test_dry_run_resolves_destination_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "teacher-data"
            result = subprocess.run(
                [
                    sys.executable,
                    "sft/dataset/download_sft_dataset.py",
                    "--repo-id",
                    "example/teacher-data",
                    "--output-dir",
                    str(destination),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(destination.resolve()), result.stdout)
            self.assertFalse(destination.exists())


class SftModelHandoffTest(unittest.TestCase):
    def test_merge_and_rl_share_the_same_default_model_path(self) -> None:
        merge_script = (ROOT / "sft/merge_lora.sh").read_text(encoding="utf-8")
        rl_script = (ROOT / "RL/scripts/train.sh").read_text(encoding="utf-8")
        expected = "RL_DIR}/models/browseragent-sft"
        self.assertIn("PROJECT_ROOT}/RL/models/browseragent-sft", merge_script)
        self.assertIn(expected, rl_script)
        self.assertIn("SFT_MODEL_PATH_OVERRIDE", merge_script)
        self.assertIn("SFT_MODEL_PATH_OVERRIDE", rl_script)


if __name__ == "__main__":
    unittest.main()
