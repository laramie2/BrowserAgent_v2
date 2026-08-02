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
            self.assertTrue(prepare.call_args.kwargs["runner"].dry_run)

    def test_prepare_sft_prints_yaml_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = SftResult(
                top_level="Qwen2.5-VL-7B-Instruct-example",
                checkpoint=Path(directory) / "checkpoint-854",
                merged_output=(
                    Path(directory)
                    / "RL/models/Qwen2.5-VL-7B-Instruct-example-merged"
                ),
                model_name="example",
            )
            output = io.StringIO()
            with patch("scripts.prepare_training.prepare_sft", return_value=result):
                with redirect_stdout(output):
                    status = main(
                        ["prepare-sft", "example"],
                        project_root=Path(directory),
                    )
            self.assertEqual(status, 0)
            self.assertIn("SFT_MODEL_NAME_OVERRIDE: example", output.getvalue())
            self.assertIn("checkpoint-854", output.getvalue())

    def test_setup_error_returns_actionable_nonzero_status(self) -> None:
        error = io.StringIO()
        with patch(
            "scripts.prepare_training.prepare_resources",
            side_effect=SetupError("missing hf"),
        ):
            with redirect_stderr(error):
                status = main(["prepare"], project_root=Path("/tmp/project"))
        self.assertEqual(status, 2)
        self.assertIn("missing hf", error.getvalue())

    def test_keyboard_interrupt_returns_shell_interrupt_status(self) -> None:
        error = io.StringIO()
        with patch(
            "scripts.prepare_training.prepare_resources",
            side_effect=KeyboardInterrupt,
        ):
            with redirect_stderr(error):
                status = main(["prepare"], project_root=Path("/tmp/project"))
        self.assertEqual(status, 130)
        self.assertIn("interrupted", error.getvalue())


class WikiStartDefaultsTest(unittest.TestCase):
    def test_wiki_start_defaults_to_one_zim_copy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "wiki_cluster/start.sh").read_text(encoding="utf-8")
        self.assertIn('ZIM_COPIES="${ZIM_COPIES:-1}"', script)


if __name__ == "__main__":
    unittest.main()
