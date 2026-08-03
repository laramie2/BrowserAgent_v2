from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from env.prepare_resources import main
from env.resource_setup import SetupError


class PrepareResourcesCliTest(unittest.TestCase):
    def test_prepare_passes_wiki_copies_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("env.prepare_resources.prepare_resources") as prepare:
                status = main(
                    ["prepare", "--wiki-copies", "4", "--dry-run"],
                    project_root=Path(directory),
                )
            self.assertEqual(status, 0)
            self.assertEqual(prepare.call_args.kwargs["wiki_copies"], 4)
            self.assertTrue(prepare.call_args.kwargs["dry_run"])
            self.assertTrue(prepare.call_args.kwargs["runner"].dry_run)

    def test_setup_error_returns_actionable_nonzero_status(self) -> None:
        error = io.StringIO()
        with patch(
            "env.prepare_resources.prepare_resources",
            side_effect=SetupError("missing hf"),
        ):
            with redirect_stderr(error):
                status = main(["prepare"], project_root=Path("/tmp/project"))
        self.assertEqual(status, 2)
        self.assertIn("missing hf", error.getvalue())

    def test_keyboard_interrupt_returns_shell_interrupt_status(self) -> None:
        error = io.StringIO()
        with patch(
            "env.prepare_resources.prepare_resources",
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
