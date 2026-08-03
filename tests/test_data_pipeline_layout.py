from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_generator_module():
    path = ROOT / "generate_sft_data.py"
    spec = importlib.util.spec_from_file_location("teacher_generator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OpenSourceLayoutTest(unittest.TestCase):
    def test_single_generator_is_flattened_to_project_root(self) -> None:
        self.assertTrue((ROOT / "generate_sft_data.py").is_file())
        self.assertFalse((ROOT / "gen_data").exists())

    def test_single_vtc_renderer_is_flattened_to_project_root(self) -> None:
        self.assertTrue((ROOT / "vtc_renderer.py").is_file())
        self.assertFalse((ROOT / "VTC_tool").exists())

    def test_top_level_scripts_directory_is_removed(self) -> None:
        self.assertFalse((ROOT / "scripts").exists())
        self.assertTrue((ROOT / "env/prepare_resources.py").is_file())
        self.assertTrue((ROOT / "env/download_prompt_tokenizer.py").is_file())

    def test_browser_runtime_assets_are_present(self) -> None:
        scripts = ROOT / "mini_webarena/scripts"
        self.assertTrue((scripts / "get_data.js").is_file())
        loader = (scripts / "__init__.py").read_text(encoding="utf-8")
        browser = (ROOT / "mini_webarena/browser_env.py").read_text(encoding="utf-8")
        self.assertIn("get_rect_script", loader)
        self.assertIn("get_rect_script", browser)


class TeacherGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_generator_module()

    def test_openai_base_url_resolves_chat_completions(self) -> None:
        self.assertEqual(
            self.module.DEFAULT_OUTPUT,
            ROOT / "sft/dataset/raw/generated_teacher.jsonl",
        )
        client = self.module.OpenAIChatClient(
            "https://provider.example/v1",
            "secret",
            "teacher",
            10,
            0,
            0.3,
            128,
            1,
        )
        self.assertEqual(
            client.endpoint, "https://provider.example/v1/chat/completions"
        )

    def test_action_and_answer_parsing(self) -> None:
        response = "<think>x</think>\n```stop [Douglas Adams]```"
        command = self.module.extract_command(response)
        self.assertEqual(command, "stop [Douglas Adams]")
        self.assertEqual(self.module.extract_stop_answer(command), "Douglas Adams")
        self.assertTrue(self.module.answer_matches("Douglas Adams", ["Adams"]))


class PipelineEntryPointTest(unittest.TestCase):
    def test_tool_server_has_no_personal_absolute_path(self) -> None:
        script = (ROOT / "start_tool_server.sh").read_text(encoding="utf-8")
        self.assertIn('PROJECT_ROOT="$(cd "$(dirname', script)
        self.assertNotIn("/DATA/", script)
        self.assertNotIn("/data/yutao", script)

    def test_public_clis_load(self) -> None:
        for command in (
            [sys.executable, "generate_sft_data.py", "--help"],
            [sys.executable, "RL/prepare_seed_data.py", "--help"],
            [sys.executable, "RL/score_difficulty.py", "--help"],
            [sys.executable, "RL/build_curriculum.py", "--help"],
            [sys.executable, "env/prepare_resources.py", "--help"],
        ):
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
