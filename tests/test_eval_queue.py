from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_eval_queue import command_for, load_jobs


class EvalQueueConfigTest(unittest.TestCase):
    def write_config(self, path: Path, value: dict) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_queue_env_variables_expand_model_output_and_extra_args(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "models"
            model_path = model_root / "merged"
            model_path.mkdir(parents=True)
            matrix_path = root / "matrix.json"
            queue_path = root / "queue.json"
            results_root = root / "results"
            self.write_config(matrix_path, {
                "defaults": {
                    "benchmarks": ["nq"],
                    "max_samples": 1000,
                    "sample_seed": 42,
                    "num_trials": 1,
                    "num_workers": 8,
                    "compression_factor": 1.2,
                },
                "groups": {
                    "main": {
                        "experiments": [{"id": "rl", "label": "RL"}],
                    },
                },
            })
            self.write_config(queue_path, {
                "name": "test",
                "defaults": {
                    "env": {
                        "MODEL_ROOT": str(model_root),
                        "MODEL_PATH": "${MODEL_ROOT}/merged",
                    },
                    "extra_args": ["--token-stats-model-path", "${MODEL_ROOT}/tokenizer"],
                },
                "jobs": [{
                    "job_id": "rl",
                    "group": "main",
                    "experiment": "rl",
                    "model_path": "${MODEL_PATH}",
                    "output_dir": "${MODEL_ROOT}/results",
                }],
            })

            with patch.dict(os.environ, {}, clear=True):
                _, jobs = load_jobs(queue_path, matrix_path, results_root, False)

            self.assertEqual(jobs[0]["model_path"], str(model_path))
            self.assertEqual(jobs[0]["output_dir"], model_root / "results")
            self.assertEqual(jobs[0]["env"]["MODEL_PATH"], str(model_path))
            self.assertEqual(
                jobs[0]["extra_args"],
                ["--token-stats-model-path", str(model_root / "tokenizer")],
            )

    def test_unset_queue_environment_variable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model"
            model_path.mkdir()
            matrix_path = root / "matrix.json"
            queue_path = root / "queue.json"
            self.write_config(matrix_path, {
                "defaults": {
                    "benchmarks": ["nq"],
                    "max_samples": 1,
                    "sample_seed": 42,
                    "num_trials": 1,
                    "num_workers": 1,
                    "compression_factor": 1.2,
                },
                "groups": {"main": {"experiments": [{"id": "rl"}]}},
            })
            self.write_config(queue_path, {
                "defaults": {"env": {"BROKEN": "${QUEUE_TEST_MISSING_VALUE}"}},
                "jobs": [{
                    "group": "main",
                    "experiment": "rl",
                    "model_path": str(model_path),
                }],
            })
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "unset variable"):
                    load_jobs(queue_path, matrix_path, root / "results", False)

    def test_retry_settings_are_forwarded_to_total_runner(self) -> None:
        job = {
            "model_path": "/models/rl",
            "output_dir": Path("/results/rl"),
            "extra_args": [],
            "settings": {
                "benchmarks": ["nq"],
                "max_samples": 1000,
                "sample_seed": 42,
                "num_trials": 1,
                "num_workers": 64,
                "compression_factor": 1.2,
                "resume": True,
                "benchmark_max_retries": 7,
                "benchmark_retry_delay": 3,
            },
        }
        command = command_for(job, False, [])
        self.assertIn("--benchmark-max-retries", command)
        self.assertEqual(command[command.index("--benchmark-max-retries") + 1], "7")
        self.assertEqual(command[command.index("--benchmark-retry-delay") + 1], "3")


if __name__ == "__main__":
    unittest.main()
