from __future__ import annotations

import http.server
import os
import socket
import subprocess
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        payload = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


@contextmanager
def health_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def reserve_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def write_fake_runtime(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import http.server
import os
import signal
import sys
from pathlib import Path


def append_pid(env_name):
    target = Path(os.environ[env_name])
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")


def serve(env_name):
    append_pid(env_name)
    port = int(sys.argv[sys.argv.index("--port") + 1])

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            payload = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if sys.argv[1:2] == ["-"]:
    sys.stdin.read()
    raise SystemExit(0)
module = sys.argv[sys.argv.index("-m") + 1]
if module == "vllm.entrypoints.openai.api_server":
    serve("FAKE_VLLM_PIDS")
if module == "verl_tool.servers.serve":
    serve("FAKE_TOOL_PIDS")
if module == "gen_seq.pipeline":
    counter = Path(os.environ["FAKE_COUNTER"])
    count = int(counter.read_text() or "0") if counter.exists() else 0
    count += 1
    counter.write_text(str(count), encoding="utf-8")
    raise SystemExit(2 if count == 1 else 0)
raise SystemExit(f"unexpected module: {module}")
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


class EvalRunnerRetryTest(unittest.TestCase):
    def run_runner(
        self,
        *,
        fail_times: int,
        exit_code: int,
        retries: int,
        no_resume: bool = False,
    ):
        with tempfile.TemporaryDirectory() as directory, health_server() as port:
            temp = Path(directory)
            prompt = temp / "prompt.txt"
            data = temp / "nq.parquet"
            output = temp / "results"
            counter = temp / "counter"
            fake_python = temp / "fake-python"
            prompt.write_text("prompt", encoding="utf-8")
            data.write_bytes(b"data")
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'count=0\n'
                '[[ ! -f "$FAKE_COUNTER" ]] || count="$(<"$FAKE_COUNTER")"\n'
                'count=$((count + 1))\n'
                'printf "%s\\n" "$count" >"$FAKE_COUNTER"\n'
                'if (( count <= FAKE_FAIL_TIMES )); then exit "$FAKE_EXIT_CODE"; fi\n'
                "exit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = {
                **os.environ,
                "PIPELINE_PYTHON": str(fake_python),
                "FAKE_COUNTER": str(counter),
                "FAKE_FAIL_TIMES": str(fail_times),
                "FAKE_EXIT_CODE": str(exit_code),
                "RUN_ID": f"eval_retry_test_{os.getpid()}_{fail_times}_{exit_code}_{retries}",
            }
            env.pop("RAY_TMPDIR", None)
            env.pop("RAY_TMPDIR_OVERRIDE", None)
            command = [
                str(ROOT / "run_eval_all.sh"),
                "--benchmarks", "nq",
                "--skip-vllm",
                "--skip-tool-server",
                "--vllm-health-url", f"http://127.0.0.1:{port}/v1/models",
                "--tool-server-port", str(port),
                "--prompt-path", str(prompt),
                "--nq-data-path", str(data),
                "--output-dir", str(output),
                "--num-workers", "1",
                "--max-samples", "1",
                "--benchmark-max-retries", str(retries),
                "--benchmark-retry-delay", "0",
                "--no-token-stats",
                "--no-use-vlm",
            ]
            if no_resume:
                command.append("--no-resume")
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
            count = int(counter.read_text(encoding="utf-8").strip())
            return result, count

    def test_dry_run_does_not_treat_existing_health_endpoint_as_stale_vllm(self) -> None:
        with health_server() as port:
            result = subprocess.run(
                [
                    str(ROOT / "run_eval_all.sh"),
                    "--benchmarks", "nq",
                    "--dry-run",
                    "--pipeline-python", "/bin/true",
                    "--vllm-health-url", f"http://127.0.0.1:{port}/v1/models",
                    "--no-token-stats",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("DRY-RUN vLLM", result.stdout)
        self.assertIn("DRY-RUN tool-server", result.stdout)

    def test_owned_tool_server_restarts_while_vllm_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            fake_runtime = temp / "fake-runtime"
            write_fake_runtime(fake_runtime)
            model = temp / "model"
            tokenizer = temp / "tokenizer"
            model.mkdir()
            tokenizer.mkdir()
            (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
            prompt = temp / "prompt.txt"
            data = temp / "nq.parquet"
            prompt.write_text("prompt", encoding="utf-8")
            data.write_bytes(b"data")
            vllm_pids = temp / "vllm-pids"
            tool_pids = temp / "tool-pids"
            counter = temp / "counter"
            vllm_port = reserve_port()
            tool_port = reserve_port()
            env = {
                **os.environ,
                "FAKE_VLLM_PIDS": str(vllm_pids),
                "FAKE_TOOL_PIDS": str(tool_pids),
                "FAKE_COUNTER": str(counter),
                "MINI_WEB_ARENA_PROMPT_MODEL": str(tokenizer),
                "RUN_ID": f"owned_retry_test_{os.getpid()}",
            }
            env.pop("RAY_TMPDIR", None)
            env.pop("RAY_TMPDIR_OVERRIDE", None)
            result = subprocess.run(
                [
                    str(ROOT / "run_eval_all.sh"),
                    "--benchmarks", "nq",
                    "--model-path", str(model),
                    "--vllm-python", str(fake_runtime),
                    "--tool-server-python", str(fake_runtime),
                    "--pipeline-python", str(fake_runtime),
                    "--vllm-port", str(vllm_port),
                    "--tool-server-port", str(tool_port),
                    "--prompt-path", str(prompt),
                    "--nq-data-path", str(data),
                    "--output-dir", str(temp / "results"),
                    "--num-workers", "1",
                    "--max-samples", "1",
                    "--benchmark-max-retries", "1",
                    "--benchmark-retry-delay", "0",
                    "--no-token-stats",
                    "--no-use-vlm",
                    "--no-clean-triton-cache",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(counter.read_text(encoding="utf-8"), "2")
            vllm_pid_values = vllm_pids.read_text().splitlines()
            self.assertEqual(len(vllm_pid_values), 1)
            tool_pid_values = tool_pids.read_text().splitlines()
            self.assertEqual(len(tool_pid_values), 2, result.stdout)
            self.assertNotEqual(tool_pid_values[0], tool_pid_values[1])
            for pid in [*vllm_pid_values, *tool_pid_values]:
                self.assertFalse(Path(f"/proc/{pid}").exists(), f"leaked pid {pid}")
            self.assertIn("_restart1.log", result.stdout)
            self.assertIn("preserving vLLM", result.stdout)

    def test_exit_two_resumes_until_success(self) -> None:
        result, count = self.run_runner(fail_times=1, exit_code=2, retries=2)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(count, 2)
        self.assertIn("AUTO_RETRY benchmark=nq", result.stdout)
        self.assertIn("preserving vLLM", result.stdout)
        self.assertIn("benchmarks=1", result.stdout)

    def test_exit_two_stops_after_retry_budget(self) -> None:
        result, count = self.run_runner(fail_times=99, exit_code=2, retries=2)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertEqual(count, 3)
        self.assertIn("exhausted 2 automatic retries", result.stdout)

    def test_no_resume_automatically_disables_retries(self) -> None:
        result, count = self.run_runner(
            fail_times=0,
            exit_code=2,
            retries=5,
            no_resume=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(count, 1)
        self.assertIn("retries disabled because resume is disabled", result.stdout)
        self.assertIn("benchmark_max_retries=0", result.stdout)
        self.assertIn("ray_tmpdir=/tmp/ba-ray-", result.stdout)

    def test_non_retryable_exit_code_fails_immediately(self) -> None:
        result, count = self.run_runner(fail_times=99, exit_code=1, retries=5)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(count, 1)
        self.assertIn("non-retryable exit code 1", result.stdout)
        self.assertNotIn("AUTO_RETRY", result.stdout)


if __name__ == "__main__":
    unittest.main()
