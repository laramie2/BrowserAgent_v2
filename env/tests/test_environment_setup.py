from __future__ import annotations

import subprocess
import sys
import unittest
import importlib.util
import os
from pathlib import Path
import tempfile


ENV_DIR = Path(__file__).resolve().parents[1]


def load_verifier():
    spec = importlib.util.spec_from_file_location("environment_verifier", ENV_DIR / "verify_env.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load verify_env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def requirement_lines(filename: str) -> set[str]:
    lines = (ENV_DIR / filename).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")}


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def run_installer_with_preflight_stubs(
    filename: str, *, env_name: str, recreate: str, create_exit: str = "42"
) -> tuple[subprocess.CompletedProcess[str], str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        bin_dir = temp / "bin"
        bin_dir.mkdir()
        conda_log = temp / "conda.log"
        conda_stub = bin_dir / "conda"
        write_executable(
            conda_stub,
            """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_CONDA_LOG"
if [[ "${1:-}" == "shell.bash" && "${2:-}" == "hook" ]]; then
  printf ':\n'
elif [[ "${1:-}" == "env" && "${2:-}" == "list" ]]; then
  printf '%s\n' "$FAKE_ENV_LIST"
elif [[ "${1:-}" == "env" && "${2:-}" == "remove" ]]; then
  exit 0
elif [[ "${1:-}" == "create" ]]; then
  exit "$FAKE_CREATE_EXIT"
fi
""",
        )
        write_executable(bin_dir / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{bin_dir}:{environment['PATH']}",
                "ENV_NAME": env_name,
                "RECREATE": recreate,
                "FAKE_CONDA_LOG": str(conda_log),
                "FAKE_ENV_LIST": f"{env_name} * /fake/{env_name}",
                "FAKE_CREATE_EXIT": create_exit,
            }
        )
        completed = subprocess.run(
            ["bash", str(ENV_DIR / filename)],
            capture_output=True,
            text=True,
            env=environment,
        )
        return completed, conda_log.read_text(encoding="utf-8")


def run_browser_installer_with_failing_verifier() -> tuple[subprocess.CompletedProcess[str], str, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        bin_dir = temp / "bin"
        prefix_bin = temp / "prefix" / "bin"
        cuda_include = temp / "prefix" / "targets" / "x86_64-linux" / "include"
        nvtx_include = temp / "prefix" / "site-packages" / "nvidia" / "nvtx" / "include"
        bin_dir.mkdir()
        prefix_bin.mkdir(parents=True)
        cuda_include.mkdir(parents=True)
        (nvtx_include / "nvtx3").mkdir(parents=True)
        (cuda_include / "cuda_runtime_api.h").write_text("", encoding="utf-8")
        (nvtx_include / "nvtx3" / "nvToolsExt.h").write_text("", encoding="utf-8")
        conda_log = temp / "conda.log"
        python_log = temp / "python.log"
        conda_stub = bin_dir / "conda"
        write_executable(
            conda_stub,
            """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_CONDA_LOG"
if [[ "${1:-}" == "shell.bash" && "${2:-}" == "hook" ]]; then
  printf '%s\n' \
    'conda() {' \
    '  if [[ "$1" == "activate" ]]; then : "${NVCC_PREPEND_FLAGS}"; export CONDA_PREFIX="$FAKE_CONDA_PREFIX"; return 0; fi' \
    '  "$FAKE_CONDA_EXECUTABLE" "$@"' \
    '}'
elif [[ "${1:-}" == "env" && "${2:-}" == "list" ]]; then
  exit 0
else
  exit 0
fi
""",
        )
        write_executable(bin_dir / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")
        write_executable(prefix_bin / "nvcc", "#!/usr/bin/env bash\necho 'release 12.8'\n")
        write_executable(
            prefix_bin / "python",
            """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_PYTHON_LOG"
if [[ "${1:-}" == "-" ]]; then printf '%s\n' "$FAKE_NVTX_INCLUDE"; exit 0; fi
if [[ "${1:-}" == *'/verify_env.py' ]]; then exit 23; fi
exit 0
""",
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{bin_dir}:{environment['PATH']}",
                "ENV_NAME": "browseragent-test",
                "RECREATE": "0",
                "FAKE_CONDA_LOG": str(conda_log),
                "FAKE_PYTHON_LOG": str(python_log),
                "FAKE_CONDA_PREFIX": str(temp / "prefix"),
                "FAKE_CONDA_EXECUTABLE": str(conda_stub),
                "FAKE_NVTX_INCLUDE": str(nvtx_include),
            }
        )
        completed = subprocess.run(
            ["bash", str(ENV_DIR / "install_browseragent_v2.sh")],
            capture_output=True,
            text=True,
            env=environment,
        )
        return (
            completed,
            conda_log.read_text(encoding="utf-8"),
            python_log.read_text(encoding="utf-8") if python_log.exists() else "",
        )


class EnvironmentSetupTests(unittest.TestCase):
    def test_browseragent_core_versions_are_pinned(self) -> None:
        requirements = requirement_lines("requirements_browseragent_v2.txt")
        self.assertIn("numpy==1.26.4", requirements)
        self.assertIn("vllm==0.11.0", requirements)
        self.assertIn("flash-attn==2.8.1", requirements)
        self.assertIn("transformer-engine[pytorch]==2.6.0.post1", requirements)

    def test_browseragent_tokenizer_stack_is_pinned_and_directly_verified(self) -> None:
        requirements = requirement_lines("requirements_browseragent_v2.txt")
        self.assertIn("transformers==4.57.6", requirements)
        self.assertIn("tokenizers==0.22.2", requirements)

        verifier = load_verifier()
        spec = verifier.SPECS["browseragent-v2"]
        self.assertEqual(spec.packages["transformers"], "4.57.6")
        self.assertEqual(spec.packages["tokenizers"], "0.22.2")
        self.assertIn("transformers.tokenization_utils_base", spec.imports)
        self.assertIn("tokenizers", spec.imports)

    def test_browseragent_protobuf_stack_is_compatible_and_verified(self) -> None:
        requirements = requirement_lines("requirements_browseragent_v2.txt")
        self.assertIn("tensorboard==2.20.0", requirements)
        self.assertIn("protobuf==5.29.6", requirements)
        self.assertIn("comm==0.2.3", requirements)

        verifier = load_verifier()
        spec = verifier.SPECS["browseragent-v2"]
        self.assertEqual(spec.packages["tensorboard"], "2.20.0")
        self.assertEqual(spec.packages["protobuf"], "5.29.6")
        self.assertEqual(spec.packages["comm"], "0.2.3")
        self.assertIn("tensorboard.compat.proto.event_pb2", spec.imports)
        self.assertIn("comm", spec.imports)

    def test_browseragent_uses_only_the_pytorch_wheel_cudnn_runtime(self) -> None:
        script = (ENV_DIR / "install_browseragent_v2.sh").read_text(encoding="utf-8")
        conda_create = script.split("conda create", 1)[1].split("# Some CUDA", 1)[0]

        self.assertNotIn("cudnn=", conda_create)
        self.assertIn('PYTORCH_CUDNN_ROOT=', script)
        self.assertIn('PYTORCH_CUDNN_INCLUDE_DIR="$PYTORCH_CUDNN_ROOT/include"', script)
        self.assertIn('PYTORCH_CUDNN_LIBRARY_DIR="$PYTORCH_CUDNN_ROOT/lib"', script)
        self.assertIn(
            'RUNTIME_LIBRARY_PATH="$PYTORCH_CUDNN_LIBRARY_DIR:$CUDA_HOME/lib:', script
        )
        self.assertIn('"CUDNN_HOME=$PYTORCH_CUDNN_ROOT"', script)
        self.assertIn('$PYTORCH_CUDNN_INCLUDE_DIR:$CUDA_INCLUDE_DIR', script)

        verifier = load_verifier()
        spec = verifier.SPECS["browseragent-v2"]
        self.assertIn("transformer_engine", getattr(spec, "isolated_imports", ()))

    def test_swift_training_stack_is_pinned(self) -> None:
        requirements = requirement_lines("requirements_swift_sft.txt")
        expected = {
            "ms-swift==3.12.6",
            "deepspeed==0.19.2",
            "transformers==4.54.1",
            "accelerate==1.9.0",
            "datasets==3.2.0",
            "peft==0.17.1",
            "trl==0.19.1",
            "qwen-vl-utils==0.0.8",
            "ipywidgets==8.1.8",
            "comm==0.2.3",
            "ipython==8.37.0",
            "traitlets==5.14.3",
            "sentence-transformers==5.6.0",
            "scikit-learn==1.7.2",
        }
        self.assertLessEqual(expected, requirements)
        self.assertNotIn("decord==0.6.0", requirements)

    def test_browseragent_installer_uses_approved_cuda_and_torch(self) -> None:
        script = (ENV_DIR / "install_browseragent_v2.sh").read_text(encoding="utf-8")
        self.assertIn('CUDA_VERSION="${CUDA_VERSION:-12.8}"', script)
        self.assertIn("torch==2.8.0", script)
        self.assertIn("https://download.pytorch.org/whl/cu128", script)
        self.assertIn('RECREATE="${RECREATE:-0}"', script)
        self.assertIn('verify_env.py" browseragent-v2', script)
        self.assertIn("browseragent_v2_repo.pth", script)
        self.assertIn('[[ "$PYTHON_VERSION" == "3.10" ]]', script)
        self.assertIn('[[ "$CUDA_VERSION" == "12.8" ]]', script)

    def test_browseragent_installer_supports_rootless_playwright_dependencies(self) -> None:
        script = (ENV_DIR / "install_browseragent_v2.sh").read_text(encoding="utf-8")
        conda_create = script.split("conda create", 1)[1].split("# Some CUDA", 1)[0]

        self.assertIn(
            'INSTALL_PLAYWRIGHT_DEPS="${INSTALL_PLAYWRIGHT_DEPS:-0}"', script
        )
        for package in (
            "nspr",
            "nss",
            "at-spi2-atk",
            "alsa-lib",
            "libgbm=1.0.7",
            "libudev1",
            "libxkbcommon",
            "xorg-libx11",
            "xorg-libxcomposite",
            "xorg-libxdamage",
            "xorg-libxfixes",
            "xorg-libxrandr",
        ):
            self.assertIn(package, conda_create)
        self.assertIn('conda env config vars set --name "$ENV_NAME"', script)
        self.assertIn('"LD_LIBRARY_PATH=$RUNTIME_LIBRARY_PATH"', script)
        self.assertIn("python -m playwright install --with-deps chromium", script)
        self.assertIn("python -m playwright install chromium", script)

    def test_browseragent_installer_exposes_pip_nvtx_headers_to_compiler(self) -> None:
        script = (ENV_DIR / "install_browseragent_v2.sh").read_text(encoding="utf-8")
        torch_install = script.index("torch==2.8.0")
        nvtx_preflight = script.index("nvtx3/nvToolsExt.h")
        cuda_preflight = script.index("cuda_runtime_api.h")
        requirements_install = script.index('--requirement "$REQUIREMENTS_FILE"')

        self.assertLess(torch_install, nvtx_preflight)
        self.assertLess(torch_install, cuda_preflight)
        self.assertLess(nvtx_preflight, requirements_install)
        self.assertLess(cuda_preflight, requirements_install)
        self.assertIn(
            'export CPATH="$NVTX_INCLUDE_DIR:$PYTORCH_CUDNN_INCLUDE_DIR:$CUDA_INCLUDE_DIR${CPATH:+:$CPATH}"',
            script,
        )
        self.assertIn(
            'export CPLUS_INCLUDE_PATH="$NVTX_INCLUDE_DIR:$PYTORCH_CUDNN_INCLUDE_DIR:$CUDA_INCLUDE_DIR${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"',
            script,
        )

    def test_swift_installer_uses_approved_cuda_and_torch(self) -> None:
        script = (ENV_DIR / "install_swift_sft.sh").read_text(encoding="utf-8")
        self.assertIn('CUDA_VERSION="${CUDA_VERSION:-12.6}"', script)
        self.assertIn("torch==2.7.1", script)
        self.assertIn("https://download.pytorch.org/whl/cu126", script)
        self.assertIn('RECREATE="${RECREATE:-0}"', script)
        self.assertIn('python "$VERIFIER_FILE" swift-sft', script)
        self.assertNotIn("ms-swift[all]", script)
        self.assertNotIn("--channel defaults", script)
        self.assertNotIn("nvidia/label", script)
        self.assertNotIn("cuda-toolkit", script)
        self.assertNotIn("nsight-compute", script)
        for package in (
            '"cuda-version=${CUDA_VERSION}"',
            "cuda-nvcc",
            "cuda-cudart-dev",
            "cuda-libraries-dev",
            "cuda-driver-dev",
            "numpy=1.26.4",
            "decord=0.6.0",
        ):
            self.assertIn(package, script)
        self.assertIn('VERIFIER_REVISION="2026-07-22-swift-sft-v2"', script)
        verifier = (ENV_DIR / "verify_env.py").read_text(encoding="utf-8")
        self.assertIn('CONFIG_REVISION = "2026-07-22-swift-sft-v2"', verifier)
        self.assertIn('grep -Fqx "CONFIG_REVISION =', script)
        self.assertIn('[[ "$PYTHON_VERSION" == "3.10" ]]', script)
        self.assertIn('[[ "$CUDA_VERSION" == "12.6" ]]', script)

    def test_shell_installers_have_valid_syntax(self) -> None:
        scripts = sorted(str(path) for path in ENV_DIR.glob("install_*.sh"))
        completed = subprocess.run(["bash", "-n", *scripts], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_existing_environment_is_refused_without_mutation(self) -> None:
        for filename, env_name in (
            ("install_browseragent_v2.sh", "browseragent-test"),
            ("install_swift_sft.sh", "swift-test"),
        ):
            with self.subTest(filename=filename):
                completed, conda_log = run_installer_with_preflight_stubs(
                    filename, env_name=env_name, recreate="0"
                )
                self.assertEqual(completed.returncode, 1)
                self.assertIn("already exists", completed.stderr)
                self.assertNotIn("env remove", conda_log)
                self.assertNotIn("create --name", conda_log)

    def test_recreate_removes_only_the_selected_environment(self) -> None:
        completed, conda_log = run_installer_with_preflight_stubs(
            "install_browseragent_v2.sh",
            env_name="selected-env",
            recreate="1",
        )
        self.assertEqual(completed.returncode, 42)
        self.assertIn("env remove --name selected-env --yes", conda_log)
        self.assertEqual(conda_log.count("env remove"), 1)
        self.assertIn("create --name selected-env", conda_log)

    def test_verification_failure_prevents_snapshots_and_success(self) -> None:
        completed, conda_log, python_log = run_browser_installer_with_failing_verifier()
        self.assertEqual(completed.returncode, 23)
        self.assertIn("verify_env.py browseragent-v2", python_log)
        self.assertNotIn("list --explicit", conda_log)
        self.assertNotIn("pip freeze", python_log)
        self.assertNotIn("is ready", completed.stdout)

    def test_verifier_exposes_both_environment_types(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ENV_DIR / "verify_env.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("browseragent-v2", completed.stdout)
        self.assertIn("swift-sft", completed.stdout)

    def test_verifier_checks_repo_package_conda_metadata_and_chromium(self) -> None:
        source = (ENV_DIR / "verify_env.py").read_text(encoding="utf-8")
        self.assertIn('"mini_webarena"', source)
        self.assertIn('"verl_tool.workers.reward_manager.BrowserAgent"', source)
        self.assertIn('conda_package="cuda-version"', source)
        self.assertEqual(source.count('conda_package="cuda-version"'), 2)
        self.assertIn("nvtx3/nvToolsExt.h", source)
        self.assertIn("sync_playwright", source)
        self.assertIn("repair the Conda runtime libraries", source)
        self.assertIn("libnspr4.so", source)
        self.assertIn("libnss3.so", source)
        self.assertIn("libgbm.so.1", source)
        self.assertIn("LD_LIBRARY_PATH does not include", source)

    def test_path_ownership_helper_rejects_system_nvcc(self) -> None:
        verifier = load_verifier()
        self.assertTrue(
            verifier.path_is_within(Path("/opt/conda/envs/test/bin/nvcc"), Path("/opt/conda/envs/test"))
        )
        self.assertFalse(
            verifier.path_is_within(Path("/usr/local/cuda/bin/nvcc"), Path("/opt/conda/envs/test"))
        )

    def test_installers_capture_environment_local_snapshots(self) -> None:
        for filename in ("install_browseragent_v2.sh", "install_swift_sft.sh"):
            script = (ENV_DIR / filename).read_text(encoding="utf-8")
            self.assertIn("conda-explicit.txt", script)
            self.assertIn("pip-freeze.txt", script)

    def test_readme_documents_reproducible_install_and_training(self) -> None:
        readme = (ENV_DIR / "README.md").read_text(encoding="utf-8")
        for required_text in (
            "PyTorch 2.8.0",
            "CUDA 12.8",
            "PyTorch 2.7.1",
            "CUDA 12.6",
            "RECREATE=1",
            "verify_env.py",
            "bash sft/01_run_sft_1.sh",
            "decord",
            "CUDA_VISIBLE_DEVICES",
            "SFT_ROOT",
            "默认无需 root",
            "libgbm=1.0.7",
            "nsight-compute",
        ):
            self.assertIn(required_text, readme)


if __name__ == "__main__":
    unittest.main()
