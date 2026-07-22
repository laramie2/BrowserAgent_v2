#!/usr/bin/env python3
"""Verify the reproducible BrowserAgent v2 Conda environments."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CONFIG_REVISION = "2026-07-22-swift-sft-v2"


@dataclass(frozen=True)
class EnvironmentSpec:
    packages: dict[str, str]
    imports: tuple[str, ...]
    torch_version: str
    torch_cuda: str
    toolkit_cuda: str
    conda_package: str
    conda_version: str
    chromium_smoke: bool = False
    isolated_imports: tuple[str, ...] = ()


SPECS = {
    "browseragent-v2": EnvironmentSpec(
        packages={
            "torch": "2.8.0",
            "torchvision": "0.23.0",
            "torchaudio": "2.8.0",
            "vllm": "0.11.0",
            "transformers": "4.57.6",
            "tokenizers": "0.22.2",
            "tensorboard": "2.20.0",
            "protobuf": "5.29.6",
            "comm": "0.2.3",
            "numpy": "1.26.4",
            "flash-attn": "2.8.1",
            "megatron-core": "0.16.1",
            "transformer-engine": "2.6.0.post1",
        },
        imports=(
            "torch",
            "tokenizers",
            "transformers.tokenization_utils_base",
            "tensorboard.compat.proto.event_pb2",
            "comm",
            "vllm",
            "flash_attn",
            "transformer_engine",
            "megatron.core",
            "ray",
            "playwright",
            "verl",
            "verl_tool",
            "mini_webarena",
            "verl_tool.workers.reward_manager.BrowserAgent",
        ),
        torch_version="2.8.0",
        torch_cuda="12.8",
        toolkit_cuda="12.8",
        conda_package="cuda-version",
        conda_version="12.8",
        chromium_smoke=True,
        isolated_imports=("transformer_engine",),
    ),
    "swift-sft": EnvironmentSpec(
        packages={
            "torch": "2.7.1",
            "torchvision": "0.22.1",
            "torchaudio": "2.7.1",
            "ms-swift": "3.12.6",
            "deepspeed": "0.19.2",
            "transformers": "4.54.1",
            "accelerate": "1.9.0",
            "datasets": "3.2.0",
            "peft": "0.17.1",
            "trl": "0.19.1",
            "qwen-vl-utils": "0.0.8",
            "decord": "0.6.0",
            "numpy": "1.26.4",
        },
        imports=(
            "torch",
            "swift",
            "deepspeed",
            "transformers",
            "accelerate",
            "datasets",
            "peft",
            "trl",
            "qwen_vl_utils",
            "decord",
        ),
        torch_version="2.7.1",
        torch_cuda="12.6",
        toolkit_cuda="12.6",
        conda_package="cuda-version",
        conda_version="12.6",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate versions, imports, CUDA, nvcc, and pip consistency."
    )
    parser.add_argument("environment", choices=tuple(SPECS))
    return parser.parse_args()


def check_package_versions(spec: EnvironmentSpec, errors: list[str]) -> None:
    for package, expected in spec.packages.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing package: {package}=={expected}")
            continue
        if actual.split("+", 1)[0] != expected:
            errors.append(f"{package}: expected {expected}, found {actual}")
        else:
            print(f"[ok] {package} {actual}")


def check_imports(spec: EnvironmentSpec, errors: list[str]) -> None:
    for module in spec.imports:
        try:
            importlib.import_module(module)
        except Exception as exc:  # Import-time ABI errors are part of this check.
            errors.append(f"cannot import {module}: {type(exc).__name__}: {exc}")
        else:
            print(f"[ok] import {module}")


def check_isolated_imports(spec: EnvironmentSpec, errors: list[str]) -> None:
    """Catch shared-library failures that another import could mask by preload order."""
    for module in spec.isolated_imports:
        code = f"import importlib; importlib.import_module({module!r})"
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        if completed.returncode:
            details = (completed.stdout + completed.stderr).strip()
            errors.append(f"cannot import {module} in a clean process:\n{details}")
        else:
            print(f"[ok] isolated import {module}")


def check_torch(spec: EnvironmentSpec, errors: list[str]) -> None:
    try:
        import torch
    except Exception as exc:
        errors.append(f"cannot import torch: {type(exc).__name__}: {exc}")
        return

    actual_version = torch.__version__.split("+", 1)[0]
    if actual_version != spec.torch_version:
        errors.append(f"torch runtime: expected {spec.torch_version}, found {torch.__version__}")

    compiled_cuda = torch.version.cuda or "none"
    if not compiled_cuda.startswith(spec.torch_cuda):
        errors.append(f"torch CUDA: expected {spec.torch_cuda}.x, found {compiled_cuda}")

    try:
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
            return
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        errors.append(f"cannot initialize CUDA device 0: {type(exc).__name__}: {exc}")
        return

    print(f"[ok] torch runtime {torch.__version__}, CUDA {compiled_cuda}")
    print(f"[ok] {device_count} GPU(s), first GPU: {device_name}")


def path_is_within(path: Path, root: Path) -> bool:
    """Return whether path resolves under root, without requiring it to exist."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def check_conda_package(spec: EnvironmentSpec, errors: list[str]) -> Path | None:
    conda_prefix_text = os.environ.get("CONDA_PREFIX")
    if not conda_prefix_text:
        errors.append("CONDA_PREFIX is not set; activate the target Conda environment first")
        return None
    conda_prefix = Path(conda_prefix_text).resolve()

    conda = shutil.which("conda")
    if conda is None:
        errors.append("conda is not in PATH")
        return conda_prefix

    completed = subprocess.run(
        [conda, "list", "--json", spec.conda_package], capture_output=True, text=True
    )
    if completed.returncode:
        details = (completed.stdout + completed.stderr).strip()
        errors.append(f"cannot query Conda package {spec.conda_package}: {details}")
        return conda_prefix
    try:
        records = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"invalid conda list JSON for {spec.conda_package}: {exc}")
        return conda_prefix

    matches = [record for record in records if record.get("name") == spec.conda_package]
    if not matches:
        errors.append(f"missing Conda package: {spec.conda_package}=={spec.conda_version}")
    elif matches[0].get("version") != spec.conda_version:
        errors.append(
            f"Conda {spec.conda_package}: expected {spec.conda_version}, "
            f"found {matches[0].get('version', 'unknown')}"
        )
    else:
        print(f"[ok] Conda {spec.conda_package} {spec.conda_version}")
    return conda_prefix


def check_nvcc(spec: EnvironmentSpec, conda_prefix: Path | None, errors: list[str]) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        errors.append("nvcc is not in PATH; activate the Conda environment first")
        return
    if conda_prefix is None or not path_is_within(Path(nvcc), conda_prefix):
        errors.append(f"nvcc must come from CONDA_PREFIX, found {nvcc}")
        return

    completed = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if completed.returncode != 0 or match is None:
        errors.append(f"could not determine nvcc version from {nvcc}")
        return
    actual = match.group(1)
    if actual != spec.toolkit_cuda:
        errors.append(f"nvcc: expected CUDA {spec.toolkit_cuda}.x, found {actual}")
    else:
        print(f"[ok] nvcc CUDA {actual}: {nvcc}")


def check_nvtx_header(errors: list[str]) -> None:
    relative_header = Path("nvidia/nvtx/include/nvtx3/nvToolsExt.h")
    try:
        distribution = importlib.metadata.distribution("nvidia-nvtx-cu12")
    except importlib.metadata.PackageNotFoundError:
        errors.append("missing package: nvidia-nvtx-cu12 (required to build Transformer Engine)")
        return

    header = Path(distribution.locate_file(relative_header)).resolve()
    if not header.is_file():
        errors.append(f"missing NVTX development header: {header}")
        return
    print(f"[ok] NVTX development header: {header}")


def check_chromium_runtime(conda_prefix: Path | None, errors: list[str]) -> None:
    if conda_prefix is None:
        return

    library_dir = (conda_prefix / "lib").resolve()
    for library in ("libnspr4.so", "libnss3.so", "libgbm.so.1"):
        path = library_dir / library
        if not path.is_file():
            errors.append(f"missing Conda Chromium runtime library: {path}")
        else:
            print(f"[ok] Chromium runtime library: {path}")

    configured_paths = {
        Path(path).resolve()
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if path
    }
    if library_dir not in configured_paths:
        errors.append(
            f"LD_LIBRARY_PATH does not include {library_dir}; reactivate the Conda "
            "environment before starting the browser server or Ray actors"
        )
    else:
        print(f"[ok] LD_LIBRARY_PATH includes {library_dir}")


def check_chromium(errors: list[str]) -> None:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_content("<title>BrowserAgent verification</title>")
                if page.title() != "BrowserAgent verification":
                    errors.append("Chromium smoke test returned an unexpected page title")
                    return
            finally:
                browser.close()
    except Exception as exc:
        errors.append(
            "cannot launch Playwright Chromium. Re-run the BrowserAgent installer or "
            "repair the Conda runtime libraries as documented in env/README.md. "
            f"Original error: {type(exc).__name__}: {exc}"
        )
    else:
        print("[ok] Playwright Chromium headless launch")


def check_pip(errors: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    if completed.returncode:
        details = (completed.stdout + completed.stderr).strip()
        errors.append(f"pip check failed:\n{details}")
    else:
        print("[ok] pip check")


def main() -> int:
    args = parse_args()
    spec = SPECS[args.environment]
    errors: list[str] = []

    if sys.version_info[:2] != (3, 10):
        errors.append(f"Python: expected 3.10, found {sys.version.split()[0]}")

    check_package_versions(spec, errors)
    check_torch(spec, errors)
    conda_prefix = check_conda_package(spec, errors)
    check_nvcc(spec, conda_prefix, errors)
    if args.environment == "browseragent-v2":
        check_nvtx_header(errors)
        check_chromium_runtime(conda_prefix, errors)
    check_isolated_imports(spec, errors)
    check_imports(spec, errors)
    if spec.chromium_smoke:
        check_chromium(errors)
    check_pip(errors)

    if errors:
        print("\nEnvironment verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"\n{args.environment} verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
