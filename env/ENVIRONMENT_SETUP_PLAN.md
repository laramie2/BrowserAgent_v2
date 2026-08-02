# BrowserAgent v2 Environment Setup Implementation Plan

> **For agentic workers:** Execute inline in the current workspace because the user explicitly requested outputs under this checkout's `env/` directory. Preserve unrelated dirty-worktree changes and the existing cu124 alternative files.

**Goal:** Make the two requested Conda environments reproducible on a new Linux x86_64 NVIDIA machine.

**Architecture:** Keep CUDA/Torch installation in ordered shell installers, keep maintainable direct Python pins in separate requirements files, and share runtime checks through one Python verifier. Document the observed drift separately from the recommended clean installation.

**Tech Stack:** Bash, Conda, Python 3.10, CUDA 12.8/12.6, PyTorch, vLLM, verl-tool, MS-Swift, DeepSpeed, unittest.

## Global Constraints

- Do not modify or delete `install_swift_sft_cu124_pinned.sh`, `requirements_swift_sft_cu124.txt`, or `swift-sft-cu124-py310.lock.txt`.
- Do not mutate either currently installed Conda environment while developing these files.
- Default installers must refuse to overwrite an existing environment unless `RECREATE=1` is explicit.
- All deliverables must live under `env/`.

### Task 1: Add failing configuration tests

**Files:**
- Create: `env/tests/test_environment_setup.py`

- [x] Assert the two direct-requirements files contain the approved exact core versions.
- [x] Assert the installers contain the approved CUDA/Torch pair, safe recreation behavior, and unified verification call.
- [x] Assert the Swift installer has no unbounded `ms-swift[all] -U` operation.
- [x] Run `python -m unittest discover -s env/tests -p 'test_*.py' -v`; expect failure because the new production files are absent.

### Task 2: Implement pinned dependency manifests and runtime verification

**Files:**
- Create: `env/requirements_browseragent_v2.txt`
- Create: `env/requirements_swift_sft.txt`
- Create: `env/verify_env.py`

- [x] Record the direct package versions proven by the repository metadata and installed environments.
- [x] Implement `verify_env.py browseragent-v2` and `verify_env.py swift-sft` with clear errors for version, CUDA, nvcc, import, and pip consistency failures.
- [x] Run the unit tests and confirm manifest/verifier tests pass.

### Task 3: Replace the two canonical installers

**Files:**
- Modify: `env/install_browseragent_v2.sh`
- Modify: `env/install_swift_sft.sh`
- Modify: `env/install_all.sh`

- [x] Add platform, driver, Conda, existing-environment, and repository preflight checks.
- [x] Install each approved CUDA/Torch stack in ABI-safe order.
- [x] Install the pinned requirements and project-local editable packages.
- [x] Invoke `verify_env.py` after installation.
- [x] Run unit tests and `bash -n` for every shell installer.

### Task 4: Write the Chinese handoff guide

**Files:**
- Modify: `env/README.md`

- [x] Document prerequisites, version matrix, one-command and per-environment installation.
- [x] Explain `RECREATE=1`, CUDA driver requirements, Playwright system libraries, Swift training usage, validation, and troubleshooting.
- [x] Explain why current `pip freeze` output is an audit source rather than the canonical installer.
- [x] Run the complete static verification suite and inspect the final diff.

### Task 5: Make Chromium installation work without root

**Files:**
- Modify: `env/install_browseragent_v2.sh`
- Modify: `env/tests/test_environment_setup.py`
- Modify: `env/verify_env.py`
- Modify: `env/README.md`
- Modify: `env/ENVIRONMENT_SETUP_DESIGN.md`

- [x] Add the conda-forge libraries required by Chromium headless shell, including the split `libgbm=1.0.7` package.
- [x] Persist the environment-local library search path so newly started Ray actors inherit it after later Conda activations.
- [x] Make the rootless Conda path the default while retaining explicit `INSTALL_PLAYWRIGHT_DEPS=1` support for Playwright's official system-package installer.
- [x] Keep the real Chromium launch smoke test mandatory and document repair/restart behavior.
- [x] Run the complete unit, shell syntax, Python compile, Conda solve, and Chromium smoke verification suite.

### Task 6: Remove Swift's unnecessary Nsight dependency

**Files:**
- Modify: `env/install_swift_sft.sh`
- Modify: `env/tests/test_environment_setup.py`
- Modify: `env/verify_env.py`
- Modify: `env/README.md`
- Modify: `env/ENVIRONMENT_SETUP_DESIGN.md`

- [x] Replace NVIDIA's full `cuda-toolkit=12.6.3` metapackage with conda-forge CUDA 12.6 compiler/runtime/development packages.
- [x] Assert the Swift environment contains `cuda-version=12.6`, environment-local nvcc 12.6.x, and no Nsight dependency.
- [x] Document why checksum failures for `nsight-compute` are avoided and how to rebuild a partial environment.
- [x] Run unit, syntax, compile, diff, and full Conda solve verification.

### Task 7: Make Swift's final consistency check clean

**Files:**
- Modify: `env/install_swift_sft.sh`
- Modify: `env/requirements_swift_sft.txt`
- Modify: `env/verify_env.py`
- Modify: `env/tests/test_environment_setup.py`
- Modify: `env/README.md`

- [x] Install Decord from its native conda-forge Python 3.10 build instead of the incorrectly tagged PyPI CPython 3.6 wheel.
- [x] Pin the missing ipywidgets and sentence-transformers dependency chains required by the resolved MS-Swift stack.
- [x] Add a shared installer/verifier configuration revision preflight to reject stale mixed file versions before environment creation.
- [x] Verify dependency resolution, Decord import/platform metadata, pip consistency, tests, and syntax.

### Task 8: Detect incomplete BrowserAgent tokenizer installations

**Files:**
- Modify: `env/requirements_browseragent_v2.txt`
- Modify: `env/verify_env.py`
- Modify: `env/tests/test_environment_setup.py`
- Modify: `env/README.md`
- Modify: `env/ENVIRONMENT_SETUP_DESIGN.md`

- [x] Pin the Transformers/Tokenizers pair already proven in the reference BrowserAgent environment.
- [x] Import the eager tokenizer module during final environment verification.
- [x] Verify the regression test, real imports, complete unit suite, syntax, and dependency consistency.

### Task 9: Resolve BrowserAgent Protobuf and mixed-cuDNN failures

**Files:**
- Modify: `env/install_browseragent_v2.sh`
- Modify: `env/requirements_browseragent_v2.txt`
- Modify: `env/verify_env.py`
- Modify: `env/tests/test_environment_setup.py`
- Modify: `env/README.md`
- Modify: `env/ENVIRONMENT_SETUP_DESIGN.md`

- [x] Pin the TensorBoard 2.20 / Protobuf 5.29 compatibility pair and install comm.
- [x] Use the PyTorch wheel's matched cuDNN headers and runtime instead of mixing Pip and Conda cuDNN.
- [x] Persist the cuDNN-first library path for later activations and Ray workers.
- [x] Verify Transformer Engine in an isolated process so preload order cannot hide ABI failures.
- [x] Run real import, dependency resolution, Conda solve, unit, syntax, compile, and diff verification.
