#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-browseragent-v2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-12.8}"
RECREATE="${RECREATE:-0}"
MAX_JOBS="${MAX_JOBS:-8}"
INSTALL_PLAYWRIGHT_DEPS="${INSTALL_PLAYWRIGHT_DEPS:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_browseragent_v2.txt"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
  die "browseragent-v2 requires Linux x86_64."
[[ "$RECREATE" == "0" || "$RECREATE" == "1" ]] || \
  die "RECREATE must be 0 or 1."
[[ "$INSTALL_PLAYWRIGHT_DEPS" == "0" || "$INSTALL_PLAYWRIGHT_DEPS" == "1" ]] || \
  die "INSTALL_PLAYWRIGHT_DEPS must be 0 or 1."
[[ "$PYTHON_VERSION" == "3.10" ]] || \
  die "This pinned stack supports only PYTHON_VERSION=3.10."
[[ "$CUDA_VERSION" == "12.8" ]] || \
  die "This pinned stack supports only CUDA_VERSION=12.8."
command -v conda >/dev/null 2>&1 || die "conda was not found in PATH."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi was not found in PATH."
nvidia-smi >/dev/null || die "The NVIDIA driver is not working."
[[ -f "$REQUIREMENTS_FILE" ]] || die "Missing $REQUIREMENTS_FILE."
[[ -f "$REPO_ROOT/verl-tool/pyproject.toml" ]] || \
  die "Run this installer from a complete BrowserAgent_v2 checkout with verl-tool present."

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$RECREATE" != "1" ]]; then
    die "Conda environment '$ENV_NAME' already exists. Use RECREATE=1 to replace it."
  fi
  conda env remove --name "$ENV_NAME" --yes
fi

echo "Creating $ENV_NAME (Python $PYTHON_VERSION, CUDA $CUDA_VERSION)..."
conda create \
  --name "$ENV_NAME" \
  --yes \
  --override-channels \
  --channel conda-forge \
  --strict-channel-priority \
  "python=${PYTHON_VERSION}" \
  "cuda-version=${CUDA_VERSION}" \
  cuda-nvcc \
  cuda-cudart-dev \
  cuda-libraries-dev \
  cuda-driver-dev \
  libcublas-dev \
  nccl \
  cuda-nvtx \
  cmake \
  ninja \
  make \
  git \
  pkg-config \
  gcc_linux-64 \
  gxx_linux-64 \
  binutils_linux-64 \
  nspr \
  nss \
  glib \
  atk-1.0 \
  at-spi2-atk \
  dbus \
  alsa-lib \
  libcups \
  libdrm \
  libegl \
  mesalib \
  libgbm=1.0.7 \
  libudev1 \
  libxkbcommon \
  libexpat \
  libxcb \
  fontconfig \
  freetype \
  fonts-conda-ecosystem \
  xorg-libx11 \
  xorg-libxcomposite \
  xorg-libxdamage \
  xorg-libxext \
  xorg-libxfixes \
  xorg-libxrandr \
  xorg-libxshmfence

# Some CUDA packages ship activate.d hooks that read optional variables before
# checking whether they exist. Temporarily disable nounset only while Conda
# sources third-party activation hooks, then restore strict mode immediately.
set +u
conda activate "$ENV_NAME"
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
RUNTIME_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS

command -v nvcc >/dev/null 2>&1 || die "CUDA Toolkit installation did not provide nvcc."
nvcc --version

python -m pip install --upgrade \
  pip==25.3 \
  setuptools==79.0.1 \
  wheel==0.45.1 \
  packaging==25.0 \
  ninja==1.13.0
python -m pip install --no-cache-dir \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# PyTorch's cu128 wheels install a matched cuDNN runtime and headers under
# site-packages/nvidia/cudnn. Do not also install Conda cuDNN: putting a
# different Conda libcudnn_graph.so.9 ahead of this runtime causes undefined
# symbols when Transformer Engine loads the remaining cuDNN components.
PYTORCH_CUDNN_ROOT="$(python - <<'PY'
from pathlib import Path
import sysconfig

root = Path(sysconfig.get_path("purelib")) / "nvidia" / "cudnn"
required = (root / "include" / "cudnn.h", root / "lib" / "libcudnn.so.9")
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"PyTorch cuDNN installation is incomplete; missing: {missing}")
print(root)
PY
)"
PYTORCH_CUDNN_INCLUDE_DIR="$PYTORCH_CUDNN_ROOT/include"
PYTORCH_CUDNN_LIBRARY_DIR="$PYTORCH_CUDNN_ROOT/lib"
RUNTIME_LIBRARY_PATH="$PYTORCH_CUDNN_LIBRARY_DIR:$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
export CUDNN_HOME="$PYTORCH_CUDNN_ROOT"
export LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Persist the same ordering for later shells and Ray actors. The PyTorch cuDNN
# directory must remain first; the following Conda paths provide Chromium and
# CUDA Toolkit libraries that are not bundled with PyTorch.
conda env config vars set --name "$ENV_NAME" \
  "CUDNN_HOME=$PYTORCH_CUDNN_ROOT" \
  "LD_LIBRARY_PATH=$RUNTIME_LIBRARY_PATH"

# The PyTorch CUDA wheels provide the NVTX development header through the
# nvidia-nvtx-cu12 wheel, under site-packages rather than CONDA_PREFIX/include.
# Transformer Engine 2.6 is source-built on Linux and otherwise cannot find it.
NVTX_INCLUDE_DIR="$(python - <<'PY'
from pathlib import Path
import site
import sysconfig

roots = {
    Path(sysconfig.get_path("purelib")),
    Path(sysconfig.get_path("platlib")),
    *(Path(path) for path in site.getsitepackages()),
}
for root in roots:
    candidate = root / "nvidia" / "nvtx" / "include"
    if (candidate / "nvtx3" / "nvToolsExt.h").is_file():
        print(candidate)
        break
else:
    raise SystemExit(
        "nvidia-nvtx-cu12 is installed without nvtx3/nvToolsExt.h; "
        "the PyTorch CUDA wheel installation is incomplete"
    )
PY
)"
CUDA_INCLUDE_DIR="$CUDA_HOME/targets/x86_64-linux/include"
[[ -f "$CUDA_INCLUDE_DIR/cuda_runtime_api.h" ]] || \
  die "CUDA Toolkit is missing $CUDA_INCLUDE_DIR/cuda_runtime_api.h."
export CPATH="$NVTX_INCLUDE_DIR:$PYTORCH_CUDNN_INCLUDE_DIR:$CUDA_INCLUDE_DIR${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$NVTX_INCLUDE_DIR:$PYTORCH_CUDNN_INCLUDE_DIR:$CUDA_INCLUDE_DIR${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
echo "Using NVTX headers from: $NVTX_INCLUDE_DIR"
echo "Using PyTorch cuDNN from: $PYTORCH_CUDNN_ROOT"
echo "Using CUDA headers from: $CUDA_INCLUDE_DIR"

cd "$REPO_ROOT/verl-tool"
python -m pip install -e verl --no-cache-dir
python -m pip install -e ".[vllm,torl,search_tool]" --no-cache-dir
python -m pip install --no-cache-dir \
  "AceCoder @ git+https://github.com/TIGER-AI-Lab/AceCoder.git@ad017bcef56a95e5b06dc8da564c8a8ab6e3d62e"
python -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  --requirement "$REQUIREMENTS_FILE"

# mini_webarena is a repository-local package without packaging metadata.
# A .pth file gives it the same checkout-bound behavior as the editable
# verl/verl-tool installs, including when Python starts outside the repo root.
BROWSERAGENT_REPO_ROOT="$REPO_ROOT" python - <<'PY'
import os
import site
from pathlib import Path

repo_root = Path(os.environ["BROWSERAGENT_REPO_ROOT"]).resolve()
pth_file = Path(site.getsitepackages()[0]) / "browseragent_v2_repo.pth"
pth_file.write_text(f"{repo_root}\n", encoding="utf-8")
print(f"Wrote {pth_file} -> {repo_root}")
PY

# Chromium is stored in the user's Playwright cache. Its Linux runtime libraries
# are provided by this Conda environment by default, so installation needs no
# root access. Set INSTALL_PLAYWRIGHT_DEPS=1 only to additionally ask Playwright
# to install the distribution's official system packages through root/sudo.
if [[ "$INSTALL_PLAYWRIGHT_DEPS" == "1" ]]; then
  python -m playwright install --with-deps chromium
else
  python -m playwright install chromium
fi

python "$SCRIPT_DIR/verify_env.py" browseragent-v2
conda list --explicit > "$CONDA_PREFIX/conda-explicit.txt"
python -m pip freeze --all | LC_ALL=C sort > "$CONDA_PREFIX/pip-freeze.txt"
echo "$ENV_NAME is ready. Activate it with: conda activate $ENV_NAME"
echo "Environment snapshots: $CONDA_PREFIX/conda-explicit.txt and $CONDA_PREFIX/pip-freeze.txt"
