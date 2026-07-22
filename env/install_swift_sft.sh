#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-swift-sft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-12.6}"
RECREATE="${RECREATE:-0}"
VERIFIER_REVISION="2026-07-22-swift-sft-v2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_swift_sft.txt"
VERIFIER_FILE="$SCRIPT_DIR/verify_env.py"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
  die "swift-sft requires Linux x86_64."
[[ "$RECREATE" == "0" || "$RECREATE" == "1" ]] || \
  die "RECREATE must be 0 or 1."
[[ "$PYTHON_VERSION" == "3.10" ]] || \
  die "This pinned stack supports only PYTHON_VERSION=3.10."
[[ "$CUDA_VERSION" == "12.6" ]] || \
  die "This pinned stack supports only CUDA_VERSION=12.6."
command -v conda >/dev/null 2>&1 || die "conda was not found in PATH."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi was not found in PATH."
nvidia-smi >/dev/null || die "The NVIDIA driver is not working."
[[ -f "$REQUIREMENTS_FILE" ]] || die "Missing $REQUIREMENTS_FILE."
[[ -f "$VERIFIER_FILE" ]] || die "Missing $VERIFIER_FILE."
grep -Fqx "CONFIG_REVISION = \"$VERIFIER_REVISION\"" "$VERIFIER_FILE" || \
  die "install_swift_sft.sh and verify_env.py are from different config revisions; sync the complete env/ directory."

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
  numpy=1.26.4 \
  decord=0.6.0 \
  cmake \
  ninja \
  make \
  gcc_linux-64 \
  gxx_linux-64

# Conda activation hooks are third-party shell code and some CUDA releases
# read optional variables while they are unset. Limit the nounset exception to
# activation itself and restore strict mode before installation continues.
set +u
conda activate "$ENV_NAME"
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

command -v nvcc >/dev/null 2>&1 || die "CUDA Toolkit installation did not provide nvcc."
nvcc --version

python -m pip install --upgrade \
  pip==25.3 \
  setuptools==75.8.2 \
  wheel==0.45.1 \
  packaging==25.0
python -m pip install --no-cache-dir \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

# Do not prebuild optional DeepSpeed CUDA operators. They are compiled lazily
# only if a selected training configuration needs them.
DS_BUILD_OPS=0 python -m pip install \
  --no-cache-dir \
  --requirement "$REQUIREMENTS_FILE"

python "$VERIFIER_FILE" swift-sft
conda list --explicit > "$CONDA_PREFIX/conda-explicit.txt"
python -m pip freeze --all | LC_ALL=C sort > "$CONDA_PREFIX/pip-freeze.txt"
echo "$ENV_NAME is ready. Activate it with: conda activate $ENV_NAME"
echo "For training, keep CUDA_HOME set to the active environment and use bfloat16 on A100."
echo "Environment snapshots: $CONDA_PREFIX/conda-explicit.txt and $CONDA_PREFIX/pip-freeze.txt"
