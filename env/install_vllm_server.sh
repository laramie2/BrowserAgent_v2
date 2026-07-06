#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vllm-server}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VLLM_VERSION="${VLLM_VERSION:-0.13.0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install "vllm==${VLLM_VERSION}"
python -m pip install qwen-vl-utils modelscope

python - <<'PY'
import torch
import vllm
print("vllm-server ready")
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("cuda available:", torch.cuda.is_available())
PY
