#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-swift-sft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

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
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install "$FLASH_ATTN_WHEEL"
python -m pip install "ms-swift[all]" -U
python -m pip install qwen-vl-utils
python -m pip install deepspeed decord

python - <<'PY'
import torch
print("swift-sft ready")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
