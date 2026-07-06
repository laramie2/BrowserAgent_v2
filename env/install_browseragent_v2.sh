#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-browseragent-v2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

conda activate "$ENV_NAME"

conda install -c conda-forge cudnn=9.8 -y
conda install -c nvidia nccl -y

python -m pip install --upgrade pip setuptools wheel packaging ninja

cd "$REPO_ROOT/verl-tool"
python -m pip install -e verl --no-cache-dir
python -m pip install -e ".[vllm,acecoder,torl,search_tool]" --no-cache-dir

python -m pip install "flash-attn==2.8.3" --no-build-isolation
python -m pip install megatron-core
python -m pip install --no-build-isolation "transformer-engine[pytorch]"
python -m pip install beartype gymnasium playwright
python -m playwright install chromium
python -m pip install text-generation
python -m pip install fire uvicorn fastapi httptools lxml tiktoken scipy fsspec modelscope

python - <<'PY'
import torch
print("browseragent-v2 ready")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
