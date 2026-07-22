#!/usr/bin/env bash
# Linux x86_64 / CPython 3.10 / CUDA 12.4.  Does not modify existing env files.
set -eo pipefail

ENV_NAME="${ENV_NAME:-swift-sft}"
RECREATE="${RECREATE:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_swift_sft_cu124.txt"
LOCK_FILE="${SCRIPT_DIR}/swift-sft-cu124-py310.lock.txt"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Requires Linux x86_64.' >&2; exit 1; }
command -v conda >/dev/null || { echo 'conda was not found in PATH.' >&2; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null || { echo 'A working NVIDIA driver is required.' >&2; exit 1; }
[[ -f "${REQUIREMENTS_FILE}" ]] || { echo "Missing ${REQUIREMENTS_FILE}" >&2; exit 1; }

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  [[ "${RECREATE}" == 1 ]] || { echo "${ENV_NAME} exists; use RECREATE=1 to replace it." >&2; exit 1; }
  conda env remove --name "${ENV_NAME}" --yes
fi

conda create --name "${ENV_NAME}" python=3.10 --yes
conda activate "${ENV_NAME}"
conda install --yes -c nvidia/label/cuda-12.4.1 cuda-toolkit=12.4.1
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
command -v nvcc >/dev/null
nvcc --version

python -m pip install --upgrade pip==25.3 setuptools==75.8.2 wheel==0.45.1
python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-cache-dir "${FLASH_ATTN_WHEEL}"
DS_BUILD_OPS=0 python -m pip install --no-cache-dir --requirement "${REQUIREMENTS_FILE}"
python -m pip check
python -m pip freeze --all | sort > "${LOCK_FILE}"

python - <<'PY'
import importlib.metadata as md
import os
import torch
assert torch.__version__ == '2.6.0+cu124', torch.__version__
assert torch.cuda.is_available(), 'PyTorch cannot access an NVIDIA GPU'
assert os.environ.get('CUDA_HOME'), 'CUDA_HOME is missing'
from torch.distributed.fsdp import FSDPModule
import deepspeed
import flash_attn
import swift
for pkg, expected in {'ms-swift': '3.5.0', 'deepspeed': '0.15.4', 'flash-attn': '2.7.4.post1'}.items():
    assert md.version(pkg) == expected, (pkg, md.version(pkg))
print('Environment verification passed')
print('torch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('DeepSpeed:', deepspeed.__version__)
print('MS-Swift:', md.version('ms-swift'))
print('FlashAttention:', flash_attn.__version__)
PY

echo "${ENV_NAME} is ready. Before training: conda activate ${ENV_NAME}; export CUDA_HOME=\$CONDA_PREFIX"
echo 'Use --torch_dtype bfloat16 on A100; float16 caused the recorded DeepSpeed loss-scale overflow.'
