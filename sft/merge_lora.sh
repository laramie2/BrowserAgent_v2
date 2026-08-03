#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-7B-Instruct}"
LORA_OUTPUT_DIR="${LORA_OUTPUT_DIR:-${SCRIPT_DIR}/output/lora}"
# RL/scripts/train.sh uses this exact default path.
MERGED_MODEL_PATH="${SFT_MODEL_PATH_OVERRIDE:-${PROJECT_ROOT}/RL/models/browseragent-sft}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"

has_model_weights() {
    compgen -G "${MERGED_MODEL_PATH}/*.safetensors" >/dev/null ||
        compgen -G "${MERGED_MODEL_PATH}/*.bin" >/dev/null
}

if ! command -v swift >/dev/null 2>&1; then
    echo "error: swift is not available; activate the swift-sft environment" >&2
    exit 2
fi
if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
    echo "error: base model not found: ${BASE_MODEL_PATH}" >&2
    exit 2
fi

if [[ -z "${SFT_CHECKPOINT_DIR:-}" ]]; then
    latest_checkpoint=""
    while IFS= read -r -d '' candidate; do
        if [[ -z "${latest_checkpoint}" || "${candidate}" -nt "${latest_checkpoint}" ]]; then
            latest_checkpoint="${candidate}"
        fi
    done < <(find "${LORA_OUTPUT_DIR}" -type d -name 'checkpoint-*' -print0 2>/dev/null)
    SFT_CHECKPOINT_DIR="${latest_checkpoint}"
fi

if [[ -z "${SFT_CHECKPOINT_DIR}" || ! -d "${SFT_CHECKPOINT_DIR}" ]]; then
    echo "error: no LoRA checkpoint found below ${LORA_OUTPUT_DIR}" >&2
    echo "Set SFT_CHECKPOINT_DIR=/path/to/checkpoint-N to select one explicitly." >&2
    exit 2
fi

if [[ -e "${MERGED_MODEL_PATH}" ]]; then
    if [[ -f "${MERGED_MODEL_PATH}/config.json" ]] && \
       has_model_weights; then
        echo "Merged SFT model already exists: ${MERGED_MODEL_PATH}"
        exit 0
    fi
    echo "error: refusing to replace incomplete output: ${MERGED_MODEL_PATH}" >&2
    echo "Move it aside or set SFT_MODEL_PATH_OVERRIDE to a new path." >&2
    exit 2
fi

mkdir -p "${SCRIPT_DIR}/logs" "$(dirname "${MERGED_MODEL_PATH}")"
LOG_FILE="${SFT_MERGE_LOG_FILE:-${SCRIPT_DIR}/logs/merge_$(date +%Y%m%d_%H%M%S).log}"

echo "Checkpoint: ${SFT_CHECKPOINT_DIR}"
echo "Merged model / RL input: ${MERGED_MODEL_PATH}"
echo "Log: ${LOG_FILE}"

swift export \
    --model "${BASE_MODEL_PATH}" \
    --adapters "${SFT_CHECKPOINT_DIR}" \
    --merge_lora true \
    --output_dir "${MERGED_MODEL_PATH}" \
    2>&1 | tee "${LOG_FILE}"

# Preserve the base processor metadata expected by BrowserAgent inference.
find "${BASE_MODEL_PATH}" -maxdepth 1 -type f \
    ! -name '*.safetensors' ! -name '*.bin' ! -name '*index.json' \
    -exec cp -f {} "${MERGED_MODEL_PATH}/" \;
rm -f "${MERGED_MODEL_PATH}/processor_config.json" \
      "${MERGED_MODEL_PATH}/chat_template.jinja"

if [[ ! -f "${MERGED_MODEL_PATH}/config.json" ]] || \
   ! has_model_weights; then
    echo "error: swift export completed without a valid merged model" >&2
    exit 2
fi

echo "Merge completed. RL will load: ${MERGED_MODEL_PATH}"
