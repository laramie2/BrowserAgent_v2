#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Paths can be overridden without editing this file.
SFT_DATASET_NAME="${SFT_DATASET_NAME:-browseragent-sft}"
SFT_DATASET_DIR="${SFT_DATASET_DIR:-${SCRIPT_DIR}/dataset/${SFT_DATASET_NAME}}"
SFT_DATASET_FILE="${SFT_DATASET_FILE:-data.jsonl}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-7B-Instruct}"
LORA_OUTPUT_DIR="${LORA_OUTPUT_DIR:-${SCRIPT_DIR}/output/lora}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29503}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-16240}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.2}"
DEEPSPEED="${DEEPSPEED:-zero3}"
FREEZE_VIT="${FREEZE_VIT:-false}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"

if ! command -v swift >/dev/null 2>&1; then
    echo "error: swift is not available; activate the swift-sft environment" >&2
    exit 2
fi
if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
    echo "error: base model not found: ${BASE_MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -f "${SFT_DATASET_DIR}/${SFT_DATASET_FILE}" ]]; then
    echo "error: prepared dataset not found: ${SFT_DATASET_DIR}/${SFT_DATASET_FILE}" >&2
    exit 2
fi

mkdir -p "${SCRIPT_DIR}/logs" "${LORA_OUTPUT_DIR}"
LOG_FILE="${SFT_LOG_FILE:-${SCRIPT_DIR}/logs/sft_$(date +%Y%m%d_%H%M%S).log}"

echo "Dataset: ${SFT_DATASET_DIR}/${SFT_DATASET_FILE}"
echo "Base model: ${BASE_MODEL_PATH}"
echo "LoRA output: ${LORA_OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"

# Swift resolves relative image paths against the working directory.
cd "${SFT_DATASET_DIR}"
swift sft \
    --model "${BASE_MODEL_PATH}" \
    --model_type qwen2_5_vl \
    --tuner_type lora \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --freeze_vit "${FREEZE_VIT}" \
    --torch_dtype bfloat16 \
    --dataset "${SFT_DATASET_FILE}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --split_dataset_ratio 0.001 \
    --output_dir "${LORA_OUTPUT_DIR}" \
    --learning_rate "${LEARNING_RATE}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_length "${MAX_LENGTH}" \
    --deepspeed "${DEEPSPEED}" \
    --eval_steps 2000 \
    --save_strategy epoch \
    --logging_steps 1 \
    --gradient_checkpointing true \
    2>&1 | tee "${LOG_FILE}"

echo "SFT completed. Run: bash sft/merge_lora.sh"
