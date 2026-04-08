#!/bin/bash
# 确保在 sft 根目录下运行此脚本

# ==========================================
# 1. 基础路径配置区
# ==========================================
SFT_ROOT="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft"
MODEL_NAME="Qwen2.5-VL-7B-Instruct"
DATASET_NAME="task-opsrc-2500stp"
DATASET_DIR="${SFT_ROOT}/dataset/${DATASET_NAME}"
LOG_FILE="${SFT_ROOT}/logs/train_${MODEL_NAME}_${DATASET_NAME}.log"

# ==========================================
# 2. 硬件与环境配置区
# ==========================================
# export CUDA_VISIBLE_DEVICES=1,2,3
export CUDA_VISIBLE_DEVICES=5
export NPROC_PER_NODE=1
export MASTER_PORT=29500

# ==========================================
# 3. 模型与数据配置区
# ==========================================
MODEL_PATH="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/models/${MODEL_NAME}"
DATASET_FILE="data.jsonl"
OUTPUT_DIR="${SFT_ROOT}/output/${MODEL_NAME}-${DATASET_NAME}-sft"
SYSTEM_PROMPT=$(< ${SFT_ROOT}/system_prompt_with_history_info.txt)

# ==========================================
# 4. 训练超参数配置区
# ==========================================
LORA_RANK=16
LORA_ALPHA=32
LEARNING_RATE=1e-4
BATCH_SIZE=1
GRAD_ACCUMULATION_STEPS=1
EVAL_STEPS=50
SAVE_STEPS=50
EPOCHS=3
MAX_LENGTH=8192
DEEPSPEED="zero3"
GRAD_CHECKPOINTING="true"

# ==========================================
# 5. 执行训练逻辑
# ==========================================
mkdir -p ${SFT_ROOT}/logs

# 临时切换到数据集所在的目录！这样 swift 就能天然找到相对路径下的 ./images
cd ${DATASET_DIR}

# 执行训练
nohup swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --learning_rate "$LEARNING_RATE" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --num_train_epochs "$EPOCHS" \
    --gradient_checkpointing "$GRAD_CHECKPOINTING" \
    --max_length "$MAX_LENGTH" \
    --deepspeed "$DEEPSPEED" \
    --system "$SYSTEM_PROMPT" \
    > "$LOG_FILE" 2>&1 &

echo "SFT training started with ${NPROC_PER_NODE} GPUs."
echo "Check ${LOG_FILE} for progress."

# 如果要停止微调 pkill -f swift 或者查看 nvidia-smi 找到进程 PID 后全部 kill 掉