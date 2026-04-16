#!/bin/bash
# 确保在 sft 根目录下运行此脚本

# ==========================================
# 1. 基础路径配置区
# ==========================================
SFT_ROOT="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft"
MODEL_NAME="Qwen2.5-VL-7B-Instruct"
DATASET_NAME="task-opsrc" 
DATASET_DIR="${SFT_ROOT}/dataset/${DATASET_NAME}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ==========================================
# 2. 硬件与环境配置区
# ==========================================
export CUDA_VISIBLE_DEVICES=1,2,3,4
export NPROC_PER_NODE=4
export MASTER_PORT=29500
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==========================================
# 3. 模型与数据配置区
# ==========================================
MODEL_PATH="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/models/${MODEL_NAME}"
DATASET_FILE="data.jsonl"
SYSTEM_PROMPT=$(< ${SFT_ROOT}/system_prompt_with_history_info.txt)

# ==========================================
# 4. 训练超参数配置区 (对齐第一个脚本的逻辑 + 保留 LoRA)
# ==========================================
TRAIN_TYPE="lora"
LORA_RANK=16
LORA_ALPHA=32
LEARNING_RATE=3e-5 # 学习率 5e-6 1e-6 1e-5
BATCH_SIZE=1                # 从 1 提升至 2，对齐第一个脚本
GRAD_ACCUMULATION_STEPS=8   # 对齐第一个脚本，全局 BS = 8卡 * 2 * 4 = 64，全局bs需要至少为32（显卡数*bs*ga）
MAX_LENGTH=16240
EPOCHS=2                    # epoch 1，2
WARMUP_RATIO=0.2           # 引入第一个脚本的预热比例
DEEPSPEED="zero3"           # 虽然是 LoRA，Zero3 依然可以节省显存处理长文本
FREEZE_VIT=false

EXTRA_CONFIG=${FREEZE_VIT:+freeze_$FREEZE_VIT}

OUTPUT_DIR="${SFT_ROOT}/output/${MODEL_NAME}-${DATASET_NAME}-sft-${LEARNING_RATE}lr-${EXTRA_CONFIG// /-}"
LOG_FILE="${SFT_ROOT}/logs/train_${MODEL_NAME}_${DATASET_NAME}-sft-${LEARNING_RATE}lr-${EXTRA_CONFIG// /-}.log"

# ==========================================
# 5. 执行训练逻辑
# ==========================================
mkdir -p "${SFT_ROOT}/logs"
mkdir -p "$OUTPUT_DIR"

# 切换到数据集所在目录，以确保能够正确加载相对路径下的 ./images
cd "${DATASET_DIR}"

# 执行训练
nohup swift sft \
    --model "$MODEL_PATH" \
    --model_type qwen2_5_vl \
    --tuner_type "$TRAIN_TYPE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --freeze_vit $FREEZE_VIT \
    --torch_dtype bfloat16 \
    --dataset "$DATASET_FILE" \
    --dataset_num_proc 100 \
    --dataloader_num_workers 48 \
    --split_dataset_ratio 0.001 \
    --system "$SYSTEM_PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
    --num_train_epochs "$EPOCHS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_length "$MAX_LENGTH" \
    --deepspeed "$DEEPSPEED" \
    --eval_steps 1000 \
    --save_strategy epoch \
    --logging_steps 1 \
    --gradient_checkpointing true \
    > "$LOG_FILE" 2>&1 &

echo "SFT LoRA training started with ${NPROC_PER_NODE} GPUs."
echo "Check ${LOG_FILE} for progress."
touch done

# 如果要停止微调 pkill -f swift 或者查看 nvidia-smi 找到进程 PID 后全部 kill 掉