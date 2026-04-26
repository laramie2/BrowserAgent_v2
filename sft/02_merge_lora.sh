#!/bin/bash
# 确保在项目根目录下运行此脚本

# ==========================================
# 1. 基础路径配置区
# ==========================================
SFT_ROOT="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft"
MODEL_NAME="Qwen2.5-VL-7B-Instruct"
DATASET_NAME="task-opsrc-without_content-newadd2720-sft-5e-5lr-freeze_false"
EPOCH=2
# 原始模型路径
MODEL_PATH="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/models/${MODEL_NAME}"

# 【重要：请替换为训练完成后生成的具体时间戳目录名和 checkpoint 步数】
CHECKPOINT_SUBDIR="v3-20260424-200127/checkpoint-510"
# ==========================================
# 2. 导出路径配置区
# ==========================================
# 训练产生的 LoRA 权重路径
CKPT_DIR="${SFT_ROOT}/output/${MODEL_NAME}-${DATASET_NAME}/${CHECKPOINT_SUBDIR}"

# 合并后的完整模型输出路径
MERGED_OUTPUT_DIR="${SFT_ROOT}/output/${MODEL_NAME}-${DATASET_NAME}-${EPOCH}epoch-merged"

LOG_FILE="${SFT_ROOT}/logs/merge_${MODEL_NAME}_${DATASET_NAME}-${EPOCH}epoch.log"

# ==========================================
# 3. 执行合并逻辑
# ==========================================
mkdir -p "${SFT_ROOT}/logs"

echo "Starting to merge LoRA weights..."
echo "Checkpoint: $CKPT_DIR"
echo "Output: $MERGED_OUTPUT_DIR"

# 使用 nohup 后台运行，将 export 和 完美的“换头术” 结合
CUDA_VISIBLE_DEVICES=0,1 \
nohup bash -c "
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu:\$LIBRARY_PATH && \
    swift export \
        --model '$MODEL_PATH' \
        --adapters '$CKPT_DIR' \
        --merge_lora true \
        --output_dir '$MERGED_OUTPUT_DIR' && \
    echo '[Post-Merge] 正在执行完美物理换头术...' && \
    find '$MODEL_PATH' -maxdepth 1 -type f ! -name '*.safetensors' ! -name '*.bin' ! -name '*index.json' -exec cp -f {} '$MERGED_OUTPUT_DIR/' \; && \
    rm -f '$MERGED_OUTPUT_DIR'/processor_config.json "$MERGED_OUTPUT_DIR"/chat_template.jinja && \
    echo '[Post-Merge] 完美配置修补完成！'
" > "$LOG_FILE" 2>&1 &

echo "Merge process started in background."
echo "Check ${LOG_FILE} for progress."