#!/bin/bash
# run_vllm_instance1.sh

MODEL_NAME="Qwen2.5-VL-7B-Instruct-step-opsrc-5000-200"
INSTANCE_ID="instance_step-opsrc-5000-200"
PORT=8008
LOG_FILE="./logs/vllm_${INSTANCE_ID}.log"

# 清理同名实例（只清理自己）
pkill -9 -f "VLLM.*--port $PORT" 2>/dev/null
rm -rf ~/.triton/cache
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 $CONDA_PREFIX/lib/libcuda.so

mkdir -p ./logs

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

# MODEL_PATH=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/output/${MODEL_NAME}-merged
MODEL_PATH=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/models/Qwen2.5-VL-7B-Instruct-step-opsrc-5000-200

CUDA_VISIBLE_DEVICES=1,2,3,4 \
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name custom-llm-1 \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --max-model-len 8192 \
    --tensor-parallel-size 4 \
    > $LOG_FILE 2>&1

# 退出时清理
pkill -9 -f "VLLM.*--port $PORT" 2>/dev/null