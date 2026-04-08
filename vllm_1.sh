#!/bin/bash
# run_vllm_instance1.sh

MODEL_NAME="Qwen2.5-VL-7B-Instruct-task-opsrc-2500stp"
INSTANCE_ID="instance_task-2500stp"
PORT=8009
LOG_FILE="./logs/vllm_${INSTANCE_ID}.log"

# 清理同名实例（只清理自己）
pkill -9 -f "VLLM.*--port $PORT" 2>/dev/null
rm -rf ~/.triton/cache
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 $CONDA_PREFIX/lib/libcuda.so

mkdir -p ./logs

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH


CUDA_VISIBLE_DEVICES=3,4 \
python -m vllm.entrypoints.openai.api_server \
    --model /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/output/${MODEL_NAME}-merged \
    --served-model-name custom-llm-2 \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    > $LOG_FILE 2>&1

# 退出时清理
pkill -9 -f "VLLM.*--port $PORT" 2>/dev/null