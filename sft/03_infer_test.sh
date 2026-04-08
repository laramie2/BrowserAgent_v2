#!/bin/bash

# ==========================================
# 1. 路径配置
# ==========================================
# 指向你刚刚合并完的完整模型目录
MERGED_MODEL_PATH="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/output/Qwen2.5-VL-7B-Instruct-step-opsrc-5000-merged"

# ==========================================
# 2. 执行推理
# ==========================================
# 使用 swift infer 启动交互式测试
swift infer \
    --model "$MERGED_MODEL_PATH" \
    --infer_backend pt \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1.05 \
    --merge_lora false  # 因为已经合并过了，这里设为 false