#!/bin/bash
# evaluate.sh


DATA=without_content-newadd2720
LR=5e-5
FREEZE=false
EPOCH=2

OUTPUT_DIR="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/results/Qwen2.5-VL-7B-Instruct_task-opsrc-${DATA}-sft-${LR}lr-freeze_${FREEZE}-${EPOCH}epoch"
PROMPT_PATH="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/prompt/system_prompt_with_history_info_without_content.txt"
MAX_SAMPLES=300
NUM_TRIALS=4
VLLM_BASE_URL=http://localhost:8008/v1/
MODEL='custom-llm-1'
NUM_WORKERS=20

python -m gen_seq.pipeline \
    --output_file=$OUTPUT_DIR/nq_test_results.jsonl \
    --data_path=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/benchmark/v1/nq/test-00000-of-00001.parquet \
    --system_prompt=$PROMPT_PATH \
    --max_samples=$MAX_SAMPLES \
    --num_trials=$NUM_TRIALS \
    --base_url=$VLLM_BASE_URL \
    --model=$MODEL \
    --use_vlm \
    --image_output_dir=$OUTPUT_DIR/nq_obs_images \
    --num_workers $NUM_WORKERS


