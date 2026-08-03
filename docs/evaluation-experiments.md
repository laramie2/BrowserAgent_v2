# Evaluation experiments

The evaluation stack is configured top-down:

`evaluate/configs/eval_matrix.json` → `evaluate/run_experiments.py` → `evaluate/run.sh` → `evaluate.pipeline`.

The top-level runner owns ports, concurrency, timeouts, resume, VTC compression ratio, maximum browser steps, and generation length. The pipeline no longer hardcodes them.

## New-server setup

Download the prompt tokenizer once before starting text-browser actors:

```bash
python env/download_prompt_tokenizer.py
```

The default local path is `models/Qwen2.5-14B-Instruct`. For another path:

```bash
export MINI_WEB_ARENA_PROMPT_MODEL=/local/path/Qwen2.5-14B-Instruct
export MINI_WEB_ARENA_TOKENIZER_LOCAL_ONLY=1
```

This prevents every concurrent Ray actor from calling Hugging Face during initialization.

## Direct evaluation

```bash
./evaluate/run.sh all \
  --model-path /models/my-checkpoint \
  --output-dir evaluate/results/my-checkpoint \
  --num-workers 128 \
  --tool-workers 128 \
  --tool-max-requests 128 \
  --browser-max-actors 128 \
  --compression-factor 1.2 \
  --tool-server-port 5000
```

Resume is enabled by default and skips completed `(trial_idx, sample_idx)` pairs. `--no-resume` intentionally reruns everything.

Each benchmark logs `BENCHMARK_TIMING`. The final `EVALUATION_TIMING_SUMMARY` includes total time, completed sample-evaluations, evaluations/second, and seconds/evaluation.

Token counting runs after evaluation and writes `token_usage_summary.json` plus `token_usage_by_benchmark.csv`. The summary contains raw-text and compressed-image Avg/Max tokens per step and token savings. Use `--no-token-stats` to defer it.

## Experiment matrix

Set model paths for the groups you will run:

```bash
export BASE_VL_MODEL_PATH=/models/base-vl
export SFT_MODEL_PATH=/models/sft
export RL_MODEL_PATH=/models/sft-rl
export RL_ALL_REWARDS_MODEL_PATH=/models/rl-all
export RL_WO_DENSE_MODEL_PATH=/models/rl-wo-dense
export RL_WO_ACTION_PENALTY_MODEL_PATH=/models/rl-wo-action
export RL_WO_FORMAT_MODEL_PATH=/models/rl-wo-format
export COMPRESSION_MODEL_PATH=/models/sft-rl
```

Inspect, dry-run, or execute:

```bash
python evaluate/run_experiments.py --list
python evaluate/run_experiments.py --group compression_ablation --dry-run
python evaluate/run_experiments.py --group main
python evaluate/run_experiments.py --group rl_ablation --experiment wo_dense_reward
```

Arguments following `--` pass through to every total-eval invocation:

```bash
python evaluate/run_experiments.py --group compression_ablation -- \
  --vllm-cuda-devices 0,1,2,3 --vllm-tensor-parallel-size 4
```

Main experiments use all available rows. Both ablation groups use up to 1000 rows per benchmark with seed 42. These values live in `evaluate/configs/eval_matrix.json`.

## Ordered evaluation queue

Use a queue when one machine should evaluate several already-merged models without manual intervention. Jobs run strictly in JSON order and may mix main, RL-ablation, and compression-ablation entries.

Start from the mixed example:

```bash
cp evaluate/configs/eval_queue.example.json evaluate/configs/eval_queue.machine01.json
```

Each job references a matrix experiment and overrides its local model path:

```json
{
  "job_id": "sft_compression_1_4",
  "group": "compression_ablation",
  "experiment": "ratio_1_4",
  "model_path": "${MODEL_ROOT}/sft-compression-1.4-merged"
}
```

The selected matrix entry supplies benchmark list, sample count, seed, and compression factor. Queue-level or job-level fields override those values. `env` supplies per-job environment variables and `extra_args` passes options to `evaluate/run.sh`.

Preflight the exact order and commands before occupying GPUs:

```bash
export MODEL_ROOT=/models/browseragent
export PIPELINE_PYTHON=/path/to/browseragent-env/bin/python
export TOOL_SERVER_PYTHON=/path/to/browseragent-env/bin/python
export VLLM_PYTHON=/path/to/vllm-env/bin/python
export MINI_WEB_ARENA_PROMPT_MODEL=/models/Qwen2.5-14B-Instruct
export TOKEN_STATS_MODEL_PATH=/models/Qwen2.5-VL-7B-Instruct

python evaluate/run_queue.py \
  --queue evaluate/configs/eval_queue.machine01.json --list

python evaluate/run_queue.py \
  --queue evaluate/configs/eval_queue.machine01.json --dry-run
```

Run the queue in tmux:

```bash
python evaluate/run_queue.py \
  --queue evaluate/configs/eval_queue.machine01.json
```

For every job, `evaluate/run.sh` starts the selected model, evaluates it, computes token statistics, and stops its vLLM/tool-server process groups before the next job starts. Do not add `--skip-vllm` to a multi-model queue.

A benchmark that exits with code `2` has unsaved environment/request failures. The runner now keeps the healthy vLLM process, restarts the tool server and its Ray runtime, and resumes the benchmark from its JSONL automatically. It retries five times by default. Code/configuration failures with other exit codes fail immediately instead of being hidden. Override the policy per queue or per job when needed:

```json
{
  "benchmark_max_retries": 8,
  "benchmark_retry_delay": 10
}
```

Set `benchmark_max_retries` to `0` to disable automatic retries. `--no-resume` automatically disables retries, because retrying without resume would duplicate experimental work. Every attempt is included in timing and throughput totals, while the completed benchmark count is incremented only once.

The runner uses a short, run-specific Ray directory under `/tmp` by default to stay below the Unix socket path limit. `RAY_TMPDIR_OVERRIDE` remains available when a machine needs a specific short location. Queue-defined environment variables may reference one another, so `model_path: "${MODEL_ROOT}/model"` works when `MODEL_ROOT` is declared in `defaults.env`.

Before loading a model, the runner checks the vLLM and tool-server TCP ports rather than relying only on a successful health response. A stale matching vLLM/tool-server process is stopped before startup, including a tool router that owns the port but returns HTTP 502. An unknown process is never killed automatically; the run fails with an `ss` inspection command. Use `--skip-vllm` or `--skip-tool-server` only to reuse an intentionally managed service, and use `--no-kill-existing-vllm` or `--no-kill-existing-tool-server` to disable replacement explicitly.

Queue state is written atomically to:

```text
evaluate/results/experiments/_queues/<queue-name>.json
```

Restarting the same command skips jobs recorded as completed with the same model/settings and resumes the interrupted job through its existing JSONL output. A changed model path or setting changes the job signature and causes that job to run again. Use `--rerun-completed` to intentionally rerun every completed item.

By default the queue stops at the first failed experiment. After fixing the problem, rerun the same command. Use `--continue-on-error` only when later independent jobs should run despite a failure.

The queue holds `evaluate/results/experiments/_queues/evaluation.lock` for its full lifetime, preventing a second queue in the same repository from taking the same ports/GPUs. Use `--lock-file` only when intentionally running isolated queues on different GPU sets and ports.

## Existing BrowserAgent-v1 results

Count tokens without rerunning inference:

```bash
export BROWSERAGENT_V1_RESULT_DIR=/path/to/browseragent-v1/results
python -m evaluate.token_stats \
  --input "$BROWSERAGENT_V1_RESULT_DIR" \
  --model_path /local/path/to/base-vlm \
  --system_prompt prompt/system_prompt_with_history_info.txt \
  --output_json "$BROWSERAGENT_V1_RESULT_DIR/token_usage_summary.json" \
  --output_csv "$BROWSERAGENT_V1_RESULT_DIR/token_usage_by_benchmark.csv"
```

## Generate paper tables

```bash
export BROWSERAGENT_V1_RESULT_DIR=/path/to/browseragent-v1/results
python evaluate/summarize_experiments.py
```

Outputs are `evaluate/reports/evaluation_tables.md` and `.csv`. The Markdown contains all three requested tables and a raw-vs-compressed token detail table.

## Concurrency tuning

Defaults match the RL setup at 128. On smaller hosts, lower pipeline, tool, actor, and Ray capacity together:

```bash
./evaluate/run.sh nq --num-workers 64 --tool-workers 64 \
  --tool-max-requests 64 --browser-max-actors 64 --browser-ray-cpus 64
```

Keep the idle pool modest (default 16). With the tokenizer local, remaining high-concurrency failures point to tool timeouts, Ray capacity, host RAM/process limits, or vLLM queue latency rather than Hugging Face rate limiting.
