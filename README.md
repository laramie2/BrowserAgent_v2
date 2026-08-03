# BrowserAgent v2

BrowserAgent v2 trains a browser agent in three stages:

1. generate or download teacher trajectories and render text observations with VTC;
2. supervised fine-tune Qwen2.5-VL with LoRA and merge the adapter;
3. construct NQ/Hotpot RL seeds, estimate difficulty with K rollouts, and train with curriculum data.

The browser operates against a local Kiwix Wikipedia service through the
`verl-tool` text-browser server. Teacher models and rollout models are accessed
through an OpenAI-compatible `/v1/chat/completions` API.

## Repository layout

| Path | Purpose |
| --- | --- |
| `env/` | Conda installers, dependency lists, verification, and public resource download tools |
| `generate_sft_data.py` | Parallel OpenAI-compatible teacher trajectory generation |
| `sft/` | Hugging Face teacher-data download, VTC preprocessing, SFT, and LoRA merge |
| `RL/` | Seed extraction, K-rollout difficulty scoring, curriculum construction, and RL training |
| `evaluate/` | Evaluation runners, experiment queues, metrics, and result analysis |
| `vtc_renderer.py` | Text accessibility-tree to image renderer |
| `wiki_cluster/` | Local Kiwix Wikipedia service |

Generated datasets, model weights, logs, and evaluation results are ignored by Git.

## 1. Install environments

Clone submodules and install the two isolated Conda environments:

```bash
git submodule update --init --recursive
bash env/install_all.sh
```

- `browseragent-v2` runs the tool server, data generation, RL, and evaluation.
- `swift-sft` runs MS-Swift LoRA training and merge.

See [`env/README.md`](env/README.md) for CUDA versions and installation options.

Download the base model, BrowserAgent SeedData, prompt tokenizer, and Wiki resources:

```bash
conda activate browseragent-v2
hf auth login
python env/prepare_resources.py prepare
python env/download_prompt_tokenizer.py
```

SeedData is stored once under `RL/dataset/BrowserAgent-SeedData/` and is shared by
teacher generation, RL seed extraction, and evaluation.

## 2. Start browser services

Start the local Wiki service:

```bash
./wiki_cluster/start.sh
```

In another terminal, start the text-browser tool server:

```bash
conda activate browseragent-v2
./start_tool_server.sh
```

The default observation endpoint is
`http://127.0.0.1:5000/get_observation`. Concurrency and timeouts can be changed
with the `TOOL_SERVER_*` and `TEXT_BROWSER_*` environment variables.

## 3. Generate teacher trajectories

Configure any OpenAI-compatible provider. Keep credentials in environment
variables, never in tracked files:

```bash
export OPENAI_BASE_URL=https://your-provider.example/v1
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=your_teacher_model
```

Generate trajectories in parallel from a SeedData parquet:

```bash
conda activate browseragent-v2
python generate_sft_data.py \
  --data-path RL/dataset/BrowserAgent-SeedData/nq/train-00000-of-00001.parquet \
  --output-file sft/dataset/raw/generated_teacher.jsonl \
  --max-samples 5000 \
  --workers 16
```

Only successful trajectories are written to the SFT JSONL by default. All task
outcomes are recorded in `generated_teacher_results.jsonl`, which also enables
resume. Use `--keep-failed` only for diagnostics and `--overwrite` to start over.

Alternatively, download a published teacher dataset without hard-coding its repository:

```bash
python sft/dataset/download_sft_dataset.py --repo-id <org/dataset>
```

## 4. Render VTC data and run SFT

Convert the generated text observations into Swift-compatible multimodal records:

```bash
python sft/dataset/prepare_sft_dataset.py \
  --input sft/dataset/raw/generated_teacher.jsonl \
  --dataset-name browseragent-sft \
  --system-msg-path prompt/system_prompt_with_history_info.txt
```

The output is `sft/dataset/browseragent-sft/data.jsonl` plus a relative `images/`
directory. Train and merge:

```bash
conda activate swift-sft
SFT_DATASET_NAME=browseragent-sft bash sft/train.sh
bash sft/merge_lora.sh
```

The merged model is written to `RL/models/browseragent-sft`, which is also the
default model path used by RL and evaluation. Export `SFT_MODEL_PATH_OVERRIDE`
before merge and RL training to use another shared location.

## 5. Build RL data

Extract reproducible NQ/Hotpot training seeds and a balanced 100-row
validation set:

```bash
conda activate browseragent-v2
python RL/prepare_seed_data.py --datasets nq hotpot --num-samples 5000 --seed 42
```

This writes `RL/dataset/nq/`, `RL/dataset/hotpot/`, and
`RL/dataset/validation_100/data.parquet`.

Serve the merged SFT model with an OpenAI-compatible vLLM endpoint:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model RL/models/browseragent-sft \
  --served-model-name browseragent-sft \
  --port 8008

export OPENAI_BASE_URL=http://127.0.0.1:8008/v1
export OPENAI_API_KEY=EMPTY
export OPENAI_MODEL=browseragent-sft
```

With Wiki and `start_tool_server.sh` still running, generate K trajectories per
sample and aggregate reward-based difficulty:

```bash
python RL/score_difficulty.py \
  --data-paths RL/dataset/nq/train_5000_labelled.parquet \
  --output-dir RL/filter_results/sft_rollout_k8_nq \
  --k 8 --num_workers 16

python RL/score_difficulty.py \
  --data-paths RL/dataset/hotpot/train_5000_labelled.parquet \
  --output-dir RL/filter_results/sft_rollout_k8_hotpot \
  --k 8 --num_workers 16
```

Build balanced curriculum stages. The default inputs are the two score files above:

```bash
python RL/build_curriculum.py --stage_size 1000 --mode disjoint
```

Stages are written below
`RL/dataset/curriculum_medium_disjoint_n1000_seed42/`, each with a trainable
`data.parquet`.

## 6. Run RL training

Select a curriculum stage through the existing benchmark override:

```bash
BENCHMARK_OVERRIDE=curriculum_medium_disjoint_n1000_seed42/stage_1_medium_warmup \
  bash RL/scripts/train.sh
```

For experiment grids, edit `RL/configs/train.yaml` and run:

```bash
bash RL/scripts/run_experiments.sh
```

## 7. Evaluate

```bash
./evaluate/run.sh all
python evaluate/summarize_results.py \
  evaluate/results/browseragent-sft/*_test_results.jsonl
```

See [`evaluate/README.md`](evaluate/README.md) and
[`docs/evaluation-experiments.md`](docs/evaluation-experiments.md) for queues,
resume behavior, token statistics, and ablation matrices.

## Configuration and secrets

- Use `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` for teacher and rollout APIs.
- Use `TOOL_SERVER_URL` and `BROWSER_URL` to override browser endpoints.
- Use `HF_TOKEN` or `hf auth login` for gated Hugging Face resources.
- Do not commit API keys, Hugging Face tokens, W&B keys, generated data, or model weights.
