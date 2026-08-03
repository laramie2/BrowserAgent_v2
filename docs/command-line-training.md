# BrowserAgent command-line training

The public workflow is organized as data download, VTC preprocessing, SFT,
LoRA merge, RL, and evaluation. All paths are project-relative and can be
overridden with environment variables.

## 1. Prepare the environments and public resources

Follow [`env/README.md`](../env/README.md) to create the `browseragent-v2` and
`swift-sft` environments. Then download the base model, BrowserAgent SeedData, and Wiki
resources:

```bash
python3 env/prepare_resources.py prepare --dry-run
python3 env/prepare_resources.py prepare
```

Hugging Face authentication should be supplied by `hf auth login` or the
`HF_TOKEN` environment variable. Never put access tokens in a tracked script.

## 2. Generate or download teacher SFT trajectories

Start the Wiki and tool services, then configure an OpenAI-compatible teacher:

```bash
./wiki_cluster/start.sh
./start_tool_server.sh
export OPENAI_BASE_URL=https://your-provider.example/v1
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=your_teacher_model
python generate_sft_data.py \
  --data-path RL/dataset/BrowserAgent-SeedData/nq/train-00000-of-00001.parquet \
  --output-file sft/dataset/raw/generated_teacher.jsonl \
  --workers 16
```

Only successful trajectories are written by default and can be passed directly to `prepare_sft_dataset.py`. Alternatively, download a published teacher dataset without hard-coding its repository:

```bash
python sft/dataset/download_sft_dataset.py --repo-id <org/dataset>
```

## 3. Render the VTC SFT dataset

```bash
python sft/dataset/prepare_sft_dataset.py \
  --input sft/dataset/raw/generated_teacher.jsonl \
  --dataset-name browseragent-sft \
  --system-msg-path prompt/system_prompt_with_history_info.txt
```

The output is self-contained under `sft/dataset/browseragent-sft/`:

```text
sft/dataset/browseragent-sft/
├── data.jsonl
└── images/
```

The converter defaults to task-level, open-source multimodal records. Use
`--level step`, `--format openai`, `--workers N`, or `--simple` when needed.

## 4. Train and merge SFT

Activate the Swift environment and run:

```bash
SFT_DATASET_NAME=browseragent-sft bash sft/train.sh
bash sft/merge_lora.sh
```

The first command writes LoRA checkpoints below `sft/output/lora/`. The merge
script selects the newest checkpoint unless `SFT_CHECKPOINT_DIR` is set.

The merged model is written to `RL/models/browseragent-sft`. This is also the
default path consumed by RL, so no manual copy or rename is required. To use a
different shared location, set the same override for merge and RL:

```bash
export SFT_MODEL_PATH_OVERRIDE=/models/my-browseragent-sft
bash sft/merge_lora.sh
bash RL/scripts/train.sh
```

Other training variables, including GPU selection, batch size, learning rate,
epochs, and output paths, can be overridden without editing the Bash scripts.

## 5. Run RL

Edit experiment settings in `RL/configs/train.yaml`, then run either a single
training job or the experiment driver:

```bash
bash RL/scripts/train.sh
bash RL/scripts/run_experiments.sh
```

The RL algorithm, rewards, samplers, and rollout behavior are otherwise
unchanged.

## 6. Evaluate

Evaluation runners, experiment configs, result computation, and error-analysis
utilities now live together under `evaluate/`:

```bash
./evaluate/run.sh all
python evaluate/summarize_results.py \
  evaluate/results/browseragent-sft/nq_test_results.jsonl
python evaluate/analyze_failures.py evaluate/results/browseragent-sft
```

For evaluation matrices and resumable queues, see
[`docs/evaluation-experiments.md`](evaluation-experiments.md).
