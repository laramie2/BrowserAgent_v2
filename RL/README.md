# RL data and training

The RL data path has three explicit stages:

1. `prepare_seed_data.py` samples NQ/Hotpot training seeds from
   `dataset/BrowserAgent-SeedData/` into `dataset/nq/` and `dataset/hotpot/`.
   It excludes questions listed in `configs/excluded_sft_questions.json` by
   default; pass `--no-exclude` to disable leakage filtering.
   It also creates `dataset/validation_100/data.parquet`, the default RL
   validation split.
2. `score_difficulty.py` runs the merged SFT model K times per seed through the tool
   server and writes rollouts plus `sample_scores.jsonl` difficulty labels.
3. `build_curriculum.py` combines the NQ/Hotpot score files into balanced
   trainable stage directories below `RL/dataset/`.

The training entry points are `scripts/train.sh` and `scripts/run_experiments.sh`.
See the root [`README.md`](../README.md#5-build-rl-data) for commands and service
requirements.
