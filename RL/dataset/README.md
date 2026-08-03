# Generated RL datasets

This directory is intentionally empty in Git. Run the public pipeline to
populate it:

1. `python env/prepare_resources.py prepare` downloads BrowserAgent SeedData.
2. `python RL/prepare_seed_data.py` creates NQ and Hotpot seed splits.
3. `python RL/score_difficulty.py` writes rollout scores under
   `RL/filter_results/`.
4. `python RL/build_curriculum.py` creates trainable curriculum stages here.

Downloaded and generated parquet, JSONL, Hugging Face cache, and curriculum
files are ignored by Git.
