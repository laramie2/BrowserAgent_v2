# Evaluation

This directory is the single home for evaluation and result computation:

- `run.sh` runs benchmarks and manages local vLLM/tool services.
- `pipeline.py` generates trajectories and benchmark metrics.
- `token_stats.py`, `summarize_results.py`, and
  `summarize_experiments.py` compute result summaries.
- `analyze_failures.py` analyzes failed trajectories.
- `run_experiments.py` and `run_queue.py` run configured experiment
  matrices and resumable queues from `configs/`.

Quick start:

```bash
./evaluate/run.sh all
python evaluate/summarize_results.py evaluate/results/<model>/*_test_results.jsonl
```

Generated logs, results, queue state, and reports stay below `evaluate/` and are
ignored by Git. See [`docs/evaluation-experiments.md`](../docs/evaluation-experiments.md)
for complete options.
