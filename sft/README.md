# Supervised fine-tuning

The SFT workflow has two runtime scripts:

1. `train.sh` trains a LoRA adapter from a prepared dataset.
2. `merge_lora.sh` merges the latest adapter into the model path consumed by
   RL (`RL/models/browseragent-sft` by default).

Dataset download and VTC conversion live in [`dataset/`](dataset/README.md).
Both Bash scripts use project-relative defaults and environment-variable
overrides; see [`docs/command-line-training.md`](../docs/command-line-training.md)
for the full workflow.
