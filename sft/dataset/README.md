# SFT dataset preparation

This directory contains the two public data-entry points. Downloaded and
generated datasets are ignored by Git; only these scripts and this guide are
tracked.

Authenticate once if the dataset is gated:

```bash
hf auth login
```

Download teacher-generated trajectories. The public repository ID is an input,
not a hard-coded project constant:

```bash
python sft/dataset/download_sft_dataset.py --repo-id <org/dataset>
```

The default destination is `sft/dataset/raw/`. Convert a JSONL trajectory file
to the VTC-rendered Swift format:

```bash
python sft/dataset/prepare_sft_dataset.py \
  --input sft/dataset/raw/generated_teacher.jsonl \
  --dataset-name browseragent-sft \
  --system-msg-path prompt/system_prompt_with_history_info.txt
```

This produces `sft/dataset/browseragent-sft/data.jsonl` and its relative
`images/` directory. Use the same dataset name when launching SFT:

```bash
SFT_DATASET_NAME=browseragent-sft bash sft/train.sh
```
