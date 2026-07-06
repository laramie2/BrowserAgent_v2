# Environment Install Guide

不建议把 `browseragent-v2`、`vllm-server`、`swift-sft` 合并成一个 conda 环境。

主要冲突点：

- `verl-tool/verl` 的 vLLM 约束是 `vllm>=0.8.5,<=0.11.0`，`verl-tool` 根包也限制 `vllm<=0.11.0`。
- 独立推理服务使用 `vllm==0.13.0`，会超过 `verl-tool` 的上限。
- Swift 微调环境固定 `torch==2.4.0/cu124`，而 vLLM 服务环境固定 `torch==2.8.0/cu128`。
- `flash-attn` 和 `transformer-engine` 都和 torch/CUDA ABI 强相关，放在同一个环境里很容易被后续 pip install 覆盖。

当前推荐结构是三个环境：

- `browseragent-v2`：项目主环境，严格服务 `verl-tool` 训练/RL。
- `vllm-server`：OpenAI 兼容推理服务，适合 Qwen2.5-VL-7B-Instruct 这类多模态模型。
- `swift-sft`：Swift SFT/LoRA 微调环境。

## 一键安装

在仓库根目录执行：

```bash
bash env/install_all.sh
```

只安装某一个环境：

```bash
bash env/install_browseragent_v2.sh
bash env/install_vllm_server.sh
bash env/install_swift_sft.sh
```

## 自定义环境名

```bash
ENV_NAME=my-vllm bash env/install_vllm_server.sh
ENV_NAME=my-swift bash env/install_swift_sft.sh
```

## vLLM 启动示例

```bash
conda activate vllm-server
vllm serve /data/yutao/lzt/BrowserAgent_v2/models/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8008 \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --trust-remote-code
```

## Swift 使用

```bash
conda activate swift-sft
bash sft/01_run_sft_1.sh
```

## BrowserAgent/verl-tool 使用

```bash
conda activate browseragent-v2
cd verl-tool
```
