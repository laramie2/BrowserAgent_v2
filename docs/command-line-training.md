# BrowserAgent 命令行训练准备

这套 CLI 负责准备模型、benchmark、Wiki ZIM 和 SFT merged 模型。RL 训练仍通过 `RL/configs/train.yaml` 与现有 bash 脚本启动。

## 前置条件

先按 [`env/README.md`](../env/README.md) 创建 `browseragent-v2` 与 `swift-sft` conda 环境。CLI 只检查依赖，不会安装或修改环境。

Hugging Face 认证使用以下任一方式：

```bash
hf auth login
# 或仅在当前 shell 中设置
export HF_TOKEN=hf_your_token
```

不要把 token 写进仓库或命令参数。

## 1. 准备公共资源

先预览操作：

```bash
python3 scripts/prepare_training.py prepare --dry-run
```

开始下载和准备：

```bash
python3 scripts/prepare_training.py prepare
```

该命令会：

- 下载 `Qwen/Qwen2.5-VL-7B-Instruct` 到 `models/`；
- 下载 `TIGER-Lab/BrowserAgent-SeedData` 到 `benchmark/`；
- 校验并解压仓库内置的 Kiwix 3.3.0；
- 下载 `cogito233/WikiEnv` 的 ZIM 分片并原子合并；
- 检查随 Git 提供的 `RL/dataset/` 数据。

默认只保留一份实体 ZIM。需要四个兼容路径时：

```bash
python3 scripts/prepare_training.py prepare --wiki-copies 4
```

额外路径是相对符号链接，不会复制 ZIM 内容。

## 2. 下载并合并 SFT

参数只需填写数据集的唯一标识：

```bash
python3 scripts/prepare_training.py prepare-sft \
  hotpot6500-nq6300-cr1_2-2048
```

CLI 会在 `Laramie2/browseragent-sft-lora` 中查找唯一匹配目录，选择最新时间戳目录中的最大 `checkpoint-N`，然后在前台运行 `swift export`。如有多个匹配目录，命令会列出冲突项并退出。

如果上一次合并留下了不完整目录，检查路径无误后可显式重建：

```bash
python3 scripts/prepare_training.py prepare-sft \
  hotpot6500-nq6300-cr1_2-2048 \
  --force
```

完成后终端会打印 `SFT_MODEL_NAME_OVERRIDE`。把该值填入 `RL/configs/train.yaml`：

```yaml
common:
  env:
    SFT_MODEL_NAME_OVERRIDE: task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false
```

## 3. 启动 RL 训练

按实验需要继续手工修改 `RL/configs/train.yaml`，然后运行：

```bash
bash RL/scripts/train.sh
# 或批量实验
bash RL/scripts/auto_train.sh
```

CLI 不改变现有训练参数、奖励、采样器或启动逻辑。

## Wiki 服务

单 ZIM 默认布局可直接启动：

```bash
./wiki_cluster/start.sh
```

使用四个链接路径时：

```bash
ZIM_COPIES=4 ./wiki_cluster/start.sh
```

检查和停止：

```bash
./wiki_cluster/check.sh
./wiki_cluster/stop.sh
```
