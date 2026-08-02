# BrowserAgent 命令行训练准备设计

## 目标

在 `slim-server` 最新代码上提供适合纯终端服务器的训练准备 CLI，把当前依赖人工编辑绝对路径、查找 checkpoint 和后台合并模型的流程收敛成两个可重复执行的命令。RL 训练本身继续使用现有 `RL/configs/train.yaml`、`RL/scripts/train.sh` 和 `RL/scripts/auto_train.sh`。

目标分支为 `feat/command-line-training-setup`。

## 用户流程

第一步准备公共资源：

```bash
python3 scripts/prepare_training.py prepare
```

需要为现有 Wiki 集群创建更多路径时：

```bash
python3 scripts/prepare_training.py prepare --wiki-copies 4
```

第二步按数据集标识下载并合并 SFT LoRA：

```bash
python3 scripts/prepare_training.py prepare-sft hotpot6500-nq6300-cr1_2-2048
```

第三步由用户按实验需要手工修改 `RL/configs/train.yaml`，然后保持现有启动方式：

```bash
bash RL/scripts/train.sh
# 或
bash RL/scripts/auto_train.sh
```

两个 CLI 子命令均支持 `--dry-run`，用于打印将执行的操作而不下载、解压、复制或合并。

## 非目标

- CLI 不创建、更新或删除 conda 环境。
- CLI 不修改 RL 算法、超参数、采样器、奖励或训练启动逻辑。
- CLI 不替代 `train.yaml`，也不自动开始 RL 训练。
- CLI 不负责从 Hugging Face 下载 RL 数据；RL 数据随 Git 分支提供。
- CLI 不支持同时存在多个 SFT 超参数变体时自动猜测应选哪一个。

## 架构

实现采用 Python 标准库为主的单一 CLI。`scripts/prepare_training.py` 只负责参数解析和用户可读输出；可测试的路径推导、资源检查、命令构建、SFT 匹配、checkpoint 选择、ZIM 合并和模型合并编排放在独立模块中。外部下载和模型导出继续调用项目已经使用的 `hf` 与 `swift` 命令。

CLI 从自身位置推导项目根目录，禁止依赖 `/data/yutao/...`、`/home/nvidia/...` 或调用者当前工作目录。子进程以参数数组执行，不通过 shell 拼接用户输入。

## 资源布局

```text
<project>/
├── benchmark/
│   └── ...                                      # TIGER-Lab/BrowserAgent-SeedData
├── models/
│   └── Qwen2.5-VL-7B-Instruct/                  # Qwen/Qwen2.5-VL-7B-Instruct
├── webarena/webarena_zim/
│   ├── 1/wikipedia_en_all_maxi_2022-05.zim      # 唯一实体文件
│   └── 2..N/wikipedia_en_all_maxi_2022-05.zim   # 可选符号链接
├── wiki_cluster/tools/
│   ├── kiwix-tools_linux-x86_64-3.3.0.tar.gz    # 随 Git 提供
│   └── kiwix-tools_linux-x86_64-3.3.0/
│       └── kiwix-serve                          # prepare 本地解压
├── sft/output/
│   └── Qwen2.5-VL-7B-Instruct-<SFT 变体>/
└── RL/models/
    └── Qwen2.5-VL-7B-Instruct-<SFT 变体>-merged/
```

## `prepare` 子命令

### 前置检查

`prepare` 检查以下条件：

- `hf` 在 `PATH` 中可用；
- 项目目录可写；
- bundled Kiwix 压缩包存在且 SHA-256 为 `cdea8226b479515c9495868dec196de9286cba57bc024df7cd15a83690dfbafc`；
- 当前平台能够运行 bundled Linux x86_64 Kiwix 二进制。

检查不会调用 `env/install_all.sh`，也不会修改任何 conda 环境。Hugging Face 认证仅使用 `HF_TOKEN` 或本机 `hf auth login` 状态；CLI 不接受 token 参数，避免 token 进入 shell history 或进程列表。

### 下载

CLI 通过以下固定资源标识下载：

- `Qwen/Qwen2.5-VL-7B-Instruct` 下载到 `models/Qwen2.5-VL-7B-Instruct/`；
- `TIGER-Lab/BrowserAgent-SeedData` 下载到 `benchmark/`；
- dataset 仓库 `cogito233/WikiEnv` 的 `wikipedia_en_all_maxi_2022-05.zim.part-*` 下载到 Wiki 暂存目录。

下载沿用 `hf download --local-dir` 的缓存与断点续传能力。已经存在的资源先验证，再跳过或继续缺失文件，默认不删除已有目录。

### Kiwix

Kiwix 3.3.0 压缩包以普通 Git 文件提交，部署机无需联网下载该工具。`prepare` 校验固定 SHA-256 后安全解压到 `wiki_cluster/tools/kiwix-tools_linux-x86_64-3.3.0/`。解压必须拒绝绝对路径或包含父目录跳转的归档成员。若目标 `kiwix-serve` 已存在且可执行，则复用它。

### ZIM 合并

下载完成后，CLI 按分片文件名排序，将内容流式写入同目录临时文件。只有当临时文件大小等于所有分片大小之和时，才原子替换为：

```text
webarena/webarena_zim/1/wikipedia_en_all_maxi_2022-05.zim
```

成功后删除本地分片，避免同时保留分片与完整 ZIM。失败时删除临时文件并保留原有完整 ZIM。若完整 ZIM 已存在且非空，默认复用，不重新拼接。

`--wiki-copies` 默认值为 `1`。值大于 `1` 时，为目录 `2..N` 创建指向 `1` 中实体 ZIM 的相对符号链接；这满足 `wiki_cluster/start.sh` 的默认路径约定，同时不重复占用 ZIM 大小的磁盘空间。

### RL 数据

旧仓库 `/home/nvidia/yutao/lzt/BrowserAgent_v2/RL/dataset/` 中当前分支缺少的 15 个文件将作为普通 Git 文件提交到对应的 `RL/dataset/` 路径。已有 41 个文件和当前工作区中用户修改的 parquet 不被覆盖或纳入本功能提交。

`prepare` 仅检查所需 RL 数据路径是否存在；缺失时给出当前 clone 不完整的错误，不尝试从旧机器绝对路径复制，也不联网下载。

## `prepare-sft` 子命令

### 目录匹配

SFT 仓库固定为 `Laramie2/browseragent-sft-lora`。CLI 通过 Hugging Face Hub 文件列表获取仓库顶层目录，并查找名称中包含用户数据集标识的目录。例如输入：

```text
hotpot6500-nq6300-cr1_2-2048
```

可匹配：

```text
Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot6500-nq6300-cr1_2-2048-sft-5e-5lr-freeze_false
```

必须恰好匹配一个顶层目录。零匹配时列出可用候选；多匹配时列出所有冲突项并退出，要求用户提供更具体的数据集标识，不按更新时间或名称自动猜测。

CLI 使用 `hf download` 的 include pattern 只下载唯一匹配目录，并保持仓库相对路径写入 `sft/output/`。

### checkpoint 选择

版本目录必须匹配包含可排序时间戳的 `v*-YYYYMMDD-HHMMSS` 形式。CLI 先按解析后的时间戳选择最新版本目录，再在该目录中按整数编号选择最大的 `checkpoint-N`。缺少合法版本或 checkpoint 时退出并列出已发现目录。

### Swift 执行

CLI 优先使用当前 `PATH` 中的 `swift`。若不存在，则检查 conda 和默认环境 `swift-sft`，通过 `conda run --no-capture-output -n swift-sft` 执行。CLI 不安装 Swift 环境。

合并前必须验证：

- 基座模型目录 `models/Qwen2.5-VL-7B-Instruct/` 存在；
- 所选 checkpoint 目录存在；
- `swift` 可执行；
- 输出父目录 `RL/models/` 可写。

执行等价于：

```bash
swift export \
  --model <project>/models/Qwen2.5-VL-7B-Instruct \
  --adapters <selected-checkpoint> \
  --merge_lora true \
  --output_dir <project>/RL/models/<full-sft-directory>-merged
```

命令在前台运行，stdout 和 stderr 直接显示在终端。成功后沿用现有 `sft/02_merge_lora_rl.sh` 的配置修补语义：从基座模型顶层复制非权重、非索引文件到 merged 目录，并移除 merged 目录中的 `processor_config.json` 和 `chat_template.jinja`。

如果 merged 目录已经包含模型权重与配置文件，则复用并输出结果。若目录存在但不完整，默认报错；只有用户显式传入 `--force` 才允许清理该特定输出目录并重新合并。

### 训练交接

完成后 CLI 打印：

- 选中的 HF 顶层目录；
- 选中的 checkpoint；
- merged 模型绝对路径；
- `SFT_MODEL_NAME_OVERRIDE` 应填写的值，即完整目录名去掉 `Qwen2.5-VL-7B-Instruct-` 前缀。

该命名保证现有 `RL/scripts/train.sh` 的以下路径拼接能够直接找到模型：

```text
RL/models/Qwen2.5-VL-7B-Instruct-${SFT_MODEL_NAME_OVERRIDE}-merged
```

## 幂等性与失败行为

- 所有外部命令失败时，CLI 保留其退出码并打印正在执行的阶段。
- `--dry-run` 不创建目录、不下载、不解压、不删除分片、不创建链接、不运行 Swift。
- 已验证完成的模型、benchmark、Kiwix、ZIM 和 merged 模型重复执行时跳过。
- CLI 不覆盖与目标无关的文件。
- `--force` 只作用于由当前 SFT 标识解析出的单一 merged 输出目录，不接受宽泛目录或 glob。
- Ctrl-C 返回非零状态；下载缓存和未完成的临时文件可供下一次执行恢复或清理。

## 文件边界

- `scripts/prepare_training.py`：命令行入口、参数解析和终端输出。
- `scripts/training_setup.py`：路径模型、命令执行器、资源准备、SFT 匹配、checkpoint 选择、ZIM/Kiwix 操作和 merge 编排。
- `tests/test_training_setup.py`：纯函数与临时文件系统单元测试。
- `tests/test_prepare_training_cli.py`：使用假的 `hf`、`swift` 和临时项目目录验证 CLI 行为。
- `docs/command-line-training.md`：面向实验部署者的简明使用文档。
- `.gitignore`：允许提交固定 Kiwix 压缩包和指定 RL 数据，同时继续忽略解压目录、下载模型、benchmark、ZIM、SFT 输出与 RL merged 模型。
- `RL/dataset/...`：补入旧仓库中缺失的 15 个数据文件。

## 测试策略

测试不下载大型资源，也不要求 GPU：

- 顶层 SFT 目录零匹配、唯一匹配和多匹配；
- 时间戳版本排序与最大 checkpoint 整数排序；
- ZIM 分片排序、流式合并、大小验证、原子替换和重复执行；
- Kiwix checksum 与安全解压路径验证；
- Wiki 单副本默认行为和多副本相对符号链接；
- merged 输出完整性判断与受限 `--force`；
- `hf`/`swift` 子进程命令参数、失败退出码和前台输出；
- `--dry-run` 不产生文件系统副作用；
- CLI `--help`、参数错误和成功摘要。

实现过程遵循测试先行：每项行为先写会因缺少实现而失败的测试，确认失败原因正确，再添加最小实现并运行完整测试集。最终执行 Python 编译检查、全部新增测试、CLI `--help` 和离线 `--dry-run`。

## 安全与可移植性

- 不在代码、配置、日志或命令参数中存储 Hugging Face token。
- 外部命令不使用 `shell=True`。
- 删除操作限制在已解析并验证位于 `RL/models/` 下的单一 merged 目录。
- tar 解压防止路径穿越。
- 项目路径允许包含空格。
- bundled Kiwix 明确限制为 Linux x86_64；不兼容平台直接给出说明，不静默下载其他二进制。
