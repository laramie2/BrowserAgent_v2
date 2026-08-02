# BrowserAgent v2 环境安装说明

本文档整理项目实际使用的两个 Conda 环境：

- `browseragent-v2`：BrowserAgent、`verl-tool`、vLLM 0.11、RL/训练和浏览器工具。
- `swift-sft`：基于 MS-Swift 的 Qwen2.5-VL SFT、LoRA、DeepSpeed 训练和权重合并。

不要合并这两个环境。它们使用不同的 PyTorch/CUDA 组合，FlashAttention、vLLM、DeepSpeed 和 Transformer Engine 都涉及 CUDA ABI，混装后很容易被后续的 `pip install -U` 覆盖。

## 推荐版本

| 环境 | Python | CUDA Toolkit | PyTorch | 关键组件 |
| --- | --- | --- | --- | --- |
| `browseragent-v2` | 3.10 | CUDA 12.8 | PyTorch 2.8.0 / cu128 | vLLM 0.11.0、FlashAttention 2.8.1、Transformer Engine 2.6.0.post1、Megatron Core 0.16.1 |
| `swift-sft` | 3.10 | CUDA 12.6 | PyTorch 2.7.1 / cu126 | MS-Swift 3.12.6、DeepSpeed 0.19.2、Transformers 4.54.1、PEFT 0.17.1、TRL 0.19.1 |

Swift 组合来自本仓库 2026-07-21 的实际训练和 LoRA 合并日志。Swift 默认使用 PyTorch SDPA；不再强制安装 FlashAttention，因为当前成功训练没有依赖它，省去一组容易出错的编译 ABI 约束。

## 新机器前置条件

需要满足：

- Linux x86_64。
- NVIDIA GPU 和可用的 `nvidia-smi`；安装 `browseragent-v2` 建议驱动 570 或更新版本。当前审计机器是 A100、驱动 580.105.08。
- 已安装 Conda。Miniconda、Anaconda 或 Miniforge 均可，但 `conda` 必须在 `PATH` 中。
- 可以访问 conda-forge、PyPI、PyTorch wheel 仓库和 GitHub。
- 仓库及子模块完整：

```bash
git submodule update --init --recursive
```

安装器会把 CUDA Toolkit、`nvcc`、CMake、Ninja 和 GCC/G++ 放进各自 Conda 环境，不要求依赖机器上 `/usr/local/cuda` 的版本。

## 安装

在仓库根目录执行：

```bash
bash env/install_all.sh
```

该命令依次安装两个环境。也可以分别安装：

```bash
bash env/install_browseragent_v2.sh
bash env/install_swift_sft.sh
```

BrowserAgent 需要编译 FlashAttention。默认使用 8 个并行编译任务；内存较小的机器可以降低并发：

```bash
MAX_JOBS=4 bash env/install_browseragent_v2.sh
```

默认情况下，安装器发现同名环境已经存在就会退出，不会在一个旧环境上继续覆盖依赖。优先用新名字试装：

```bash
ENV_NAME=browseragent-v2-new bash env/install_browseragent_v2.sh
ENV_NAME=swift-sft-new bash env/install_swift_sft.sh
```

确认允许删除并重建同名环境时，才使用 `RECREATE=1`：

```bash
RECREATE=1 bash env/install_browseragent_v2.sh
RECREATE=1 bash env/install_swift_sft.sh
```

`RECREATE=1` 会删除对应 `ENV_NAME` 指定的 Conda 环境，环境内未另行保存的内容不能恢复。不要给 `install_all.sh` 设置统一的自定义 `ENV_NAME`，否则两个安装器会争用同一个名称。

## Playwright/Chromium 依赖（默认无需 root）

`install_browseragent_v2.sh` 默认从 conda-forge 安装 Chromium headless shell 所需的
NSPR、NSS、ATK、ALSA、GBM、UDev、X11、字体等共享库，其中显式固定
`libgbm=1.0.7` 以提供 `libgbm.so.1`。随后只下载当前 Playwright 版本对应的 Chromium：

```bash
python -m playwright install chromium
```

安装器还会把以下路径写入该 Conda 环境的持久变量 `LD_LIBRARY_PATH`：

```text
$PYTHON_SITE_PACKAGES/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib
```

首项是 PyTorch cu128 wheel 自带并与 Torch 匹配的 cuDNN；后续路径让浏览器服务和 Ray
actor 找到 Conda 提供的 `libnspr4.so`、`libnss3.so` 和 `libgbm.so.1`。安装器不会再
另外安装版本不同的 Conda cuDNN。修改依赖或环境变量后，必须停止旧的浏览器服务/训练
任务并重新启动；已经运行的 Ray actor 不会刷新环境变量。

如果机器有 root/sudo，并希望同时使用 Playwright 官方系统包安装方式，可以显式执行：

```bash
INSTALL_PLAYWRIGHT_DEPS=1 bash env/install_browseragent_v2.sh
```

无论选择哪种方式，安装结尾都会检查三个关键共享库、动态库搜索路径，并实际启动一次
无头 Chromium。任何检查失败都不会把环境报告为安装成功。

现有环境只缺 `libgbm.so.1` 时，可原地修复而无需重建：

```bash
conda activate browseragent-v2
conda install -y -c conda-forge --strict-channel-priority --freeze-installed libgbm=1.0.7
CUDNN_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cudnn")')"
RUNTIME_LIBS="$CUDNN_ROOT/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib"
conda env config vars set -n browseragent-v2 \
  "CUDNN_HOME=$CUDNN_ROOT" "LD_LIBRARY_PATH=$RUNTIME_LIBS"
conda deactivate
conda activate browseragent-v2
python -m playwright install chromium
python env/verify_env.py browseragent-v2
```

## 验证

安装器结尾会自动执行严格验证，也可以手动复查：

```bash
conda activate browseragent-v2
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDNN_HOME="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cudnn")')"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
python env/verify_env.py browseragent-v2
```

```bash
conda activate swift-sft
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
python env/verify_env.py swift-sft
```

验证内容包括 Python 与关键包版本、`mini_webarena`/reward manager 等核心模块导入、Torch 编译使用的 CUDA、GPU 可见性、Conda Toolkit 精确版本、`nvcc` 是否来自当前环境、实际启动无头 Chromium，以及 `pip check`。

每次安装成功还会在环境根目录生成：

- `$CONDA_PREFIX/conda-explicit.txt`：当前平台的 Conda 精确包 URL 快照。
- `$CONDA_PREFIX/pip-freeze.txt`：最终 Python 包快照。

这两份文件用于审计一次具体安装。如果以后需要逐包复刻同一台机器，可以一并保存；日常跨机器安装仍应使用仓库中的安装器和直接依赖清单。

## Swift 训练依赖

`requirements_swift_sft.txt` 显式包含：

- `deepspeed`：ZeRO-2/ZeRO-3 分布式训练。
- `accelerate`、`datasets`、`peft`、`trl`：训练调度、数据集、LoRA/PEFT 和训练器。
- `qwen-vl-utils`、`decord`、`av`：Qwen-VL 图片和视频数据。`decord=0.6.0` 使用 conda-forge 的原生 Python 3.10 构建，避免 PyPI wheel 内部错误的 CPython 3.6 平台标签。
- `ipywidgets`、`comm`、`ipython`、`traitlets`：MS-Swift 解析出的交互组件及其完整依赖。
- `sentence-transformers`、`scikit-learn`：MS-Swift 解析出的 embedding 组件及其完整依赖。
- `tensorboard`：训练日志。
- `ninja` 与环境内 CUDA/GCC 工具链：按需 JIT 编译 DeepSpeed 扩展。

A100 训练使用 `bfloat16`。仓库训练脚本已经设置 `CUDA_HOME=$CONDA_PREFIX` 和 `--torch_dtype bfloat16`，但 `sft/01_run_sft_1.sh` 仍包含当前机器专用配置。迁移到新机器后，运行前必须检查并修改脚本顶部的：

- `SFT_ROOT`：新机器上的仓库 `sft/` 绝对路径。
- `MODEL_PATH` 和数据集目录：确认模型及数据实际存在。
- `CUDA_VISIBLE_DEVICES`：改成新机器计划使用的 GPU 编号。
- `NPROC_PER_NODE`：与可见 GPU 数量一致。

确认这些配置后再启动：

```bash
conda activate swift-sft
bash sft/01_run_sft_1.sh
```

LoRA 合并示例：

```bash
conda activate swift-sft
bash sft/02_merge_lora.sh
```

## BrowserAgent/verl-tool 使用

```bash
conda activate browseragent-v2
cd verl-tool
```

`verl` 与 `verl-tool` 会以 editable 模式从当前仓库安装。仓库内的 `mini_webarena` 没有独立打包元数据，安装器会在环境中写入 `browseragent_v2_repo.pth`，让 reward manager 从仓库根目录导入它。移动或删除仓库后，这三个 checkout-bound 包都会失效；在新机器上应先把仓库放到最终位置，再运行安装器。

## 为什么没有直接复制旧环境的 pip freeze

2026-07-22 审计时，两个现有环境都已发生手动安装造成的漂移：

- 现有 `browseragent-v2` 是 Torch 2.8.0/cu128、vLLM 0.11.0，但 `numpy==2.2.6` 与 `verl` 声明的 `numpy<2.0.0` 冲突；旧脚本也遗漏了实际存在的完整 CUDA 12.8 编译工具链。
- 旧环境的 FlashAttention 2.8.3 超过 Transformer Engine 2.6.0.post1 明确支持的 2.8.1 上限，因此推荐配置回退到 2.8.1。
- 现有 `swift-sft` 是 Torch 2.7.1/cu126、MS-Swift 3.12.6、DeepSpeed 0.19.2，但环境内 `nvcc` 仍是 CUDA 12.4，并缺少训练日志要求的 `decord`。
- 旧 Swift 脚本先装 Torch 2.4/cu124，再执行无版本上限的 `ms-swift[all] -U`，后一步会重新升级 Torch 和大量依赖，因此脚本声称的版本并不是最终版本。

因此当前 `pip freeze` 只能作为找回人工补装依赖的审计材料，不能原样作为新机器安装锁。本方案固定已实际运行的核心版本，同时修正上述冲突和 CUDA 对齐问题。

## 历史 cu124 方案和独立 vLLM 环境

以下文件保留为历史备选，没有被修改，也不由 `install_all.sh` 调用：

- `install_swift_sft_cu124_pinned.sh`
- `requirements_swift_sft_cu124.txt`
- `swift-sft-cu124-py310.lock.txt`

它们对应 Torch 2.6/cu124、MS-Swift 3.5.0，不是当前推荐训练环境。

`install_vllm_server.sh` 也继续保留，但独立 vLLM 服务环境不属于本次两个环境的默认安装范围。

## 常见问题

### 环境已经存在

使用不同的 `ENV_NAME` 试装，或确认旧环境可以删除后使用 `RECREATE=1`。不要手工在旧环境中反复运行 pip 安装命令。

### `NVCC_PREPEND_FLAGS: unbound variable`

这是部分 Conda CUDA `activate.d` 脚本与 Bash `set -u` 的兼容问题。当前安装器只在 `conda activate` 执行期间临时关闭 `nounset`，激活后立即恢复。若旧版安装器在此处中止，环境只完成了 Conda 创建，Torch 和项目依赖尚未安装；更新仓库后直接重建，不要在残缺环境中手动补包：

```bash
conda deactivate
RECREATE=1 bash env/install_browseragent_v2.sh
```

判断旧环境是否残缺：

```bash
conda activate browseragent-v2
python -c 'import torch; print(torch.__version__)'
```

如果提示 `ModuleNotFoundError: No module named 'torch'`，说明必须重新运行完整安装。

### Transformer Engine 缺少 `nvtx3/nvToolsExt.h`

PyTorch 的 cu128 wheel 会把 NVTX 头文件安装到
`site-packages/nvidia/nvtx/include`，而不是 Conda 常见的 `$CONDA_PREFIX/include`。
Transformer Engine 2.6.0.post1 在 Linux 上从源码构建，如果编译器没有搜索该目录，
就会出现 `fatal error: nvtx3/nvToolsExt.h: No such file or directory`。

当前安装器会在 Torch 安装后检查 NVTX 和 CUDA Toolkit 头文件，并把两个 include
目录显式加入 `CPATH`/`CPLUS_INCLUDE_PATH`。该配置已通过实际构建
`transformer_engine_torch-2.6.0.post1` wheel 验证。旧脚本失败后会留下不完整环境；
更新 `env/` 目录后重建：

```bash
cd /data1/yutao/BrowserAgent_v2
conda deactivate
RECREATE=1 bash env/install_browseragent_v2.sh
```

### `NameError: name 'EncodingFast' is not defined`

这表示 Transformers 初始化时没有检测到一个完整可用的 Tokenizers 安装。常见状态是
`tokenizers` 的模块文件或 `.dist-info` 元数据在中断/重复安装后只剩一部分；此时
`pip list` 仍可能显示版本，但 Transformers 的可用性检查会返回 false。该错误不是模型
或 tokenizer 配置文件造成的。

当前 BrowserAgent requirements 固定已验证的 `transformers==4.57.6` 与
`tokenizers==0.22.2`，统一验证器会直接导入 `tokenizers` 和
`transformers.tokenization_utils_base`，从而在安装结束时捕获这种残缺状态。

已完成其他安装步骤的环境可以原地重新安装这两个包：

```bash
cd /data1/yutao/BrowserAgent_v2
conda activate browseragent-v2
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  transformers==4.57.6 tokenizers==0.22.2
python -c 'from tokenizers import Encoding; from transformers import AutoTokenizer; print("tokenizer stack: ok")'
python env/verify_env.py browseragent-v2
```

若环境此前经历过多次失败安装，推荐同步完整 `env/` 目录后干净重建：

```bash
conda deactivate
RECREATE=1 bash env/install_browseragent_v2.sh
```

### Protobuf 版本错误或 cuDNN `undefined symbol`

BrowserAgent 的 AceCoder/evalplus 依赖链仍需要 `protobuf<6`。TensorBoard 2.21 的生成代码
则要求 Protobuf 6.31.1 或更新版本，因此两者不能共存。当前配置固定参考环境中已经验证的
`tensorboard==2.20.0` 与 `protobuf==5.29.6`，同时补齐 `comm==0.2.3`。

PyTorch 2.8/cu128 wheel 自带 cuDNN 9.10。如果环境同时安装 Conda cuDNN 9.8，并把
`$CONDA_PREFIX/lib` 放在搜索路径前面，动态链接器会把 Pip 的
`libcudnn_heuristic.so.9` 与 Conda 的 `libcudnn_graph.so.9` 混用，最终出现
`ExpandBandMatrixOperation` undefined symbol。当前安装器只使用 PyTorch wheel 的 cuDNN，
并在干净子进程中验证 Transformer Engine 导入。

现有环境可以原地修复：

```bash
cd /data1/yutao/BrowserAgent_v2
conda activate browseragent-v2
python -m pip install --no-cache-dir \
  tensorboard==2.20.0 protobuf==5.29.6 comm==0.2.3

CUDNN_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cudnn")')"
test -f "$CUDNN_ROOT/lib/libcudnn.so.9"
RUNTIME_LIBS="$CUDNN_ROOT/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib"
conda env config vars set -n browseragent-v2 \
  "CUDNN_HOME=$CUDNN_ROOT" "LD_LIBRARY_PATH=$RUNTIME_LIBS"
conda deactivate
conda activate browseragent-v2

python -c 'import transformer_engine; import megatron.core; print("TE/Megatron: ok")'
python env/verify_env.py browseragent-v2
```

修复后必须重新启动训练任务和 Ray worker，让新进程继承更新后的动态库路径。

### Swift 安装出现 `nsight-compute` ChecksumMismatchError

旧版 Swift 安装器使用 NVIDIA channel 的完整 `cuda-toolkit=12.6.3` 元包，因此会下载
训练不需要的 Nsight Compute。若 NVIDIA 仓库文件内容与 repodata 中的 SHA256 不一致，
Conda 会以 `ChecksumMismatchError` 终止。

当前安装器已改用 conda-forge 的 CUDA 12.6 核心开发栈：`cuda-nvcc`、
`cuda-cudart-dev`、`cuda-libraries-dev` 和 `cuda-driver-dev`。它仍支持 DeepSpeed/JIT
扩展编译，但依赖图中不再包含 `nsight-compute`。同步新版 `env/` 后重建失败留下的环境：

```bash
cd /data1/yutao/BrowserAgent_v2
conda deactivate
RECREATE=1 bash env/install_swift_sft.sh
```

新的依赖图不会使用旧的 Nsight 下载缓存，因此不必为了本方案手动清空整个 Conda 缓存。

### Swift 安装末尾仍查询 `cuda-toolkit` 或 `pip check` 报 Decord

如果输出中仍有 `cannot query Conda package cuda-toolkit`，说明只更新了
`install_swift_sft.sh`，但实际运行的 `verify_env.py` 仍是旧版本。当前两个文件共享配置版本
`2026-07-22-swift-sft-v2`；新版安装器会在创建环境前检查版本，不匹配就直接退出。
迁移时必须同步完整的 `env/` 目录。

PyPI 的 `decord==0.6.0` 文件名看似支持 Python 3，但 wheel 内部实际标记为
`cp36-cp36m-manylinux2010_x86_64`，所以在 Python 3.10 下即使能够导入，`pip check`
仍会正确报告 `not supported on this platform`。当前安装器改用 conda-forge 的
`cp310-cp310-linux_x86_64` 构建，并显式补齐 ipywidgets 与 sentence-transformers 的依赖。

对于已经完成主要安装、只在最终验证失败的环境，可以原地修复：

```bash
cd /data1/yutao/BrowserAgent_v2
conda activate swift-sft
python -m pip uninstall -y decord
conda install -y -c conda-forge --strict-channel-priority numpy=1.26.4 decord=0.6.0
python -m pip install --no-cache-dir -r env/requirements_swift_sft.txt
python env/verify_env.py swift-sft
```

### `torch.cuda.is_available()` 为 false

先检查 `nvidia-smi`。如果它失败，这是主机驱动或容器 GPU 映射问题，不是 Conda 包缺失。容器中还需确认启动时暴露了 GPU。

### `nvcc` 与 Torch CUDA 不一致

确认已经激活正确环境，并检查：

```bash
which python
which nvcc
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
nvcc --version
```

两条路径都应位于同一个 `$CONDA_PREFIX` 下。不要把 `/usr/local/cuda/bin` 放在环境的 `bin` 前面。

### FlashAttention 编译失败或内存不足

确认是在 `browseragent-v2` 中、Torch 2.8/cu128 已先安装、`CUDA_HOME=$CONDA_PREFIX`，然后降低编译并发，例如 `MAX_JOBS=2`。Swift 环境不需要为当前训练配置安装 FlashAttention。

### `pip check` 失败

不要直接用 `pip install -U` 修补。先比较：

```bash
python -m pip check
python -m pip freeze --all
```

若核心版本已经变化，通常重建环境比在原环境继续覆盖更可靠。

## 维护配置

升级 CUDA、Torch、vLLM、MS-Swift 或 DeepSpeed 时，必须把对应安装脚本、requirements、`verify_env.py` 和本文版本矩阵一起修改。至少运行：

```bash
python -m unittest discover -s env/tests -p 'test_*.py' -v
bash -n env/install_*.sh
```

不要重新引入 `ms-swift[all] -U` 或其他无版本上限的全环境升级命令。
