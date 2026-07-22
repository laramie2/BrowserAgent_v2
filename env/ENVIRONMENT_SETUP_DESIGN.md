# BrowserAgent v2 环境整理设计

## 目标

在 Linux x86_64 NVIDIA GPU 机器上，用仓库内脚本稳定重建 `browseragent-v2` 与 `swift-sft` 两个 Conda 环境。安装过程必须显式记录 CUDA、PyTorch 和关键训练依赖的版本，避免无边界的 `pip install -U` 再次改变 ABI 组合。

## 已确认的基线

- `browseragent-v2` 使用 Python 3.10、CUDA Toolkit 12.8、PyTorch 2.8.0、vLLM 0.11.0、Transformers 4.57.6、Tokenizers 0.22.2、TensorBoard 2.20.0、Protobuf 5.29.6、FlashAttention 2.8.1。该 Transformers/Tokenizers 组合满足 vLLM 约束并已实际通过 eager tokenizer 导入；TensorBoard/Protobuf 组合兼容 AceCoder 传递依赖中的 `protobuf<6`；FlashAttention 版本不超过 Transformer Engine 2.6.0.post1 在运行时代码中声明的 2.8.1 上限。
- `swift-sft` 使用 Python 3.10、conda-forge CUDA 12.6 核心开发栈、PyTorch 2.7.1+cu126、MS-Swift 3.12.6、DeepSpeed 0.19.2。
- Swift 的这组核心版本已在本仓库 2026-07-21 的训练与 LoRA 合并日志中实际运行。
- Swift 默认使用 PyTorch SDPA，不把 FlashAttention 作为必装训练依赖；这样避免 Torch/CUDA/编译 ABI 不匹配。
- Swift 的 Decord 使用 conda-forge Python 3.10 原生构建，不使用内部仍带 CPython 3.6 标签的 PyPI wheel；MS-Swift 解析出的 widgets 和 sentence-transformers 依赖链显式固定，确保 `pip check` 完整。

## 文件职责

- `install_browseragent_v2.sh`：创建主训练/RL 环境，安装完整 CUDA 12.8 编译工具链、项目 editable 包、vLLM、FlashAttention 和 Transformer Engine。
- `requirements_browseragent_v2.txt`：固定主环境的直接 Python 依赖。
- `install_swift_sft.sh`：创建 SFT 环境，安装 CUDA 12.6 的 nvcc/runtime/development 子包、PyTorch cu126 和训练依赖；不安装会额外拉取 Nsight Compute 的完整 `cuda-toolkit` 元包。
- `requirements_swift_sft.txt`：固定 MS-Swift、DeepSpeed、Transformers、PEFT、TRL、数据与多模态依赖。
- `verify_env.py`：按环境类型检查 Python、关键版本、GPU、Torch CUDA、Conda Toolkit 元数据、环境内 nvcc、核心模块、Chromium 与 pip 依赖一致性。
- `README.md`：提供新机器安装、验证、使用和排错说明。
- `install_swift_sft_cu124_pinned.sh` 及其两份依赖文件：保留为历史备选，不由默认入口调用。

## 安全与失败行为

安装器默认不修改已经存在的同名环境；用户必须显式设置 `RECREATE=1` 才会删除并重建。安装前检查 Linux x86_64、Conda 和 NVIDIA 驱动，安装后必须通过统一验证器。BrowserAgent 安装器还会用环境内 `.pth` 文件暴露仓库根目录的 `mini_webarena`，并把 PyTorch wheel 内的 NVTX/cuDNN 头文件目录及 Conda CUDA Toolkit 头文件目录显式加入编译搜索路径，保证 Transformer Engine 源码构建不依赖隐式激活状态。运行时只使用 PyTorch wheel 自带的匹配 cuDNN，不安装另一套 Conda cuDNN；其库目录固定在 `LD_LIBRARY_PATH` 首位并写入 `CUDNN_HOME`，避免同一进程混用不同的 `libcudnn_graph.so.9` 和 `libcudnn_heuristic.so.9`。Chromium 默认使用 conda-forge 提供的用户态运行库，并固定独立的 `libgbm=1.0.7` 包；安装器把这些环境内库目录持久写入 `LD_LIBRARY_PATH`，使以后启动的 Ray actor 继承。`INSTALL_PLAYWRIGHT_DEPS=1` 保留为有 root/sudo 时的官方系统依赖安装选项，但最终共享库检查和 Chromium 启动测试始终不可跳过。任何阶段失败立即退出，不继续留下一个被宣称为可用的环境。

Swift 安装器与统一验证器共享配置版本标识；安装器在删除或创建环境前检查该标识，避免新安装逻辑配合旧验证规则造成错误结论。

## 验证策略

静态测试检查关键版本固定、危险的无界升级不存在、脚本具备安全重建开关且 Bash 语法有效。运行时验证检查包版本、CUDA 可用性、Toolkit/torch CUDA 系列一致性、工具链属于当前 Conda 环境、Tokenizers 模块及 Transformers eager tokenizer 模块导入、TensorBoard protobuf 生成代码、干净子进程中的 Transformer Engine/cuDNN 加载、核心 reward 模块导入、Chromium 启动与 `pip check`。
