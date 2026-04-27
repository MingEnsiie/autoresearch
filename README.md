# autoresearch

![实验进展](progress.png)

让 AI 智能体在无人值守的情况下自主进行语言模型训练实验。智能体修改代码、训练 10 分钟、检查结果是否提升、保留或丢弃修改，如此循环往复。你醒来时便可看到一份完整的实验记录，以及（希望是）更优的模型。

训练代码基于单 GPU 实现，核心思路是：你不直接修改 Python 文件，而是编写 `program.md`——这份 Markdown 文件为 AI 智能体提供上下文、定义研究流程。

## 工作原理

仓库有意保持精简，核心文件只有三个：

- **`prepare.py`** — 固定常量、一次性数据准备（下载训练数据、训练 BPE 分词器）及运行时工具（数据加载、评估）。**不可修改。**
- **`train.py`** — 智能体唯一可编辑的文件。包含完整的 GPT 模型、优化器（Muon + AdamW）和训练循环。架构、超参数、批次大小等均可自由调整。**由智能体迭代修改。**
- **`program.md`** — 智能体的行为指令。定义实验流程、约束条件和记录规范。**由人类维护和迭代。**

训练时间固定为 **10 分钟**（实际训练时钟时间，不含启动和编译）。评估指标为 **val_bpb**（验证集比特每字节），越低越好，且与词表大小无关，可公平比较不同架构。

## 快速开始

**环境要求：** 单张 NVIDIA GPU，Python 3.10+，[uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装 uv 包管理器（若尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖
uv sync

# 3. 下载数据并训练分词器（一次性，约 2 分钟）
uv run prepare.py

# 4. 手动运行一次训练实验（约 10 分钟）
uv run train.py
```

以上命令全部正常运行后，环境即已就绪，可进入自主研究模式。

## 运行智能体

在本仓库目录下启动 Claude Code / Codex 或其他智能体，然后发送如下提示：

```
请阅读 program.md，我们开始一轮新实验！先完成初始化设置。
```

`program.md` 本质上是一份轻量级"智能体运行规程"。

## 项目结构

```
prepare.py              — 固定常量、数据准备、分词器、数据加载器、评估（不可修改）
train.py                — GPT 模型、优化器、训练循环（智能体修改此文件）
program.md              — 智能体行为指令（人类维护此文件）
pyproject.toml          — 项目依赖声明
tests/
  test_train_config.py  — train.py 配置合法性校验（单 GPU 友好检查）
analysis.ipynb          — 实验结果可视化分析
results.tsv             — 实验记录（git 不追踪，由智能体写入）
run.log                 — 最近一次训练的完整输出日志
```

## 当前模型配置

以下为 `train.py` 中的关键超参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `DEPTH` | 12 | Transformer 层数 |
| `ASPECT_RATIO` | 48 | 模型维度 = DEPTH × ASPECT_RATIO（当前 576） |
| `TOTAL_BATCH_SIZE` | 2¹⁹ ≈ 524K tokens | 每步优化器更新的 token 数 |
| `MATRIX_LR` | 0.06 | 矩阵参数学习率（Muon 优化器） |
| `WINDOW_PATTERN` | `SSSL` | 注意力模式：L=全局，S=半上下文滑动窗口 |
| `WARMDOWN_RATIO` | 0.5 | 学习率衰减占总时间的比例 |

## 实验记录摘要

| 提交 | val_bpb | 显存(GB) | 状态 | 说明 |
|------|---------|---------|------|------|
| e2a49f1 | 1.112932 | 11.4 | 保留 | 基线 |
| 6877a44 | 1.103275 | 8.8 | 保留 | ASPECT_RATIO=48，模型维度384，33M参数 |
| f2ed755 | 1.101157 | 8.8 | 保留 | MATRIX_LR=0.05 略有提升 |
| 4f68a0e | **1.100356** | 8.8 | 保留 | MATRIX_LR=0.06 进一步提升（当前最优） |

## 设计决策

- **单文件修改原则。** 智能体只修改 `train.py`，范围可控，diff 可审查。
- **固定时间预算。** 训练始终运行 10 分钟，与具体计算平台无关。不同实验之间（无论模型大小、批次大小、架构如何变化）均可直接比较；同时也意味着 autoresearch 会在该时间预算内为你的平台找到最优模型。但注意，不同计算平台之间的结果不具可比性。
- **自洽封闭。** 除 PyTorch 及少量小型包外无外部依赖，不涉及分布式训练和复杂配置。单卡、单文件、单指标。

## 平台支持

当前需要单张 NVIDIA GPU（在 H100 / 24GB 消费级 GPU 上测试）。在显存受限的设备（如 24GB 卡）上，以下调整可降低显存占用：

1. 降低 `DEPTH`（默认 12），例如改为 8 或 6。
2. 降低 `TOTAL_BATCH_SIZE`，保持为 2 的幂次，例如 `2**17`。
3. 降低 `DEVICE_BATCH_SIZE`，例如从 32 改为 16。
4. 在 `prepare.py` 中，可降低 `MAX_SEQ_LEN`（默认 2048）以减少每步显存。
5. 将 `WINDOW_PATTERN` 改为纯 `"L"`（全局注意力），在较小模型上可能更高效。

如需在 Mac / AMD / Windows 平台运行，可参考以下社区 fork：

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)（macOS）
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)（macOS MLX）
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx)（Windows）
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch)（AMD）

## 许可证

MIT
