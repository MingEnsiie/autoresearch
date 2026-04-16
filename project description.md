# autoresearch 项目汇总

## 项目目的

`autoresearch` 是一个面向自治实验代理的小型 LLM 预训练研究项目。它把真实训练代码压缩到一个可控范围内，让代理在单 GPU 上反复修改 `train.py`、运行固定预算训练、读取评估指标，并根据结果保留或丢弃实验。

项目的核心目标不是提供一个通用训练框架，而是建立一个最小、可比较、可自动迭代的研究闭环：

1. 固定数据、tokenizer、评估指标和训练时间预算。
2. 只开放 `train.py` 作为实验面。
3. 用 `val_bpb` 衡量实验效果，数值越低越好。
4. 让代理持续提出、验证和记录实验假设。

这种设计把研究自由度集中在模型结构、优化器、超参数和训练循环上，同时避免代理通过修改评估、数据或依赖来制造不可比较的结果。

## 核心文件职责

| 文件 | 职责 | 实验代理是否应修改 |
| --- | --- | --- |
| `README.md` | 说明项目背景、快速开始、设计选择和平台限制。 | 否，除非任务明确要求更新文档。 |
| `program.md` | 定义自治研究代理的工作流，包括建分支、跑基线、记录结果、实验循环和回退规则。 | 通常由人类维护；实验运行时按其执行。 |
| `prepare.py` | 固定基础设施：下载数据、训练 tokenizer、构造 dataloader、定义 `TIME_BUDGET`、`MAX_SEQ_LEN` 和 `evaluate_bpb`。 | 否。它是评估和数据边界。 |
| `train.py` | 主要实验面：GPT 模型、优化器、超参数、训练循环、最终评估和日志输出。 | 是。自治实验主要只改这个文件。 |
| `pyproject.toml` / `uv.lock` | 固定依赖和 Python 项目元数据。 | 否。自治实验不安装新包、不改依赖。 |
| `run.log` | 单次训练运行日志，常用于提取指标或排查崩溃。 | 可覆盖生成，不作为实验代码提交。 |
| `results.tsv` | 实验结果记录表，包含 commit、`val_bpb`、显存、状态和说明。 | 可更新，但按 `program.md` 要求保持未跟踪。 |

## 整体流程

### 1. 环境和数据准备

项目使用 `uv` 管理依赖，目标平台是单张 NVIDIA GPU。首次使用时应运行：

```bash
uv sync
uv run prepare.py
```

`prepare.py` 会把数据和 tokenizer 放在 `~/.cache/autoresearch/` 下。数据缓存不属于仓库内容，训练脚本运行前必须存在。

### 2. 建立实验分支

自治实验应在专用分支上运行，例如：

```bash
git checkout -b autoresearch/<tag>
```

分支代表一次连续研究运行。每个实验思路先提交，再训练。如果效果变好，保留提交；如果变差或崩溃，按规则记录后回退。

### 3. 跑基线

第一次实验必须不改代码，直接运行当前 `train.py`，建立 baseline：

```bash
uv run train.py > run.log 2>&1
```

训练结束后从日志提取核心结果：

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

### 4. 修改并运行实验

每轮实验围绕一个明确假设修改 `train.py`，例如模型深度、注意力窗口、batch size、学习率、优化器参数、结构组件或训练调度。修改后提交，再运行训练。

实验循环的基本判断标准：

- `val_bpb` 降低：结果变好，保留提交。
- `val_bpb` 持平或升高：结果没有改善，记录后回退。
- 训练崩溃或 OOM：先看是否是简单错误；若不是，记录为 `crash` 并放弃该思路。

### 5. 记录结果

每轮实验结果写入 `results.tsv`，使用 tab 分隔，列为：

```text
commit	val_bpb	memory_gb	status	description
```

`status` 只使用：

- `keep`：实验改善指标并被保留。
- `discard`：实验可运行但指标未改善。
- `crash`：实验崩溃、OOM 或无法产生有效指标。

## 核心机制

### 固定时间预算

`prepare.py` 定义：

```python
TIME_BUDGET = 300
```

`train.py` 的训练循环按训练阶段 wall clock 统计，目标是固定 5 分钟训练预算。启动、编译和预热前若干步不计入主要训练时间。这个机制让不同实验在同一机器上可比较：不管模型变大还是变小，实验都必须在同一时间预算内竞争。

### 固定评估指标

最终指标是 `val_bpb`，即 validation bits per byte。它由 `prepare.py` 中的 `evaluate_bpb` 计算，基于固定验证 shard、固定 `MAX_SEQ_LEN` 和固定评估 token 数。

`val_bpb` 越低表示模型在验证数据上的压缩效果越好。相比普通 token loss，BPB 对 vocab size 更不敏感，因此更适合比较 tokenizer 固定但模型结构变化的实验。

实验代理不得修改 `evaluate_bpb`，否则结果失去可信度。

### 数据和 tokenizer

`prepare.py` 从 Hugging Face 数据集下载 parquet shard，并固定最后一个 shard 作为验证集：

- 训练数据：除 pinned validation shard 以外的数据 shard。
- 验证数据：`shard_06542.parquet`。
- tokenizer：基于训练数据训练 BPE，词表大小为 `8192`。

tokenizer 和 token byte lookup 会保存到 `~/.cache/autoresearch/tokenizer/`。BPB 评估依赖 token byte lookup 统计目标 token 对应的 UTF-8 字节数。

### dataloader

`make_dataloader` 使用 BOS 对齐和 best-fit packing：

- 每行以 BOS 开始。
- 文档被打包进固定长度序列。
- 尽量减少裁剪。
- 不使用 padding，保持 100% token 利用率。

训练和验证都走同一套 dataloader 机制，但 split 不同。训练 split 排除验证 shard，验证 split 只使用 pinned validation shard。

### 模型结构

`train.py` 定义单文件 GPT：

- RMSNorm 风格归一化。
- Rotary embedding。
- causal self-attention。
- 可按层使用短窗口或长窗口注意力，默认 `WINDOW_PATTERN = "SSSL"`。
- MLP 使用 `relu(x).square()`。
- 部分层使用 value embedding 和 gate 混入 value residual。
- 每层有 residual scaling 参数。

模型大小主要由 `DEPTH`、`ASPECT_RATIO`、`HEAD_DIM` 推导。默认 `DEPTH = 8`，模型维度由深度和 aspect ratio 计算后对齐到 head dimension。

### 注意力实现和编译

`train.py` 根据 CUDA capability 选择实现：

- 支持时使用 FlashAttention 3 kernel。
- 否则回退到 PyTorch SDPA。
- 支持时对模型和优化器步骤使用 `torch.compile`。
- 对不支持的 CUDA capability 跳过 compile 或使用兼容路径。

这部分影响速度、显存和可运行性。实验代理修改注意力相关代码时，需要同时考虑 Hopper 与非 Hopper GPU 的行为。

### 优化器机制

训练使用自定义 `MuonAdamW` 组合优化器：

- AdamW 用于 embedding、value embedding、lm head 和标量参数。
- Muon 用于 transformer 中的矩阵参数。
- Muon 参数按 shape 分组，使用 momentum、正交化和谨慎 weight decay。
- 学习率按模型维度做缩放。

训练过程中还会根据时间进度调整学习率、Muon momentum 和 weight decay。

### 训练循环

训练循环的关键点：

- 使用 `TOTAL_BATCH_SIZE` 和 `DEVICE_BATCH_SIZE * MAX_SEQ_LEN` 推导 gradient accumulation steps。
- 使用 bfloat16 autocast。
- 每个 optimizer step 前累积多个 micro step。
- 训练超过固定预算后停止。
- 若 loss 为 NaN 或大于 100，直接失败退出。
- 训练结束后调用 `evaluate_bpb`，再打印最终摘要。

最终摘要包含：

- `val_bpb`
- `training_seconds`
- `total_seconds`
- `peak_vram_mb`
- `mfu_percent`
- `total_tokens_M`
- `num_steps`
- `num_params_M`
- `depth`

## 结果影响和取舍

### 可比性

可比性依赖固定项不被破坏：

- 固定验证数据。
- 固定 BPB 评估函数。
- 固定训练时间预算。
- 固定 tokenizer 和数据准备逻辑。

任何修改 `prepare.py`、验证集选择、`evaluate_bpb`、`TIME_BUDGET` 或依赖环境的行为，都会让结果难以和历史实验比较。

### 指标收益

实验是否值得保留，主要看 `val_bpb` 是否降低。改动越复杂，越需要更明确的指标收益。`program.md` 的标准是：微小收益如果伴随明显复杂度，通常不值得；如果通过删除代码获得相同或更好结果，则是有价值的简化。

### 显存影响

`peak_vram_mb` 是软约束。允许为了显著改善 `val_bpb` 增加显存，但不应让显存大幅膨胀或频繁 OOM。若实验导致 OOM，应优先判断是参数设置过大还是实现错误。

### 吞吐影响

固定时间预算下，吞吐直接影响总训练 token 数。更大的模型可能单步更强但步数更少；更小的模型可能训练更多 token。实验代理应比较最终 `val_bpb`，而不是只看单个中间指标。

### 维护影响

本项目刻意小而集中。实验代码应保持可读、可回退、可解释。堆叠临时兼容逻辑、隐藏状态或难以解释的技巧，会降低后续代理继续研究的效率。

## 实验代理执行注意事项

1. 自治实验开始前，确认数据和 tokenizer 已存在；缺失时让用户运行 `uv run prepare.py`。
2. 第一次运行必须建立 baseline，不先改代码。
3. 每轮只围绕一个清晰实验假设修改 `train.py`。
4. 修改后先提交，再运行训练，便于保留或回退。
5. 训练命令使用重定向，避免日志刷满上下文：

   ```bash
   uv run train.py > run.log 2>&1
   ```

6. 结果提取优先使用：

   ```bash
   grep "^val_bpb:\|^peak_vram_mb:" run.log
   ```

7. 如果没有指标输出，查看最后错误：

   ```bash
   tail -n 50 run.log
   ```

8. 不修改 `prepare.py`、`evaluate_bpb`、依赖文件或数据缓存逻辑。
9. 不把 `results.tsv` 当作代码提交内容。
10. 指标改善时保留提交；指标未改善或复杂度不值得时回退。

## 适合尝试的实验面

实验代理可以优先从 `train.py` 中的低风险旋钮开始：

- `DEPTH`、`ASPECT_RATIO`、`HEAD_DIM`
- `TOTAL_BATCH_SIZE`、`DEVICE_BATCH_SIZE`
- `EMBEDDING_LR`、`UNEMBEDDING_LR`、`MATRIX_LR`、`SCALAR_LR`
- `WEIGHT_DECAY`、`ADAM_BETAS`
- `WARMUP_RATIO`、`WARMDOWN_RATIO`、`FINAL_LR_FRAC`
- `WINDOW_PATTERN`
- MLP、attention、residual scaling、value embedding 等结构组件

每次实验应记录“改了什么”和“为什么值得尝试”。如果一个实验失败，应把失败原因转化为下一轮更清晰的假设，而不是无目的地堆改动。

## 判断一次实验是否成功

一次成功实验至少满足：

- `uv run train.py` 能完整结束。
- 日志输出有效 `val_bpb`。
- `val_bpb` 低于当前最佳结果。
- 显存增长在可接受范围内。
- 代码复杂度与指标收益匹配。
- 结果已写入 `results.tsv`。

如果只改善了训练 loss、吞吐或 MFU，但最终 `val_bpb` 没有改善，则不能视为主要目标成功。
