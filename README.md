# autoresearch

让 AI 智能体在无人值守的情况下自主进行机器学习训练实验。智能体修改代码、运行训练、评估结果、保留或丢弃修改，如此循环往复。

## 子项目

| 目录 | 说明 |
|------|------|
| [`autoGPT/`](autoGPT/README.md) | 原始版本：自主 GPT 语言模型训练，评估指标 val_bpb |
| [`cifar100/`](cifar100/) | CIFAR-100 图像分类自动调参实验 |
| [`cifar-res101/`](cifar-res101/) | CIFAR ResNet-101 图像分类实验 |

## 环境要求

单张 NVIDIA GPU，Python 3.10+，[uv](https://docs.astral.sh/uv/)。

```bash
# 安装所有依赖
uv sync
```

## 快速上手

各子项目均有独立的 `prepare.py`、`train.py` 和 `program.md`，详见对应目录的 README。

```bash
# 以 autoGPT 为例
uv run autoGPT/prepare.py   # 一次性数据准备
uv run autoGPT/train.py     # 运行训练
```

## 项目结构

```
autoGPT/          — 自主 GPT 训练（原始版本）
cifar100/         — CIFAR-100 自动调参
cifar-res101/     — CIFAR ResNet-101 实验
docs/             — 设计文档与计划
pyproject.toml    — 统一依赖声明
uv.lock           — 依赖版本锁定
```

## 许可证

MIT
