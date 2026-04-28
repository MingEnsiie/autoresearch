"""
CIFAR-100 autoresearch training script. Single-GPU, single-file.
Uses ImageNet-pretrained backbone fine-tuned on CIFAR-100.
Usage: uv run train.py
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import math
import time

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from prepare import TIME_BUDGET, NUM_CLASSES, CIFAR100_MEAN, CIFAR100_STD
from prepare import evaluate_top1, make_train_loader

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model
BACKBONE = (
    "seresnextaa101d_32x8d.sw_in12k_ft_in1k"  # IN-12k pretrained, then IN-1k finetuned
)
PRETRAINED = True  # use ImageNet pretrained weights

# Training
IMAGE_SIZE = 224  # training crop size
BATCH_SIZE = 128  # restored with gradient checkpointing to save VRAM
LR = 3e-3  # head learning rate (AdamW) - higher for faster head convergence
BACKBONE_LR = 5e-5  # backbone learning rate - lower to preserve IN-12k features
WEIGHT_DECAY = 1e-3  # weight decay (10x higher regularization)
LABEL_SMOOTHING = 0.1  # label smoothing for cross-entropy

# LR schedule (cosine with warmup)
WARMUP_RATIO = 0.05  # fraction of time budget for LR warmup

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# Training augmentation
train_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
)

train_loader = make_train_loader(BATCH_SIZE, train_transform)
train_iter = iter(train_loader)

# Model: ImageNet pretrained backbone, replace head for CIFAR-100
model = timm.create_model(BACKBONE, pretrained=PRETRAINED, num_classes=NUM_CLASSES)
model.set_grad_checkpointing(True)  # trade compute for memory → enables larger BS
model = model.to(device)
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Backbone: {BACKBONE} (pretrained={PRETRAINED}), params: {num_params:.1f}M")
print(f"Time budget: {TIME_BUDGET}s | batch_size: {BATCH_SIZE} | lr: {LR}")

# Optimizer + loss: differential LR (head 10x higher than backbone)
# Access pre-compile model params through model._orig_mod if compiled, else model
_model_ref = model
head_params = (
    list(_model_ref._orig_mod.fc.parameters())
    if hasattr(_model_ref, "_orig_mod")
    else list(_model_ref.fc.parameters())
)
head_param_ids = {id(p) for p in head_params}
backbone_params = [p for p in _model_ref.parameters() if id(p) not in head_param_ids]
optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": LR},
    ],
    weight_decay=WEIGHT_DECAY,
)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    t = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
    return 0.5 * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0

while True:
    model.train()
    torch.cuda.synchronize()
    t0 = time.time()

    try:
        images, labels = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        images, labels = next(train_iter)

    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        outputs = model(images)
        loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
    remaining = max(0, TIME_BUDGET - total_training_time)

    if step % 50 == 0:
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {loss.item():.4f} | lrm: {lrm:.3f} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r log

# ---------------------------------------------------------------------------
# Final eval
# ---------------------------------------------------------------------------

val_top1 = evaluate_top1(model, device)
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_top1:         {val_top1:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params:.1f}")
print(f"backbone:         {BACKBONE}")
print(f"pretrained:       {PRETRAINED}")
