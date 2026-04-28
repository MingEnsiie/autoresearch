"""
CIFAR-100 auto-tuning training script. Single-GPU, single-file.
Uses ImageNet-pretrained resnet50 fine-tuned on CIFAR-100.
Hyperparameters are injected via environment variables by auto_tune.py.
Usage: BACKBONE_LR=1e-4 HEAD_LR=1e-3 python train.py
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import math
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from prepare import NUM_CLASSES, CIFAR100_MEAN, CIFAR100_STD
from prepare import evaluate_top1, make_train_loader

TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "600"))

BACKBONE = "resnet50"
PRETRAINED = True

BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "5e-4"))
HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.1"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.05"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
RANDAUG_NUM_OPS = int(os.environ.get("RANDAUG_NUM_OPS", "2"))
RANDAUG_MAGNITUDE = int(os.environ.get("RANDAUG_MAGNITUDE", "9"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine")
FREEZE_BACKBONE_STEPS = int(os.environ.get("FREEZE_BACKBONE_STEPS", "0"))

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

augmentation_ops = [
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
]
if RANDAUG_NUM_OPS > 0:
    augmentation_ops.append(
        transforms.RandAugment(num_ops=RANDAUG_NUM_OPS, magnitude=RANDAUG_MAGNITUDE)
    )
augmentation_ops.extend(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
)
train_transform = transforms.Compose(augmentation_ops)

train_loader = make_train_loader(BATCH_SIZE, train_transform)
train_iter = iter(train_loader)

model = timm.create_model(BACKBONE, pretrained=PRETRAINED, num_classes=NUM_CLASSES)
model = model.to(device)
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters()) / 1e6

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
        {"params": head_params, "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if LR_SCHEDULE == "linear":
        return max(0.0, (1.0 - progress) / (1.0 - WARMUP_RATIO))
    elif LR_SCHEDULE == "constant":
        return 1.0
    else:
        t = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
        return 0.5 * (1.0 + math.cos(math.pi * t))


t_start_training = time.time()
total_training_time = 0.0
step = 0

while True:
    model.train()

    if step < FREEZE_BACKBONE_STEPS:
        for p in backbone_params:
            p.requires_grad = False
    else:
        for p in backbone_params:
            p.requires_grad = True

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

    if step % 50 == 0:
        pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0, TIME_BUDGET - total_training_time)
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {loss.item():.4f} | lrm: {lrm:.3f} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

val_top1 = evaluate_top1(model, device)
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

save_path = Path(__file__).parent / "best_model.pt"
torch.save(model, save_path)

print("---")
print(f"val_top1:         {val_top1:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params:.1f}")
print(f"backbone:         {BACKBONE}")
print(f"pretrained:       {PRETRAINED}")
