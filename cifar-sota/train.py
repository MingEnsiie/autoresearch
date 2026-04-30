"""
CIFAR-100 training script — flexible backbone with SOTA strategies.
Supports: multiple timm backbones, EMA, gradient accumulation, drop_path,
mixup/cutmix, progressive resizing, cosine/linear schedule.
Hyperparameters injected via environment variables.
Usage: BACKBONE=convnext_base BACKBONE_LR=5e-4 python train.py
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import copy, json, math, random, time
import numpy as np
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from prepare import NUM_CLASSES, CIFAR100_MEAN, CIFAR100_STD
from prepare import evaluate_top1, make_train_loader

TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "1200"))
BACKBONE = os.environ.get("BACKBONE", "convnext_base.fb_in22k_ft_in1k")
PRETRAINED = True

BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "5e-4"))
HEAD_LR = float(os.environ.get("HEAD_LR", "2e-3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.1"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.05"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
RANDAUG_NUM_OPS = int(os.environ.get("RANDAUG_NUM_OPS", "2"))
RANDAUG_MAGNITUDE = int(os.environ.get("RANDAUG_MAGNITUDE", "9"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine")
FREEZE_BACKBONE_STEPS = int(os.environ.get("FREEZE_BACKBONE_STEPS", "0"))
MIXUP_ALPHA = float(os.environ.get("MIXUP_ALPHA", "0.0"))
CUTMIX_ALPHA = float(os.environ.get("CUTMIX_ALPHA", "0.0"))
DROP_PATH_RATE = float(os.environ.get("DROP_PATH_RATE", "0.0"))
USE_EMA = int(os.environ.get("USE_EMA", "0"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.9998"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "0.0"))
HISTORY_FILE = os.environ.get(
    "HISTORY_FILE", str(Path(__file__).parent / "history.json")
)

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
scaler = torch.amp.GradScaler("cuda", enabled=False)

aug_ops = [
    transforms.Resize(
        IMAGE_SIZE + 32, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
]
if RANDAUG_NUM_OPS > 0:
    aug_ops.append(
        transforms.RandAugment(num_ops=RANDAUG_NUM_OPS, magnitude=RANDAUG_MAGNITUDE)
    )
aug_ops.extend(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
)
train_transform = transforms.Compose(aug_ops)
train_loader = make_train_loader(BATCH_SIZE, train_transform)
train_iter = iter(train_loader)

model_kwargs = {"pretrained": PRETRAINED, "num_classes": NUM_CLASSES}
if DROP_PATH_RATE > 0:
    model_kwargs["drop_path_rate"] = DROP_PATH_RATE
model = timm.create_model(BACKBONE, **model_kwargs)
model = model.to(device)
model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {BACKBONE}  params: {num_params:.1f}M")

_ref = model._orig_mod if hasattr(model, "_orig_mod") else model
head_name = None
for name in ["fc", "head", "classifier", "head.fc"]:
    parts = name.split(".")
    obj = _ref
    try:
        for p in parts:
            obj = getattr(obj, p)
        head_name = name
        break
    except AttributeError:
        continue

if head_name:
    parts = head_name.split(".")
    obj = _ref
    for p in parts:
        obj = getattr(obj, p)
    head_params = list(obj.parameters())
else:
    head_params = list(_ref.parameters())[-2:]

head_ids = {id(p) for p in head_params}
backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
for g in optimizer.param_groups:
    g["initial_lr"] = g["lr"]

ema_model = None
if USE_EMA:
    ema_model = copy.deepcopy(model._orig_mod if hasattr(model, "_orig_mod") else model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def update_ema(ema, model, decay):
    src = model._orig_mod if hasattr(model, "_orig_mod") else model
    for ema_p, model_p in zip(ema.parameters(), src.parameters()):
        ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)


def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.size()
    cut = math.sqrt(1 - lam)
    rw, rh = int(W * cut), int(H * cut)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, x2 = max(0, cx - rw // 2), min(W, cx + rw // 2)
    y1, y2 = max(0, cy - rh // 2), min(H, cy + rh // 2)
    mx = x.clone()
    mx[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mx, y, y[idx], lam


def mix_criterion(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)


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
accum_loss = 0.0

while True:
    model.train()

    if step < FREEZE_BACKBONE_STEPS:
        for p in backbone_params:
            p.requires_grad = False
    elif step == FREEZE_BACKBONE_STEPS and FREEZE_BACKBONE_STEPS > 0:
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
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * lrm

    use_mix = False
    if CUTMIX_ALPHA > 0 and random.random() < 0.5:
        images, ya, yb, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
        use_mix = True
    elif MIXUP_ALPHA > 0:
        images, ya, yb, lam = mixup_data(images, labels, MIXUP_ALPHA)
        use_mix = True

    with autocast_ctx:
        outputs = model(images)
        loss = (
            mix_criterion(criterion, outputs, ya, yb, lam)
            if use_mix
            else criterion(outputs, labels)
        )
        loss = loss / GRAD_ACCUM

    loss.backward()
    accum_loss += loss.item()

    if (step + 1) % GRAD_ACCUM == 0:
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema_model is not None:
            update_ema(ema_model, model, EMA_DECAY)
        accum_loss = 0.0

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 10:
        total_training_time += dt

    time_up = step > 10 and total_training_time >= TIME_BUDGET

    if step % 200 == 0:
        pct = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0, TIME_BUDGET - total_training_time)
        print(
            f"\rstep {step:05d} ({pct:.1f}%) | loss={loss.item() * GRAD_ACCUM:.4f}"
            f" | lrm={lrm:.3f} | rem={remaining:.0f}s    ",
            end="",
            flush=True,
        )

    step += 1
    if time_up:
        break

print()

eval_model = ema_model if ema_model is not None else model
val_top1 = evaluate_top1(eval_model, device)
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

Path(HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
with open(HISTORY_FILE, "w") as f:
    json.dump(
        {"elapsed": round(total_training_time, 1), "val_top1": round(val_top1, 6)}, f
    )

save_path = Path(__file__).parent / "best_model.pt"
save_model = (
    ema_model
    if ema_model is not None
    else (model._orig_mod if hasattr(model, "_orig_mod") else model)
)
torch.save(save_model.state_dict(), save_path)

print("---")
print(f"val_top1:         {val_top1:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params:.1f}")
print(f"backbone:         {BACKBONE}")
print(f"pretrained:       {PRETRAINED}")
