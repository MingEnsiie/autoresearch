"""
Data utilities for CIFAR-100 — identical constants and functions as
cifar100/prepare.py so both directories share the same cached dataset.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms

# ── constants (must match cifar100/prepare.py exactly) ────────────────────────
NUM_CLASSES = 100
EVAL_IMAGE_SIZE = 224

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cifar100autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def make_val_loader(batch_size=256, image_size=EVAL_IMAGE_SIZE, num_workers=4):
    """Fixed validation loader — do not modify."""
    transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    dataset = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=False, download=False, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_train_loader(batch_size, transform, num_workers=8):
    """Training loader — transform provided by caller."""
    dataset = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=True, download=False, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


@torch.no_grad()
def evaluate_top1(model, device, batch_size=512):
    """Top-1 accuracy on CIFAR-100 test set (10 000 images, 100 classes)."""
    model.eval()
    val_loader = make_val_loader(batch_size=batch_size)
    correct = total = 0
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx:
            outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total
