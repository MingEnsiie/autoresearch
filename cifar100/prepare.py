"""
One-time data preparation for CIFAR-100 autoresearch experiments.
Downloads CIFAR-100 dataset via torchvision.

Usage:
    python prepare.py  # download CIFAR-100

Data is stored in ~/.cache/cifar100autoresearch/.
"""

import os

import torch
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # training time budget in seconds (10 minutes)
NUM_CLASSES = 100
EVAL_IMAGE_SIZE = 224  # fixed evaluation image size

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cifar100autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# CIFAR-100 dataset statistics
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


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
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_train_loader(batch_size, transform, num_workers=4):
    """Training loader — transform is provided by train.py."""
    dataset = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_top1(model, device, batch_size=256):
    """
    Top-1 accuracy on CIFAR-100 test set (10,000 images, 100 classes).
    Fixed evaluation — do not modify.
    Uses fixed EVAL_IMAGE_SIZE so results are comparable across configs.
    """
    model.eval()
    val_loader = make_val_loader(batch_size=batch_size)
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")
    print("Downloading CIFAR-100 train split...")
    torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
    print("Downloading CIFAR-100 test split...")
    torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True)
    print("Done! Ready to train.")
