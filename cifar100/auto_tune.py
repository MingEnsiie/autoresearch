"""
CIFAR-100 auto-tuning: 20 rounds, 3-min each, resnet50 backbone.
Runs train.py with different hyperparameters injected via env vars.
Usage: python auto_tune.py
"""

import os
import subprocess
import re
import time
from pathlib import Path

PYTHON = Path(__file__).parent.parent / ".auto" / "bin" / "python"
TRAIN_SCRIPT = Path(__file__).parent / "train.py"
LOG_FILE = Path(__file__).parent / "run.log"
RESULTS_FILE = Path(__file__).parent / "auto_results.tsv"

ROUNDS = [
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "5e-5",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "5e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "3e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "5e-4",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "64",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "256",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.0",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.05",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.1",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.0",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.05",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.15",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "192",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "160",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "15",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "5",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.0",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.10",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "0",
    },
    {
        "BACKBONE_LR": "1e-4",
        "HEAD_LR": "1e-3",
        "BATCH_SIZE": "128",
        "WEIGHT_DECAY": "0.01",
        "LABEL_SMOOTHING": "0.1",
        "WARMUP_RATIO": "0.05",
        "IMAGE_SIZE": "224",
        "RANDAUG_NUM_OPS": "2",
        "RANDAUG_MAGNITUDE": "9",
        "LR_SCHEDULE": "cosine",
        "FREEZE_BACKBONE_STEPS": "200",
    },
]

HEADERS = ["round", "val_top1", "memory_gb", "status", "description"]
PARAM_KEYS = [
    "BACKBONE_LR",
    "HEAD_LR",
    "BATCH_SIZE",
    "WEIGHT_DECAY",
    "LABEL_SMOOTHING",
    "WARMUP_RATIO",
    "IMAGE_SIZE",
    "RANDAUG_NUM_OPS",
    "RANDAUG_MAGNITUDE",
    "LR_SCHEDULE",
    "FREEZE_BACKBONE_STEPS",
]


def run_round(round_idx, params):
    env = os.environ.copy()
    env.update(params)
    env["TIME_BUDGET"] = "180"
    desc = " ".join(f"{k}={v}" for k, v in params.items())
    print(f"\n{'=' * 60}")
    print(f"Round {round_idx + 1}/20: {desc}")
    print(f"{'=' * 60}")

    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(PYTHON), str(TRAIN_SCRIPT)],
            env=env,
            stdout=open(LOG_FILE, "w"),
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 300s")
        return 0.0, 0.0, "crash", desc

    elapsed = time.time() - t0

    val_top1 = 0.0
    peak_vram = 0.0
    status = "crash"

    try:
        log_text = LOG_FILE.read_text()
        for line in log_text.splitlines():
            if line.startswith("val_top1:"):
                val_top1 = float(line.split()[1])
            if line.startswith("peak_vram_mb:"):
                peak_vram = float(line.split()[1]) / 1024
    except Exception:
        pass

    if val_top1 > 0:
        status = "done"

    memory_gb = round(peak_vram, 1)
    print(
        f"  Result: val_top1={val_top1:.6f}, memory={memory_gb}GB, time={elapsed:.0f}s, status={status}"
    )
    return val_top1, memory_gb, status, desc


def main():
    with open(RESULTS_FILE, "w") as f:
        f.write("\t".join(HEADERS) + "\n")

    results = []
    best_top1 = 0.0
    best_round = -1
    best_params = None

    for i, params in enumerate(ROUNDS):
        val_top1, memory_gb, status, desc = run_round(i, params)
        results.append((i, val_top1, memory_gb, status, desc))

        with open(RESULTS_FILE, "a") as f:
            f.write(f"{i + 1}\t{val_top1:.6f}\t{memory_gb}\t{status}\t{desc}\n")

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_round = i + 1
            best_params = params.copy()

    print(f"\n\n{'=' * 60}")
    print(f"AUTO-TUNING COMPLETE: 20 rounds finished")
    print(f"{'=' * 60}")
    print(f"\nBest round: {best_round} | val_top1: {best_top1:.6f}")
    print(f"\nBest hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}={v}")
    print(f"\nFull results saved to: {RESULTS_FILE}")
    print(f"\n--- Results Summary ---")
    print(f"{'Round':>5} {'val_top1':>10} {'Mem(GB)':>8} {'Status':>8} Description")
    print("-" * 80)
    for i, val_top1, memory_gb, status, desc in results:
        marker = " *** BEST" if i + 1 == best_round else ""
        print(
            f"{i + 1:>5} {val_top1:>10.6f} {memory_gb:>8.1f} {status:>8} {desc[:50]}{marker}"
        )


if __name__ == "__main__":
    main()
