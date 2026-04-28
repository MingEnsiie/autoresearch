# ResNet50 Auto-Tuning: 20-Round 3-Min Search Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automated 20-round hyperparameter search for CIFAR-100 using fixed resnet50 backbone, 3-min training budget per round, reporting the best configuration found.

**Architecture:** A single Python script (`cifar100/auto_tune.py`) that: (1) defines a search space of hyperparameters, (2) iterates 20 rounds of train-evaluate-record, (3) writes results to `cifar100/auto_results.tsv`, (4) prints the best config at the end. Each round modifies `cifar100/train.py` with new hyperparameters, runs training (3-min budget), and extracts val_top1 from the log.

**Tech Stack:** Python, PyTorch, timm, torchvision — all already in the project.

---

### Task 1: Reset train.py to resnet50 baseline with 3-min budget

**Files:**
- Modify: `cifar100/train.py`
- Modify: `cifar100/prepare.py` (read-only reference — do NOT modify)

**Step 1: Rewrite train.py for resnet50 baseline + 3-min budget**

The new `train.py` must:
- Fix `BACKBONE = "resnet50"`, `PRETRAINED = True` (never changed by the auto-tuner)
- Override `TIME_BUDGET` to 180 seconds (3 minutes) via a module-level override since prepare.py sets it to 600
- Support reading hyperparameters from environment variables so the auto-tuner can inject them without rewriting the file each time
- Keep differential LR (backbone vs head), cosine LR schedule with warmup, AdamW optimizer
- Use same augmentation pipeline (RandomCrop, HFlip, RandAugment)

Environment variable interface:
```
BACKBONE_LR   (default 1e-4)
HEAD_LR       (default 1e-3)
BATCH_SIZE    (default 128)
WEIGHT_DECAY  (default 0.01)
LABEL_SMOOTHING (default 0.1)
WARMUP_RATIO  (default 0.05)
IMAGE_SIZE    (default 224)
RANDAUG_NUM_OPS (default 2)
RANDAUG_MAGNITUDE (default 9)
LR_SCHEDULE   (default "cosine", options: "cosine", "linear", "constant")
FREEZE_BACKBONE_EPOCHS (default 0, freeze backbone for N epochs at start)
```

**Step 2: Test that train.py runs**

Run: `cd /home/mingzh/Documents/Workplace/autoresearch && uv run cifar100/train.py > cifar100/run.log 2>&1`
Expected: Completes in ~3 minutes, prints `val_top1` and `peak_vram_mb`

**Step 3: Verify output**

Run: `grep "^val_top1:\|^peak_vram_mb:" cifar100/run.log`
Expected: Two lines with numeric values

---

### Task 2: Create the auto-tuning script

**Files:**
- Create: `cifar100/auto_tune.py`

**Step 1: Write auto_tune.py**

The script must:
1. Define a list of 20 hyperparameter configurations to try (see search space below)
2. For each round:
   a. Set environment variables for the hyperparameters
   b. Run `uv run cifar100/train.py > cifar100/run.log 2>&1`
   c. Parse `val_top1` and `peak_vram_mb` from the log
   d. Record result to `cifar100/auto_results.tsv`
   e. Print round summary
3. After 20 rounds, print the best configuration found
4. Handle crashes (OOM, NaN) gracefully — record as 0.0 and move on

**Search space (20 rounds):**

The search explores these dimensions systematically:
- **Backbone LR**: log scale from 1e-5 to 1e-3
- **Head LR**: log scale from 1e-4 to 1e-2
- **Batch size**: 64, 128, 256
- **Weight decay**: 0.0, 0.01, 0.05, 0.1
- **Label smoothing**: 0.0, 0.05, 0.1, 0.15
- **Warmup ratio**: 0.0, 0.05, 0.10, 0.15
- **Image size**: 160, 192, 224
- **RandAugment magnitude**: 5, 9, 15
- **LR schedule**: cosine, linear
- **Freeze backbone epochs**: 0, 1, 2 (where 1 epoch ≈ 500 steps for CIFAR-100)

Rounds are ordered as a coarse-to-fine search:
- Rounds 1-5: Broad exploration (vary backbone_lr, head_lr, batch_size)
- Rounds 6-10: Regularization exploration (weight_decay, label_smoothing)
- Rounds 11-15: Augmentation & image size exploration
- Rounds 16-18: LR schedule & warmup refinement
- Rounds 19-20: Best-so-far refinement (slight perturbation of best config)

**Step 2: Test the auto-tuner**

Run: `cd /home/mingzh/Documents/Workplace/autoresearch && uv run cifar100/auto_tune.py`
Expected: Runs 20 rounds × ~3 min ≈ 60 min, writes results to TSV

---

### Task 3: Run the full 20-round auto-tuning

**Step 1: Execute auto_tune.py**

Run: `cd /home/mingzh/Documents/Workplace/autoresearch && uv run cifar100/auto_tune.py 2>&1 | tee cifar100/auto_tune.log`

This will take approximately 60 minutes. Monitor progress via the live log output.

**Step 2: Review results**

Run: `cat cifar100/auto_results.tsv`
Expected: 20 rows with val_top1 values, the best highlighted

---

### Task 4: Generate the best model training script

**Files:**
- Modify: `cifar100/train.py` (apply best hyperparameters as hardcoded defaults)

**Step 1: Update train.py with best hyperparameters**

Read `auto_results.tsv` to find the best round, then hardcode its hyperparameters into `train.py` as defaults. Set `TIME_BUDGET` back to 600 for a proper 10-minute final training run.

**Step 2: Verify the final model**

Run: `cd /home/mingzh/Documents/Workplace/autoresearch && uv run cifar100/train.py > cifar100/run.log 2>&1`
Then: `grep "^val_top1:" cifar100/run.log`
Expected: val_top1 at least as good as the best 3-min round (likely better with 10-min budget)

**Step 3: Report results**

Print summary table of all 20 rounds and the final best configuration.
