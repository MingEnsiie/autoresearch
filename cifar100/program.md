# cifar100 autoresearch

This is an experiment to have the LLM autonomously research CIFAR-100 image classification.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr28`). The branch `cifar100autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b cifar100autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `cifar100/prepare.py` — fixed constants, data prep, evaluation. Do not modify.
   - `cifar100/train.py` — the file you modify. Model, optimizer, augmentation, training loop.
4. **Verify data exists**: Run `uv run cifar100/prepare.py` to ensure CIFAR-100 is downloaded.
5. **Initialize results.tsv**: Create `cifar100/results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it from the repo root as:

```
uv run cifar100/train.py > cifar100/run.log 2>&1
```

**What you CAN do:**
- Modify `cifar100/train.py` — this is the only file you edit. Everything is fair game: model backbone, optimizer, augmentation, hyperparameters, training loop, batch size, learning rate schedule, etc.

**What you CANNOT do:**
- Modify `cifar100/prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, etc).
- Install new packages outside the existing environment. You can use anything already available: `timm`, `torchvision`, `torch`.
- Modify the evaluation harness. The `evaluate_top1` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_top1 (Top-1 accuracy on CIFAR-100 test set).** Since the time budget is fixed, you don't need to worry about training time. Everything is fair game: change the backbone, optimizer, augmentation pipeline, learning rate schedule, batch size, fine-tuning strategy (freeze/unfreeze layers), etc. The only hard constraint is **VRAM must not exceed 12GB**.

**VRAM** is a hard constraint. If a run OOMs, it's a crash — reduce batch size or model size.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement with lots of complexity is not worth it. Removing something while getting equal or better results is a win.

**The first run**: Your very first run should always establish the baseline — run the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_top1:         0.823500
training_seconds: 600.1
total_seconds:    635.2
peak_vram_mb:     5520.0
num_steps:        3750
num_params_M:     25.6
backbone:         resnet50
pretrained:       True
```

You can extract the key metric from the log file:

```
grep "^val_top1:" cifar100/run.log
```

## Logging results

When an experiment is done, log it to `cifar100/results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_top1	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_top1 achieved (e.g. 0.823500) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 5.4 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_top1	memory_gb	status	description
a1b2c3d	0.823500	5.4	keep	baseline resnet50 pretrained
b2c3d4e	0.831200	5.4	keep	increase LR to 3e-3
c3d4e5f	0.815000	5.4	discard	switch to SGD optimizer
d4e5f6g	0.000000	0.0	crash	batch size 512 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `cifar100autoresearch/apr28`).

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Tune `cifar100/train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run cifar100/train.py > cifar100/run.log 2>&1`
5. Read out results: `grep "^val_top1:\|^peak_vram_mb:" cifar100/run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 cifar100/run.log` to read the stack trace and attempt a fix. If unfixable after a few attempts, give up on that idea.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file — leave it untracked by git)
8. If val_top1 improved (higher), "advance" the branch — keep the git commit
9. If val_top1 is equal or worse, `git reset --hard` back to the previous best commit

**Timeout**: Each experiment takes ~10 minutes total (+ startup/eval overhead). If a run exceeds 20 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, bug, etc.), use judgment: easy fix → fix and re-run. Fundamentally broken → skip, log "crash", move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder — try different backbones, augmentation strategies, LR schedules, fine-tuning strategies (differential LR, layer freezing), mixup/cutmix, test-time augmentation. The loop runs until the human interrupts you, period.
