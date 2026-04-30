"""
CIFAR-100 SOTA auto-tuning: 30 greedy rounds x 20 min each.
Explores modern backbones + training strategies on RTX 4090.
Constraints: params < 100M, model file < 500MB.

Strategy: each round applies ONE delta on top of current best config.
If val_top1 improves -> keep; else -> discard.
"""

import json, os, subprocess, time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PYTHON = Path(__file__).parent.parent / ".auto" / "bin" / "python"
TRAIN_SCRIPT = Path(__file__).parent / "train.py"
LOG_FILE = Path(__file__).parent / "auto_tune.log"
RESULTS_FILE = Path(__file__).parent / "auto_results.tsv"
RESULTS_JSON = Path(__file__).parent / "results" / "results.json"
FIGURE_FILE = Path(__file__).parent / "results" / "experiment_results.png"
HISTORY_DIR = Path(__file__).parent / "results" / "histories"

for d in (HISTORY_DIR, FIGURE_FILE.parent):
    d.mkdir(parents=True, exist_ok=True)

TIME_BUDGET = 1200  # 20 minutes per round

BASELINE = {
    "BACKBONE": "convnext_base.fb_in22k_ft_in1k",
    "BACKBONE_LR": "5e-4",
    "HEAD_LR": "2e-3",
    "BATCH_SIZE": "128",
    "GRAD_ACCUM": "1",
    "WEIGHT_DECAY": "0.01",
    "LABEL_SMOOTHING": "0.1",
    "WARMUP_RATIO": "0.05",
    "IMAGE_SIZE": "224",
    "RANDAUG_NUM_OPS": "2",
    "RANDAUG_MAGNITUDE": "9",
    "LR_SCHEDULE": "cosine",
    "FREEZE_BACKBONE_STEPS": "0",
    "MIXUP_ALPHA": "0.0",
    "CUTMIX_ALPHA": "0.0",
    "DROP_PATH_RATE": "0.0",
    "USE_EMA": "0",
    "EMA_DECAY": "0.9998",
    "GRAD_CLIP": "0.0",
}

STRATEGY_DELTAS = [
    ("Baseline-ConvNeXtB-22k", {}),
    ("Swin-Base-22k", {"BACKBONE": "swin_base_patch4_window7_224.ms_in22k_ft_in1k"}),
    (
        "EfficientNetV2-M-21k",
        {"BACKBONE": "tf_efficientnetv2_m.in21k_ft_in1k", "BATCH_SIZE": "128"},
    ),
    ("ConvNeXtV2-Base-22k", {"BACKBONE": "convnextv2_base.fcmae_ft_in22k_in1k"}),
    ("CAFormer-B36-22k", {"BACKBONE": "caformer_b36.sail_in22k_ft_in1k"}),
    ("BackboneLR-1e-3", {"BACKBONE_LR": "1e-3", "HEAD_LR": "4e-3"}),
    ("BackboneLR-2e-4", {"BACKBONE_LR": "2e-4", "HEAD_LR": "1e-3"}),
    ("BackboneLR-3e-4", {"BACKBONE_LR": "3e-4", "HEAD_LR": "1.5e-3"}),
    ("WeightDecay-0.05", {"WEIGHT_DECAY": "0.05"}),
    ("WeightDecay-0.005", {"WEIGHT_DECAY": "0.005"}),
    ("LabelSmooth-0.15", {"LABEL_SMOOTHING": "0.15"}),
    ("LabelSmooth-0.05", {"LABEL_SMOOTHING": "0.05"}),
    ("Warmup-10pct", {"WARMUP_RATIO": "0.10"}),
    ("Mixup-0.3", {"MIXUP_ALPHA": "0.3"}),
    ("CutMix-1.0", {"CUTMIX_ALPHA": "1.0"}),
    ("Mixup0.2+CutMix0.5", {"MIXUP_ALPHA": "0.2", "CUTMIX_ALPHA": "0.5"}),
    ("DropPath-0.1", {"DROP_PATH_RATE": "0.1"}),
    ("DropPath-0.2", {"DROP_PATH_RATE": "0.2"}),
    ("EMA", {"USE_EMA": "1", "EMA_DECAY": "0.9998"}),
    ("BatchSize-64-Accum2", {"BATCH_SIZE": "64", "GRAD_ACCUM": "2"}),
    ("BatchSize-256", {"BATCH_SIZE": "256"}),
    ("ImageSize-288", {"IMAGE_SIZE": "288", "BATCH_SIZE": "64", "GRAD_ACCUM": "2"}),
    ("RandAug-Mag12", {"RANDAUG_MAGNITUDE": "12"}),
    ("RandAug-Mag6", {"RANDAUG_MAGNITUDE": "6"}),
    ("RandAug-Ops3", {"RANDAUG_NUM_OPS": "3"}),
    ("GradClip-1.0", {"GRAD_CLIP": "1.0"}),
    ("Warmup-15pct", {"WARMUP_RATIO": "0.15"}),
    ("WeightDecay-0.001", {"WEIGHT_DECAY": "0.001"}),
    ("EMA-0.9995", {"USE_EMA": "1", "EMA_DECAY": "0.9995"}),
    ("LabelSmooth-0.2", {"LABEL_SMOOTHING": "0.2"}),
]
assert len(STRATEGY_DELTAS) == 30, f"Need 30 strategies, got {len(STRATEGY_DELTAS)}"


def run_round(round_idx, params, delta, best_cfg, hist_file):
    name = STRATEGY_DELTAS[round_idx][0]
    total_rounds = len(STRATEGY_DELTAS)
    print(f"\n{'=' * 62}")
    print(f"Round {round_idx + 1:02d}/{total_rounds} — {name}")
    inherited = {k: v for k, v in best_cfg.items() if k not in delta}
    print(f"  Inherited : {' '.join(f'{k}={v}' for k, v in sorted(inherited.items()))}")
    if delta:
        print(f"  Delta     : {' '.join(f'{k}={v}' for k, v in sorted(delta.items()))}")
    else:
        print(f"  Delta     : (none — baseline)")
    print(f"{'=' * 62}", flush=True)

    env = os.environ.copy()
    env.update(params)
    env["TIME_BUDGET"] = str(TIME_BUDGET)
    env["HISTORY_FILE"] = str(hist_file)

    t0 = time.time()
    try:
        subprocess.run(
            [str(PYTHON), str(TRAIN_SCRIPT)],
            env=env,
            stdout=open(LOG_FILE, "w"),
            stderr=subprocess.STDOUT,
            timeout=TIME_BUDGET + 300,
        )
    except subprocess.TimeoutExpired:
        print(f"  !! TIMEOUT after {TIME_BUDGET + 300}s")
        return 0.0, 0.0, "timeout", []
    elapsed = time.time() - t0

    val_top1 = 0.0
    peak_vram = 0.0
    status = "crash"
    history = []

    try:
        log = LOG_FILE.read_text()
        for line in log.splitlines():
            if line.startswith("val_top1:"):
                val_top1 = float(line.split()[1])
            if line.startswith("peak_vram_mb:"):
                peak_vram = float(line.split()[1]) / 1024
        if val_top1 > 0:
            status = "done"
        if hist_file.exists():
            history = json.loads(hist_file.read_text())
    except Exception as e:
        print(f"  Parse error: {e}")

    print(
        f"  => val_top1={val_top1:.4f}  vram={peak_vram:.1f}GB"
        f"  wall={elapsed:.0f}s  status={status}",
        flush=True,
    )
    return val_top1, peak_vram, status, history


def main():
    best_cfg = BASELINE.copy()
    best_acc = 0.0
    results = []
    resume_from = 0

    if RESULTS_JSON.exists():
        try:
            saved = json.loads(RESULTS_JSON.read_text())
            if saved:
                results = saved
                resume_from = len(results)
                for rec in results:
                    if rec["kept"]:
                        best_cfg = rec["config"].copy()
                        best_acc = rec["val_top1"]
                print(f"[RESUME] Loaded {resume_from} completed rounds")
                print(f"[RESUME] best_acc={best_acc:.6f}")
                kept_so_far = [r["name"] for r in results if r["kept"]]
                print(f"[RESUME] Kept: {' -> '.join(kept_so_far) or '(none)'}")
                print(
                    f"[RESUME] Continuing from round {resume_from + 1}...", flush=True
                )
        except Exception as e:
            print(f"[RESUME] Could not load: {e} — starting fresh")

    if resume_from == 0:
        with open(RESULTS_FILE, "w") as f:
            f.write("round\tname\tval_top1\tvram_gb\tkept\tstatus\tdescription\n")

    for i, (name, delta) in enumerate(STRATEGY_DELTAS):
        if i < resume_from:
            continue
        candidate = best_cfg.copy()
        candidate.update(delta)

        hist_file = HISTORY_DIR / f"exp{i:02d}_{name}.json"
        val_top1, peak_vram, status, history = run_round(
            i, candidate, delta, best_cfg, hist_file
        )

        kept = val_top1 > best_acc
        if kept:
            best_cfg = candidate.copy()
            best_acc = val_top1
            kept_names = [r["name"] for r in results if r["kept"]] + [name]
            print(f"  *** KEPT [{name}]  best -> {best_acc:.6f}", flush=True)
            print(f"  Accumulated kept: {' -> '.join(kept_names)}", flush=True)
        else:
            kept_names = [r["name"] for r in results if r["kept"]]
            print(f"  --- DROP [{name}]  best stays {best_acc:.6f}", flush=True)

        rec = {
            "exp": i,
            "name": name,
            "val_top1": float(val_top1),
            "best_so_far": float(best_acc),
            "kept": kept,
            "status": status,
            "vram_gb": round(peak_vram, 1),
            "history": history,
            "config": candidate,
        }
        results.append(rec)

        desc = " ".join(f"{k}={v}" for k, v in delta.items()) or "baseline"
        with open(RESULTS_FILE, "a") as f:
            f.write(
                f"{i + 1}\t{name}\t{val_top1:.6f}\t{peak_vram:.1f}"
                f"\t{kept}\t{status}\t{desc}\n"
            )

        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2, default=str)

        plot_results(results)

    print(f"\n{'=' * 62}")
    print("AUTO-TUNING COMPLETE")
    print(f"Best val_top1 : {best_acc:.6f}")
    print("Best config   :")
    for k, v in best_cfg.items():
        print(f"  {k}={v}")
    print(f"\nResults TSV : {RESULTS_FILE}")
    print(f"Figure      : {FIGURE_FILE}")


def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(22, 12),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.55},
    )

    xs = list(range(len(results)))
    accs = [r["val_top1"] * 100 for r in results]
    bests = [r["best_so_far"] * 100 for r in results]
    colors = ["#27ae60" if r["kept"] else "#e74c3c" for r in results]
    best_val = max(accs) if accs else 0

    bars = ax1.bar(xs, accs, color=colors, alpha=0.85, zorder=3, width=0.6)
    ax1.plot(xs, bests, "b--o", lw=2.2, ms=5, zorder=5, label="Running best")
    ax1.axhline(best_val, color="navy", lw=1.2, ls=":", alpha=0.55)

    for r, bar, acc in zip(results, bars, accs):
        color = "#1a5e35" if r["kept"] else "#7f1010"
        weight = "bold" if r["kept"] else "normal"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.05,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=6,
            color=color,
            fontweight=weight,
            rotation=90,
        )

    ax1.set_xticks(xs)
    ax1.set_xticklabels(
        [f"#{r['exp']:02d}\n{r['name']}" for r in results],
        fontsize=6,
        ha="center",
    )
    ax1.set_ylabel("Val Top-1 (%)", fontsize=11)
    ax1.set_title(
        "CIFAR-100 SOTA Backbone — Greedy Strategy Search (30 rounds x 20 min)\n"
        "green = kept  ·  red = discarded",
        fontsize=12,
    )
    ax1.grid(True, axis="y", alpha=0.3, zorder=0)
    if accs:
        ax1.set_ylim(min(accs) - 1, best_val + 2.5)
    handles = [
        mpatches.Patch(color="#27ae60", label="Strategy kept"),
        mpatches.Patch(color="#e74c3c", label="Strategy discarded"),
        plt.Line2D([0], [0], color="blue", ls="--", marker="o", label="Running best"),
    ]
    ax1.legend(handles=handles, fontsize=9, loc="lower right")

    kept_xs = [r["exp"] for r in results if r["kept"]]
    kept_accs = [r["val_top1"] * 100 for r in results if r["kept"]]
    kept_names = [r["name"] for r in results if r["kept"]]

    if kept_xs:
        ax2.step(kept_xs, kept_accs, where="post", color="#27ae60", lw=2.5, zorder=4)
        ax2.scatter(kept_xs, kept_accs, s=60, color="#27ae60", zorder=5)
        for x, y, name in zip(kept_xs, kept_accs, kept_names):
            ax2.annotate(
                f"{name}\n{y:.2f}%",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#1a5e35",
            )
        ax2.set_ylim(min(kept_accs) - 0.5, max(kept_accs) + 1.5)

    ax2.set_xlim(-0.5, max(len(results) - 0.5, 0.5))
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"#{i}" for i in xs], fontsize=7)
    ax2.set_ylabel("Best Val Top-1 (%)", fontsize=10)
    ax2.set_title("Cumulative accuracy gain — kept strategies only", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved -> {FIGURE_FILE}", flush=True)


if __name__ == "__main__":
    main()
