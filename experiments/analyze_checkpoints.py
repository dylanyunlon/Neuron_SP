#!/usr/bin/env python3
"""
experiments/analyze_checkpoints.py

Scans checkpoints/ directory, extracts training metrics from rank_0.pt files,
plots loss curve / lr schedule / grad norm, saves figure, and prints JSON summary.
"""

import os
import re
import json
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict


# ── paths ────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR = Path("checkpoints")
FIGURES_DIR     = Path("experiments/figures")
OUTPUT_FIG      = FIGURES_DIR / "training_curves.png"


# ── helpers ──────────────────────────────────────────────────────────────────

def discover_steps(checkpoints_dir: Path) -> list[int]:
    """Return sorted list of step numbers found under checkpoints/step_XXXXX/."""
    pattern = str(checkpoints_dir / "step_*")
    dirs = glob.glob(pattern)
    steps = []
    for d in dirs:
        m = re.search(r"step_(\d+)$", d)
        if m and Path(d).is_dir():
            steps.append(int(m.group(1)))
    return sorted(steps)


def extract_metrics(checkpoint_path: Path) -> dict | None:
    """
    Load rank_0.pt and attempt to extract loss, learning_rate, grad_norm.
    Returns a dict or None if the file cannot be parsed.

    Supports several common checkpoint layouts:
      • {"loss": ..., "learning_rate": ..., "grad_norm": ...}
      • {"metrics": {"loss": ..., ...}}
      • {"optimizer_state": ..., "loss": ...}
      • HuggingFace-style trainer_state inside the .pt
      • Top-level tensor attributes
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  [warn] could not load {checkpoint_path}: {exc}")
        return None

    metrics = {}

    def _try_keys(obj, *keys):
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                v = obj[k]
                if isinstance(v, torch.Tensor):
                    v = v.item()
                return float(v)
        return None

    # ── loss ─────────────────────────────────────────────────────────────────
    loss = (
        _try_keys(ckpt, "loss", "train_loss", "current_loss")
        or _try_keys(ckpt.get("metrics", {}), "loss", "train_loss")
        or _try_keys(ckpt.get("trainer_state", {}), "log_history") and None  # handled below
    )

    # HuggingFace trainer_state / log_history list
    if loss is None and isinstance(ckpt.get("trainer_state"), dict):
        log_history = ckpt["trainer_state"].get("log_history", [])
        for entry in reversed(log_history):
            if "loss" in entry:
                loss = float(entry["loss"])
                break

    if loss is None and isinstance(ckpt.get("log_history"), list):
        for entry in reversed(ckpt["log_history"]):
            if "loss" in entry:
                loss = float(entry["loss"])
                break

    metrics["loss"] = loss

    # ── learning_rate ─────────────────────────────────────────────────────────
    lr = (
        _try_keys(ckpt, "learning_rate", "lr", "current_lr")
        or _try_keys(ckpt.get("metrics", {}), "learning_rate", "lr")
    )

    # try optimizer state
    if lr is None and "optimizer_state_dict" in ckpt:
        try:
            lr = float(ckpt["optimizer_state_dict"]["param_groups"][0]["lr"])
        except Exception:
            pass

    if lr is None and "optimizer" in ckpt:
        try:
            lr = float(ckpt["optimizer"]["param_groups"][0]["lr"])
        except Exception:
            pass

    # HuggingFace trainer_state
    if lr is None and isinstance(ckpt.get("trainer_state"), dict):
        lr = _try_keys(ckpt["trainer_state"], "learning_rate")
        if lr is None:
            log_history = ckpt["trainer_state"].get("log_history", [])
            for entry in reversed(log_history):
                if "learning_rate" in entry:
                    lr = float(entry["learning_rate"])
                    break

    metrics["learning_rate"] = lr

    # ── grad_norm ─────────────────────────────────────────────────────────────
    gn = (
        _try_keys(ckpt, "grad_norm", "gradient_norm", "global_grad_norm")
        or _try_keys(ckpt.get("metrics", {}), "grad_norm", "gradient_norm")
    )

    if gn is None and isinstance(ckpt.get("trainer_state"), dict):
        log_history = ckpt["trainer_state"].get("log_history", [])
        for entry in reversed(log_history):
            if "grad_norm" in entry:
                gn = float(entry["grad_norm"])
                break

    metrics["grad_norm"] = gn

    return metrics


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_curves(steps, losses, lrs, grad_norms, output_path: Path):
    fig = plt.figure(figsize=(16, 12), facecolor="#0f1117")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    common_kw = dict(facecolor="#0f1117")
    line_kw   = dict(linewidth=1.6, alpha=0.9)

    # ── palette ──────────────────────────────────────────────────────────────
    C_LOSS = "#4cc9f0"
    C_LR   = "#f72585"
    C_GRAD = "#7bed9f"
    C_GRID = "#2a2d3e"

    def _style_ax(ax, title, ylabel, color):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Step", color="#8b949e", fontsize=10)
        ax.set_ylabel(ylabel, color=color, fontsize=10)
        ax.tick_params(colors="#8b949e", which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        ax.grid(True, color=C_GRID, linewidth=0.6, linestyle="--")

    # ── 1. Loss ───────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0], **common_kw)
    valid = [(s, l) for s, l in zip(steps, losses) if l is not None]
    if valid:
        sx, lx = zip(*valid)
        ax1.plot(sx, lx, color=C_LOSS, **line_kw, label="loss")
        # rolling average (window = min(50, 10% of points))
        win = max(1, min(50, len(lx) // 10))
        if win > 1:
            smoothed = np.convolve(lx, np.ones(win)/win, mode="valid")
            ax1.plot(sx[win-1:], smoothed, color="white", linewidth=1.0,
                     alpha=0.5, linestyle="--", label=f"rolling avg ({win})")
        ax1.legend(facecolor="#0f1117", edgecolor=C_GRID, labelcolor="white", fontsize=9)
    _style_ax(ax1, "Loss Curve", "Loss", C_LOSS)

    # ── 2. Learning Rate ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], **common_kw)
    valid = [(s, l) for s, l in zip(steps, lrs) if l is not None]
    if valid:
        sx, lx = zip(*valid)
        ax2.plot(sx, lx, color=C_LR, **line_kw)
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(-4, 4))
        ax2.yaxis.get_offset_text().set_color("#8b949e")
    _style_ax(ax2, "Learning Rate Schedule", "LR", C_LR)

    # ── 3. Grad Norm ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], **common_kw)
    valid = [(s, g) for s, g in zip(steps, grad_norms) if g is not None]
    if valid:
        sx, gx = zip(*valid)
        ax3.plot(sx, gx, color=C_GRAD, **line_kw, label="grad norm")
        win = max(1, min(50, len(gx) // 10))
        if win > 1:
            smoothed = np.convolve(gx, np.ones(win)/win, mode="valid")
            ax3.plot(sx[win-1:], smoothed, color="white", linewidth=1.0,
                     alpha=0.5, linestyle="--", label=f"rolling avg ({win})")
        ax3.legend(facecolor="#0f1117", edgecolor=C_GRID, labelcolor="white", fontsize=9)
    _style_ax(ax3, "Gradient Norm", "Grad Norm", C_GRAD)

    fig.suptitle("Training Curves", color="white", fontsize=16, fontweight="bold", y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Figure saved → {output_path}")


# ── JSON summary ──────────────────────────────────────────────────────────────

def build_summary(steps, losses, lrs, grad_norms) -> dict:
    """
    Returns:
      - per_1000_steps: avg loss for each 1k-step bucket
      - overall: min loss, final lr, total steps scanned
    """
    bucket: dict[int, list[float]] = defaultdict(list)
    for s, l in zip(steps, losses):
        if l is not None:
            key = (s // 1000) * 1000
            bucket[key].append(l)

    per_1000 = {}
    for k in sorted(bucket):
        vals = bucket[k]
        label = f"step_{k:06d}"
        per_1000[label] = {
            "avg_loss": round(float(np.mean(vals)), 6),
            "min_loss": round(float(np.min(vals)),  6),
            "count":    len(vals),
        }

    valid_losses = [l for l in losses if l is not None]
    valid_lrs    = [l for l in lrs    if l is not None]

    summary = {
        "per_1000_steps": per_1000,
        "overall": {
            "total_steps_scanned": len(steps),
            "min_loss":   round(float(np.min(valid_losses)),  6) if valid_losses else None,
            "final_loss": round(float(valid_losses[-1]),       6) if valid_losses else None,
            "current_lr": float(valid_lrs[-1])                   if valid_lrs    else None,
        },
    }
    return summary


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not CHECKPOINTS_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {CHECKPOINTS_DIR.resolve()}\n"
            "Run this script from the project root, or adjust CHECKPOINTS_DIR."
        )

    steps = discover_steps(CHECKPOINTS_DIR)
    if not steps:
        raise RuntimeError(f"No step_XXXXX/ subdirectories found under {CHECKPOINTS_DIR}")

    print(f"Found {len(steps)} checkpoint steps: {steps[0]} … {steps[-1]}")

    losses, lrs, grad_norms = [], [], []

    for step in steps:
        ckpt_path = CHECKPOINTS_DIR / f"step_{step:05d}" / "rank_0.pt"
        if not ckpt_path.exists():
            # also try zero-padded with more digits
            candidates = list((CHECKPOINTS_DIR / f"step_{step:05d}").glob("rank_0.pt"))
            if not candidates:
                print(f"  [skip] {ckpt_path} not found")
                losses.append(None); lrs.append(None); grad_norms.append(None)
                continue
            ckpt_path = candidates[0]

        print(f"  loading step {step:>6d} … ", end="", flush=True)
        m = extract_metrics(ckpt_path)
        if m is None:
            losses.append(None); lrs.append(None); grad_norms.append(None)
            print("failed")
            continue

        losses.append(m["loss"])
        lrs.append(m["learning_rate"])
        grad_norms.append(m["grad_norm"])
        print(
            f"loss={m['loss']}, lr={m['learning_rate']}, grad_norm={m['grad_norm']}"
        )

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_curves(steps, losses, lrs, grad_norms, OUTPUT_FIG)

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = build_summary(steps, losses, lrs, grad_norms)
    summary_json = json.dumps(summary, indent=2)
    print("\n── Training Summary ──────────────────────────────────────────────")
    print(summary_json)

    summary_path = FIGURES_DIR / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_json)
    print(f"[✓] Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
