#!/usr/bin/env python
"""Parse a 3DINO detection fine-tuning log and plot training curves.

Usage:
    python plot_training_curve.py <log_file> [-o output.png]
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# [Iter 100], Train loss: 5.224015712738037
ITER_RE = re.compile(r"^\[Iter (\d+)\],\s*Train loss:\s*([\d.eE+-]+)")
# [Iter 1500] Train loss: 3.4356, Val F4: 0.0000
EVAL_RE = re.compile(r"^\[Iter (\d+)\]\s*Train loss:\s*([\d.eE+-]+),\s*Val F4:\s*([\d.eE+-]+)")


def parse(log_path: Path):
    train_iters, train_loss = [], []
    eval_iters, eval_loss, val_f4 = [], [], []
    for line in log_path.read_text().splitlines():
        m = EVAL_RE.match(line)
        if m:
            eval_iters.append(int(m.group(1)))
            eval_loss.append(float(m.group(2)))
            val_f4.append(float(m.group(3)))
            continue
        m = ITER_RE.match(line)
        if m:
            train_iters.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
    return train_iters, train_loss, eval_iters, eval_loss, val_f4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_file", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    train_iters, train_loss, eval_iters, eval_loss, val_f4 = parse(args.log_file)
    out = args.output or args.log_file.with_suffix(".training_curve.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_iters, train_loss, color="tab:blue", lw=1, alpha=0.6,
             label="Train loss (per 100 iters)")
    if eval_iters:
        ax1.plot(eval_iters, eval_loss, "o-", color="tab:red", lw=1.5,
                 label="Train loss (eval avg)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(eval_iters, val_f4, "o-", color="tab:green", lw=1.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Val F4")
    ax2.set_title("Validation F4")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(args.log_file.name)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")
    print(f"  train points: {len(train_iters)}, eval points: {len(eval_iters)}")
    if eval_loss:
        print(f"  final train loss: {eval_loss[-1]:.4f}, final Val F4: {val_f4[-1]:.4f}")


if __name__ == "__main__":
    main()
