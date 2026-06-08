#!/usr/bin/env python
"""
Compare CryoDINO's ported `object_detection_loss` against the original Kaggle one.

CryoDINO:  dinov2/eval/detection_3d/loss.py          (called in detection3d.py via
                                                        loss_fn(cls_map, off_map, labels=labels))
Kaggle:    cryoet/modelling/detection/functional.py  (object_detection_loss)

Both are run on byte-identical inputs in the same process, and the scalar loss +
loss_dict are compared. The loss involves a TaskAlignedAssigner with topk / argmax
operations, so we test across several random seeds and input shapes to be confident
the two implementations are exactly equivalent (not just equal on one lucky draw).

Run:
    conda activate loss_cmp
    python scripts_debugging/compare_object_detection_loss.py
"""
import sys

# --- make both repos importable -------------------------------------------------
CRYODINO_ROOT = "/home/sumin/Documents/cryoSumin/CryoDINO/3DINO"
KAGGLE_ROOT = "/home/sumin/Documents/cryoSumin/Kaggle-2024-CryoET"
sys.path.insert(0, CRYODINO_ROOT)
sys.path.insert(0, KAGGLE_ROOT)

import torch

from dinov2.eval.detection_3d.loss import object_detection_loss as cryodino_loss
from cryoet.modelling.detection.functional import object_detection_loss as kaggle_loss


def make_inputs(seed, B, C, D, H, W, n_objects, stride, device="cpu"):
    """Build a random (logits, offsets, labels, stride) batch.

    labels: [B, N, 5] = (x, y, z, class_id, sigma); padding rows have class_id == -100.
    The center coords are placed inside the decoded anchor grid so the assigner finds
    positive matches (otherwise the loss degenerates to the trivial all-background case).
    """
    g = torch.Generator(device=device).manual_seed(seed)

    logits = torch.randn(B, C, D, H, W, generator=g, device=device)
    offsets = torch.randn(B, 3, D, H, W, generator=g, device=device)

    # decoded grid spans roughly [0, stride*dim]; keep GT centers inside it
    max_xyz = torch.tensor([W * stride, H * stride, D * stride], dtype=torch.float32, device=device)
    labels = torch.full((B, n_objects, 5), -100.0, device=device)
    for b in range(B):
        # vary how many real objects each sample has, including the all-padding case
        n_real = int(torch.randint(0, n_objects + 1, (1,), generator=g, device=device).item())
        for i in range(n_real):
            xyz = torch.rand(3, generator=g, device=device) * max_xyz
            cls = float(torch.randint(0, C, (1,), generator=g, device=device).item())
            sigma = float(torch.rand(1, generator=g, device=device).item() * 3.0 + 1.0)  # [1, 4)
            labels[b, i, :3] = xyz
            labels[b, i, 3] = cls
            labels[b, i, 4] = sigma
    return logits, offsets, labels, stride


def run_case(name, seed, B, C, D, H, W, n_objects, stride):
    logits, offsets, labels, stride = make_inputs(seed, B, C, D, H, W, n_objects, stride)

    # clone so neither call can observe in-place mutations made by the other
    cd_loss, cd_dict = cryodino_loss(logits.clone(), offsets.clone(), strides=stride, labels=labels.clone())
    kg_loss, kg_dict = kaggle_loss(logits.clone(), offsets.clone(), strides=stride, labels=labels.clone())

    cd_loss_v = float(cd_loss)
    kg_loss_v = float(kg_loss)
    abs_diff = abs(cd_loss_v - kg_loss_v)

    keys = ["loss", "cls_loss", "reg_loss", "num_items_in_batch"]
    dict_diffs = {k: abs(cd_dict[k] - kg_dict[k]) for k in keys}
    max_dict_diff = max(dict_diffs.values())

    # bit-exact equality expected (same ops, same order, same dtype)
    exact = (cd_loss_v == kg_loss_v) and all(cd_dict[k] == kg_dict[k] for k in keys)
    status = "EXACT" if exact else ("CLOSE" if abs_diff < 1e-5 and max_dict_diff < 1e-5 else "DIFF")

    print(
        f"[{status:5}] {name:28} seed={seed:<3} shape=B{B}C{C}D{D}H{H}W{W} n={n_objects} stride={stride}\n"
        f"          cryodino loss={cd_loss_v:.10f}  kaggle loss={kg_loss_v:.10f}  |Δ|={abs_diff:.2e}\n"
        f"          dict |Δ| max={max_dict_diff:.2e}  "
        f"(cls {dict_diffs['cls_loss']:.2e}, reg {dict_diffs['reg_loss']:.2e}, "
        f"div {dict_diffs['num_items_in_batch']:.2e})"
    )
    return status


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)

    # (name, seed, B, C, D, H, W, n_objects, stride)
    cases = [
        ("single-obj small",        0, 1, 5, 8, 8, 8,  1, 2),
        ("multi-obj small",         1, 2, 5, 8, 8, 8,  6, 2),
        ("many-obj medium",         2, 2, 5, 10, 12, 12, 12, 2),
        ("single class",            3, 1, 1, 8, 8, 8,  4, 2),
        ("stride 4",                4, 2, 5, 8, 8, 8,  6, 4),
        ("more classes",            5, 2, 7, 8, 10, 10, 8, 2),
        ("possible all-padding",    6, 3, 5, 6, 6, 6,  2, 2),
        ("larger batch",            7, 4, 5, 8, 8, 8,  5, 2),
    ]

    statuses = []
    for case in cases:
        statuses.append(run_case(*case))
        print()

    n_exact = statuses.count("EXACT")
    n_close = statuses.count("CLOSE")
    n_diff = statuses.count("DIFF")
    print("=" * 70)
    print(f"SUMMARY: {n_exact} EXACT, {n_close} CLOSE (<1e-5), {n_diff} DIFF  out of {len(statuses)} cases")
    if n_diff == 0 and n_close == 0:
        print("RESULT: The two implementations produce BIT-IDENTICAL outputs. ✓")
    elif n_diff == 0:
        print("RESULT: Outputs match to <1e-5 (floating-point equivalent). ✓")
    else:
        print("RESULT: Outputs DIFFER on some cases. ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
