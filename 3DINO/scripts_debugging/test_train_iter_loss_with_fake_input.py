#!/usr/bin/env python
"""
Exercise the ACTUAL detection3d.py training path with fake input and verify the
loss matches the original Kaggle implementation.

Unlike `compare_object_detection_loss.py` (which calls the two loss functions
directly), this script reproduces the production code path from
`dinov2/eval/detection3d.py`:

    loss_fn = partial(object_detection_loss, strides=detection_strides)   # line 463/466
    ...
    cls_map, off_map = model(x)                                           # line 189
    loss, loss_dict = loss_fn(cls_map, off_map, labels=labels)            # line 190  <-- STOP HERE

We:
  1. build `loss_fn` via `functools.partial` exactly as detection3d.py does,
  2. define a fake detection model `model(x) -> (cls_map, off_map)`,
  3. build a fake batch dict {"image": ..., "points": ...} like the dataloader yields,
  4. run a device-agnostic copy of `train_iter` that STOPS right after the loss
     line (no backward / optimizer / scheduler), and
  5. compare the CryoDINO loss against the original Kaggle `object_detection_loss`
     on byte-identical (cls_map, off_map, labels) inputs.

This confirms both that the production call wiring works end-to-end on fake input
AND that it yields the same loss as the Kaggle reference.

Run:
    conda activate loss_cmp
    python scripts_debugging/test_train_iter_loss_with_fake_input.py
"""
import sys
from functools import partial

# --- make both repos importable -------------------------------------------------
CRYODINO_ROOT = "/home/sumin/Documents/cryoSumin/CryoDINO/3DINO"
KAGGLE_ROOT = "/home/sumin/Documents/cryoSumin/Kaggle-2024-CryoET"
sys.path.insert(0, CRYODINO_ROOT)
sys.path.insert(0, KAGGLE_ROOT)

import torch
from torch import nn

# the exact loss imported by detection3d.py (dinov2/eval/detection3d.py:14)
from dinov2.eval.detection_3d.loss import object_detection_loss as cryodino_loss
# the original Kaggle reference
from cryoet.modelling.detection.functional import object_detection_loss as kaggle_loss


# ---------------------------------------------------------------------------
# Fake detection model: forward(x) -> (cls_map, off_map), matching the contract
# in detection3d.py (UNETRHead etc. return cls_logits [B,C,Ds,Hs,Ws] and
# off_map [B,3,Ds,Hs,Ws], where Ds = D // stride).
# ---------------------------------------------------------------------------
class FakeDetectionModel(nn.Module):
    def __init__(self, num_classes, stride):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride
        # a trivial conv stack with stride to produce a downsampled feature map,
        # so the path "x -> model -> (cls_map, off_map)" runs for real.
        self.backbone = nn.Conv3d(1, 8, kernel_size=3, stride=stride, padding=1)
        self.cls_head = nn.Conv3d(8, num_classes, kernel_size=1)
        self.off_head = nn.Conv3d(8, 3, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        cls_map = self.cls_head(feat)   # [B, C, Ds, Hs, Ws]
        off_map = self.off_head(feat)   # [B, 3, Ds, Hs, Ws]
        return cls_map, off_map


def make_fake_batch(seed, B, D, H, W, num_classes, n_objects, stride, device):
    """Build a fake dataloader batch: {"image": [B,1,D,H,W], "points": [B,N,5]}.

    points: (x, y, z, class_id, sigma); padding rows have class_id == -100,
    matching the GT convention consumed by object_detection_loss.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    image = torch.randn(B, 1, D, H, W, generator=g, device=device)

    # GT centers live in the decoded grid space ~ [0, stride * feature_dim]
    Ds, Hs, Ws = D // stride, H // stride, W // stride
    max_xyz = torch.tensor([Ws * stride, Hs * stride, Ds * stride], dtype=torch.float32, device=device)

    points = torch.full((B, n_objects, 5), -100.0, device=device)
    for b in range(B):
        n_real = int(torch.randint(0, n_objects + 1, (1,), generator=g, device=device).item())
        for i in range(n_real):
            xyz = torch.rand(3, generator=g, device=device) * max_xyz
            cls = float(torch.randint(0, num_classes, (1,), generator=g, device=device).item())
            sigma = float(torch.rand(1, generator=g, device=device).item() * 3.0 + 1.0)  # [1, 4)
            points[b, i, :3] = xyz
            points[b, i, 3] = cls
            points[b, i, 4] = sigma
    return {"image": image, "points": points}


def train_iter_stop_after_loss(model, batch, loss_fn, device):
    """Device-agnostic copy of detection3d.py:train_iter, STOPPING right after
    the loss line (line 190). No backward / optimizer.step / scheduler.step.

    Original (detection3d.py:186-196):
        def train_iter(model, batch, optimizer, scheduler, scaler, loss_fn):
            x      = batch["image"].cuda()
            labels = batch["points"].cuda()
            cls_map, off_map = model(x)
            loss, loss_dict = loss_fn(cls_map, off_map, labels=labels)   # <-- stop here
            ...
    """
    x = batch["image"].to(device)
    labels = batch["points"].to(device)
    cls_map, off_map = model(x)                                  # line 189
    loss, loss_dict = loss_fn(cls_map, off_map, labels=labels)   # line 190  <-- STOP
    return loss, loss_dict, cls_map, off_map, labels


def run_case(name, seed, B, D, H, W, num_classes, n_objects, stride, device="cpu"):
    torch.manual_seed(seed)
    model = FakeDetectionModel(num_classes, stride).to(device).eval()

    batch = make_fake_batch(seed, B, D, H, W, num_classes, n_objects, stride, device)

    # build loss_fn EXACTLY as detection3d.py does (line 463/466)
    detection_strides = stride
    loss_fn = partial(cryodino_loss, strides=detection_strides)

    # --- production path: stops right after `loss_fn(...)` on line 190 ---------
    with torch.no_grad():
        cd_loss, cd_dict, cls_map, off_map, labels = train_iter_stop_after_loss(
            model, batch, loss_fn, device
        )

    # --- reference path: original Kaggle loss on the SAME model outputs --------
    with torch.no_grad():
        kg_loss, kg_dict = kaggle_loss(
            cls_map.clone(), off_map.clone(), strides=detection_strides, labels=labels.clone()
        )

    cd_v, kg_v = float(cd_loss), float(kg_loss)
    abs_diff = abs(cd_v - kg_v)
    keys = ["loss", "cls_loss", "reg_loss", "num_items_in_batch"]
    dict_diffs = {k: abs(cd_dict[k] - kg_dict[k]) for k in keys}
    max_dict_diff = max(dict_diffs.values())

    exact = (cd_v == kg_v) and all(cd_dict[k] == kg_dict[k] for k in keys)
    finite = torch.isfinite(cd_loss).all().item()
    status = (
        "EXACT" if exact
        else ("CLOSE" if abs_diff < 1e-5 and max_dict_diff < 1e-5 else "DIFF")
    )

    Ds, Hs, Ws = D // stride, H // stride, W // stride
    print(
        f"[{status:5}] {name:24} seed={seed:<3} img=B{B}D{D}H{H}W{W} "
        f"-> map=C{num_classes}D{Ds}H{Hs}W{Ws} stride={stride} finite={finite}\n"
        f"          cryodino(train_iter) loss={cd_v:.10f}  kaggle loss={kg_v:.10f}  |Δ|={abs_diff:.2e}\n"
        f"          dict |Δ| max={max_dict_diff:.2e}  "
        f"(cls {dict_diffs['cls_loss']:.2e}, reg {dict_diffs['reg_loss']:.2e}, "
        f"div {dict_diffs['num_items_in_batch']:.2e})"
    )
    return status, finite


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")

    # (name, seed, B, D, H, W, num_classes, n_objects, stride)
    cases = [
        ("single-obj small",     0, 1, 16, 16, 16, 5, 1, 2),
        ("multi-obj small",      1, 2, 16, 16, 16, 5, 6, 2),
        ("many-obj medium",      2, 2, 20, 24, 24, 5, 12, 2),
        ("single class",         3, 1, 16, 16, 16, 1, 4, 2),
        ("stride 4",             4, 2, 32, 32, 32, 5, 6, 4),
        ("more classes",         5, 2, 16, 20, 20, 7, 8, 2),
        ("possible all-padding", 6, 3, 12, 12, 12, 5, 2, 2),
        ("larger batch",         7, 4, 16, 16, 16, 5, 5, 2),
    ]

    statuses, all_finite = [], True
    for case in cases:
        status, finite = run_case(*case, device=device)
        statuses.append(status)
        all_finite = all_finite and finite
        print()

    n_exact = statuses.count("EXACT")
    n_close = statuses.count("CLOSE")
    n_diff = statuses.count("DIFF")
    print("=" * 70)
    print(f"SUMMARY: {n_exact} EXACT, {n_close} CLOSE (<1e-5), {n_diff} DIFF  out of {len(statuses)} cases")
    print(f"All losses finite: {all_finite}")
    if n_diff == 0 and all_finite:
        print("RESULT: train_iter path runs on fake input AND matches the Kaggle loss. ✓")
        if n_close == 0:
            print("        (bit-identical)")
    else:
        print("RESULT: Mismatch or non-finite loss detected. ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
