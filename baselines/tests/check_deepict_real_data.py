"""Step-by-step run of the per-class DeePiCt scoring on real data, with timing + memory after every step.

Mirrors baseline_common.emit_and_score, but each step is explicit so you can see where the time and
RAM go. Needs a lot of RAM for a full tomogram (unmeasured; a plain run was OOM-killed at ~118GB),
so a memory cap is on by default. Run from the repo root in an env with torch, monai, nibabel,
mrcfile (e.g. conda env `cryoet`):

    python -u baselines/tests/check_deepict_real_data.py
    python -u baselines/tests/check_deepict_real_data.py --only 3          # just the membrane class
    python -u baselines/tests/check_deepict_real_data.py --no-hd           # Dice only, fast
    python -u baselines/tests/check_deepict_real_data.py --crop 500        # centred 500^3 crop

Each step prints:  [elapsed s | +step s] RSS now / peak GB  message

--mem-limit-gb caps the ADDRESS SPACE (virtual memory, not RSS) and is off by default: an allocation past
it raises MemoryError at the exact line instead of the OS OOM-killing the process. Because virtual memory
runs ahead of RSS, a 100 GB cap fired at ~73 GB RSS and made two HD95 calls fail that were not failing on
their own, so only use it with a value well above the RSS you expect (the machine's RAM minus ~10 GB).
Per-class HD95 failures are printed and skipped so the other classes still report; the production
code does not do that.
"""
import argparse
import gc
import os
import resource
import sys
import tempfile
import time

import mrcfile
import numpy as np
import torch
from monai.data import decollate_batch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from baseline_common import (CryoMetrics, hd95_exact, load_volume, remap_12049,  # noqa: E402
                             save_prediction)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


class Tracker:
    def __init__(self):
        self.t0 = self.last = time.time()

    @staticmethod
    def rss_gb():
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
        return float("nan")

    def step(self, msg):
        now = time.time()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"[{now - self.t0:6.0f}s | +{now - self.last:5.0f}s] RSS {self.rss_gb():5.1f} / peak {peak:5.1f} GB  {msg}",
              flush=True)
        self.last = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=os.path.join(
        REPO, "deepict_run/deepict_ds001/out/predictions/model/TS_0003"),
        help="folder holding <class>/probability_map.mrc for each class")
    ap.add_argument("--label", default=os.path.join(REPO, "TS_0003_l.nii.gz"),
                    help="ground-truth label volume (also the affine reference when saving)")
    ap.add_argument("--classes", nargs="+", default=["cytoplasm", "organelles", "membrane"],
                    help="class names in label-value order (value 1, 2, ...)")
    ap.add_argument("--name", default="TS_0003")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--remap-12049", action="store_true")
    ap.add_argument("--gpu-metrics", action="store_true", help="default is CPU metrics, as in the sbatch scripts")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="score only these class indices (0 = background, 1..N); default all")
    ap.add_argument("--no-hd", action="store_true", help="skip HD95, compute Dice only")
    ap.add_argument("--compare-cryometrics", action="store_true",
                    help="also run the old stage-by-stage CryoMetrics path (one-hot, Dice, HD95 via "
                         "torch.quantile) and print it next to the new numbers, to check they agree. "
                         "Much slower and ~5x the memory; it fails on a full tomogram (quantile cap).")
    ap.add_argument("--crop", type=int, default=0,
                    help="score a centred N x N x N crop (N clipped to each axis) of the label and maps; 0 = full "
                         "volume. A crop has far fewer surface voxels, so it will not reproduce the torch.quantile cap.")
    ap.add_argument("--mem-limit-gb", type=float, default=0.0, help="address-space (virtual) cap, 0 = off")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.mem_limit_gb > 0:
        lim = int(args.mem_limit_gb * 1e9)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    tr = Tracker()
    tr.step(f"start (mem cap {args.mem_limit_gb:g} GB, threshold {args.threshold}, "
            f"{'GPU' if args.gpu_metrics else 'CPU'} metrics)")

    # ---- ground truth ------------------------------------------------------------------------
    gt = load_volume(args.label)
    crop = (slice(None),) * 3
    if args.crop:
        crop = tuple(slice((d - min(args.crop, d)) // 2, (d - min(args.crop, d)) // 2 + min(args.crop, d))
                     for d in gt.shape)
        full_shape = gt.shape
        gt = np.ascontiguousarray(gt[crop])   # drop the full-size array
        tr.step(f"cropped {full_shape} -> {gt.shape} (slices {[(c.start, c.stop) for c in crop]})")
    tr.step(f"loaded label {gt.shape} {gt.dtype}, values {np.unique(gt).astype(int).tolist()}")
    if args.remap_12049:
        gt = remap_12049(gt)
        tr.step(f"remapped label (12049), values {np.unique(gt).astype(int).tolist()}")
    gt = gt.astype(np.uint8)
    tr.step("label held as uint8")

    def load_prob(cls):
        with mrcfile.open(os.path.join(args.pred_dir, cls, "probability_map.mrc"), permissive=True) as m:
            return np.array(m.data[crop], dtype=np.float32)

    n_cls = len(args.classes) + 1
    todo = set(args.only) if args.only is not None else set(range(n_cls))
    summary = {}
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="check_deepict_")
    os.makedirs(out_dir, exist_ok=True)

    def score(k, name, pk_bool):
        """Score one class as its own binary problem, with a step line per stage."""
        # new path: pooled-count Dice + whole-volume HD95 with np.percentile (what production uses)
        n_pred, n_gt = int(pk_bool.sum()), int((gt == k).sum())
        tp = int(np.count_nonzero(pk_bool & (gt == k)))
        dice_new = 2.0 * tp / (n_pred + n_gt) if (n_pred + n_gt) else float("nan")
        tr.step(f"Dice (pooled counts) {dice_new:.4f}   [pred {n_pred:,}  gt {n_gt:,}  tp {tp:,}]")
        hd_new = float("nan")
        if not args.no_hd:
            try:
                hd_new = hd95_exact(pk_bool, gt == k)
                tr.step(f"HD95 (np.percentile, whole volume) {hd_new:.4f}")
            except Exception as e:  # noqa: BLE001 -- diagnostic script: report and move on
                tr.step(f"HD95 FAILED for class {k} ({name}): {type(e).__name__}: {e}")
        summary[k] = (name, dice_new, None if not np.isfinite(hd_new) else hd_new)
        if not args.compare_cryometrics:
            return

        # old path, for comparison: CryoMetrics stage by stage
        m = torch.as_tensor(pk_bool, dtype=torch.float32)
        logits = torch.stack([1.0 - m, m], dim=0).unsqueeze(0)      # (1, 2, D, H, W), as score_mask does
        del m
        target = torch.as_tensor(gt == k, dtype=torch.float32)[None, None]
        if args.gpu_metrics and torch.cuda.is_available():
            logits, target = logits.cuda(), target.cuda()
        tr.step(f"logits {tuple(logits.shape)} + target {tuple(target.shape)} built")

        metric = CryoMetrics(num_classes=2)
        pred_list = [metric.post_pred(x) for x in decollate_batch(logits)]
        tgt_list = [metric.post_label(x) for x in decollate_batch(target)]
        del logits, target
        tr.step("post_pred (argmax + one-hot) and post_label (one-hot) done")

        metric.dice_metric_batch(y_pred=pred_list, y=tgt_list)
        dice = [float(d) for d in metric.dice_metric_batch.aggregate()]
        metric.dice_metric_batch.reset()
        tr.step(f"Dice [bg_k, fg_k] = {[round(d, 4) for d in dice]}")

        hd = None
        if not args.no_hd:
            try:
                metric.hd_metric_batch(y_pred=pred_list, y=tgt_list)
                hd = [float(h) for h in metric.hd_metric_batch.aggregate()]
                tr.step(f"HD95 [bg_k, fg_k] = {[round(h, 4) for h in hd]}  "
                        f"(CryoMetrics.__call__ runs HD95 twice per call, so production costs 2x this)")
            except Exception as e:  # noqa: BLE001 -- diagnostic script: report and move on
                tr.step(f"HD95 FAILED for class {k} ({name}): {type(e).__name__}: {e}")
            finally:
                metric.hd_metric_batch.reset()
        tr.step(f"CryoMetrics fg values: Dice {dice[1]:.4f}   HD95 {'n/a' if hd is None else f'{hd[1]:.4f}'}")
        del pred_list, tgt_list, metric
        gc.collect()
        tr.step("freed")

    # ---- single pass: load a class, fill the label map, score it, free it ---------------------
    mask = anyfg = None
    try:
        for k, cls in enumerate(args.classes, 1):
            print(f"\n===== class {k} ({cls}) =====", flush=True)
            prob = load_prob(cls)
            tr.step(f"loaded {cls} probability map {prob.shape}, mean {prob.mean():.4f}")
            fg = prob >= args.threshold
            del prob
            tr.step(f"thresholded {cls}: {int(fg.sum()):,} voxels ({fg.mean() * 100:.2f}%) >= {args.threshold}")
            if mask is None:
                mask = np.zeros(fg.shape, dtype=np.uint8)
                anyfg = np.zeros(fg.shape, dtype=bool)
            mask[fg] = k
            anyfg |= fg
            tr.step(f"filled label map: {cls} -> value {k} (later classes overwrite earlier ones)")
            if k in todo:
                tr.step(f"scoring: pred {int(fg.sum()):,} voxels, gt {int((gt == k).sum()):,} voxels")
                score(k, cls, fg)
            del fg

        if 0 in todo:
            print("\n===== class 0 (bg) =====", flush=True)
            tr.step(f"scoring: pred {int((~anyfg).sum()):,} voxels, gt {int((gt == 0).sum()):,} voxels")
            score(0, "bg", ~anyfg)

    finally:
        if mask is not None:   # saved even if a step raised
            out_path = save_prediction(mask, args.label, os.path.join(out_dir, args.name + (f"_crop{args.crop}" if args.crop else "") + ".nii.gz"))
            v, c = np.unique(mask, return_counts=True)
            tr.step(f"saved prediction {out_path}  counts {dict(zip(v.tolist(), c.tolist()))}")

    # ---- summary -------------------------------------------------------------------------------
    print("\n===== summary (fg channel of each class's binary problem) =====")
    for k, (name, d, h) in summary.items():
        print(f"class {k} {name:12s} Dice {d:.4f}   HD95 {'n/a' if h is None else f'{h:.4f}'}")
    tr.step(f"done, outputs in {out_dir}")


if __name__ == "__main__":
    main()
