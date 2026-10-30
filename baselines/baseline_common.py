"""
Shared plumbing for the MemBrain-seg / DeePiCt baseline inference scripts.

The point of this module is *comparability*: `membrain_inference.py` and `deepict_inference.py`
must emit predictions and metrics in exactly the same shape as
`inference/segmentation3d_inference.py`, so all three methods drop into one results table.

`CryoMetrics` here is a VERBATIM COPY of the class in `segmentation3d_inference.py`. That file is
deliberately left untouched, and it cannot be imported from anyway: its dinov2 imports run at
module scope, so `import segmentation3d_inference` dies with `ModuleNotFoundError: omegaconf` in
the membrain venv, and its metrics.json aggregation is inline inside `main()` rather than being a
function. Copying is therefore the only way to share the metric.

The copy is not left to trust: `test_baseline_eval.py` compares the two class bodies as text and
fails if they diverge. If you change the metric in one place, change it in the other, or that
guard will tell you.

The rest of this module is what the two baselines share and the reference script has no equivalent
of: MRC/NIfTI volume I/O, sample resolution, and the bridge from a binary mask to the logit tensor
`CryoMetrics` expects. Heavier optional deps (nibabel, mrcfile) are imported lazily.
"""

import gc
import glob
import json
import os

import numpy as np
import torch

from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

__all__ = [
    "CryoMetrics", "clear_cuda_memory", "write_metrics_json",
    "strip_ext", "sample_name", "load_volume", "binarize", "save_mrc", "save_prediction",
    "resolve_samples", "score_mask", "record_metrics", "emit_and_score",
    "foreground_values", "split_classes", "remap_12049", "target_class",
]

VOLUME_EXTS = (".nii.gz", ".nii", ".mrc", ".rec", ".pt", ".h5")


# --------------------------------------------------------------------------------------
# volume I/O  (ported from membrain_kit/datalist_utils.py so the repo is self-contained)
# --------------------------------------------------------------------------------------

def strip_ext(fname):
    """'UF5.nii.gz' -> 'UF5' ; 'TE7_0000_patch_0_0_0.pt' -> 'TE7_0000_patch_0_0_0'."""
    b = os.path.basename(fname)
    for ext in VOLUME_EXTS:
        if b.endswith(ext):
            return b[: -len(ext)]
    return os.path.splitext(b)[0]


def sample_name(image_path, label_path=None):
    """Canonical output name for a sample: prefer the label's stem, else strip the _0000 suffix."""
    if label_path:
        return strip_ext(label_path)
    name = strip_ext(image_path)
    return name[: -len("_0000")] if name.endswith("_0000") else name


def load_volume(path):
    """Load a .pt (torch tensor) or .nii/.nii.gz/.mrc/.rec volume as a float32 numpy array."""
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu").numpy().astype(np.float32)
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return np.asarray(nib.load(path).dataobj).astype(np.float32)
    if path.endswith((".mrc", ".rec")):
        import mrcfile
        with mrcfile.open(path, permissive=True) as m:
            return np.asarray(m.data).astype(np.float32)
    raise ValueError(f"unsupported volume format: {path}")


def binarize(label_arr):
    """CZII multi-class {0,1,2,3} -> binary foreground {0,1} (uint8).

    Both baselines are binary, so labels are ALWAYS binarised here. This sidesteps the
    `--dataset-name` footgun in segmentation3d_inference.py, where label binarisation only
    happens when the string "10010" appears in --dataset-name (default: "dataset").
    """
    return (np.asarray(label_arr) > 0).astype(np.uint8)


def target_class(label_arr, value):
    """Extract one raw label value as a binary mask (uint8), ignoring every other class.

    Used to scope membrain-seg (binary-only architecture) to one specific class on a
    multi-class dataset -- e.g. membrane-only on Dataset001 (raw value 3) or Dataset049
    (raw value 2, pre-remap) -- instead of lumping every foreground class together via
    `binarize`. Copied from `baselines/membrain_kit/datalist_utils.py`.
    """
    return (np.asarray(label_arr) == value).astype(np.uint8)


def foreground_values(label_arr):
    """Sorted non-zero label values present in one volume, as ints."""
    return [int(v) for v in np.unique(label_arr) if int(v) != 0]


def split_classes(label_arr, class_values):
    """Multi-label volume -> one binary uint8 mask per class value, in the given order.

    Copied from `baselines/deepict/datalist_utils.py` (same reasoning as the rest of this
    module's duplicated I/O helpers: no cross-kit import).
    """
    return [(label_arr == v).astype(np.uint8) for v in class_values]


def remap_12049(label_arr):
    """Dataset049/EMPIAR-12049 raw {0..5} -> remapped {0,1,2,3}.

    Matches `_remap_12049` in `3DINO/dinov2/eval/segmentation_3d/augmentations.py` and
    `baselines/deepict/datalist_utils.py`'s copy: raw classes 1 (matrix) and 5 (centrioles)
    merge into background, the rest renumber contiguously (2->1, 3->2, 4->3).
    """
    out = np.zeros_like(label_arr)
    out[label_arr == 2] = 1
    out[label_arr == 3] = 2
    out[label_arr == 4] = 3
    return out.astype(label_arr.dtype)


def save_mrc(arr, path, dtype=np.float32):
    """Write a numpy array to .mrc (both baselines read MRC, not NIfTI)."""
    import mrcfile
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(arr, dtype=dtype))


def save_prediction(mask, ref_image_path, out_path):
    """Save a uint8 mask as NIfTI, reusing the reference image's affine/header when it is NIfTI.

    Mirrors segmentation3d_inference.py, which writes `nib.Nifti1Image(pred, affine=nib_img.affine,
    header=nib_img.header)`. For .mrc/.pt inputs there is no affine to inherit, so identity is used.
    """
    import nibabel as nib
    mask = np.ascontiguousarray(np.asarray(mask), dtype=np.uint8)
    if ref_image_path.endswith((".nii", ".nii.gz")):
        ref = nib.load(ref_image_path)
        img = nib.Nifti1Image(mask, affine=ref.affine, header=ref.header)
        img.set_data_dtype(np.uint8)
    else:
        img = nib.Nifti1Image(mask, affine=np.eye(4, dtype=np.float32))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    nib.save(img, out_path)
    return out_path


# --------------------------------------------------------------------------------------
# sample resolution  (same two input modes as segmentation3d_inference.py)
# --------------------------------------------------------------------------------------

def resolve_samples(datalist=None, input_dir=None, label_dir=None, split="test"):
    """Return [(image_path, label_path_or_None)].

    Mode 1 (--datalist): a MONAI-style datalist JSON; uses the `split` key (default "test").
                         A top-level JSON *list* is also accepted (the membrain kit's test_list.json).
    Mode 2 (--input-dir): a directory of volumes, optionally matched against --label-dir by
                          filename with the `_0000` image suffix stripped.
    """
    samples = []
    if datalist:
        with open(datalist) as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else data.get(split, [])
        if not entries:
            raise SystemExit(f"No '{split}' entries found in {datalist}")
        for entry in entries:
            img_path = entry["image"]
            label_path = entry.get("label") or None
            if label_path and not os.path.exists(label_path):
                print(f"  Warning: label not found, skipping metrics for {label_path}")
                label_path = None
            samples.append((img_path, label_path))
    elif input_dir:
        files = []
        for ext in ("*.nii.gz", "*.nii", "*.mrc", "*.rec"):
            files.extend(glob.glob(os.path.join(input_dir, ext)))
        for img_path in sorted(files):
            label_path = None
            if label_dir:
                stem = sample_name(img_path)
                for ext in (".nii.gz", ".nii", ".mrc", ".rec"):
                    cand = os.path.join(label_dir, stem + ext)
                    if os.path.exists(cand):
                        label_path = cand
                        break
            samples.append((img_path, label_path))
    else:
        raise SystemExit("Error: must provide either --datalist or --input-dir")

    if not samples:
        raise SystemExit("No images found")
    return samples


# --------------------------------------------------------------------------------------
# metrics
#
# CryoMetrics below is copied VERBATIM from segmentation3d_inference.py. Do not "improve" it here:
# the whole point is that CryoDINO and the two baselines compute the identical metric, and
# test_baseline_eval.py fails if these two definitions stop matching character for character.
# ---------------------------------------------------------------------------------------

class CryoMetrics:

    def __init__(self, num_classes):
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.dice_metric       = DiceMetric(include_background=True, reduction="mean",       get_not_nans=False)
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
        self.hd_metric         = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
        self.hd_metric_batch   = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    def __call__(self, pred, target):
        target_list = decollate_batch(target)
        target_list = [self.post_label(t) for t in target_list]
        pred_list   = decollate_batch(pred)
        pred_list   = [self.post_pred(p) for p in pred_list]

        self.dice_metric(y_pred=pred_list, y=target_list)
        self.dice_metric_batch(y_pred=pred_list, y=target_list)
        self.hd_metric(y_pred=pred_list, y=target_list)
        self.hd_metric_batch(y_pred=pred_list, y=target_list)

        avg_dice   = self.dice_metric.aggregate().item()
        class_dice = [d.item() for d in self.dice_metric_batch.aggregate()]
        avg_hd     = self.hd_metric.aggregate().item()
        class_hd   = [d.item() for d in self.hd_metric_batch.aggregate()]

        self.dice_metric.reset()
        self.dice_metric_batch.reset()
        self.hd_metric.reset()
        self.hd_metric_batch.reset()

        return avg_dice, class_dice, avg_hd, class_hd

def clear_cuda_memory():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"Failed to clear CUDA memory: {e}")


def write_metrics_json(results, output_dir, extra=None):
    """Aggregate per-image results and write <output_dir>/metrics.json.

    Same schema segmentation3d_inference.py emits (it builds this inline in main()), so all three
    methods land in one table. `extra` adds top-level provenance keys.
    """
    if not results:
        print("\nNo metrics computed (no labels provided or no matches found)")
        return None

    all_avg_dice = np.mean([r['avg_dice'] for r in results.values()])
    all_avg_hd   = np.mean([r['avg_hd95'] for r in results.values()])
    num_cls = len(next(iter(results.values()))['per_class_dice'])
    all_per_cls_dice = [
        np.mean([r['per_class_dice'][c] for r in results.values()])
        for c in range(num_cls)
    ]
    all_per_cls_hd = [
        np.mean([r['per_class_hd95'][c] for r in results.values()])
        for c in range(num_cls)
    ]
    output = {
        'per_image': results,
        'overall_avg_dice':       float(all_avg_dice),
        'overall_per_class_dice': [float(d) for d in all_per_cls_dice],
        'overall_avg_hd95':       float(all_avg_hd),
        'overall_per_class_hd95': [float(h) for h in all_per_cls_hd],
    }
    if extra:
        output.update(extra)

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nOverall Dice: {all_avg_dice:.4f}, Per-class: {all_per_cls_dice}")
    print(f"Overall HD95: {all_avg_hd:.4f}, Per-class: {all_per_cls_hd}")
    print(f"Metrics saved to: {metrics_path}")
    return metrics_path


# ---------------------------------------------------------------------------------------
# mask -> logits bridge
# --------------------------------------------------------------------------------------

def score_mask(metric, pred_mask, gt_mask, cpu_metrics=False):
    """Score a *binary mask* (not logits) against a binary ground truth with `metric`.

    The baselines emit a hard mask / a probability map, whereas CryoMetrics.__call__ expects the
    argmax-able logit tensor CryoDINO produces. Lifting the mask to `[1 - m, m]` makes the argmax
    reproduce the mask exactly, so the same code path -- and therefore the same numbers -- is used
    for all three methods.
    """
    pred_mask = np.asarray(pred_mask)
    gt_mask = np.asarray(gt_mask)
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"prediction/label shape mismatch: pred {pred_mask.shape} vs label {gt_mask.shape}"
        )
    m = torch.as_tensor(pred_mask, dtype=torch.float32)
    logits = torch.stack([1.0 - m, m], dim=0).unsqueeze(0)              # (1, 2, D, H, W)
    target = torch.as_tensor(gt_mask, dtype=torch.float32)[None, None]  # (1, 1, D, H, W)
    # HD95 builds one-hot distance transforms; on a full tomogram those can exceed 10+ GiB,
    # hence the CPU escape hatch (same rationale as segmentation3d_inference.py --cpu-metrics).
    if not cpu_metrics and torch.cuda.is_available():
        logits, target = logits.cuda(), target.cuda()
    return metric(logits, target)


def hd95_exact(pk, gk, percentile=95):
    """Whole-volume HD95 of one binary problem (`pk` = prediction mask, `gk` = ground-truth mask, bool).

    The same computation as MONAI's HausdorffDistanceMetric -- the edges and the two directed surface
    distances come from monai.metrics.utils.get_edge_surface_distance, and the result is the larger
    of the two directed percentiles -- except that the percentile is taken with np.percentile
    instead of torch.quantile. torch.quantile refuses inputs over 2**24 elements, which a fragmented
    prediction of a full tomogram exceeds. np.percentile uses the same linear interpolation.

    It is also cheaper than going through CryoMetrics: one distance transform per direction, where
    CryoMetrics(2) runs HD95 twice (mean and per-class) over both channels, i.e. 4x as many.
    Returns nan when the prediction or the ground truth is empty (HD95 is undefined).
    """
    from monai.metrics.utils import get_edge_surface_distance

    if not pk.any() or not gk.any():
        return float("nan")
    try:
        _, distances, _ = get_edge_surface_distance(
            torch.from_numpy(np.ascontiguousarray(pk)), torch.from_numpy(np.ascontiguousarray(gk)),
            distance_metric="euclidean", symetric=True, class_index=1,
        )
        directed = [float(np.percentile(d.numpy(), percentile)) if d.numel() else float("nan") for d in distances]
        return float(np.max(directed))   # np.max propagates nan, like torch.max in MONAI
    finally:
        # The edge and distance arrays are full-volume (tens of GB on a tomogram) and were observed to
        # survive the call -- resident memory grew ~26 GB per class until the OOM killer fired -- so
        # collect them here instead of leaving it to the next allocation pressure.
        distances = None
        gc.collect()


def score_class(pk, gk, with_hd=True):
    """Score one class's binary problem (`pk` prediction mask, `gk` ground-truth mask, both bool).

    Returns (dice, hd95). Dice is 2*TP / (|pred| + |gt|) from voxel counts over the whole volume,
    which is what MONAI's DiceMetric computes for the foreground channel of a one-hot binary
    problem. HD95 is `hd95_exact`.
    """
    n_pred, n_gt = int(np.count_nonzero(pk)), int(np.count_nonzero(gk))
    tp = int(np.count_nonzero(pk & gk))
    dice = 2.0 * tp / (n_pred + n_gt) if (n_pred + n_gt) else float("nan")
    return dice, (hd95_exact(pk, gk) if with_hd else float("nan"))


def emit_and_score(name, img_path, label_path, pred, output_dir, results, metric,
                   cpu_metrics=False, provenance=None, remap_12049_gt=False,
                   target_class_value=None, threshold=0.5):
    """Save the prediction as <output_dir>/<name>.nii.gz and, when a label exists, score it.

    The common tail of every per-sample loop in both baseline scripts, in either mode (running the
    model or reading an existing prediction off disk).

    `pred` is either a single hard mask (binary) or a list of N zero-argument callables, one per
    class, each returning that class's probability map (multi-class). Each map is loaded, scored and
    freed before the next is loaded -- never all N at once. DeePiCt's multi-class head is multi-label (N independent
    sigmoids, no background channel, classes may overlap), so multi-class is scored as N+1
    independent binary problems rather than one mutually-exclusive argmax: class k is
    `prob_k >= threshold` against `label == k`, and class 0 (background) is "no class >= threshold"
    against `label == 0`. The per-class values are reported as [bg, class 1..N] and avg_dice /
    avg_hd95 are their plain means. Dice and HD95 are both whole-volume values (see `score_class`;
    HD95 uses np.percentile, so it is not limited by torch.quantile's input size).

    The saved volume is one ordinal label map for viewing, filled class by class: voxels where
    class k passes the threshold are set to k, so where classes overlap the LATER class wins (it is
    order-dependent, unlike an argmax). It is not what is scored.

    `remap_12049_gt=True` applies Dataset049's background-merge remap to the loaded ground truth
    before scoring (multi-class mode only).

    `target_class_value` (binary mode only): instead of `binarize(gt)` (every class > 0 counts as
    foreground), extract this one raw label value as the ground truth -- e.g. membrain-seg scoped
    to membrane-only on a multi-class dataset. Mutually exclusive with `remap_12049_gt` in
    practice (binary mode ignores the latter).

    Pass `provenance` to rewrite metrics.json after every sample. Scoring a full tomogram is slow
    (HD95 runs four distance transforms over 200-450 M voxels), so a run that is cut short -- a
    SLURM walltime, an OOM on a later sample -- would otherwise lose every tomogram already
    finished. Aggregates are recomputed from `results` on each call, so a partial file is
    internally consistent; its `overall_*` keys are simply means over fewer tomograms.
    """
    multiclass = isinstance(pred, (list, tuple))
    if not multiclass:
        out_path = save_prediction(pred, img_path, os.path.join(output_dir, name + ".nii.gz"))
        print(f"  Saved prediction: {out_path}")
        if label_path:
            gt = load_volume(label_path)
            if target_class_value is not None:
                gt = target_class(gt, target_class_value)
            else:
                gt = binarize(gt)
            record_metrics(results, name, metric, pred, gt, cpu_metrics)
            del gt
            if provenance is not None:
                write_metrics_json(results, output_dir, extra=provenance)
        clear_cuda_memory()
        return out_path

    # Multi-class: a single pass over the classes. Each map is loaded, thresholded, written into the
    # label map, scored, and freed before the next one is loaded. Only the label (uint8), the label
    # map (uint8) and `anyfg` (bool) stay resident; background is scored last, from `anyfg`.
    loaders = list(pred)
    n_cls = len(loaders) + 1
    gt = None
    if label_path:
        gt = load_volume(label_path)
        if remap_12049_gt:
            gt = remap_12049(gt)
        gt = gt.astype(np.uint8)   # labels are small integers; float32 would cost 4x the memory
    dice, hd = [None] * n_cls, [None] * n_cls

    def score(k, fg):
        dice[k], hd[k] = score_class(fg, gt == k)
        print(f"  class {k}: Dice {dice[k]:.4f}  HD95 {hd[k]:.4f}", flush=True)

    out_path = os.path.join(output_dir, name + ".nii.gz")
    mask = anyfg = None
    complete = False
    try:
        for k, load in enumerate(loaders, 1):
            fg = np.asarray(load()) >= threshold
            if mask is None:
                mask = np.zeros(fg.shape, dtype=np.uint8)
                anyfg = np.zeros(fg.shape, dtype=bool)
            mask[fg] = k   # later classes overwrite earlier ones where they overlap
            anyfg |= fg
            if gt is not None:
                score(k, fg)
            del fg
        if gt is not None:
            score(0, ~anyfg)
        complete = True
    finally:
        if mask is not None:   # saved even if a score raised, so the prediction is not lost
            save_prediction(mask, img_path, out_path)
            note = "" if complete else "  (PARTIAL: the run stopped before the last classes were filled in)"
            print(f"  Saved prediction: {out_path}{note}")
    del mask, anyfg
    if gt is not None:
        results[name] = {
            "avg_dice": float(np.mean(dice)),
            "per_class_dice": dice,
            "avg_hd95": float(np.mean(hd)),
            "per_class_hd95": hd,
        }
        print(f"  Dice: {results[name]['avg_dice']:.4f}, Per-class: {dice}")
        print(f"  HD95: {results[name]['avg_hd95']:.4f}, Per-class: {hd}")
        del gt
        if provenance is not None:
            write_metrics_json(results, output_dir, extra=provenance)
    clear_cuda_memory()
    return out_path


def record_metrics(results, name, metric, pred_mask, gt_mask, cpu_metrics=False):
    """Score one sample and append it to `results` in the per-image schema."""
    avg_dice, per_cls_dice, avg_hd, per_cls_hd = score_mask(metric, pred_mask, gt_mask, cpu_metrics)
    results[name] = {
        "avg_dice": float(avg_dice),
        "per_class_dice": [float(d) for d in per_cls_dice],
        "avg_hd95": float(avg_hd),
        "per_class_hd95": [float(h) for h in per_cls_hd],
    }
    print(f"  Dice: {avg_dice:.4f}, Per-class: {results[name]['per_class_dice']}")
    print(f"  HD95: {avg_hd:.4f}, Per-class: {results[name]['per_class_hd95']}")
    return results[name]
