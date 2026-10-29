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
    "foreground_values", "split_classes", "remap_12049",
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

def score_mask(metric, pred, gt_mask, cpu_metrics=False):
    """Score a prediction (not logits) against a ground-truth label with `metric`.

    `CryoMetrics.__call__` expects an argmax-able, multi-channel logit tensor -- the shape
    CryoDINO's network output naturally has. The baselines don't have that, so this lifts
    whatever they do have into the same shape, in one of two ways:

      - `pred` is a single binary mask / probability array `m` (DeePiCt-binary, membrain-seg):
        lifted to `[1 - m, m]`, so the argmax reproduces `m` exactly.
      - `pred` is a list of N independent per-class probability maps (DeePiCt multi-class: a
        multi-label sigmoid head, no background channel, classes may overlap -- see
        baselines/deepict/README.md's "Binary and multi-class"): lifted to
        `[bg, p_1, ..., p_N]` where `bg = clip(1 - max(p_1..p_N), 0, 1)`. Each p_i is itself a
        sigmoid output confirmed bounded in (0, 1) (DeePiCt's assemble_tomo_from_subtomos is
        called with final_activation='sigmoid'), so the clip is defensive, not load-bearing.

    Either way, the same `CryoMetrics` call computes the same numbers, so DS010 (binary) and
    DS001/Dataset049 (multi-class) land in one comparable table.
    """
    gt_mask = np.asarray(gt_mask)
    if isinstance(pred, (list, tuple)):
        probs = [np.asarray(p) for p in pred]
        for p in probs:
            if p.shape != gt_mask.shape:
                raise ValueError(
                    f"prediction/label shape mismatch: pred {p.shape} vs label {gt_mask.shape}"
                )
        stacked = np.stack(probs, axis=0)                          # (N, D, H, W)
        bg = np.clip(1.0 - stacked.max(axis=0), 0.0, 1.0)           # (D, H, W)
        channels = np.concatenate([bg[None], stacked], axis=0)     # (N+1, D, H, W)
        logits = torch.as_tensor(channels, dtype=torch.float32).unsqueeze(0)  # (1, N+1, D, H, W)
    else:
        pred_mask = np.asarray(pred)
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


def emit_and_score(name, img_path, label_path, pred, output_dir, results, metric,
                   cpu_metrics=False, provenance=None, remap_12049_gt=False):
    """Save the prediction as <output_dir>/<name>.nii.gz and, when a label exists, score it.

    The common tail of every per-sample loop in both baseline scripts, in either mode (running the
    model or reading an existing prediction off disk).

    `pred` is either a single hard mask (binary) or a list of N per-class probability maps
    (multi-class, see `score_mask`). In the multi-class case, the saved volume is the real
    prediction -- `argmax([bg, p_1, ..., p_N])`, giving an ordinal 0=bg/1..N=class label map, the
    same shape every other method's saved prediction already has -- not a per-class probability
    map.

    `remap_12049_gt=True` applies Dataset049's background-merge remap to the loaded ground truth
    before scoring (ignored in binary mode, where the ground truth is binarized instead).

    Pass `provenance` to rewrite metrics.json after every sample. Scoring a full tomogram is slow
    (HD95 runs four distance transforms over 200-450 M voxels), so a run that is cut short -- a
    SLURM walltime, an OOM on a later sample -- would otherwise lose every tomogram already
    finished. Aggregates are recomputed from `results` on each call, so a partial file is
    internally consistent; its `overall_*` keys are simply means over fewer tomograms.
    """
    multiclass = isinstance(pred, (list, tuple))
    if multiclass:
        stacked = np.stack([np.asarray(p) for p in pred], axis=0)
        bg = np.clip(1.0 - stacked.max(axis=0), 0.0, 1.0)
        mask = np.argmax(np.concatenate([bg[None], stacked], axis=0), axis=0).astype(np.uint8)
    else:
        mask = pred

    out_path = save_prediction(mask, img_path, os.path.join(output_dir, name + ".nii.gz"))
    print(f"  Saved prediction: {out_path}")
    if label_path:
        gt = load_volume(label_path)
        if multiclass:
            if remap_12049_gt:
                gt = remap_12049(gt)
        else:
            gt = binarize(gt)
        record_metrics(results, name, metric, pred, gt, cpu_metrics)
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
