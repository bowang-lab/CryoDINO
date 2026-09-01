"""
MemBrain-seg baseline inference, in the same shape as segmentation3d_inference.py.

Runs a trained membrain-seg (teamtomo) binary U-Net over a set of tomograms and writes
predictions + Dice/HD95 metrics using CryoDINO's exact output schema, so the baseline can be
compared against a CryoDINO segmentation head without any post-hoc reconciliation.

Inference only -- training lives in membrain_kit/run_datalist.sh.

Environment: run in the membrain venv (it needs `membrain_seg`), NOT the cryoet conda env:
    module load python/3.11 && source /scratch/cyyu/venvs/membrain/bin/activate

Two input modes (identical to segmentation3d_inference.py):
  1. --datalist: a MONAI datalist JSON; the "test" split is used. Entries need an "image" key;
     a "label" key that resolves to an existing file enables metrics. A bare JSON list (the
     membrain kit's test_list.json) also works.
  2. --input-dir: a directory of volumes, optionally with --label-dir for metrics.

Usage:

  python inference/membrain_inference.py \\
    --checkpoint /path/to/membrain_binary-<best>.ckpt \\
    --datalist /path/to/datalist.json \\
    --output-dir /path/to/output/

  python inference/membrain_inference.py \\
    --checkpoint /path/to/membrain_binary-<best>.ckpt \\
    --input-dir /path/to/images/ --label-dir /path/to/labels/ \\
    --output-dir /path/to/output/ --sw-roi-size 160

Outputs:
  - <output-dir>/<name>.nii.gz: predicted binary mask (uint8)
  - <output-dir>/metrics.json:  per-image and overall Dice/HD95 (only when labels are available)
  - <output-dir>/_native/<name>/: membrain's own .mrc outputs (removed unless --keep-native)
"""

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_common import (  # noqa: E402
    binarize,
    CryoMetrics,
    emit_and_score,
    load_volume,
    resolve_samples,
    sample_name,
    save_mrc,
    write_metrics_json,
)


def get_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Trained membrain-seg lightning checkpoint (.ckpt). "
                        "Required unless --predictions-dir is given.")
    p.add_argument("--predictions-dir", type=str, default=None,
                   help="Score EXISTING membrain predictions instead of running the model. "
                        "Expects <DIR>/<name>/*_segmented.mrc (the layout membrain segment writes).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory for predictions and metrics.json")
    p.add_argument("--datalist", type=str, default=None,
                   help="Path to datalist JSON (uses the test split for inference)")
    p.add_argument("--split", type=str, default="test",
                   help="Datalist split to run (default: test)")
    p.add_argument("--input-dir", type=str, default=None,
                   help="Directory containing input volumes (.nii.gz/.nii/.mrc/.rec)")
    p.add_argument("--label-dir", type=str, default=None,
                   help="Optional directory of label volumes (used with --input-dir)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="membrain segmentation threshold on the membrane score (default: 0.0)")
    p.add_argument("--sw-roi-size", type=int, default=160,
                   help="Sliding window size for inference; must be a multiple of 32 (default: 160)")
    p.add_argument("--no-tta", action="store_true",
                   help="Disable membrain's 8-fold mirroring test-time augmentation (8x faster, worse)")
    p.add_argument("--rescale-patches", action="store_true",
                   help="Rescale patches to --out-pixel-size (off by default, matching --no-rescale-patches)")
    p.add_argument("--in-pixel-size", type=float, default=None,
                   help="Input pixel size in Angstrom (only with --rescale-patches; default: from header)")
    p.add_argument("--out-pixel-size", type=float, default=10.0,
                   help="Target pixel size in Angstrom (only with --rescale-patches)")
    p.add_argument("--store-probabilities", action="store_true",
                   help="Also keep membrain's <name>_scores.mrc probability map")
    p.add_argument("--keep-native", action="store_true",
                   help="Keep <output-dir>/_native/ (membrain's raw .mrc outputs)")
    p.add_argument("--cpu-metrics", action="store_true",
                   help="Compute Dice/HD95 on CPU (avoids GPU OOM on large volumes)")
    return p


def find_segmentation(predictions_dir, name):
    """Locate an existing membrain segmentation for `name` under <predictions_dir>/<name>/."""
    hits = sorted(glob.glob(os.path.join(predictions_dir, name, "*_segmented.mrc")))
    if not hits:
        raise SystemExit(
            f"Error: no *_segmented.mrc for {name} under "
            f"{os.path.join(predictions_dir, name)} -- check --predictions-dir"
        )
    if len(hits) > 1:
        print(f"  Warning: {len(hits)} segmentations for {name}, using the last: "
              f"{os.path.basename(hits[-1])}")
    return hits[-1]


def main():
    args = get_args_parser().parse_args()
    scoring_only = args.predictions_dir is not None
    if not scoring_only and not args.checkpoint:
        raise SystemExit("Error: --checkpoint is required unless --predictions-dir is given")
    if scoring_only and not os.path.isdir(args.predictions_dir):
        raise SystemExit(f"Error: --predictions-dir not found: {args.predictions_dir}")
    if not scoring_only and args.sw_roi_size % 32 != 0:
        raise SystemExit(f"--sw-roi-size must be a multiple of 32, got {args.sw_roi_size}")
    os.makedirs(args.output_dir, exist_ok=True)

    segment = None
    if not scoring_only:
        # Imported here so --help works without the membrain venv active.
        from membrain_seg.segmentation.segment import segment

    samples = resolve_samples(args.datalist, args.input_dir, args.label_dir, args.split)
    has_any_labels = any(lbl for _, lbl in samples)
    metric = CryoMetrics(num_classes=2) if has_any_labels else None
    print(f"Found {len(samples)} images ({sum(1 for _, l in samples if l)} with labels)")

    native_root = os.path.join(args.output_dir, "_native")
    results = {}

    if scoring_only:
        provenance = {
            "method": "membrain-seg",
            "source": "existing-predictions",
            "predictions_dir": os.path.abspath(args.predictions_dir),
        }
    else:
        provenance = {
            "method": "membrain-seg",
            "source": "inference",
            "checkpoint": os.path.abspath(args.checkpoint),
            "segmentation_threshold": args.threshold,
            "sw_roi_size": args.sw_roi_size,
            "test_time_augmentation": not args.no_tta,
        }


    for i, (img_path, label_path) in enumerate(samples, 1):
        name = sample_name(img_path, label_path)
        print(f"\n[{i}/{len(samples)}] Processing: {name}")

        if scoring_only:
            seg_path = find_segmentation(args.predictions_dir, name)
            print(f"  Existing segmentation: {seg_path}")
        else:
            work = os.path.join(native_root, name)
            os.makedirs(work, exist_ok=True)
            # membrain's segment() reads MRC only -> materialize non-MRC inputs.
            if img_path.endswith((".mrc", ".rec")):
                img_mrc = img_path
            else:
                img_mrc = os.path.join(work, name + ".mrc")
                save_mrc(load_volume(img_path), img_mrc)

            seg_path = segment(
                tomogram_path=img_mrc,
                ckpt_path=args.checkpoint,
                out_folder=work,
                rescale_patches=args.rescale_patches,
                in_pixel_size=args.in_pixel_size,
                out_pixel_size=args.out_pixel_size,
                store_probabilities=args.store_probabilities,
                sw_roi_size=args.sw_roi_size,
                test_time_augmentation=not args.no_tta,
                segmentation_threshold=args.threshold,
            )
            print(f"  membrain segmentation: {seg_path}")

        pred = binarize(load_volume(seg_path))
        emit_and_score(name, img_path, label_path, pred, args.output_dir,
                       results, metric, args.cpu_metrics, provenance)
        del pred

    write_metrics_json(results, args.output_dir, extra=provenance)

    if not scoring_only and not args.keep_native and os.path.isdir(native_root):
        shutil.rmtree(native_root, ignore_errors=True)
        print(f"Removed intermediates: {native_root} (pass --keep-native to keep them)")


if __name__ == "__main__":
    main()
