"""
DeePiCt baseline inference, in the same shape as segmentation3d_inference.py.

Runs a trained DeePiCt 3D U-Net over a set of tomograms and writes predictions + Dice/HD95
metrics using CryoDINO's exact output schema, so the baseline can be compared against a CryoDINO
segmentation head without any post-hoc reconciliation.

Inference only -- training lives in deepict/run_datalist.sh.

DeePiCt's prediction path is config-driven and split across three scripts, so this driver
generates a prediction-only config and shells out to the upstream stages unchanged:
    generate_prediction_partition.py -> segment.py -> assemble_prediction.py
generate_prediction_partition.py and segment.py are class-independent (the network emits every
channel in one pass), so they run once per tomogram; assemble_prediction.py is per-class, so it
runs once per class (same "segment once, assemble N times" pattern as deepict/predict.sh).

Binary vs multi-class is inherited from --train-config's semantic_classes list (one name ->
binary, N>1 -> multi-class) -- the same auto-detection convention as
deepict/prepare_from_datalist.py. In binary mode the assembled sigmoid probability map is
thresholded at --threshold, exactly as before. In multi-class mode (DeePiCt's head is multi-label:
N independent sigmoid channels, no background channel -- see deepict/README.md's "Binary and
multi-class") each class is scored as its own binary problem: `prob_k >= --threshold` against
`label == k`, plus a background entry ("no class >= --threshold" against `label == 0`), each through
whole-volume Dice and HD95 (baseline_common.score_class; HD95 uses np.percentile so it is not capped
by torch.quantile's input size). Per-class values are reported as [bg, class 1..N] and the averages
are their plain means -- see baseline_common.py's emit_and_score. The saved <name>.nii.gz is still one ordinal
label map (argmax over [bg, p_1..p_N]) for viewing only. Dataset049/EMPIAR-12049 additionally needs
--remap-12049 (merges matrix/centrioles into background before scoring, see
baselines/deepict/datalist_utils.py's remap_12049).

Environment: run in the venv built for DeePiCt/membrain, NOT the cryoet conda env:
    module load python/3.11 && source /scratch/cyyu/venvs/membrain/bin/activate

Two input modes (identical to segmentation3d_inference.py):
  1. --datalist: a MONAI datalist JSON; the "test" split is used.
  2. --input-dir: a directory of volumes, optionally with --label-dir for metrics.

Usage:

  # binary (e.g. Dataset010)
  python inference/deepict_inference.py \\
    --model-path /path/to/out/model_best.pth \\
    --train-config /path/to/the/training/config.yaml \\
    --deepict-root /path/to/DeePiCt \\
    --datalist /path/to/datalist.json \\
    --output-dir /path/to/output/

  # multi-class (e.g. Dataset001; add --remap-12049 for Dataset049)
  python inference/deepict_inference.py \\
    --model-path /path/to/out/model_best.pth \\
    --train-config /path/to/the/training/config.yaml \\
    --deepict-root /path/to/DeePiCt \\
    --datalist /path/to/datalist.json \\
    --output-dir /path/to/output/

Outputs:
  - <output-dir>/<name>.nii.gz: predicted label map (uint8) -- a hard binary mask in binary mode,
    or the ordinal 0=bg/1..N=class argmax prediction in multi-class mode
  - <output-dir>/metrics.json:  per-image and overall Dice/HD95 (only when labels are available)
  - <work-dir>/: converted .mrc inputs, metadata.csv, the generated config(s), DeePiCt partitions
    and the raw probability maps (removed unless --keep-intermediates)
"""

import argparse
import copy
import csv
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_common import (  # noqa: E402
    binarize,
    CryoMetrics,
    emit_and_score,
    load_volume,
    remap_12049,
    resolve_samples,
    sample_name,
    save_mrc,
    split_classes,
    write_metrics_json,
)

# Minimal config skeleton. constants/config.py::Config reads every one of these keys eagerly,
# so none may be omitted even though training is inactive. Network hyperparameters are
# placeholders: segment.py rebuilds UNet3D from checkpoint['model_descriptor'], not from here.
CONFIG_SKELETON = {
    "dataset_table": None,
    "output_dir": None,
    "work_dir": None,
    "model_path": None,
    "cluster": {"logdir": None},
    "tomos_sets": {"training_list": [], "prediction_list": []},
    "cross_validation": {"active": False, "folds": 2, "statistics_file": "cv_statistics.csv"},
    "training": {
        "active": False,
        "semantic_classes": ["particle"],
        "processing_tomo": "tomo",
        "box_size": 64,
        "min_label_fraction": 0.001,
        "overlap": 12,
        "batch_size": 4,
        "force_retrain": False,
        "iterations_per_epoch": 300,
        "unet_hyperparameters": {
            "depth": 2, "initial_features": 8, "epochs": 100, "train_split": 0.8,
            "batch_norm": True, "encoder_dropout": 0, "decoder_dropout": 0.2, "loss": "Dice",
        },
        "data_augmentation": {
            "rounds": 0, "rot_angle": 180, "elastic_alpha": 0, "sigma_gauss": 1,
            "salt_pepper_p": 0.01, "salt_pepper_ampl": 0.1,
        },
    },
    "prediction": {"active": True, "semantic_class": "particle"},
    "postprocessing_clustering": {
        "active": False, "threshold": 0.5, "min_cluster_size": 100, "max_cluster_size": None,
        "clustering_connectivity": 3, "calculate_motl": False,
        "ignore_border_thickness": [10, 20, 10], "region_mask": "lamella_file",
        "contact_mode": "intersection", "contact_distance": 10,
    },
    "evaluation": {
        "particle_picking": {"active": False, "pr_tolerance_radius": 10,
                             "statistics_file": "pr_statistics.csv"},
        "segmentation_evaluation": {"active": False, "statistics_file": "dice_eval.csv"},
    },
    "debug": False,
}


def get_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-path", type=str, default=None,
                   help="Trained DeePiCt model (.pth, e.g. out/model_best.pth). "
                        "Required unless --predictions-dir is given.")
    p.add_argument("--predictions-dir", type=str, default=None,
                   help="Score EXISTING DeePiCt predictions instead of running the model. "
                        "Expects <DIR>/<name>/<class>/probability_map.mrc per declared class, "
                        "i.e. point it at .../out/predictions/<model_name>.")
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
    p.add_argument("--deepict-root", type=str, default=os.environ.get("DEEPICT"),
                   help="Path to the DeePiCt clone (default: $DEEPICT)")
    p.add_argument("--train-config", type=str, default=None,
                   help="Config the model was trained with; box_size/overlap/semantic_classes "
                        "are inherited from it (strongly recommended). len(semantic_classes)>1 "
                        "switches to multi-class scoring.")
    p.add_argument("--box-size", type=int, default=None,
                   help="Partition box size; overrides --train-config (default: 64)")
    p.add_argument("--overlap", type=int, default=None,
                   help="Partition overlap; overrides --train-config (default: 12)")
    p.add_argument("--semantic-class", type=str, default=None,
                   help="Force binary mode on this single class, overriding --train-config's "
                        "(possibly multi-class) semantic_classes list.")
    p.add_argument("--remap-12049", action="store_true",
                   help="Dataset049/EMPIAR-12049 only: merge matrix/centrioles into background "
                        "before scoring (matches augmentations.py's _remap_12049). Only affects "
                        "the ground truth used for scoring; ignored in binary mode.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Threshold on each sigmoid probability map (default: 0.5). Binary mode: the "
                        "foreground mask. Multi-class mode: the per-class mask, and the background "
                        "is 'no class above it'.")
    p.add_argument("--work-dir", type=str, default=None,
                   help="Scratch dir for .mrc inputs, partitions and probability maps "
                        "(default: <output-dir>/_deepict_work)")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Keep the work dir (converted .mrc, partitions, probability maps)")
    p.add_argument("--cpu-metrics", action="store_true",
                   help="Compute Dice/HD95 on CPU (avoids GPU OOM on large volumes)")
    return p


def build_dataset(samples, data_dir, class_names, class_values, remap=False):
    """Materialize each sample as the .mrc tomogram + region + one mask per class.

    Ported from deepict/prepare_from_datalist.py::convert_entries, generalized the same way:
    `class_values` is None in binary mode (single 'particle' mask, old behaviour unchanged);
    otherwise one binary mask file is written per class value. The region mask column is NOT
    optional: generate_prediction_partition.py and segment.py index it unconditionally.
    """
    raw_d, msk_d = os.path.join(data_dir, "raw"), os.path.join(data_dir, "masks")
    os.makedirs(raw_d, exist_ok=True)
    os.makedirs(msk_d, exist_ok=True)
    rows, shapes = [], {}
    for i, (img_path, label_path) in enumerate(samples, 1):
        name = sample_name(img_path, label_path)
        raw_p = os.path.join(raw_d, name + ".mrc")
        reg_p = os.path.join(msk_d, name + "_region.mrc")

        img = load_volume(img_path)
        shape = img.shape
        save_mrc(img, raw_p)
        region = (img != 0).astype(np.int8)
        del img
        save_mrc(region, reg_p, dtype=np.int8)
        del region

        lab = None
        if label_path:
            lab = load_volume(label_path)
            if remap:
                lab = remap_12049(lab)
        if class_values is None:
            masks = [binarize(lab) if lab is not None else np.zeros(shape, np.uint8)]
        else:
            masks = (split_classes(lab, class_values) if lab is not None
                    else [np.zeros(shape, np.uint8) for _ in class_values])
        mask_ps = []
        for cname, m in zip(class_names, masks):
            p = os.path.join(msk_d, f"{name}_{cname}.mrc")
            save_mrc(m, p, dtype=np.int8)
            mask_ps.append(p)
        del masks

        rows.append(tuple([name, raw_p, reg_p] + mask_ps))
        shapes[name] = shape
        print(f"  [{i}/{len(samples)}] {name:32s} shape={shape}", flush=True)
    return rows, shapes


def write_config(args, rows, work_dir, class_names, box_size, overlap):
    cfg = copy.deepcopy(CONFIG_SKELETON)
    meta = os.path.join(work_dir, "data", "metadata.csv")
    with open(meta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tomo_name", "tomo", "lamella_file"] + [f"{n}_mask" for n in class_names])
        w.writerows(rows)

    cfg["dataset_table"] = meta
    cfg["output_dir"] = os.path.join(work_dir, "out")
    cfg["work_dir"] = os.path.join(work_dir, "work")
    cfg["model_path"] = os.path.abspath(args.model_path)
    cfg["cluster"]["logdir"] = os.path.join(work_dir, "logs")
    cfg["tomos_sets"]["training_list"] = []
    cfg["tomos_sets"]["prediction_list"] = [r[0] for r in rows]
    cfg["training"]["active"] = False
    cfg["training"]["semantic_classes"] = class_names
    cfg["training"]["box_size"] = box_size
    cfg["training"]["overlap"] = overlap
    cfg["prediction"]["active"] = True
    # In binary mode this is the one class predict reads. In multi-class mode
    # assemble_prediction.py only ever reads one channel per call, so the main loop below writes
    # a per-class override of this same config (same trick as deepict/predict.sh's
    # .predict_<class>.yaml) rather than relying on this default.
    cfg["prediction"]["semantic_class"] = class_names[0]

    cfg_path = os.path.join(work_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path, cfg


def write_class_config(cfg_path, class_name, out_path):
    """Copy cfg_path with prediction.semantic_class overridden -- assemble_prediction.py takes
    only --config_file, so each class needs its own config to select that channel."""
    cfg = yaml.safe_load(open(cfg_path))
    cfg["prediction"]["semantic_class"] = class_name
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_stage(script, src, cfg_path, name, cwd):
    """Run one upstream DeePiCt stage.

    No --gpu is passed on purpose: DeePiCt's to_device(gpu=None) then keeps SLURM's (MIG)
    CUDA_VISIBLE_DEVICES intact. cwd matters because segment.py writes its .done_patterns/
    marker relative to the working directory.
    """
    cmd = [sys.executable, script, "--pythonpath", src, "--config_file", cfg_path,
           "--fold", "None", "--tomo_name", name]
    print("  $ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=cwd).returncode
    if rc != 0:
        stage = os.path.basename(script)
        hint = ""
        if rc < 0 or rc == 137:
            # DeePiCt holds the tomogram as float64 and every partition box in RAM; a full
            # tomogram needs a real-memory node (a ~4 GB login shell gets OOM-killed).
            hint = ("\n  This looks like an OOM kill. DeePiCt loads the whole tomogram as "
                    "float64 plus every partition box in RAM -- run on a compute node with "
                    "enough memory, not a login shell.")
        raise SystemExit(f"Error: DeePiCt stage {stage} failed for {name} (exit {rc}).{hint}")


def main():
    args = get_args_parser().parse_args()
    scoring_only = args.predictions_dir is not None
    src = scripts = None
    if scoring_only:
        if not os.path.isdir(args.predictions_dir):
            raise SystemExit(f"Error: --predictions-dir not found: {args.predictions_dir}")
    else:
        if not args.model_path:
            raise SystemExit("Error: --model-path is required unless --predictions-dir is given")
        if not args.deepict_root:
            raise SystemExit("Error: --deepict-root not given and $DEEPICT is unset")
        src = os.path.join(args.deepict_root, "3d_cnn", "src")
        scripts = os.path.join(args.deepict_root, "3d_cnn", "scripts")
        if not os.path.isdir(src):
            raise SystemExit(f"Error: {src} not found -- is --deepict-root a DeePiCt clone?")
        if not os.path.isfile(args.model_path):
            raise SystemExit(f"Error: model not found: {args.model_path}")

    # box_size / overlap must match training (they drive partitioning and reassembly); the
    # network shape does not, since segment.py reads it from checkpoint['model_descriptor'].
    box_size, overlap, class_names = 64, 12, ["particle"]
    if args.train_config:
        tc = yaml.safe_load(open(args.train_config))["training"]
        box_size, overlap = tc["box_size"], tc["overlap"]
        class_names = list(tc["semantic_classes"])
        print(f"Inherited from {args.train_config}: box_size={box_size} overlap={overlap} "
              f"classes={class_names}")
    box_size = args.box_size or box_size
    overlap = args.overlap if args.overlap is not None else overlap
    if args.semantic_class:
        class_names = [args.semantic_class]
    multiclass = len(class_names) > 1
    class_values = list(range(1, len(class_names) + 1)) if multiclass else None
    if args.remap_12049 and not multiclass:
        print("Warning: --remap-12049 has no effect in binary mode (GT is binarized instead).")

    os.makedirs(args.output_dir, exist_ok=True)
    work_dir = os.path.abspath(args.work_dir or os.path.join(args.output_dir, "_deepict_work"))

    samples = resolve_samples(args.datalist, args.input_dir, args.label_dir, args.split)
    has_any_labels = any(lbl for _, lbl in samples)
    metric = CryoMetrics(num_classes=len(class_names) + 1) if has_any_labels else None
    print(f"Found {len(samples)} images ({sum(1 for _, l in samples if l)} with labels)")
    print(f"Mode: {'multi-class ' + str(class_names) if multiclass else 'binary (particle)'}")

    rows = shapes = cfg = cfg_path = model_name = None
    if not scoring_only:
        os.makedirs(work_dir, exist_ok=True)
        print("\n=== 1) converting inputs to DeePiCt .mrc + metadata.csv ===")
        rows, shapes = build_dataset(samples, os.path.join(work_dir, "data"), class_names,
                                     class_values, remap=args.remap_12049)
        cfg_path, cfg = write_config(args, rows, work_dir, class_names, box_size, overlap)
        print(f"config -> {cfg_path}")
        model_name = os.path.basename(args.model_path)[:-4]

    results = {}

    if scoring_only:
        provenance = {
            "method": "deepict",
            "source": "existing-predictions",
            "predictions_dir": os.path.abspath(args.predictions_dir),
            "threshold": args.threshold,
            "semantic_classes": class_names,
        }
    else:
        provenance = {
            "method": "deepict",
            "source": "inference",
            "model_path": os.path.abspath(args.model_path),
            "threshold": args.threshold,
            "box_size": box_size,
            "overlap": overlap,
            "semantic_classes": class_names,
        }

    print("\n=== predicting ===" if not scoring_only else "\n=== scoring existing predictions ===")
    for i, (img_path, label_path) in enumerate(samples, 1):
        name = rows[i - 1][0] if rows else sample_name(img_path, label_path)
        print(f"\n[{i}/{len(samples)}] Processing: {name}")

        if scoring_only:
            expected = load_volume(img_path).shape
            prob_paths = []
            for cname in class_names:
                prob_path = os.path.join(args.predictions_dir, name, cname, "probability_map.mrc")
                if not os.path.isfile(prob_path):
                    raise SystemExit(f"Error: no probability map for {name}/{cname} at "
                                     f"{prob_path} -- check --predictions-dir / --semantic-class")
                prob_paths.append(prob_path)
        else:
            # generate_prediction_partition.py silently no-ops when the partition h5 already
            # exists, which would otherwise reuse a partition built with different settings.
            stale = os.path.join(cfg["work_dir"], "testing_data", name)
            if os.path.isdir(stale):
                shutil.rmtree(stale, ignore_errors=True)

            for stage in ("generate_prediction_partition.py", "segment.py"):
                run_stage(os.path.join(scripts, stage), src, cfg_path, name, work_dir)

            expected = shapes[name]
            prob_paths = []
            for cname in class_names:
                # Binary mode: cfg_path's prediction.semantic_class is already class_names[0],
                # so this reproduces the pre-multi-class code path exactly (same config, no
                # override file written).
                class_cfg_path = cfg_path
                if multiclass:
                    class_cfg_path = os.path.join(work_dir, f".predict_{cname}.yaml")
                    write_class_config(cfg_path, cname, class_cfg_path)
                run_stage(os.path.join(scripts, "assemble_prediction.py"), src, class_cfg_path,
                         name, work_dir)
                prob_path = os.path.join(cfg["output_dir"], "predictions", model_name, name,
                                         cname, "probability_map.mrc")
                if not os.path.isfile(prob_path):
                    raise SystemExit(f"Error: DeePiCt produced no probability map at {prob_path}")
                prob_paths.append(prob_path)

        def load_prob(path, expected=expected):
            prob = load_volume(path)
            if prob.shape != expected:
                raise SystemExit(f"Error: probability map shape {prob.shape} != input {expected}")
            return prob

        if multiclass:
            # One loader per class: emit_and_score loads a single map at a time rather than
            # holding all N full-volume maps in memory.
            pred = [(lambda p=p: load_prob(p)) for p in prob_paths]
            print(f"  probability maps: classes={class_names}")
        else:
            prob = load_prob(prob_paths[0])
            pred = (prob >= args.threshold).astype(np.uint8)
            del prob
            print(f"  probability map (class={class_names[0]}): fg voxels at "
                  f"t={args.threshold}: {int(pred.sum())}")

        emit_and_score(name, img_path, label_path, pred, args.output_dir,
                       results, metric, args.cpu_metrics, provenance,
                       remap_12049_gt=args.remap_12049, threshold=args.threshold)
        del pred

    write_metrics_json(results, args.output_dir, extra=provenance)

    if not scoring_only and not args.keep_intermediates:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"Removed intermediates: {work_dir} (pass --keep-intermediates to keep them)")


if __name__ == "__main__":
    main()
