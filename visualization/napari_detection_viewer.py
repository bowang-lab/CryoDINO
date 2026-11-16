"""
napari viewer for particle-detection results (CryoDINO detection3d.py and the DeepETPicker baseline).

Same layout as the segmentation viewer (napari_viewer.py): the tomogram plus each method's predicted
picks (discs coloured by class; only the first method's layer starts visible, toggle the others
with their eye icons), at the per-class thresholds chosen on validation, and a hidden ground_truth
layer (rings coloured by class). --check scores the picks
with the F-beta metric (3DINO/dinov2/eval/detection_3d/metrics.py, compute_metrics).

Inputs (all coordinates are (x, y, z) = NIfTI array axes 0, 1, 2, the datalist convention):
    --datalist      JSON datalist; the --split entries give GT points, voxel_size and image path
    --cryodino      test_predictions.pkl (or .csv) from detection3d_test_dump.py
    --deepet        DeepETPicker pred root holding <split>/<name>.coords and metrics_<dataset>.json
                    (from baselines/deepetpicker_kit/infer_splits.py + evaluate_czi.py)

Usage:
    python visualization/napari_detection_viewer.py \
        --datalist czi_100_datalist.json --split test --image-root ./tomos \
        --deepet ./deepet_czii_100 --cryodino ./test_eval/test_predictions.pkl

    # no GUI: per-tomogram counts and the split F-beta at the thresholds in use, to check that the
    # coordinates line up (should reproduce test_f4_at_val_thresholds over the same tomograms)
    python visualization/napari_detection_viewer.py ... --check

napari is an optional dependency -- install with:  pip install "napari[all]"
"""

import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_METRICS_PY = os.path.join(REPO, "3DINO", "dinov2", "eval", "detection_3d", "metrics.py")

CLASS_COLORS = ["#1f77b4", "#9467bd", "#17becf", "#e377c2", "#bcbd22", "#8c564b"]
CLASS_COLOR_NAMES = ["blue", "purple", "cyan", "pink", "olive", "brown"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def strip_ext(fname):
    b = os.path.basename(fname)
    for ext in (".nii.gz", ".nii", ".mrc", ".rec", ".pt", ".h5"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return os.path.splitext(b)[0]


def load_metrics_module(path):
    """Load metrics.py straight from its file (needs only numpy/scipy, not torch or dinov2)."""
    spec = importlib.util.spec_from_file_location("det_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Tomo:
    name: str
    image_path: str
    voxel_size: float
    gt: np.ndarray            # (N, 4): x, y, z, class (0-based), -100 rows removed


@dataclass
class Method:
    name: str
    preds: Dict[str, np.ndarray]     # tomogram name -> (K, 5): x, y, z, class (0-based), score
    thresholds: Dict[str, float]     # class name -> score threshold (keep score >= thr)


def load_tomos(datalist, split, image_root):
    with open(datalist) as f:
        entries = json.load(f).get(split, [])
    if not entries:
        raise SystemExit(f"datalist has no '{split}' entries: {datalist}")
    tomos = []
    for e in entries:
        p = np.asarray(e.get("points", []), dtype=np.float64).reshape(-1, 5)
        p = p[p[:, 3] >= 0]
        path = e["image"]
        if not os.path.exists(path) and image_root:
            path = os.path.join(image_root, os.path.basename(path))
        if "voxel_size" not in e:
            raise SystemExit(f"{e['image']}: datalist entry has no voxel_size")
        tomos.append(Tomo(strip_ext(e["image"]), path, float(e["voxel_size"]), p[:, :4]))
    return tomos


def _thr_dict(thr, class_names):
    if thr is None:
        return {n: 0.0 for n in class_names}
    if isinstance(thr, dict):
        return {n: float(thr[n]) for n in class_names}
    return {n: float(thr) for n in class_names}     # BYU: one global threshold


def load_cryodino(path, class_names, name="CryoDINO"):
    """test_predictions.pkl / .csv written by dinov2/eval/detection_3d/prediction_io.py."""
    preds = {}
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            d = pickle.load(f)
        for img, vs, rec in zip(d["images"], d["voxel_sizes"], d["data"]):
            if "pred_centers_ang" in rec:                      # CZI
                xyz = np.asarray(rec["pred_centers_ang"], dtype=np.float64).reshape(-1, 3) / vs
                lab = np.asarray(rec["pred_labels"], dtype=np.float64)
            else:                                              # BYU
                xyz = np.asarray(rec["pred_centers_vox"], dtype=np.float64).reshape(-1, 3)
                lab = np.zeros(len(xyz))
            sc = np.asarray(rec["pred_scores"], dtype=np.float64)
            preds[strip_ext(img)] = np.column_stack([xyz, lab, sc]).reshape(-1, 5)
        return Method(name, preds, _thr_dict(d.get("val_thresholds"), class_names))

    # CSV: thresholds are not stored, so take them from the sibling .pkl if present, otherwise
    # recover them as the lowest score per class flagged above_val_threshold.
    rows = {}
    lowest_above = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            c, s = int(r["class_id"]), float(r["score"])
            rows.setdefault(r["tomogram"], []).append(
                [float(r["x_vox"]), float(r["y_vox"]), float(r["z_vox"]), c, s])
            if int(r["above_val_threshold"]):
                cn = class_names[c]
                lowest_above[cn] = min(lowest_above.get(cn, np.inf), s)
    preds = {k: np.asarray(v, dtype=np.float64).reshape(-1, 5) for k, v in rows.items()}
    sib = os.path.splitext(path)[0] + ".pkl"
    if os.path.exists(sib):
        with open(sib, "rb") as f:
            thr = _thr_dict(pickle.load(f).get("val_thresholds"), class_names)
    else:
        print(f"  {name}: no {os.path.basename(sib)} next to the CSV; thresholds recovered from "
              f"above_val_threshold (exact only up to the next-lower score)")
        thr = {n: float(lowest_above.get(n, 1.0)) for n in class_names}
    return Method(name, preds, thr)


def load_deepet(pred_root, split, dataset, class_names, name="DeepETPicker"):
    """<pred_root>/<split>/<name>.coords (cls 1-based, x, y, z, score) + metrics_<dataset>.json."""
    pdir = os.path.join(pred_root, split)
    if not os.path.isdir(pdir):
        raise SystemExit(f"{name}: no {split}/ folder under {pred_root}")
    preds = {}
    for fn in sorted(os.listdir(pdir)):
        if not fn.endswith(".coords"):
            continue
        fp = os.path.join(pdir, fn)
        if os.path.getsize(fp) == 0:
            a = np.zeros((0, 5))
        else:
            a = np.loadtxt(fp, delimiter="\t", ndmin=2)
        sc = a[:, 4] if a.shape[1] > 4 else np.ones(len(a))
        preds[fn[: -len(".coords")]] = np.column_stack([a[:, 1:4], a[:, 0] - 1, sc]).reshape(-1, 5)
    thr = None
    mj = os.path.join(pred_root, f"metrics_{dataset}.json")
    if os.path.exists(mj):
        with open(mj) as f:
            thr = json.load(f).get("val_thresholds")
    if thr is None:
        print(f"  {name}: no val_thresholds in {mj}; starting every class at threshold 0")
    return Method(name, preds, _thr_dict(thr, class_names))


def load_volume_zyx(path):
    """NIfTI (x, y, z) -> (z, y, x) float32 view for napari (Z is the scroll axis)."""
    import nibabel as nib

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.asarray(nib.load(path).dataobj, dtype=np.float32).T


# ---------------------------------------------------------------------------
# Matching (same rule as metrics.compute_metrics)
# ---------------------------------------------------------------------------

class Matcher:
    def __init__(self, dataset, mod, byu_min_radius=None):
        self.dataset = dataset
        self.mod = mod
        if dataset == "czi":
            m = mod.CZIDetectionMetrics()
            self.class_names = list(m.PARTICLE_NAMES)
            self.radii_ang = list(m.PARTICLE_RADII_ANG)
            self.weights = list(m.WEIGHTS)
            self.mult = m.distance_multiplier
            self.beta = m.beta
        else:
            kw = {} if byu_min_radius is None else {"min_radius": byu_min_radius}
            m = mod.BYUDetectionMetrics(**kw)
            self.class_names = list(m.PARTICLE_NAMES)
            self.min_radius = m.min_radius
            self.weights = [1]
            self.beta = m.beta

    def match_radius_vox(self, c, vs):
        if self.dataset == "czi":
            return self.radii_ang[c] * self.mult / vs
        return self.min_radius / vs

    def classify(self, pred, gt, vs, thr):
        """-> pred status array ('tp'|'dup'|'fp'|'below'), gt hit mask, per-class (tp, fp, fn)."""
        status = np.full(len(pred), "below", dtype=object)
        hit = np.zeros(len(gt), dtype=bool)
        counts = np.zeros((len(self.class_names), 3), dtype=int)
        for c, cname in enumerate(self.class_names):
            pi = np.where((pred[:, 3] == c) & (pred[:, 4] >= thr[cname]))[0]
            gi = np.where(gt[:, 3] == c)[0]
            r = self.match_radius_vox(c, vs)
            counts[c] = self.mod.compute_metrics(gt[gi, :3], r, pred[pi, :3])
            if len(pi) == 0:
                continue
            status[pi] = "fp"
            if len(gi) == 0:
                continue
            ptree = cKDTree(pred[pi, :3])
            d, j = ptree.query(gt[gi, :3])
            ok = d <= r
            hit[gi[ok]] = True
            d_gt, _ = cKDTree(gt[gi, :3]).query(pred[pi, :3])
            status[pi[d_gt <= r]] = "dup"
            status[pi[j[ok]]] = "tp"
        return status, hit, counts

    def fbeta(self, counts):
        """Weighted F-beta from per-class (tp, fp, fn), as in metrics._f4_at_threshold."""
        b2 = self.beta ** 2
        per = []
        for tp, fp, fn in counts:
            p = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            per.append((1 + b2) * p * rc / (b2 * p + rc) if p + rc else 0.0)
        w = np.asarray(self.weights, dtype=float)
        return float(np.dot(per, w) / w.sum()), per


# ---------------------------------------------------------------------------
# Check mode (no GUI)
# ---------------------------------------------------------------------------

def run_check(tomos, methods, matcher):
    for m in methods:
        total = np.zeros((len(matcher.class_names), 3), dtype=int)
        n = 0
        print(f"\n== {m.name}  thresholds: " + ", ".join(f"{k}={v:.4f}" for k, v in m.thresholds.items()))
        for t in tomos:
            if t.name not in m.preds:
                continue
            _, _, cnt = matcher.classify(m.preds[t.name], t.gt, t.voxel_size, m.thresholds)
            total += cnt
            n += 1
            f, _ = matcher.fbeta(cnt)
            tp, fp, fn = cnt.sum(axis=0)
            print(f"  {t.name:30s} tp={tp:4d} fp={fp:5d} fn={fn:4d}  F{matcher.beta:g}={f:.4f}")
        f, per = matcher.fbeta(total)
        print(f"  split F{matcher.beta:g} over {n}/{len(tomos)} tomograms = {f:.4f}   per class: "
              + ", ".join(f"{c}={v:.3f}" for c, v in zip(matcher.class_names, per)))


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def launch_viewer(tomos, methods, matcher, byu_marker_radius_ang=150.0):
    """Same layout as the segmentation viewer (visualization/_napari_utils.py launch_viewer):
    an 'image' layer, one prediction layer per method, a hidden 'ground_truth' layer, and a
    Navigation dock with Sample / Previous / Next / Z slice / info. Picks are the ones above each
    method's validation-chosen threshold, coloured by class; only the first method's layer
    starts visible."""
    from visualization._napari_utils import _import_napari

    napari = _import_napari()
    from magicgui import magicgui
    from magicgui.widgets import Container, Label, SpinBox

    viewer = napari.Viewer(ndisplay=2, title="CryoET Detection Viewer")
    names = [t.name for t in tomos]
    cls_names = matcher.class_names
    current_idx = [None]
    _syncing = [False]

    def display_radius_vox(c, vs):
        if matcher.dataset == "czi":
            return matcher.radii_ang[c] / vs
        return byu_marker_radius_ang / vs

    z_slider = SpinBox(value=0, min=0, max=1, label="Z slice", step=1)

    def _on_z_slider_changed(val):
        if not _syncing[0]:
            _syncing[0] = True
            viewer.dims.set_point(0, val)
            _syncing[0] = False

    z_slider.changed.connect(_on_z_slider_changed)

    def _on_dims_change(event=None):
        if not _syncing[0]:
            _syncing[0] = True
            z_slider.value = int(viewer.dims.current_step[0])
            _syncing[0] = False

    viewer.dims.events.current_step.connect(_on_dims_change)

    info_w = Label(value="")

    def _kept(m, t):
        pred = m.preds.get(t.name)
        if pred is None:
            return None
        thr = np.array([m.thresholds[c] for c in cls_names])
        cls = pred[:, 3].astype(int)
        return pred[pred[:, 4] >= thr[cls]]

    def _format_info(t, shape):
        lines = [f"Sample: {t.name}", f"Shape:  {shape}"]
        for m in methods:
            p = _kept(m, t)
            if p is None:
                lines.append(f"{m.name}: no predictions")
                continue
            cnt = np.bincount(p[:, 3].astype(int), minlength=len(cls_names))
            lines.append(f"{m.name}: {len(p)} picks")
            if len(cls_names) > 1:
                lines.append("  " + ", ".join(f"{c} {n}" for c, n in zip(cls_names, cnt)))
        lines.append("")
        lines.append("Colours (class):")
        lines += [f"  {CLASS_COLOR_NAMES[i % 6]}: {n}" for i, n in enumerate(cls_names)]
        lines.append("Method layers (eye icon to toggle): "
                     + ", ".join(f"{m.name} prediction" for m in methods))
        lines.append("ground_truth layer: hidden")
        return "\n".join(lines)

    def _add_points(data_xyz, name, **kw):
        data = data_xyz[:, [2, 1, 0]] if len(data_xyz) else np.zeros((0, 3))
        if not len(data):
            kw = {k: v for k, v in kw.items() if k in ("symbol", "opacity", "visible")}
        try:    # napari >= 0.9: thick slices along Z, picks drawn as the sphere's cross-section
            viewer.add_points(data, name=name, ndim=3, projection_mode="rescale_spherical", **kw)
        except (TypeError, ValueError):     # older napari
            viewer.add_points(data, name=name, ndim=3, out_of_slice_display=True, **kw)

    def _display_sample(idx):
        current_idx[0] = idx
        t = tomos[idx]
        try:
            vol = load_volume_zyx(t.image_path)
        except FileNotFoundError:
            info_w.value = f"{t.name}: image not found at {t.image_path} (use --image-root)"
            return
        z_max = vol.shape[0]
        z_mid = z_max // 2
        z_slider.max = z_max - 1

        viewer.layers.clear()

        # Raw image (percentile contrast: raw tomograms are too noisy for min/max)
        lo, hi = np.percentile(vol[::4, ::4, ::4], [1, 99])
        kw = dict(name="image", colormap="gray", opacity=0.7, contrast_limits=(float(lo), float(hi)))
        try:    # keep the image a single slice when the Z margin below turns on thick slicing
            viewer.add_image(vol, projection_mode="none", **kw)
        except TypeError:
            viewer.add_image(vol, **kw)
        # Z margin = largest particle radius, so a pick shows on every slice its sphere crosses
        if hasattr(viewer.dims, "margin_left"):
            r = max(display_radius_vox(c, t.voxel_size) for c in range(len(cls_names)))
            viewer.dims.margin_left = (r,) + tuple(viewer.dims.margin_left[1:])
            viewer.dims.margin_right = (r,) + tuple(viewer.dims.margin_right[1:])

        # Predictions, one layer per method
        for mi, m in enumerate(methods):
            p = _kept(m, t)
            if p is None:
                p = np.zeros((0, 5))
            pc = p[:, 3].astype(int)
            lname = "prediction" if len(methods) == 1 else f"{m.name} prediction"
            _add_points(p[:, :3], lname,
                        size=[2 * display_radius_vox(c, t.voxel_size) for c in pc],
                        face_color=[CLASS_COLORS[c % len(CLASS_COLORS)] for c in pc],
                        border_width=0, symbol="disc", opacity=0.5, visible=(mi == 0),
                        features={"class": [cls_names[c] for c in pc], "score": p[:, 4]})

        # Ground truth, hidden like the segmentation viewer's ground_truth layer
        gc = t.gt[:, 3].astype(int)
        _add_points(t.gt[:, :3], "ground_truth",
                    size=[2 * display_radius_vox(c, t.voxel_size) for c in gc],
                    face_color="transparent",
                    border_color=[CLASS_COLORS[c % len(CLASS_COLORS)] for c in gc],
                    border_width=0.08, symbol="disc", visible=False,
                    features={"class": [cls_names[c] for c in gc]})

        z_slider.value = z_mid
        viewer.dims.set_point(0, z_mid)
        viewer.reset_view()
        info_w.value = _format_info(t, vol.T.shape)

    @magicgui(call_button=False, sample={"choices": names, "label": "Sample"})
    def nav_widget(sample: str = names[0]) -> None:
        idx = names.index(sample)
        if idx != current_idx[0]:
            _display_sample(idx)

    nav_widget.sample.changed.connect(lambda val: nav_widget())

    @magicgui(call_button="Previous")
    def prev_btn() -> None:
        nav_widget.sample.value = names[max(0, current_idx[0] - 1)]

    @magicgui(call_button="Next")
    def next_btn() -> None:
        nav_widget.sample.value = names[min(len(names) - 1, current_idx[0] + 1)]

    viewer.window.add_dock_widget(
        Container(widgets=[nav_widget, prev_btn, next_btn, z_slider, info_w]),
        name="Navigation", area="right")

    _display_sample(0)
    return viewer


def main():
    ap = argparse.ArgumentParser(description="Overlay predicted vs GT particle picks in napari.")
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--split", default="test", help="datalist split to browse (default: test)")
    ap.add_argument("--image-root", default=None,
                    help="folder holding the tomograms, used when the datalist path does not exist "
                         "here (matched by basename)")
    ap.add_argument("--dataset", default="czi", choices=["czi", "byu"])
    ap.add_argument("--cryodino", action="append", default=[],
                    help="test_predictions.pkl/.csv; repeat for several runs (NAME=PATH to label)")
    ap.add_argument("--deepet", action="append", default=[],
                    help="DeepETPicker pred root (holds <split>/ and metrics_<dataset>.json); "
                         "repeatable, NAME=PATH to label")
    ap.add_argument("--tomograms", default=None, help="comma-separated subset of tomogram names")
    ap.add_argument("--only-with-images", action="store_true",
                    help="list only tomograms whose image file exists locally")
    ap.add_argument("--metrics-py", default=DEFAULT_METRICS_PY)
    ap.add_argument("--byu-min-radius", type=float, default=None, help="BYU match radius in A")
    ap.add_argument("--byu-marker-radius", type=float, default=150.0,
                    help="BYU marker display radius in A (display only)")
    ap.add_argument("--check", action="store_true", help="print counts and F-beta, no GUI")
    args = ap.parse_args()

    matcher = Matcher(args.dataset, load_metrics_module(args.metrics_py), args.byu_min_radius)

    def split_name(spec, default):
        if "=" in spec and not os.path.exists(spec):
            n, p = spec.split("=", 1)
            return n, p
        return default, spec

    methods: List[Method] = []
    for i, spec in enumerate(args.cryodino):
        n, p = split_name(spec, "CryoDINO" if len(args.cryodino) == 1 else f"CryoDINO{i + 1}")
        methods.append(load_cryodino(p, matcher.class_names, n))
    for i, spec in enumerate(args.deepet):
        n, p = split_name(spec, "DeepETPicker" if len(args.deepet) == 1 else f"DeepETPicker{i + 1}")
        methods.append(load_deepet(p, args.split, args.dataset, matcher.class_names, n))
    if not methods:
        raise SystemExit("give at least one --cryodino or --deepet")

    tomos = load_tomos(args.datalist, args.split, args.image_root)
    if args.tomograms:
        keep = set(args.tomograms.split(","))
        tomos = [t for t in tomos if t.name in keep]
    if args.only_with_images:
        tomos = [t for t in tomos if os.path.exists(t.image_path)]
    tomos = [t for t in tomos if any(t.name in m.preds for m in methods)]
    if not tomos:
        raise SystemExit("no tomogram of the split has predictions from any method")
    for m in methods:
        print(f"{m.name}: predictions for {sum(t.name in m.preds for t in tomos)}/{len(tomos)} tomograms")

    if args.check:
        run_check(tomos, methods, matcher)
        return

    sys.path.insert(0, REPO)
    launch_viewer(tomos, methods, matcher, args.byu_marker_radius)
    import napari
    napari.run()


if __name__ == "__main__":
    main()
