"""
napari viewer for particle-detection results (CryoDINO detection3d.py and the DeepETPicker baseline).

Overlays ground-truth picks and one or more methods' predicted picks on each tomogram of a datalist
split. Predictions are matched to GT with the same rule as the F-beta metric
(3DINO/dinov2/eval/detection_3d/metrics.py, compute_metrics), so what you see agrees with the score:

    GT (ring, coloured by class)       every labelled particle
    <method> picks                     green  = TP  (nearest pick within the match radius of a GT)
                                       yellow = duplicate (within radius of an already-matched GT;
                                                the metric counts it as FP)
                                       red    = FP  (no GT of that class within radius)
                                       grey   = below the score threshold (only with "show below thr")
    <method> missed                    orange = GT with no pick of its class within radius (FN)

Match radius: CZI = 0.5 x class radius (Kaggle), BYU = min_radius (1000 A) / voxel_size.
Thresholds default to the per-class thresholds chosen on validation and can be changed live.

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
STATUS_COLORS = {"tp": "#2ca02c", "dup": "#ffd700", "fp": "#d62728", "below": "#808080"}
MISSED_COLOR = "#ff7f0e"
METHOD_SYMBOLS = ["disc", "square", "triangle_up", "diamond"]


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
    from visualization._napari_utils import _import_napari

    napari = _import_napari()
    from magicgui.widgets import CheckBox, ComboBox, Container, FloatSlider, Label, PushButton, SpinBox

    viewer = napari.Viewer(ndisplay=2, title="CryoET Detection Viewer")
    names = [t.name for t in tomos]
    cls_names = matcher.class_names
    thresholds = {m.name: dict(m.thresholds) for m in methods}     # live, editable copy
    state = {"idx": None, "syncing": False}

    tomo_cb = ComboBox(choices=names, value=names[0], label="Tomogram")
    prev_btn, next_btn = PushButton(text="Previous"), PushButton(text="Next")
    z_spin = SpinBox(value=0, min=0, max=1, label="Z slice")
    class_cb = ComboBox(choices=["all"] + cls_names, value="all", label="Class")
    method_cb = ComboBox(choices=[m.name for m in methods], label="Threshold of")
    thr_slider = FloatSlider(value=0.0, min=0.0, max=1.0, step=0.005, label="Score thr")
    show_below = CheckBox(text="show below thr", value=False)
    reset_btn = PushButton(text="Reset to val thresholds")
    info = Label(value="")

    def display_radius_vox(c, vs):
        if matcher.dataset == "czi":
            return matcher.radii_ang[c] / vs
        return byu_marker_radius_ang / vs

    def class_filter(cls_col):
        if class_cb.value == "all":
            return np.ones(len(cls_col), dtype=bool)
        return cls_col == cls_names.index(class_cb.value)

    def replace_points(name, data_xyz, **kw):
        visible = viewer.layers[name].visible if name in viewer.layers else kw.pop("visible", True)
        kw.pop("visible", None)
        if name in viewer.layers:
            viewer.layers.remove(name)
        data = data_xyz[:, [2, 1, 0]] if len(data_xyz) else np.zeros((0, 3))
        if not len(data):
            kw = {k: v for k, v in kw.items() if k in ("symbol", "opacity")}
        try:    # napari >= 0.9: thick slices along Z, picks drawn as the sphere's cross-section
            viewer.add_points(data, name=name, visible=visible, ndim=3,
                              projection_mode="rescale_spherical", **kw)
        except (TypeError, ValueError):     # older napari
            viewer.add_points(data, name=name, visible=visible, ndim=3, out_of_slice_display=True, **kw)

    def draw_points():
        t = tomos[state["idx"]]
        vs = t.voxel_size
        lines = [f"{t.name}   voxel {vs:.3f} A   GT {len(t.gt)}"]

        for mi, m in enumerate(methods):
            sym = METHOD_SYMBOLS[mi % len(METHOD_SYMBOLS)]
            pred = m.preds.get(t.name)
            if pred is None:
                lines.append(f"{m.name}: no predictions for this tomogram")
                for suffix in (" picks", " missed"):
                    replace_points(m.name + suffix, np.zeros((0, 3)), symbol=sym)
                continue
            status, hit, cnt = matcher.classify(pred, t.gt, vs, thresholds[m.name])
            f, per = matcher.fbeta(cnt)
            sel = class_filter(pred[:, 3])
            if class_cb.value != "all":
                ci = cls_names.index(class_cb.value)
                cnt, f = cnt[ci:ci + 1], per[ci]
            tp, fp, fn = cnt.sum(axis=0)
            dup = int(np.sum((status == "dup") & sel))
            lines.append(f"{m.name}: TP {tp}  FP {fp} (dup {dup})  FN {fn}  F{matcher.beta:g}={f:.3f}")

            keep = class_filter(pred[:, 3]) & ((status != "below") | show_below.value)
            p, st = pred[keep], status[keep]
            pc = p[:, 3].astype(int)
            replace_points(m.name + " picks", p[:, :3],
                           size=[2 * display_radius_vox(c, vs) for c in pc],
                           face_color=[STATUS_COLORS[s] for s in st], border_width=0,
                           symbol=sym, opacity=0.5,
                           features={"status": list(st), "class": [cls_names[c] for c in pc],
                                     "score": p[:, 4]})
            miss = (~hit) & class_filter(t.gt[:, 3])
            mg = t.gt[miss]
            mc = mg[:, 3].astype(int)
            replace_points(m.name + " missed", mg[:, :3],
                           size=[2.3 * display_radius_vox(c, vs) for c in mc],
                           face_color="transparent", border_color=MISSED_COLOR, border_width=0.1,
                           symbol="square", features={"class": [cls_names[c] for c in mc]})

        # GT last so its rings sit on top of the picks
        g = t.gt[class_filter(t.gt[:, 3])]
        cls = g[:, 3].astype(int)
        replace_points("GT", g[:, :3], size=[2 * display_radius_vox(c, vs) for c in cls],
                       face_color="transparent", border_color=[CLASS_COLORS[c % 6] for c in cls],
                       border_width=0.08, symbol="disc",
                       features={"class": [cls_names[c] for c in cls]})
        info.value = "\n".join(lines)

    def sync_slider():
        m = method_cb.value
        cname = class_cb.value if class_cb.value != "all" else cls_names[-1]
        state["syncing"] = True
        thr_slider.value = float(np.clip(thresholds[m][cname], 0.0, 1.0))
        thr_slider.label = f"Thr {cname}" if class_cb.value != "all" else "Thr (all classes)"
        state["syncing"] = False

    def on_thr(val):
        if state["syncing"] or state["idx"] is None:
            return
        targets = cls_names if class_cb.value == "all" else [class_cb.value]
        for c in targets:
            thresholds[method_cb.value][c] = float(val)
        draw_points()

    def on_reset():
        for m in methods:
            thresholds[m.name] = dict(m.thresholds)
        sync_slider()
        draw_points()

    def show(idx):
        state["idx"] = idx
        t = tomos[idx]
        try:
            vol = load_volume_zyx(t.image_path)
        except FileNotFoundError:
            info.value = f"{t.name}: image not found at {t.image_path} (use --image-root)"
            return
        z = vol.shape[0]
        z_spin.max = z - 1
        sub = vol[::4, ::4, ::4]
        lo, hi = np.percentile(sub, [1, 99])
        if "image" in viewer.layers:
            viewer.layers.remove("image")
        kw = dict(name="image", colormap="gray", contrast_limits=(float(lo), float(hi)))
        try:    # keep the image a single slice when the Z margin below turns on thick slicing
            viewer.add_image(vol, projection_mode="none", **kw)
        except TypeError:
            viewer.add_image(vol, **kw)
        viewer.layers.move(viewer.layers.index("image"), 0)
        # Z margin = largest particle radius, so a pick shows on every slice its sphere crosses
        if hasattr(viewer.dims, "margin_left"):
            r = max(display_radius_vox(c, t.voxel_size) for c in range(len(cls_names)))
            viewer.dims.margin_left = (r,) + tuple(viewer.dims.margin_left[1:])
            viewer.dims.margin_right = (r,) + tuple(viewer.dims.margin_right[1:])
        draw_points()
        viewer.dims.set_point(0, z // 2)
        viewer.reset_view()

    def on_dims(event=None):
        if not state["syncing"]:
            state["syncing"] = True
            z_spin.value = int(viewer.dims.current_step[0])
            state["syncing"] = False

    def on_z(val):
        if not state["syncing"]:
            state["syncing"] = True
            viewer.dims.set_point(0, val)
            state["syncing"] = False

    tomo_cb.changed.connect(lambda v: show(names.index(v)))
    prev_btn.changed.connect(lambda: setattr(tomo_cb, "value", names[max(0, names.index(tomo_cb.value) - 1)]))
    next_btn.changed.connect(lambda: setattr(tomo_cb, "value",
                                             names[min(len(names) - 1, names.index(tomo_cb.value) + 1)]))
    class_cb.changed.connect(lambda v: (sync_slider(), draw_points()))
    method_cb.changed.connect(lambda v: sync_slider())
    thr_slider.changed.connect(on_thr)
    show_below.changed.connect(lambda v: draw_points())
    reset_btn.changed.connect(on_reset)
    z_spin.changed.connect(on_z)
    viewer.dims.events.current_step.connect(on_dims)

    legend = Label(value="picks (filled): green TP, yellow dup (FP), red FP, grey below thr\n"
                         "orange square: missed GT (FN); ring colour: GT class\n"
                         "methods: " + ", ".join(f"{m.name}={METHOD_SYMBOLS[i % 4]}"
                                                 for i, m in enumerate(methods)))
    viewer.window.add_dock_widget(
        Container(widgets=[tomo_cb, prev_btn, next_btn, z_spin, class_cb, method_cb, thr_slider,
                           show_below, reset_btn, info, legend]),
        name="Detection", area="right")

    sync_slider()
    show(0)
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
