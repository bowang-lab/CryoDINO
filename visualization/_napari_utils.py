"""
napari 3D visualization utilities for segmentation QA.

Provides an interactive napari viewer for browsing (image, prediction, ground-truth)
triplets produced by segmentation3d_inference.py. Supports sample navigation,
Dice metric display, and 3D volumetric rendering.

napari is an optional dependency -- install with:
    pip install "napari[all]"
"""

import os
import glob
import json
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# Lazy napari import
# ---------------------------------------------------------------------------

def _import_napari():
    """Import napari lazily; raise a helpful error if not installed."""
    try:
        import napari
        return napari
    except ImportError:
        raise ImportError(
            "napari is required for visualization but not installed.\n"
            "Install it with:  pip install \"napari[all]\""
        )


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SampleData:
    """Holds one (image, prediction, label) triplet for the viewer."""
    name: str
    image: np.ndarray                       # raw volume (z, y, x)
    prediction: np.ndarray                  # predicted labels
    label: Optional[np.ndarray] = None      # ground-truth labels
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_nifti_volume(path: str) -> np.ndarray:
    """Load a NIfTI file and return its data as a numpy array."""
    return np.asarray(nib.load(path).dataobj)


def load_samples_from_output_dir(
    pred_dir: str,
    image_dir: str,
    label_dir: Optional[str] = None,
    metrics_path: Optional[str] = None,
) -> List[SampleData]:
    """Reconstruct SampleData triplets from saved prediction files.

    Parameters
    ----------
    pred_dir : str
        Directory containing predicted segmentation NIfTI files.
    image_dir : str
        Directory containing the raw input NIfTI images.
    label_dir : str, optional
        Directory containing ground-truth label NIfTI files.
    metrics_path : str, optional
        Path to metrics.json. If *None*, looks for ``pred_dir/metrics.json``.

    Returns
    -------
    list[SampleData]
    """
    # Discover prediction files
    pred_files = sorted(
        glob.glob(os.path.join(pred_dir, "*.nii.gz"))
        + glob.glob(os.path.join(pred_dir, "*.nii"))
    )
    # Filter out metrics.json artefacts (shouldn't match, but be safe)
    pred_files = [p for p in pred_files if not p.endswith(".json")]

    if not pred_files:
        raise FileNotFoundError(f"No NIfTI predictions found in {pred_dir}")

    # Load metrics if available
    if metrics_path is None:
        metrics_path = os.path.join(pred_dir, "metrics.json")
    per_image_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
        per_image_metrics = data.get("per_image", {})

    samples: List[SampleData] = []
    for pred_path in pred_files:
        pred_name = os.path.basename(pred_path)

        # Find matching raw image (may have _0000 suffix)
        img_path = _find_matching_image(pred_name, image_dir)
        if img_path is None:
            print(f"  Warning: no raw image found for {pred_name}, skipping")
            continue

        # Find matching label
        lbl_array = None
        if label_dir:
            lbl_path = os.path.join(label_dir, pred_name)
            if os.path.exists(lbl_path):
                lbl_array = load_nifti_volume(lbl_path)

        # Look up metrics by the _0000 image name or the pred name
        img_basename = os.path.basename(img_path)
        metrics = per_image_metrics.get(img_basename, per_image_metrics.get(pred_name, {}))

        samples.append(SampleData(
            name=pred_name,
            image=load_nifti_volume(img_path),
            prediction=load_nifti_volume(pred_path),
            label=lbl_array,
            metrics=metrics,
        ))

    if not samples:
        raise FileNotFoundError("Could not match any predictions to raw images")

    print(f"Loaded {len(samples)} samples for visualization")
    return samples


def _find_matching_image(pred_name: str, image_dir: str) -> Optional[str]:
    """Return the path to the raw image matching *pred_name*.

    Images may carry a ``_0000`` suffix that predictions do not.
    """
    # Direct match
    direct = os.path.join(image_dir, pred_name)
    if os.path.exists(direct):
        return direct

    # Try adding _0000 suffix  (e.g. foo.nii.gz -> foo_0000.nii.gz)
    for ext in (".nii.gz", ".nii"):
        if pred_name.endswith(ext):
            stem = pred_name[: -len(ext)]
            candidate = os.path.join(image_dir, f"{stem}_0000{ext}")
            if os.path.exists(candidate):
                return candidate

    return None


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def launch_viewer(samples: List[SampleData]) -> None:
    """Open an interactive napari viewer for browsing segmentation results.

    Parameters
    ----------
    samples : list[SampleData]
        Pre-loaded sample triplets.
    """
    napari = _import_napari()
    from magicgui import magicgui
    from magicgui.widgets import Label, Container, SpinBox

    # 2D viewer showing XY in-plane; Z is the scroll axis
    viewer = napari.Viewer(ndisplay=2, title="CryoET Segmentation Viewer")

    # State ----------------------------------------------------------------
    current_idx = [0]  # mutable so closures can update
    _syncing = [False]  # guard against recursive slider ↔ dims updates

    # Widget helpers -------------------------------------------------------
    sample_names = [s.name for s in samples]

    # Z-slice slider (range updated when sample changes) -------------------
    z_slider = SpinBox(value=0, min=0, max=1, label="Z slice", step=1)

    def _on_z_slider_changed(val: int) -> None:
        if not _syncing[0]:
            _syncing[0] = True
            viewer.dims.set_point(0, val)
            _syncing[0] = False

    z_slider.changed.connect(_on_z_slider_changed)

    def _on_dims_change(event=None) -> None:
        if not _syncing[0]:
            _syncing[0] = True
            z = int(viewer.dims.current_step[0])
            z_slider.value = z
            _syncing[0] = False

    viewer.dims.events.current_step.connect(_on_dims_change)

    def _display_sample(idx: int) -> None:
        """Load sample *idx* into the viewer."""
        current_idx[0] = idx
        s = samples[idx]

        # NIfTI arrays are (x, y, z); transpose to (z, y, x) so napari
        # scrolls along z and displays the in-plane XY view.
        image = s.image.T
        prediction = s.prediction.T
        label = s.label.T if s.label is not None else None

        # Update z-slider range BEFORE adding layers so the current_step
        # callback (fired by napari's _go_to_center_step) doesn't exceed max.
        z_max = image.shape[0]
        z_mid = z_max // 2
        z_slider.max = z_max - 1

        # Clear existing layers
        viewer.layers.clear()

        # Raw image
        viewer.add_image(image, name="image", colormap="gray", opacity=0.7)

        # Prediction
        viewer.add_labels(prediction.astype(np.int32), name="prediction", opacity=0.5)

        # Ground truth (hidden by default)
        if label is not None:
            viewer.add_labels(label.astype(np.int32), name="ground_truth", visible=False)

        # Jump to middle slice
        z_slider.value = z_mid
        viewer.dims.set_point(0, z_mid)

        viewer.reset_view()

        # Update info widget
        _update_info(s)

    # Info text widget -----------------------------------------------------
    info_label = {"widget": None}  # will hold reference

    def _format_metrics(s: SampleData) -> str:
        lines = [f"Sample: {s.name}"]
        lines.append(f"Shape:  {s.image.shape}")
        if s.metrics:
            avg = s.metrics.get("avg_dice")
            if avg is not None:
                lines.append(f"Avg Dice: {avg:.4f}")
            per_cls = s.metrics.get("per_class_dice")
            if per_cls:
                cls_str = ", ".join(f"{d:.4f}" for d in per_cls)
                lines.append(f"Per-class: [{cls_str}]")
        if s.label is None:
            lines.append("(no ground truth)")
        return "\n".join(lines)

    def _update_info(s: SampleData) -> None:
        if info_label["widget"] is not None:
            info_label["widget"].value = _format_metrics(s)

    # Navigation widget ----------------------------------------------------
    @magicgui(
        call_button=False,
        sample={"choices": sample_names, "label": "Sample"},
    )
    def nav_widget(sample: str = sample_names[0]) -> None:
        idx = sample_names.index(sample)
        if idx != current_idx[0]:
            _display_sample(idx)

    nav_widget.sample.changed.connect(lambda val: nav_widget())

    @magicgui(call_button="Previous")
    def prev_btn() -> None:
        idx = max(0, current_idx[0] - 1)
        nav_widget.sample.value = sample_names[idx]

    @magicgui(call_button="Next")
    def next_btn() -> None:
        idx = min(len(samples) - 1, current_idx[0] + 1)
        nav_widget.sample.value = sample_names[idx]

    # Assemble dock --------------------------------------------------------
    info_w = Label(value=_format_metrics(samples[0]))
    info_label["widget"] = info_w

    container = Container(widgets=[
        nav_widget,
        prev_btn,
        next_btn,
        z_slider,
        info_w,
    ])
    viewer.window.add_dock_widget(container, name="Navigation", area="right")

    # Show first sample
    _display_sample(0)

    napari.run()
