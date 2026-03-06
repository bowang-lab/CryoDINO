"""
Author: Ahmadreza Attarpour & Sumin Kim
Run CryoDINO segmentation inference on NIfTI volumes and visualize in napari.

Loads a pretrained 3DINO backbone + trained segmentation head, runs sliding
window inference on one or more NIfTI volumes, saves predicted segmentation
masks, and opens the results in an interactive napari 3D viewer.

Usage:
  # Run inference then visualize
  python inference/run_inference_and_visualize.py \
    --input /path/to/TS_0002_0000.nii.gz \
    --pretrained-weights /path/to/teacher_checkpoint.pth \
    --checkpoint /path/to/best_model.pth \
    --run-inference

  # Multiple input files
  python inference/run_inference_and_visualize.py \
    --input /path/to/TS_0003_0000.nii.gz /path/to/TS_0004_0000.nii.gz \
    --pretrained-weights /path/to/teacher_checkpoint.pth \
    --checkpoint /path/to/best_model.pth \
    --run-inference

  # Skip visualization (inference only)
  python inference/run_inference_and_visualize.py \
    --input /path/to/TS_0002_0000.nii.gz \
    --pretrained-weights /path/to/teacher_checkpoint.pth \
    --checkpoint /path/to/best_model.pth \
    --run-inference --no-viewer

  # Visualize existing predictions (no inference)
  python inference/run_inference_and_visualize.py \
    --input /path/to/TS_0002_0000.nii.gz \
    --pretrained-weights /path/to/teacher_checkpoint.pth \
    --checkpoint /path/to/best_model.pth
"""

import os
import sys
import gc
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

# Add project root so `visualization._napari_utils` is importable
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "3DINO/dinov2/configs/train/vit3d_highres.yaml")


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run CryoDINO segmentation inference and visualize in napari.",
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Input NIfTI file(s)",
    )
    parser.add_argument(
        "--config-file", type=str, default=DEFAULT_CONFIG,
        help="Model config YAML",
    )
    parser.add_argument(
        "--pretrained-weights", type=str, required=True,
        help="SSL pretrained backbone weights",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Trained segmentation head checkpoint",
    )
    parser.add_argument(
        "--segmentation-head", type=str, default="ViTAdapterUNETR",
        choices=["UNETR", "Linear", "ViTAdapterUNETR"],
        help="Segmentation head architecture (default: ViTAdapterUNETR)",
    )
    parser.add_argument(
        "--image-size", type=int, default=112,
        help="Patch size for sliding window (default: 112)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=4,
        help="Number of segmentation classes (default: 4)",
    )
    parser.add_argument(
        "--input-channels", type=int, default=1,
        help="Number of input channels (default: 1)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Directory to save predictions (default: ./output)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.75,
        help="Sliding window overlap (default: 0.75)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Sliding window batch size (default: 4)",
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Skip napari visualization (inference only)",
    )
    parser.add_argument(
        "--run-inference", action="store_true",
        help="Run model inference (otherwise load existing predictions from output dir)",
    )
    return parser


def clear_cuda_memory():
    try:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _cleanup_model_imports():
    """Remove dinov2 from sys.path/modules to avoid conflicts with napari/numba.

    numba iterates sys.modules looking for builtins like ``print``; if
    ``dinov2.distributed`` is present it triggers an AttributeError.
    """
    dino_path = os.path.join(PROJECT_ROOT, "3DINO")
    while dino_path in sys.path:
        sys.path.remove(dino_path)
    to_remove = [k for k in sys.modules if k == "dinov2" or k.startswith("dinov2.")]
    for k in to_remove:
        del sys.modules[k]


# ---------------------------------------------------------------------------
# Inference path
# ---------------------------------------------------------------------------

def run_inference(args):
    """Run model inference on input files, save predictions, return viz samples."""
    import torch
    import nibabel as nib
    from functools import partial

    # Add 3DINO to path for dinov2 imports
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "3DINO"))

    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst,
        ScaleIntensityRangePercentiles, EnsureType, Lambda,
    )
    from monai.inferers import sliding_window_inference
    from dinov2.eval.setup import get_args_parser as get_base_args_parser, setup_and_build_model_3d
    from dinov2.eval.segmentation_3d.segmentation_heads import (
        UNETRHead, LinearDecoderHead, ViTAdapterUNETRHead,
    )
    from visualization._napari_utils import SampleData, load_nifti_volume

    # Build transforms:
    #   1. Z-score normalise the whole tomogram
    #   2. Per-patch percentile [-1, 1] applied inside the sliding-window predictor
    image_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Lambda(func=lambda x: (x - x.mean()) / (x.std() + 1e-8)),  # z-score whole tomo
        EnsureType(),
    ])

    # Build model
    base_parser = get_base_args_parser(add_help=False)
    base_defaults = base_parser.parse_args([])
    for key, val in vars(base_defaults).items():
        if not hasattr(args, key):
            setattr(args, key, val)
    if not hasattr(args, "cache_dir"):
        args.cache_dir = ""

    feature_model, autocast_dtype = setup_and_build_model_3d(args)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)

    head_cls = {
        "UNETR": UNETRHead,
        "Linear": LinearDecoderHead,
        "ViTAdapterUNETR": ViTAdapterUNETRHead,
    }[args.segmentation_head]

    seg_model = head_cls(
        feature_model, args.input_channels, args.image_size,
        args.num_classes, autocast_ctx,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    seg_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    seg_model.cuda()
    seg_model.eval()

    # Wrap model to apply per-patch percentile [-1, 1] normalisation inside
    # the sliding window — matches the patchwise normalisation used during training.
    patch_normalize = ScaleIntensityRangePercentiles(
        lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False,
    )
    def patchwise_predictor(patch_data):
        patch_data = torch.stack([patch_normalize(patch_data[i]) for i in range(patch_data.shape[0])])
        return seg_model(patch_data)

    image_size = (args.image_size,) * 3

    print(f"Processing {len(args.input)} file(s)...")

    viz_samples = []

    with torch.no_grad():
        for img_path in args.input:
            fname = os.path.basename(img_path)
            print(f"\nProcessing: {fname}")

            img = image_transforms(img_path)
            img = img.unsqueeze(0).cuda()

            logits = sliding_window_inference(
                img, image_size, args.batch_size,
                patchwise_predictor, overlap=args.overlap,
            )

            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Save prediction
            nib_img = nib.load(img_path)
            pred_nifti = nib.Nifti1Image(pred, affine=nib_img.affine, header=nib_img.header)
            out_name = fname.replace("_0000.nii.gz", ".nii.gz").replace("_0000.nii", ".nii")
            out_path = os.path.join(args.output_dir, out_name)
            nib.save(pred_nifti, out_path)
            print(f"  Saved: {out_path}")

            viz_samples.append(SampleData(
                name=fname,
                image=load_nifti_volume(img_path),
                prediction=pred,
            ))

            clear_cuda_memory()

    print(f"\nPredictions saved to: {args.output_dir}")

    # Cleanup distributed process group
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    return viz_samples


# ---------------------------------------------------------------------------
# Load-from-disk path
# ---------------------------------------------------------------------------

def load_existing_results(args):
    """Load saved predictions from output dir for visualization."""
    from visualization._napari_utils import SampleData, load_nifti_volume

    viz_samples = []
    for img_path in args.input:
        fname = os.path.basename(img_path)
        out_name = fname.replace("_0000.nii.gz", ".nii.gz").replace("_0000.nii", ".nii")
        pred_path = os.path.join(args.output_dir, out_name)

        if not os.path.exists(pred_path):
            print(f"  Warning: no prediction found for {fname} at {pred_path}, skipping")
            continue

        viz_samples.append(SampleData(
            name=fname,
            image=load_nifti_volume(img_path),
            prediction=load_nifti_volume(pred_path),
        ))

    print(f"Loaded {len(viz_samples)} sample(s) from {args.output_dir}")
    return viz_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_inference:
        viz_samples = run_inference(args)
    else:
        viz_samples = load_existing_results(args)

    if not args.no_viewer and viz_samples:
        # Remove dinov2 from sys.modules to prevent numba/napari conflict
        _cleanup_model_imports()
        from visualization._napari_utils import launch_viewer
        print("Launching napari viewer...")
        launch_viewer(viz_samples)


if __name__ == "__main__":
    main()
