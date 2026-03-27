"""
Author: Ahmadreza Attarpour
3DINO Segmentation Inference Script

Runs sliding window inference using a pretrained 3DINO backbone + trained segmentation head
(Linear, UNETR, or ViTAdapterUNETR) on 3D NIfTI volumes. Saves predicted segmentation masks
as NIfTI files and optionally computes per-image dice metrics when labels are available.

Two input modes:
  1. --datalist: Pass a JSON datalist file. The script uses the "test" split entries.
     Each entry must have an "image" key. If a "label" key exists and the file is found,
     dice metrics are computed for that sample.
  2. --input-dir: Pass a directory of NIfTI files. Optionally pass --label-dir for metrics.
     Labels are matched by filename (stripping _0000 suffix from image names).

Usage examples:

  # Mode 1: Using a datalist JSON (test split with labels)
  python inference/segmentation3d_inference.py \\
    --config-file 3DINO/dinov2/configs/train/vit3d_highres.yaml \\
    --pretrained-weights /path/to/teacher_checkpoint.pth \\
    --checkpoint /path/to/best_model.pth \\
    --segmentation-head UNETR \\
    --image-size 112 \\
    --num-classes 4 \\
    --datalist /path/to/Dataset001_CZII_10001_100_datalist.json \\
    --output-dir /path/to/output/

  # Mode 2: Using input directory (no labels, predictions only)
  python inference/segmentation3d_inference.py \\
    --config-file 3DINO/dinov2/configs/train/vit3d_highres.yaml \\
    --pretrained-weights /path/to/teacher_checkpoint.pth \\
    --checkpoint /path/to/best_model.pth \\
    --segmentation-head Linear \\
    --image-size 112 \\
    --num-classes 4 \\
    --input-dir /path/to/images/ \\
    --output-dir /path/to/output/

  # Mode 2: With labels for evaluation
  python inference/segmentation3d_inference.py \\
    --config-file 3DINO/dinov2/configs/train/vit3d_highres.yaml \\
    --pretrained-weights /path/to/teacher_checkpoint.pth \\
    --checkpoint /path/to/best_model.pth \\
    --segmentation-head UNETR \\
    --image-size 112 \\
    --num-classes 4 \\
    --input-dir /path/to/images/ \\
    --label-dir /path/to/labels/ \\
    --output-dir /path/to/output/

Outputs:
  - <output-dir>/<image_name>.nii.gz: Predicted segmentation mask (uint8)
  - <output-dir>/metrics.json: Per-image and overall dice scores (only when labels are available)
"""

import os
import sys
import gc
import json
import glob

import torch
import numpy as np
import nibabel as nib
from functools import partial

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRangePercentiles,
    EnsureType,
    Lambda,
    AsDiscrete,
)
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

# Add 3DINO to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '3DINO'))

from dinov2.eval.setup import get_args_parser as get_base_args_parser, setup_and_build_model_3d
from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead, LinearDecoderHead, ViTAdapterUNETRHead


def get_args_parser():
    parser = get_base_args_parser(add_help=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained segmentation head checkpoint (best_model.pth)")
    parser.add_argument("--segmentation-head", type=str, required=True,
                        choices=["UNETR", "Linear", "ViTAdapterUNETR"],
                        help="Segmentation head type")
    parser.add_argument("--image-size", type=int, required=True,
                        help="Input patch size (e.g., 112)")
    parser.add_argument("--num-classes", type=int, required=True,
                        help="Number of segmentation classes")
    parser.add_argument("--input-channels", type=int, default=1,
                        help="Number of input channels")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for sliding window inference")
    parser.add_argument("--datalist", type=str, default=None,
                        help="Path to datalist JSON (uses test split for inference)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing input NIfTI images")
    parser.add_argument("--label-dir", type=str, default=None,
                        help="Optional directory containing label NIfTI files (used with --input-dir)")
    parser.add_argument("--overlap", type=float, default=0.75,
                        help="Sliding window overlap (default: 0.75)")
    parser.add_argument("--dataset-name", type=str, default="dataset",
                        help="Dataset name for metrics (used to determine metric type)" )
    parser.add_argument("--deep-supervision", action="store_true",
                        help="Must match training setting — instantiates DS heads so checkpoint loads correctly")
    parser.add_argument("--cpu-metrics", action="store_true",
                        help="Compute Dice/HD95 metrics on CPU (avoids GPU OOM on large volumes)")
    return parser

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


def build_transforms(dataset_name):
    """Build inference transforms matching the training val pipeline:
      - ZScore at tomogram level (load time)
      - Per-patch percentile norm handled inside make_patchwise_predictor
    """
    image_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Lambda(func=lambda x: (x - x.mean()) / (x.std() + 1e-8)),  # z-score whole tomo
        EnsureType(),
    ])
    # Dataset010: merge all foreground labels to class 1 (binary)
    if "10010" in dataset_name:
        label_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Lambda(func=lambda x: (x > 0).float()),
            EnsureType(),
        ])
    # Dataset049 (12049): merge classes 1,5 into bg; remap 2→1, 3→2, 4→3 → 4 classes
    elif "12049" in dataset_name:
        def _remap_12049(x):
            out = torch.zeros_like(x)
            out[x == 2] = 1
            out[x == 3] = 2
            out[x == 4] = 3
            return out.float()
        label_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Lambda(func=_remap_12049),
            EnsureType(),
        ])
    else:
        label_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            EnsureType(),
        ])
    return image_transforms, label_transforms

def make_patchwise_predictor(model):
    """Wrap model to normalize each sliding window patch before forward pass,
    matching per-patch percentile normalization used during pretraining."""
    from monai.transforms import ScaleIntensityRangePercentiles
    normalize = ScaleIntensityRangePercentiles(
        lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False
    )

    def predictor(patch_data):
        # patch_data: (N, C, H, W, D) from sliding_window_inference
        # normalize each sample in the batch independently, matching per-crop training normalization
        normalized = torch.stack([normalize(patch_data[i]) for i in range(patch_data.shape[0])])
        return model(normalized)
    return predictor

def main():
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set defaults for attributes required by config setup but unused for inference
    if not hasattr(args, 'cache_dir'):
        args.cache_dir = ""

    # Build feature model
    feature_model, autocast_dtype = setup_and_build_model_3d(args)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)

    # Build segmentation head
    if args.segmentation_head == 'UNETR':
        seg_model = UNETRHead(feature_model, args.input_channels, args.image_size, args.num_classes, autocast_ctx,
                              deep_supervision=args.deep_supervision)
    elif args.segmentation_head == 'Linear':
        seg_model = LinearDecoderHead(feature_model, args.input_channels, args.image_size, args.num_classes, autocast_ctx)
    elif args.segmentation_head == 'ViTAdapterUNETR':
        seg_model = ViTAdapterUNETRHead(feature_model, args.input_channels, args.image_size, args.num_classes, autocast_ctx,
                                        deep_supervision=args.deep_supervision)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    seg_model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    seg_model.cuda()
    seg_model.eval()

    # Build transforms
    image_transforms, label_transforms = build_transforms(args.dataset_name)

    # Build list of (image_path, label_path_or_None) pairs
    samples = []
    if args.datalist:
        with open(args.datalist) as f:
            datalist = json.load(f)
        test_entries = datalist.get('test', [])
        if not test_entries:
            print(f"No 'test' entries found in {args.datalist}")
            return
        for entry in test_entries:
            img_path = entry['image']
            label_path = entry.get('label', None)
            # Check if label file actually exists
            if label_path and not os.path.exists(label_path):
                print(f"  Warning: label not found, skipping metrics for {label_path}")
                label_path = None
            samples.append((img_path, label_path))
    elif args.input_dir:
        image_files = sorted(
            glob.glob(os.path.join(args.input_dir, '*.nii.gz')) +
            glob.glob(os.path.join(args.input_dir, '*.nii'))
        )
        for img_path in image_files:
            label_path = None
            if args.label_dir:
                fname = os.path.basename(img_path)
                label_name = fname.replace('_0000.nii.gz', '.nii.gz').replace('_0000.nii', '.nii')
                candidate = os.path.join(args.label_dir, label_name)
                if os.path.exists(candidate):
                    label_path = candidate
            samples.append((img_path, label_path))
    else:
        print("Error: must provide either --datalist or --input-dir")
        return

    if not samples:
        print("No images found")
        return

    # Set up metric if any sample has labels
    has_any_labels = any(lbl is not None for _, lbl in samples)
    if has_any_labels:
        metric = CryoMetrics(args.num_classes)

    print(f"Found {len(samples)} images ({sum(1 for _, l in samples if l)} with labels)")
    image_size = (args.image_size,) * 3
    results = {}

    with torch.no_grad():
        for img_path, label_path in samples:
            fname = os.path.basename(img_path)
            print(f"\nProcessing: {fname}")

            # Load and transform image
            img = image_transforms(img_path)
            img = img.unsqueeze(0).cuda()  # add batch dim

            # Run sliding window inference
            # logits = sliding_window_inference(
            #     img, image_size, args.batch_size,
            #     seg_model, overlap=args.overlap
            # )
            logits = sliding_window_inference(
                img, image_size, args.batch_size,
                make_patchwise_predictor(seg_model), overlap=args.overlap
            )
            # Get prediction
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Save prediction as NIfTI with same affine as input
            nib_img = nib.load(img_path)
            pred_nifti = nib.Nifti1Image(pred, affine=nib_img.affine, header=nib_img.header)
            out_name = fname.replace('_0000.nii.gz', '.nii.gz').replace('_0000.nii', '.nii')
            out_path = os.path.join(args.output_dir, out_name)
            nib.save(pred_nifti, out_path)
            print(f"  Saved prediction: {out_path}")

            # free memory before metric computation
            del img, pred 

            # Compute dice if label exists
            if label_path:
                label = label_transforms(label_path)

                # Convert to plain torch tensors to avoid MetaTensor issues in decollate_batch.
                # --cpu-metrics keeps tensors on CPU to avoid GPU OOM on large volumes
                # (HD95 one-hot tensors can exceed 10+ GiB for full CryoET tomograms).
                if args.cpu_metrics:
                    logits_t = torch.tensor(logits.cpu().numpy())
                    label_t = torch.tensor(torch.as_tensor(label).unsqueeze(0).numpy())
                else:
                    logits_t = torch.tensor(logits.cpu().numpy()).cuda()
                    label_t = torch.tensor(torch.as_tensor(label).unsqueeze(0).numpy()).cuda()
                avg_dice, per_cls_dice, avg_hd, per_cls_hd = metric(logits_t, label_t)
                per_cls_dice = [float(d) for d in per_cls_dice]
                per_cls_hd   = [float(h) for h in per_cls_hd]
                results[fname] = {
                    'avg_dice':       float(avg_dice),
                    'per_class_dice': per_cls_dice,
                    'avg_hd95':       float(avg_hd),
                    'per_class_hd95': per_cls_hd,
                }
                print(f"  Dice: {avg_dice:.4f}, Per-class: {per_cls_dice}")
                print(f"  HD95: {avg_hd:.4f}, Per-class: {per_cls_hd}")
                del label, logits_t, label_t

            del logits
            clear_cuda_memory()

    # Save metrics
    if results:
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
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nOverall Dice: {all_avg_dice:.4f}, Per-class: {all_per_cls_dice}")
        print(f"Overall HD95: {all_avg_hd:.4f}, Per-class: {all_per_cls_hd}")
        print(f"Metrics saved to: {metrics_path}")
    else:
        print("\nNo metrics computed (no labels provided or no matches found)")


if __name__ == "__main__":
    main()
