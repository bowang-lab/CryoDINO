# Diagnostic for B200 eval failures. Run from ~/projects/CryoDINO/3DINO with PYTHONPATH=.
# Usage: python diag_eval_precision.py <ckpt> <n_tomos> <mode>   mode in {fp16, bf16, fp32}
# Evaluates best_model.pth on the first n val tomograms (BYU) and reports, per tomogram,
# non-finite tile count, logit range, #detections, and F2 at the val threshold sweep.
import sys, torch, contextlib
import dinov2.eval.detection3d as D
from dinov2.eval.setup import get_args_parser, setup_and_build_model_3d
from dinov2.eval.detection_3d.detection_heads import ViTAdapterUNETRHead
from dinov2.eval.detection_3d.augmentations import make_transforms
from dinov2.eval.detection_3d.metrics import get_metric
from dinov2.data.loaders import make_detection_dataset_3d

ckpt, n_tomos, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
BASE = "/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection/byu_detection_patches_128"
args = D.add_seg_args(get_args_parser(add_help=True)).parse_args([
    "--config-file", "dinov2/configs/train/vit3d_highres.yaml", "--pretrained-weights", "",
    "--output-dir", "/tmp/diag_out", "--train-feature-model"])
feature_model, _ = setup_and_build_model_3d(args)
model = ViTAdapterUNETRHead(feature_model, 1, 128, 1, contextlib.nullcontext)
model.load_state_dict(torch.load(ckpt, map_location="cpu")); model.cuda().eval()

dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mode)
# monkey-patch the autocast used inside sliding_window_accumulate
D.torch.cuda.amp.autocast = (lambda: torch.autocast("cuda", dtype=dtype)) if dtype else (lambda: contextlib.nullcontext())
stats = {"nonfinite_tiles": 0, "tiles": 0}
orig_fwd = model.forward
def fwd(x):
    c, o = orig_fwd(x)
    stats["tiles"] += 1; stats["nonfinite_tiles"] += int(not torch.isfinite(c).all())
    return c, o
model.forward = fwd

_, val_tf = make_transforms(crop_size=128)
_, val_ds, _, _, _ = make_detection_dataset_3d("byu", 100, BASE, val_tf, val_tf, "/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/cache_dir_downstream_detection/random_init_byu_100pct_vit_adapter", 1)
metric = get_metric("byu")
with torch.no_grad():
    for i in range(n_tomos):
        b = D.detection_collate_fn([val_ds[i]])
        s, _ = D.sliding_window_accumulate(model, b["image"].cuda(), 128, 2, 0.5)
        print(f"tomo {i}: nonfinite_voxels={(~torch.isfinite(s)).sum().item()} "
              f"logit[min,max]=[{torch.nan_to_num(s).min():.2f},{torch.nan_to_num(s).max():.2f}] "
              f"tiles={stats['tiles']} nonfinite_tiles={stats['nonfinite_tiles']}", flush=True)
        D.detection_val_iter(model, b, metric, 128, 2, 0.5, "byu")
print(mode, "F2 sweep:", metric.threshold_sweep())
