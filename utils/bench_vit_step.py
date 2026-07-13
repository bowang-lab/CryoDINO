"""Single-GPU forward+backward benchmark of the CryoDINO ViT-Large backbone.

Isolates raw compute from FSDP/NCCL/dataloader: if this is fast, the training
slowdown comes from distributed communication or dispatch; if slow, from
kernels. Uses the same crop shapes as pretraining (batch 275/GPU).

Run from the 3DINO directory on a GPU node:
    cd 3DINO && PYTHONPATH=. python ../utils/bench_vit_step.py
Optionally with profiler output:
    PYTHONPATH=. python ../utils/bench_vit_step.py --profile
"""

import argparse
import time

import torch

from dinov2.models.vision_transformer import vit_large_3d

import os
if os.environ.get("CUDNN_BENCHMARK") == "1":
    torch.backends.cudnn.benchmark = True
    print(">>> torch.backends.cudnn.benchmark = True")

N_WARMUP = 3
N_ITERS = 10

# batch 275/GPU: 2 global crops 96^3, 8 local crops 48^3.
# Reduced here to fit a single GPU without FSDP sharding headroom concerns:
# scale measured time by (275 / BATCH).
BATCH = 275


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="print top CUDA ops")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--stages", action="store_true",
                        help="time patch-embed vs blocks separately (low memory)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"batch: {args.batch} (training uses 275; scale time by {275/args.batch:.2f}x)")

    model = vit_large_3d(img_size=96, patch_size=16, block_chunks=4,
                         drop_path_rate=0.3, init_values=1e-5,
                         drop_path_uniform=True).cuda().half()

    g = torch.randn(2 * args.batch, 1, 96, 96, 96, device="cuda", dtype=torch.half)
    l = torch.randn(8 * args.batch, 1, 48, 48, 48, device="cuda", dtype=torch.half)

    def fwd(g_, l_):
        return model.forward_features_list([g_, l_], [None, None])

    if args.compile:
        # compile the callable we actually invoke — torch.compile(model) would
        # be bypassed because forward_features_list is called directly
        fwd = torch.compile(fwd)
        print("torch.compile enabled (first iters include compilation)")

    def step():
        out = fwd(g, l)
        loss = sum(o["x_norm_clstoken"].float().pow(2).mean() for o in out)
        loss.backward()
        model.zero_grad(set_to_none=True)

    if args.stages:
        def timeit(fn, *a):
            for _ in range(3):
                out = fn(*a)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10):
                fn(*a)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / 10, out

        with torch.no_grad():
            for label, inp in (("globals", g), ("locals ", l)):
                ms_prep, tokens = timeit(lambda t: model.prepare_tokens_with_masks(t, None), inp)
                ms_chunk, _ = timeit(lambda t: model.blocks[0](t), tokens)
                print(f"{label}: prepare_tokens (conv3d patch embed + pos): {ms_prep:8.1f} ms")
                print(f"{label}: one BlockChunk fwd (6 blocks, x4 total):   {ms_chunk:8.1f} ms")
                print(f"{label}: est. full forward: {ms_prep + 4 * ms_chunk:8.1f} ms")
        return

    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(N_ITERS):
        step()
    torch.cuda.synchronize()
    per_iter = (time.time() - t0) / N_ITERS
    scaled = per_iter * 275 / args.batch
    print(f"\nfwd+bwd per iter (batch {args.batch}): {per_iter:.3f} s")
    print(f"scaled to batch 275:                  {scaled:.3f} s")
    print("(training does ~3 backbone passes/iter: student global+local, teacher global)")

    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
