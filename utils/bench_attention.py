"""Benchmark attention backends on the current GPU.

Compares xformers.ops.memory_efficient_attention (used by MemEffAttention in
3DINO) against PyTorch native F.scaled_dot_product_attention, on the exact
shapes CryoDINO pretraining uses (ViT-Large, batch 275/GPU, 96^3 global /
48^3 local crops).

Run on a GPU node:
    python utils/bench_attention.py
"""

import torch
import torch.nn.functional as F

N_WARMUP = 10
N_ITERS = 50

# ViT-Large: embed_dim=1024, 16 heads, head_dim=64
HEADS, HEAD_DIM = 16, 64

# (label, batch, seq_len) — batch 275/GPU: globals 2x275, locals 8x275
# tokens: (96/16)^3 + 1 cls = 217 global, (48/16)^3 + 1 = 28 local
SHAPES = [
    ("global crops B=550 N=217", 550, 217),
    ("local crops  B=2200 N=28", 2200, 28),
]


def bench(fn, *tensors):
    for _ in range(N_WARMUP):
        fn(*tensors)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N_ITERS):
        fn(*tensors)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / N_ITERS  # ms per call


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    cap = torch.cuda.get_device_capability(0)
    print(f"compute capability: sm_{cap[0]}{cap[1]}")

    try:
        import xformers
        import xformers.ops as xops
        print(f"xformers: {xformers.__version__}")
        have_xformers = True
    except ImportError:
        print("xformers: NOT AVAILABLE")
        have_xformers = False

    for label, B, N in SHAPES:
        print(f"\n=== {label} ===")
        # xformers layout: (B, N, heads, head_dim)
        q = torch.randn(B, N, HEADS, HEAD_DIM, device="cuda", dtype=torch.half)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        # SDPA layout: (B, heads, N, head_dim)
        qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))

        if have_xformers:
            ms = bench(xops.memory_efficient_attention, q, k, v)
            print(f"xformers mem_eff:              {ms:8.3f} ms")
            try:
                from xformers.ops.fmha import _dispatch_fw
                from xformers.ops.fmha.common import Inputs
                op = _dispatch_fw(Inputs(query=q, key=k, value=v), False)
                print(f"  dispatched backend: {op.NAME}")
            except Exception as e:
                print(f"  (backend introspection failed: {e})")

        ms = bench(F.scaled_dot_product_attention, qt, kt, vt)
        print(f"torch SDPA (auto backend):     {ms:8.3f} ms")

        from torch.nn.attention import SDPBackend, sdpa_kernel
        for backend in (
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
        ):
            try:
                with sdpa_kernel(backend):
                    ms = bench(F.scaled_dot_product_attention, qt, kt, vt)
                print(f"torch SDPA [{backend.name:<19}]: {ms:8.3f} ms")
            except Exception:
                print(f"torch SDPA [{backend.name:<19}]: unavailable")


if __name__ == "__main__":
    main()
