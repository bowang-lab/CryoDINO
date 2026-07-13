"""Diagnose why conv3d patch embed is slow in some envs.

Reports cuDNN version, which kernel conv3d dispatches to, and compares:
  - cuDNN enabled (default)
  - cuDNN benchmark autotune
  - cuDNN DISABLED (native fallback)

Run in each env (cryoet, cryodino) on a GPU node:
    python utils/diag_conv3d.py
"""

import torch
import torch.nn as nn

DEV, DT = "cuda", torch.half
E, C, P = 1024, 1, 16
# local branch is the worst case: many small crops
SHAPE = (800, C, 48, 48, 48)


def bench(conv, x, n=20):
    for _ in range(5):
        conv(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        conv(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"torch:          {torch.__version__}")
    print(f"cuda (torch):   {torch.version.cuda}")
    print(f"cudnn version:  {torch.backends.cudnn.version()}")
    print(f"cudnn enabled:  {torch.backends.cudnn.enabled}")
    print(f"cudnn available:{torch.backends.cudnn.is_available()}")

    conv = nn.Conv3d(C, E, kernel_size=P, stride=P).to(DEV, DT)
    x = torch.randn(*SHAPE, device=DEV, dtype=DT)

    # 1) which kernel does conv3d actually call?
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        conv(x); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            conv(x); torch.cuda.synchronize()
    print("\n--- top CUDA kernels for one conv3d call ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    with torch.no_grad():
        # 2) default
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        print(f"cudnn ON  (default):   {bench(conv, x):9.2f} ms")

        # 3) benchmark autotune
        torch.backends.cudnn.benchmark = True
        print(f"cudnn ON  (benchmark): {bench(conv, x):9.2f} ms")
        torch.backends.cudnn.benchmark = False

        # 4) cudnn disabled -> native fallback
        torch.backends.cudnn.enabled = False
        print(f"cudnn OFF (native):    {bench(conv, x):9.2f} ms")
        torch.backends.cudnn.enabled = True


if __name__ == "__main__":
    main()
