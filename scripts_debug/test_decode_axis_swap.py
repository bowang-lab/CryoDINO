"""Standalone test: confirm the X<->Z axis-swap in decode_detections.

Run:
    python scripts_debug/test_decode_axis_swap.py

Why this matters
----------------
`sliding_window_accumulate` builds a global score/offset grid in [C, X', Y', Z']
order (volume axes D=X, H=Y, W=Z), then `decode_detections` turns each cell into
an (channel0, channel1, channel2) center. GT for val/test is stored as
[x, y, z] = [X, Y, Z] (load_detection_annotations -> patchify subtract x0 along
nibabel axis 0 = X).

This test places a single "hot" detection cell at a KNOWN grid location in a
NON-cubic feature map and checks which volume axis the decoded center's
channel-0 actually tracks. If channel-0 tracks the LAST axis (Z) instead of the
first axis (X), predicted centers are in (z, y, x) order -- swapped vs GT -- and
inference on non-cubic full tomograms is wrong (predicted X capped at the Z
extent). On cubic training patches the swap is invisible (all axes equal range).
"""
import sys
import os

import torch

# Import the real decode from the repo (no model / no heavy deps needed).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "3DINO"))
from dinov2.eval.detection_3d.loss import decode_detections  # noqa: E402

STRIDE = 2

# Non-cubic feature map mimicking a (630, 630, 184) volume at stride 2:
#   Xs=315 (axis0), Ys=315 (axis1), Zs=92 (axis2)
Xs, Ys, Zs = 315, 315, 92
C = 6

# Put a single hot cell at grid index (ix, iy, iz). Choose ix large so that, if
# channel-0 == X, the decoded X is ~600 (> Z extent of 184) -- only representable
# if channel-0 truly tracks the first axis.
ix, iy, iz = 300, 150, 40
expected_x = (ix + 0.5) * STRIDE   # 601.0  if channel0 == X (first axis)
expected_y = (iy + 0.5) * STRIDE   # 301.0
expected_z = (iz + 0.5) * STRIDE   # 81.0

logits = torch.full((1, C, Xs, Ys, Zs), -10.0)
logits[0, 0, ix, iy, iz] = 10.0            # class 0 hot at (ix, iy, iz)
offsets = torch.zeros(1, 3, Xs, Ys, Zs)    # zero offset -> center == anchor

pred_logits, pred_centers, anchors = decode_detections(logits, offsets, STRIDE)
# pick the anchor flattened index of our hot cell
flat = (ix * Ys + iy) * Zs + iz
center = pred_centers[0, flat].tolist()    # [c0, c1, c2]

print(f"hot grid cell (ix,iy,iz) = ({ix},{iy},{iz})  feature map {Xs}x{Ys}x{Zs}")
print(f"decoded center [c0,c1,c2] = {center}")
print(f"GT convention expects     = [x,y,z] = [{expected_x},{expected_y},{expected_z}]")
print()

c0, c1, c2 = center
match_xyz = abs(c0 - expected_x) < 1e-3 and abs(c2 - expected_z) < 1e-3
match_zyx = abs(c0 - expected_z) < 1e-3 and abs(c2 - expected_x) < 1e-3

if match_xyz:
    print("RESULT: channel0 == X (first axis). NO swap. decode matches GT [x,y,z].")
elif match_zyx:
    print("RESULT: channel0 == Z (LAST axis). *** AXIS SWAP CONFIRMED ***")
    print("        decoded centers are (z, y, x); GT is (x, y, z).")
    print(f"        max representable channel0 = {(Zs - 0.5) * STRIDE:.0f}  (Z extent),")
    print(f"        but GT x ranges up to ~{(Xs - 0.5) * STRIDE:.0f}  -> most particles unmatchable.")
    sys.exit(1)
else:
    print("RESULT: unexpected center ordering; inspect manually.")
    sys.exit(2)
