#!/bin/bash
# One-time setup: clone DeePiCt and apply the compatibility patches needed to run its
# (Python-3.7-era) code on a modern stack (torch 2.x / numpy 2.x / pandas 2.x). Idempotent.
#
# Usage:  DEEPICT=/path/to/DeePiCt bash setup.sh
# The Python venv is built separately (see README) — needs: torch, monai, mrcfile, h5py,
# tensorboardX, pandas, scikit-image, scipy, pyyaml.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DEEPICT:=$HERE/DeePiCt}"   # clones into the pipeline folder by default; override with DEEPICT=

if [ ! -d "$DEEPICT" ]; then
  git clone https://github.com/ZauggGroup/DeePiCt "$DEEPICT"
fi
SRC="$DEEPICT/3d_cnn/src"
SCR="$DEEPICT/3d_cnn/scripts"
SF="$DEEPICT/spectrum_filter"

# 1) numpy>=1.24 removed np.int  -> int   (spectrum filter + clustering/postprocess)
sed -i 's/\.astype(np\.int)/.astype(int)/g' \
    "$SF/FilterUtils.py" \
    "$SRC/tomogram_utils/coordinates_toolbox/clustering.py" \
    "$SCR/FilterUtil.py"
# 2) torch>=2.0 removed ReduceLROnPlateau(verbose=...)
sed -i 's/patience=10, verbose=True/patience=10/' "$SCR/training.py"
# 3) pandas>=2.0 removed DataFrame.append -> pd.concat  (statistics + csv writers; idempotent)
python - "$SRC" <<'PY'
import pathlib, re, sys
src = pathlib.Path(sys.argv[1])
# matches  x = x.append(y, sort=False)  or  sort="False"  -> pd.concat([x, y], ignore_index=True)
pat = re.compile(r'(\w+)\s*=\s*\1\.append\((\w+),\s*sort=["\']?False["\']?\)')
for rel in ("constants/statistics.py", "file_actions/writers/csv.py", "plotting/statistics.py"):
    f = src / rel
    if not f.exists():
        continue
    t = f.read_text()
    t2 = pat.sub(lambda m: f"{m.group(1)} = pd.concat([{m.group(1)}, {m.group(2)}], ignore_index=True)", t)
    if t2 != t:
        f.write_text(t2)
        print("  patched append ->", rel)
PY
# 4) torch>=2.6 flipped torch.load(weights_only=True) default -> can't unpickle DeePiCt's
#    ModelDescriptor checkpoints. Add weights_only=False to every torch.load(...) call.
python - "$DEEPICT/3d_cnn" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
pat = re.compile(r'torch\.load\(([^)]*)\)')
def fix(m):
    args = m.group(1)
    return m.group(0) if "weights_only" in args else f"torch.load({args}, weights_only=False)"
for f in list((root / "scripts").glob("*.py")) + list((root / "src").rglob("*.py")):
    t = f.read_text()
    t2 = pat.sub(fix, t)
    if t2 != t:
        f.write_text(t2)
        print("  patched torch.load ->", f.relative_to(root))
PY

echo "DeePiCt patched at: $DEEPICT"
echo "Verify:"
grep -q "astype(int)" "$SF/FilterUtils.py"                                   && echo "  [ok] FilterUtils np.int"
grep -q "patience=10)" "$SCR/training.py"                                    && echo "  [ok] ReduceLROnPlateau verbose"
! grep -q "\.append(.*sort=" "$SRC/constants/statistics.py" "$SRC/file_actions/writers/csv.py" \
                                                                             && echo "  [ok] DataFrame.append -> pd.concat"
grep -q "weights_only=False" "$SCR/segment.py"                              && echo "  [ok] torch.load weights_only=False"
