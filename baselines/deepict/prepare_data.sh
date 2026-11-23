#!/bin/bash
# EXAMPLE data prep for the CZII .pt-cube dataset: .pt -> .mrc, spectrum matching, metadata.csv.
# For a DIFFERENT dataset, replace step 1 with your own raw+mask -> .mrc + metadata.csv, then
# keep step 2 (spectrum matching) if desired.
#
# Usage:  DATA=<out_dir> [ONLY="TE10_0000,TE11_0000"] bash prepare_data.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DEEPICT:=$HERE/DeePiCt}"   # override with DEEPICT= if the repo lives elsewhere
: "${VENV:=$HERE/venv}"         # override with VENV= to point at your venv
: "${DATA:=$HERE/data}"         # output dir for the .mrc + metadata.csv
ONLY="${ONLY:-}"

# Needs a real-RAM node: spectrum matching FFTs each full 512^3 cube (~a few GB/cube).
# A 4 GB login/interactive shell will be OOM-killed — use >=16 GB.
[ -f "$VENV/bin/activate" ] || { echo "ERROR: venv not found at $VENV — build it (see README), or set VENV=."; exit 1; }
module load python/3.11 2>/dev/null || true
source "$VENV/bin/activate"

echo "== 1) convert .pt -> .mrc (raw + masks) + metadata.csv =="
python "$HERE/pt_to_mrc.py" --dst "$DATA" ${ONLY:+--only-tomos "$ONLY"}

echo "== 2) spectrum matching: extract target from a reference cube, match all raw -> filt =="
mkdir -p "$DATA/filt" "$DATA/spectra"
REF=$(ls "$DATA"/raw/*.mrc | head -1)
python "$DEEPICT/spectrum_filter/extract_spectrum.py" --input "$REF" \
    --output "$DATA/spectra/target_spectrum.tsv"
for f in "$DATA"/raw/*.mrc; do
  b=$(basename "$f")
  python "$DEEPICT/spectrum_filter/match_spectrum.py" --input "$f" \
      --target "$DATA/spectra/target_spectrum.tsv" --output "$DATA/filt/$b"
done
sed -i "s#/raw/#/filt/#" "$DATA/metadata.csv"   # point the 'tomo' column at spectrum-matched cubes
echo "== data ready: $DATA/metadata.csv =="
