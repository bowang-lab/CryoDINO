#!/bin/bash
# Download full CryoPanda dataset from URL list (1725 files)
# -----------------------------------------------------------
# Setup on Fir (Compute Canada):
#   1. Copy the URL file:  scp ~/Downloads/8a504807b5f947e58a1c57d7ff7a9658.txt <user>@fir.computecanada.ca:~/cryopanda_urls.txt
#   2. Copy this script:   scp slurm_scripts/download_cryopanda.sh <user>@fir.computecanada.ca:~/
#   3. On Fir:
#        tmux new -s cryopanda
#        bash ~/download_cryopanda.sh
#   4. Detach tmux: Ctrl+B then D

URL_FILE="$HOME/cryopanda_urls.txt"
OUT_DIR="$SCRATCH/cryopanda"
LOG_DIR="$OUT_DIR/logs"
N_PARALLEL=6        # keep conservative to avoid being rate-limited

mkdir -p "$OUT_DIR/particles_h5" "$OUT_DIR/metadata" "$LOG_DIR"

DONE_LOG="$LOG_DIR/done.txt"
FAIL_LOG="$LOG_DIR/failed.txt"
touch "$DONE_LOG" "$FAIL_LOG"

download_one() {
    url="$1"
    fname=$(echo "$url" | sed 's/.*fileName=\([^&]*\).*/\1/')

    # Route h5 files to particles_h5 subdir, everything else to metadata
    if [[ "$fname" == *.h5 ]]; then
        out_path="$OUT_DIR/particles_h5/$fname"
    else
        out_path="$OUT_DIR/metadata/$fname"
    fi

    # Skip already completed files
    if [ -f "$out_path" ] && [ -s "$out_path" ]; then
        echo "[SKIP] $fname"
        return 0
    fi

    wget -q -c -O "$out_path" "$url" \
        --tries=5 --timeout=120 --waitretry=20

    if [ $? -eq 0 ] && [ -s "$out_path" ]; then
        echo "$fname" >> "$DONE_LOG"
        echo "[OK]   $fname"
    else
        rm -f "$out_path"
        echo "$url" >> "$FAIL_LOG"
        echo "[FAIL] $fname"
    fi
}

export -f download_one
export OUT_DIR DONE_LOG FAIL_LOG

# Load GNU parallel (Compute Canada module)
module load parallel 2>/dev/null || true

TOTAL=$(wc -l < "$URL_FILE")
echo "=============================================="
echo "CryoPanda download"
echo "Total files : $TOTAL"
echo "Output dir  : $OUT_DIR"
echo "Parallel    : $N_PARALLEL"
echo "Logs        : $LOG_DIR"
echo "=============================================="

parallel -j "$N_PARALLEL" --bar --joblog "$LOG_DIR/parallel.log" \
    download_one {} < "$URL_FILE"

echo "=============================================="
echo "Done   : $(wc -l < $DONE_LOG) / $TOTAL"
echo "Failed : $(wc -l < $FAIL_LOG)"
[ -s "$FAIL_LOG" ] && echo "Re-run to retry failed files (completed files are skipped automatically)"
echo "=============================================="
