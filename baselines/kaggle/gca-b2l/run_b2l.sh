#!/usr/bin/env bash
# B2-L on one stream: Popper through the 24 systems in order, earlier programs lifted into the
# background knowledge (Forgetgol setting). 120 s per word, at most 4 searches at once (one per
# core on a Kaggle CPU notebook). Progress is saved after every system; rerunning resumes.
#   bash run_b2l.sh <stream name, e.g. stream_s50_n0.0> <dataset dir>
set -euo pipefail
STREAM="$1"; SRC="$2"
POPPER_DIR="${POPPER_DIR:-/root/popper}"; WORK="${WORK:-/kaggle/working}"
export POPPER_CMD="$(command -v uv) run --project $POPPER_DIR $POPPER_DIR/popper.py"
mkdir -p "$WORK"
PY="$(command -v python3 || command -v python)"
"$PY" "$SRC/run_popper.py" "$SRC/data/$STREAM.jsonl" --out "$WORK/b2l_$STREAM.json" --accumulate \
  --checkpoint "$WORK/ckpt_b2l_$STREAM.json" --timeout 120 --max-par "${MAX_PAR:-4}" 2>&1 | tee -a "$WORK/log_b2l_$STREAM.txt"
test -f "$WORK/b2l_$STREAM.json" && echo "DONE -> $WORK/b2l_$STREAM.json"
