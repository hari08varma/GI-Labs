#!/usr/bin/env bash
# Kaggle CPU notebook setup for B2-L (needs Internet ON). Installs the same Popper as the Modal and
# laptop runs: Popper commit 897b087 (24 Oct 2026), SWI-Prolog >= 9.1.12, janus-swi via uv.
set -euo pipefail
POPPER_DIR="${POPPER_DIR:-/root/popper}"
echo "== SWI-Prolog"
if ! command -v swipl >/dev/null || ! swipl -g "current_prolog_flag(version,V), (V >= 90112 -> halt(0) ; halt(1))" -t "halt(1)"; then
  apt-get update -qq
  apt-get install -y -qq software-properties-common git build-essential pkg-config >/dev/null
  add-apt-repository -y ppa:swi-prolog/stable >/dev/null
  apt-get update -qq
  apt-get install -y -qq swi-prolog-nox >/dev/null
fi
swipl --version
swipl -g "current_prolog_flag(version,V), (V >= 90112 -> halt(0) ; (format('SWI-Prolog too old: ~w~n',[V]), halt(1)))" -t "halt(1)"
echo "== uv and Popper"
pip install -q uv
rm -rf "$POPPER_DIR"
git clone -q https://github.com/logic-and-learning-lab/Popper.git "$POPPER_DIR"
cd "$POPPER_DIR"
git checkout -q 897b08754ca648e9f63a4207028814b3c566e1bb
uv python install 3.14 >/dev/null
CFLAGS="-Wno-error=incompatible-pointer-types" uv sync --python 3.14 >/dev/null
uv run --project "$POPPER_DIR" python -c "import janus_swi as j; print('janus ok, swipl', j.query_once('current_prolog_flag(version,V)')['V'])"
echo "SETUP OK"
