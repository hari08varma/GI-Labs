#!/usr/bin/env bash
# Run run_popper.py inside WSL Ubuntu (SWI-Prolog 9.2.9 from apt; Popper from GitHub, commit
# 897b087 of 24 Oct 2026, installed with uv; janus-swi built with
# CFLAGS=-Wno-error=incompatible-pointer-types because GCC 14 rejects a pointer-type warning).
# Same Popper code and settings as the Modal runs. Usage (from Windows):
#   wsl -d Ubuntu -- bash baselines/popper_runner/run_wsl.sh <run_popper args>   (from the repo root)
set -euo pipefail
export POPPER_CMD="$HOME/.local/bin/uv run --project $HOME/popper $HOME/popper/popper.py"
cd "$(dirname "$0")/../.."
exec python3 baselines/popper_runner/run_popper.py "$@"
