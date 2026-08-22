#!/bin/bash
# Does the web-tools adapter really cut invented package names, or was that noise?
# Yesterday: base 0.20 -> web-tools 0.05. Last night: base 0.11 -> per-square 0.12-0.14.
# Both were internally paired, so put every arm in ONE comparison on the same items.
#
# One matrix_eval PROCESS PER ARM: loading several adapters in a single process
# leaked GPU memory last night and killed phases 2 and 3 with a Metal OOM.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
ROOT=runs/day/20260820-decisive
mkdir -p "$ROOT"
CELLS=packages:single,packages:swamp
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

arm_eval () {
  local name=$1 adapter=$2
  [ -f "$ROOT/$name/results.json" ] && { say "skip $name"; return 0; }
  say "arm $name"
  uv run python scripts/matrix_eval.py --cells "$CELLS" --arm "$name=$adapter" \
    --n 48 --k 2 --max-new-tokens 2048 --no-manage-machine --out "$ROOT/$name" 2>&1
}

arm_eval base ""
arm_eval web-tools ~/models/adapters/qa-arxiv-mt-arm2-60
arm_eval trivia-single "$PWD/runs/night/20260818-night/trivia-single/promoted"
arm_eval papers-single "$PWD/runs/night/20260818-night/papers-single/promoted"
arm_eval papers-single-seed1 "$PWD/runs/night/20260818-night/papers-single-seed1/promoted"
say "DECISIVE RUN DONE"
