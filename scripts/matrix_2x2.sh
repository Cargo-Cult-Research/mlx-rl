#!/bin/bash
# lifecycle: one-off (archive once the 2x2 is in the lab book)
# Complete the trivia x papers 2x2 on one instrument.
#
# Packages is parked (2026-08-25, Urs): its grader charges -9 for denials,
# class names and system libraries, so ~half the negative rewards are
# extraction bugs, not hallucinations. A 3x3 built on that column would be
# measuring the extractor. Two subjects, four cells, one ruler instead.
#
# The cells were NOT already comparable. build_eval_cells (train.py:1445)
# injects calib_file=CALIB[domain] for a held-out cell, but the TRAINING task
# is built from raw --task-kwargs. trivia-v3 passed its calib file, so its
# diagonal and papers-v1's trivia cell agree. papers-v1 passed none, so its
# diagonal ran on an uncalibrated set -- three bands instead of four, no
# `uncertain` -- and scored base at +0.762 vs +0.250 for the calibrated set.
# Same model, same subject, different instrument.
#
# So: re-score both rows through build_eval_cells at n=64, every 10 steps,
# seed 0. Same draw, same bands, same judge, all four cells.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

say "papers row: both cells, calibrated"
uv run python scripts/retrofit_eval.py \
  --run runs/curve/20260824-papers-v1 \
  --cells trivia:single,papers:single \
  --every 10 --n 64 \
  --out runs/retrofit/matrix2x2-papers-v1 2>&1

say "trivia row: diagonal at the same n and draw as everything else"
uv run python scripts/retrofit_eval.py \
  --run runs/curve/20260822-trivia-v3 \
  --also runs/curve/20260822-trivia-v3-to120 \
  --cells trivia:single \
  --every 10 --n 64 \
  --out runs/retrofit/matrix2x2-trivia-v3 2>&1

say "MATRIX 2x2 DONE"
say "  papers row -> runs/retrofit/matrix2x2-papers-v1/metrics.jsonl"
say "  trivia diag -> runs/retrofit/matrix2x2-trivia-v3/metrics.jsonl"
say "  trivia off-diag (papers cell) already at runs/retrofit/trivia-v3-heldout/metrics.jsonl"
