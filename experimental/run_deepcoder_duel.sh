#!/bin/bash
# lifecycle: one-off (archive when the qwen36-vs-qwen38 duel is published)
# DeepCoder 32k duel: qwen36 (35B-A3B MoE) vs qwen38 (27B dense), same seeded
# problems, same cap/k/temperature, one Telegram message per task covering both.
#
# Design notes (why it looks like this):
#  - ALTERNATE EVERY BATCH. Both models walk the SAME seeded problem list,
#    swapping after each rollout batch. difficulty_sweep resumes per
#    (task_id, temperature, model), so each call picks up where that model
#    left off. Loads are cheap next to rollouts, so swap as often as possible
#    — pairs then land on Telegram within minutes of the start.
#    CHUNK_x defaults to x_BATCH and must not go below it: --max-problems 1
#    with --batch-prompts 3 puts ONE problem in the batch (3 sequences instead
#    of 9), and losing that parallelism costs far more than the load it saves.
#    Per-problem alternation is therefore the wrong granularity; per-batch is
#    the finest one that keeps rollouts at full width. The two models take
#    different-sized turns (see A_BATCH/B_BATCH), so the loop runs until BOTH
#    have covered the sample rather than assuming a shared stride.
#  - SEPARATE --out per model. Non-negotiable: pointing both legs at one file
#    used to make the second leg read the first's rows as its own and report
#    "0 to do" (fixed in load_done, but separate files stay the clean layout).
#  - qwen38 SMOKE FIRST. The qwen38 MLX build is ours and unproven in this
#    harness; a 2-problem smoke fails in minutes instead of after a 12 h
#    qwen36 leg. Nothing long starts until it passes.
#  - Driver holds the EXCLUSIVE lease for the whole duel and legs run
#    --no-manage-machine, so the two models never overlap in memory and
#    nothing scheduled can load a backend on top of us. As of 2026-08-14
#    release leaves :8084 empty (resting default off) — no restore to verify.
#
# Watch:  tail -f ~/code/mlx-rl/runs/sweeps/deepcoder-32k-duel.log
set -u
cd "$(dirname "$0")/.."

SAMPLE="${SAMPLE:-200}"
# Per-model batch width. NB the harness batch is batch_prompts * k concurrent
# sequences, so at k=3 the narrowest possible is 3 — "batch 1" is not
# expressible without dropping k. Measured qwen38 curve (benchmarks/data/
# qwen38-mlx-20260814.jsonl) has a dead zone from 4..16 where aggregate barely
# moves while per-stream collapses, and a regime change at 32 (per-seq state
# 288->144 MB) that nearly doubles aggregate. At 32k context, though, memory
# decides: b4@32k already peaks at 45 GB, so the wide end is out of reach and
# these values sit in the dead zone by necessity, not by choice.
A_BATCH="${A_BATCH:-3}"      # qwen36 MoE: 9 seqs, ~360 t/s aggregate at 16k
B_BATCH="${B_BATCH:-2}"      # qwen38 dense: 6 seqs, memory-bound at 32k
CHUNK_A="${CHUNK_A:-$A_BATCH}"
CHUNK_B="${CHUNK_B:-$B_BATCH}"
CAP="${CAP:-32768}"
K="${K:-3}"
REQ_GB="${REQ_GB:-60}"
A_MODEL="${A_MODEL:-$HOME/models/mlx/Qwen3.6-35B-A3B-4bit}"
B_MODEL="${B_MODEL:-/Volumes/data/models/Qwen3.8-27B-4bit-ours}"

S=runs/sweeps
A_OUT=$S/deepcoder-32k-qwen36.jsonl
B_OUT=$S/deepcoder-32k-qwen38.jsonl
LOG=$S/deepcoder-32k-duel.log
NOTE="bash $HOME/code/housekeeping/note.sh"
MEMLEASE="python3 $HOME/code/housekeeping/memlease.py"
PY=.venv/bin/python

sweep () {  # sweep <model> <out> <n> <batch>
    $PY experimental/difficulty_sweep.py --task deepcoder --k "$K" \
        --temperature 1.0 --max-new-tokens "$CAP" --batch-prompts "$4" \
        --sample "$SAMPLE" --sample-seed 7 --max-problems "$3" \
        --model "$1" --out "$2" --required-gb "$REQ_GB" --no-manage-machine
}

[ -d "$B_MODEL" ] || { echo "FATAL: $B_MODEL missing (is /Volumes/data mounted?)"; exit 1; }
for pair in "A:$CHUNK_A:$A_BATCH" "B:$CHUNK_B:$B_BATCH"; do
    IFS=: read -r leg c b <<< "$pair"
    if [ "$c" -lt "$b" ]; then
        echo "FATAL: CHUNK_$leg=$c < ${leg}_BATCH=$b would under-fill every" \
             "rollout batch ($c problems x k instead of $b x k concurrent" \
             "sequences). Swapping models is cheap; shrinking the batch is not."
        exit 1
    fi
done

$MEMLEASE acquire deepcoder-duel --block exclusive --ensure-gb "$REQ_GB" \
    --pid $$ --note "DeepCoder 32k duel qwen36 vs qwen38, $SAMPLE problems" || exit 1
# REPORTER pre-set: the trap is armed before the reporter starts, and under
# set -u an unbound $REPORTER would make the trap itself fail (leaking the lease)
# on any early exit — e.g. the smoke failing.
REPORTER=""
trap '[ -n "$REPORTER" ] && kill "$REPORTER" 2>/dev/null; \
      $MEMLEASE release deepcoder-duel --block exclusive' EXIT

$NOTE "DeepCoder 32k duel starting: qwen36 vs qwen38, $SAMPLE problems, k=$K cap=$CAP, batch qwen36=$A_BATCH qwen38=$B_BATCH; log $LOG"

# Fail fast on the unproven checkpoint before committing hours to leg A.
echo "=== qwen38 smoke (2 problems) ==="
if ! sweep "$B_MODEL" "$B_OUT" "$B_BATCH" "$B_BATCH"; then
    $NOTE "DeepCoder duel ABORTED: qwen38 smoke failed — checkpoint or harness, not a score"
    exit 1
fi
echo "=== smoke passed ==="

# Reporter joins the two files and sends one message per completed pair.
$PY experimental/duel_report.py --a "$A_OUT" --a-name qwen36 \
    --b "$B_OUT" --b-name qwen38 --expect "$SAMPLE" \
    ${ONLY_DISAGREEMENTS:+--only-disagreements} &
REPORTER=$!

rc=0
turn=0
while [ $(( turn * CHUNK_A )) -lt "$SAMPLE" ] || [ $(( turn * CHUNK_B )) -lt "$SAMPLE" ]; do
    turn=$((turn + 1))
    echo "=== turn $turn (qwen36 x$CHUNK_A, qwen38 x$CHUNK_B) ==="
    sweep "$A_MODEL" "$A_OUT" "$CHUNK_A" "$A_BATCH" || { rc=$?; break; }
    sweep "$B_MODEL" "$B_OUT" "$CHUNK_B" "$B_BATCH" || { rc=$?; break; }
done

# Let the reporter drain the last chunk before the EXIT trap kills it.
sleep 90
$NOTE "DeepCoder 32k duel done (rc=$rc): $(wc -l < "$A_OUT") qwen36 rows, $(wc -l < "$B_OUT") qwen38 rows"
exit "$rc"
