#!/bin/bash
# qa-glovec1-arm_run.sh — 2026-08-01: the c=1 (TruthRL-threshold) arm, the
# last pre-registered scientific item. Penalty c is a threshold in
# disguise (answer iff p > c/(1+c)): c=3 -> 0.75 (all runs so far),
# c=1 -> 0.5. Recipe otherwise identical to ARM C seed 0. Prediction:
# more answering on uncertain-band items, lower hedge rates across the
# board, worse fabrication protection — the penalty-sweep point for the
# paper's threshold figure. First: C-200's missing glove-OFF inertness
# probe (5 min), then the run.
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
STAMP=20260801
LOG="runs/qa-glovec1-$STAMP.log"
CALIB=runs/qa-calib-20260724/calib.jsonl
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
tg(){
    source "$HOME/code/housekeeping/.env" 2>/dev/null || true
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -s -m 20 \
        "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" -d text="glove-c1: $*" >/dev/null
    true
}

say "C-200 glove-OFF inertness probe (was missing from the C driver)"
$PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
    --adapter "$HOME/models/adapters/qa-gloveC-200-20260731" \
    --out "runs/qa-chat-gloveC200-nosys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"

KW='{"calib_file": "runs/qa-calib-20260724/calib.jsonl",
 "band_mix": {"known": 0.65, "uncertain": 0.25, "unknown": 0.1},
 "chat_frac": 0.5, "system": "honesty", "wrong_penalty": 1.0,
 "chat_band_mix": {"known": 0.35, "uncertain": 0.35, "unknown": 0.3},
 "judge_cache": "runs/judge/qa-abstain-cache.jsonl"}'

OUT="runs/qa-glovec1-$STAMP"
mkdir -p "$OUT"
say "c=1 arm start (threshold 0.5; C recipe otherwise)"
.venv/bin/mlx-rl-train \
    --profile qwen36 --task qa_abstain \
    --task-kwargs "$KW" \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps 200 --batch-prompts 8 --group-size 8 --micro-batch 4 \
    --max-new-tokens 192 --lora-layers 12 --grad-checkpoint \
    --lr 3e-6 --kl-coef 0.01 --inject-r 1 \
    --eval-every 10 --eval-n 200 --checkpoint-every 20 \
    --seed 0 --abort-inactive-window 30 --swap-guard-margin 12 \
    --out "$OUT" >>"$LOG" 2>&1
rc=$?
say "c=1 rc=$rc"
if [ $rc -ne 0 ]; then
    tg "c=1 arm FAILED rc=$rc $([ -f "$OUT/ABORTED" ] && echo ABORTED) — $LOG"
    exit 1
fi
AD="$HOME/models/adapters/qa-glovec1-$STAMP"
mkdir -p "$AD"
cp "$OUT/adapters/adapter-00200.safetensors" "$AD/adapters.safetensors"
cp "$HOME/models/adapters/qa-chatmix-20260730/adapter_config.json" "$AD/adapter_config.json"
echo "promoted $OUT ckpt 200 (c=1 threshold arm)" > "$AD/MANIFEST.md"
say "probes[c1] chat glove-ON"
$PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
    --adapter "$AD" --system honesty \
    --out "runs/qa-chat-glovec1-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
say "probes[c1] papers-recall glove-ON"
$PY scripts/papers_recall_probe.py --k 4 --adapter "$AD" --system honesty \
    --out "runs/papers-recall-glovec1-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
say "binding[c1]"
$PY scripts/qa_binding_correlation.py --adapter "$AD" --n 200 --k 8 \
    --out "runs/qa-binding-c1-$STAMP" >>"$LOG" 2>&1 || say "binding nonzero"
tg "c=1 arm DONE rc=0 — probes + binding in runs/"
say "c=1 DONE"
