#!/bin/bash
# qa-gloveC-seed1_run.sh — 2026-08-01: seed-1 replication of ARM C (the
# winner: known-side pressure, chat bands 0.35/0.35/0.30). Single variable
# vs qa-gloveC-20260731 = --seed 1. Every headline result so far is n=1;
# this hardens the central claim. Probes + binding on ckpt 200 chained.
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
STAMP=20260801
LOG="runs/qa-gloveC-seed1-$STAMP.log"
CALIB=runs/qa-calib-20260724/calib.jsonl
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
tg(){
    source "$HOME/code/housekeeping/.env" 2>/dev/null || true
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -s -m 20 \
        "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" -d text="gloveC-seed1: $*" >/dev/null
    true
}

KW_C='{"calib_file": "runs/qa-calib-20260724/calib.jsonl",
 "band_mix": {"known": 0.65, "uncertain": 0.25, "unknown": 0.1},
 "chat_frac": 0.5, "system": "honesty",
 "chat_band_mix": {"known": 0.35, "uncertain": 0.35, "unknown": 0.3},
 "judge_cache": "runs/judge/qa-abstain-cache.jsonl"}'

OUT="runs/qa-gloveC-seed1-$STAMP"
mkdir -p "$OUT"
say "ARM C seed-1 replication start"
.venv/bin/mlx-rl-train \
    --profile qwen36 --task qa_abstain \
    --task-kwargs "$KW_C" \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps 200 --batch-prompts 8 --group-size 8 --micro-batch 4 \
    --max-new-tokens 192 --lora-layers 12 --grad-checkpoint \
    --lr 3e-6 --kl-coef 0.01 --inject-r 1 \
    --eval-every 10 --eval-n 200 --checkpoint-every 20 \
    --seed 1 --abort-inactive-window 30 --swap-guard-margin 12 \
    --out "$OUT" >>"$LOG" 2>&1
rc=$?
say "seed-1 rc=$rc"
if [ $rc -ne 0 ]; then
    tg "seed-1 FAILED rc=$rc $([ -f "$OUT/ABORTED" ] && echo ABORTED) — $LOG"
    exit 1
fi
AD="$HOME/models/adapters/qa-gloveC-seed1-$STAMP"
mkdir -p "$AD"
cp "$OUT/adapters/adapter-00200.safetensors" "$AD/adapters.safetensors"
cp "$HOME/models/adapters/qa-chatmix-20260730/adapter_config.json" "$AD/adapter_config.json"
echo "promoted $OUT ckpt 200 (ARM C seed-1 replication)" > "$AD/MANIFEST.md"
say "probes[C-seed1] chat glove-ON"
$PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
    --adapter "$AD" --system honesty \
    --out "runs/qa-chat-gloveCseed1-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
say "probes[C-seed1] papers-recall glove-ON"
$PY scripts/papers_recall_probe.py --k 4 --adapter "$AD" --system honesty \
    --out "runs/papers-recall-gloveCseed1-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
say "binding[C-seed1]"
$PY scripts/qa_binding_correlation.py --adapter "$AD" --n 200 --k 8 \
    --out "runs/qa-binding-Cseed1-$STAMP" >>"$LOG" 2>&1 || say "binding nonzero"
tg "seed-1 replication DONE rc=0 — probes + binding in runs/"
say "seed-1 DONE"
