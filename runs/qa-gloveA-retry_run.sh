#!/bin/bash
# qa-gloveA-retry_run.sh — 2026-07-31: rerun ARM A (glove + unknown-heavy
# chat bands) after the first attempt swap-guard-aborted at ~step 110.
# Diagnosis: NOT thrashing (decode held 300-400 tok/s to the end) — the
# unknown-heavy chat mix produces uncached judge work every step, and each
# per-step `claude -p` subprocess burst evicts cold trainer pages; macOS
# never pages them back in, so cumulative swap-used ratchets monotonically
# (+5.2 GB by step ~110) and trips the growth-based guard. The guard's
# purpose is catching genuine thrash; for judge-heavy runs a 12 GB margin
# keeps the fail-loud property (real thrash blows past any margin in
# minutes) without dying on the ratchet.
# Waits for the night driver (arm B + its probes) to finish first.
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
STAMP=20260731
LOG="runs/qa-gloveA-retry-$STAMP.log"
CALIB=runs/qa-calib-20260724/calib.jsonl
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
tg(){
    source "$HOME/code/housekeeping/.env" 2>/dev/null || true
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -s -m 20 \
        "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" -d text="gloveA-retry: $*" >/dev/null
    true
}

NIGHT_PID="${1:?usage: qa-gloveA-retry_run.sh <night_driver_pid>}"
say "waiting for night driver pid $NIGHT_PID to finish"
while kill -0 "$NIGHT_PID" 2>/dev/null; do sleep 60; done
say "night driver done — starting ARM A retry (swap margin 12)"
tg "night driver finished; ARM A retry starting (200 steps, margin 12GB)"

KW_A='{"calib_file": "runs/qa-calib-20260724/calib.jsonl",
 "band_mix": {"known": 0.65, "uncertain": 0.25, "unknown": 0.1},
 "chat_frac": 0.5, "system": "honesty",
 "chat_band_mix": {"known": 0.15, "uncertain": 0.35, "unknown": 0.5},
 "judge_cache": "runs/judge/qa-abstain-cache.jsonl"}'

OUT="runs/qa-gloveA-$STAMP"
mkdir -p "$OUT"
.venv/bin/mlx-rl-train \
    --profile qwen36 --task qa_abstain \
    --task-kwargs "$KW_A" \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps 200 --batch-prompts 8 --group-size 8 --micro-batch 4 \
    --max-new-tokens 192 --lora-layers 12 --grad-checkpoint \
    --lr 3e-6 --kl-coef 0.01 --inject-r 1 \
    --eval-every 10 --eval-n 200 --checkpoint-every 20 \
    --seed 0 --abort-inactive-window 30 --swap-guard-margin 12 \
    --out "$OUT" >>"$LOG" 2>&1
rc=$?
say "ARM A retry rc=$rc"
if [ $rc -eq 0 ] && [ -f "$OUT/adapters/adapter-00200.safetensors" ]; then
    dst="$HOME/models/adapters/qa-gloveA-$STAMP"
    mkdir -p "$dst"
    cp "$OUT/adapters/adapter-00200.safetensors" "$dst/adapters.safetensors"
    cp "$HOME/models/adapters/qa-chatmix-20260730/adapter_config.json" "$dst/adapter_config.json"
    echo "promoted $OUT ckpt 200 (ARM A retry; see run dir for metrics)" > "$dst/MANIFEST.md"
    say "probes: chat glove-ON"
    $PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
        --adapter "$dst" --system honesty \
        --out "runs/qa-chat-gloveA-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
    say "probes: papers-recall glove-ON"
    $PY scripts/papers_recall_probe.py --k 4 --adapter "$dst" --system honesty \
        --out "runs/papers-recall-gloveA-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
    say "probes: chat glove-OFF"
    $PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
        --adapter "$dst" \
        --out "runs/qa-chat-gloveA-nosys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
    tg "ARM A retry DONE rc=0 + probes done — runs/qa-gloveA-$STAMP"
else
    tg "ARM A retry FAILED rc=$rc $([ -f "$OUT/ABORTED" ] && echo ABORTED) — $LOG"
fi
say "DONE rc=$rc"
