#!/bin/bash
# qa-gloveA160-probes_run.sh — 2026-07-31: the retry's eval bottomed at step
# 160 (answered 0.715, wrong 0.10) and drifted back by 200 (0.855/0.165) —
# the flagship's late-drift pattern again. Probe ckpt-160 alongside the
# already-probed ckpt-200. Waits for the gloveB probe chain to finish.
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
LOG="runs/qa-gloveA-retry-20260731.log"
CALIB=runs/qa-calib-20260724/calib.jsonl
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
tg(){
    source "$HOME/code/housekeeping/.env" 2>/dev/null || true
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -s -m 20 \
        "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" -d text="gloveA-160: $*" >/dev/null
    true
}

WAIT_PID="${1:?usage: qa-gloveA160-probes_run.sh <gloveB_probes_pid>}"
say "gloveA-160 probes: waiting for pid $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done

AD="$HOME/models/adapters/qa-gloveA-160-20260731"
mkdir -p "$AD"
cp runs/qa-gloveA-20260731/adapters/adapter-00160.safetensors "$AD/adapters.safetensors"
cp "$HOME/models/adapters/qa-chatmix-20260730/adapter_config.json" "$AD/adapter_config.json"
echo "promoted runs/qa-gloveA-20260731 ckpt 160 (eval-best; late drift by 200)" > "$AD/MANIFEST.md"

say "probes[gloveA-160] chat glove-ON"
$PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
    --adapter "$AD" --system honesty \
    --out "runs/qa-chat-gloveA160-sys-20260731" >>"$LOG" 2>&1 || say "probe nonzero"
say "probes[gloveA-160] papers-recall glove-ON"
$PY scripts/papers_recall_probe.py --k 4 --adapter "$AD" --system honesty \
    --out "runs/papers-recall-gloveA160-sys-20260731" >>"$LOG" 2>&1 || say "probe nonzero"
say "probes[gloveA-160] chat glove-OFF"
$PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
    --adapter "$AD" \
    --out "runs/qa-chat-gloveA160-nosys-20260731" >>"$LOG" 2>&1 || say "probe nonzero"
tg "ckpt-160 probes done — full grid complete"
say "gloveA-160 probes DONE"
