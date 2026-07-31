#!/bin/bash
# qa-glove-night_run.sh — 2026-07-30 overnight, pre-registered two-arm test
# of the "glove": HONESTY_SYSTEM system prompt shipped with the adapter.
# The 07-30 chatmix run (no glove) was flat — chat declines never ignited
# from a ~1% propensity. The glove legalizes declining in a register that
# carries across tasks; RL's job becomes calibration, not invention.
#
#   Stage 0  control probes: BASE + glove, no adapter (does instruction
#            alone fix it? then RL has a different job)
#   Stage 1  sanity (3 steps) — gate for the fulls
#   Stage 2  ARM A full 200: glove + unknown-heavy chat band mix
#            (chat_band_mix 0.15/0.35/0.50) -> promote -> probes
#   Stage 3  ARM B full 200: glove only (chat bands = tag bands) — ablation
#            -> promote -> probes
#
# Per-arm probes: qa_chat_probe + papers_recall (chat frame), each glove-ON
# and glove-OFF (register binding: does the adapter need the glove?).
# Every stage Telegrams; a sanity failure aborts the fulls.
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
STAMP=20260730
LOG="runs/qa-glove-night-$STAMP.log"
CALIB=runs/qa-calib-20260724/calib.jsonl
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
tg(){
    source "$HOME/code/housekeeping/.env" 2>/dev/null || true
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -s -m 20 \
        "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" -d text="glove-night: $*" >/dev/null
    true
}

promote(){  # promote <run_dir> <name>: mlx_lm-loadable layout
    local run=$1 name=$2 dst="$HOME/models/adapters/$name"
    mkdir -p "$dst"
    cp "$run/adapters/adapter-00200.safetensors" "$dst/adapters.safetensors"
    cat > "$dst/adapter_config.json" <<'JSON'
{
  "fine_tune_type": "lora",
  "num_layers": 12,
  "lora_parameters": {
    "rank": 16,
    "scale": 20.0,
    "dropout": 0.0,
    "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "linear_attn.in_proj_qkv",
             "linear_attn.in_proj_z", "linear_attn.out_proj"]
  }
}
JSON
    echo "promoted $run -> $dst (ckpt 200; see $run for metrics)" > "$dst/MANIFEST.md"
}

probe_set(){  # probe_set <adapter_dir_or_empty> <tag>
    # (no arrays: empty arrays + set -u break on macOS bash 3.2; adapter
    # paths here never contain spaces)
    local ad=$1 tag=$2
    say "probes[$tag] chat-probe glove-ON"
    $PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
        ${ad:+--adapter "$ad"} --system honesty \
        --out "runs/qa-chat-$tag-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
    say "probes[$tag] papers-recall glove-ON"
    $PY scripts/papers_recall_probe.py --k 4 ${ad:+--adapter "$ad"} --system honesty \
        --out "runs/papers-recall-$tag-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
    if [ -n "$ad" ]; then
        say "probes[$tag] chat-probe glove-OFF (register binding)"
        $PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
            --adapter "$ad" \
            --out "runs/qa-chat-$tag-nosys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
    fi
}

train(){  # train <out_dir> <task_kwargs_json> <steps>
    local out=$1 kwargs=$2 steps=$3
    mkdir -p "$out"
    .venv/bin/mlx-rl-train \
        --profile qwen36 --task qa_abstain \
        --task-kwargs "$kwargs" \
        --chat-kwargs '{"enable_thinking": false}' \
        --steps "$steps" --batch-prompts 8 --group-size 8 --micro-batch 4 \
        --max-new-tokens 192 --lora-layers 12 --grad-checkpoint \
        --lr 3e-6 --kl-coef 0.01 --inject-r 1 \
        --eval-every 10 --eval-n 200 --checkpoint-every 20 \
        --seed 0 --abort-inactive-window 30 \
        --out "$out" >>"$LOG" 2>&1
}

KW_COMMON='"calib_file": "runs/qa-calib-20260724/calib.jsonl",
 "band_mix": {"known": 0.65, "uncertain": 0.25, "unknown": 0.1},
 "chat_frac": 0.5, "system": "honesty",
 "judge_cache": "runs/judge/qa-abstain-cache.jsonl"'
KW_A="{$KW_COMMON, \"chat_band_mix\": {\"known\": 0.15, \"uncertain\": 0.35, \"unknown\": 0.5}}"
KW_B="{$KW_COMMON}"

say "STAGE 0: control probes (base + glove, no adapter)"
probe_set "" base-glove
tg "stage 0 done (base+glove control probes)"

say "STAGE 1: sanity (3 steps, arm-A kwargs)"
mkdir -p "runs/qa-glove-sanity-$STAMP"
.venv/bin/mlx-rl-train --profile qwen36 --task qa_abstain \
    --task-kwargs "$KW_A" --chat-kwargs '{"enable_thinking": false}' \
    --steps 3 --batch-prompts 8 --group-size 8 --micro-batch 4 \
    --max-new-tokens 192 --lora-layers 12 --grad-checkpoint \
    --lr 3e-6 --kl-coef 0.01 --inject-r 1 --eval-every 10 --eval-n 24 \
    --seed 0 --out "runs/qa-glove-sanity-$STAMP" >>"$LOG" 2>&1
if [ $? -ne 0 ]; then
    say "SANITY FAILED — aborting the fulls"; tg "SANITY FAILED, night aborted — check $LOG"; exit 1
fi
tg "sanity passed, starting ARM A (glove + unknown-heavy chat bands, 200 steps)"

say "STAGE 2: ARM A full"
train "runs/qa-gloveA-$STAMP" "$KW_A" 200
rcA=$?
say "arm A rc=$rcA"
if [ $rcA -eq 0 ] && [ -f "runs/qa-gloveA-$STAMP/adapters/adapter-00200.safetensors" ]; then
    promote "runs/qa-gloveA-$STAMP" "qa-gloveA-$STAMP"
    probe_set "$HOME/models/adapters/qa-gloveA-$STAMP" gloveA
    tg "ARM A done rc=$rcA + probes done; starting ARM B (glove only)"
else
    tg "ARM A FAILED rc=$rcA $([ -f runs/qa-gloveA-$STAMP/ABORTED ] && echo ABORTED); continuing to ARM B"
fi

say "STAGE 3: ARM B full (ablation: glove, tag band mix for chat)"
train "runs/qa-gloveB-$STAMP" "$KW_B" 200
rcB=$?
say "arm B rc=$rcB"
if [ $rcB -eq 0 ] && [ -f "runs/qa-gloveB-$STAMP/adapters/adapter-00200.safetensors" ]; then
    promote "runs/qa-gloveB-$STAMP" "qa-gloveB-$STAMP"
    probe_set "$HOME/models/adapters/qa-gloveB-$STAMP" gloveB
fi
tg "NIGHT DONE — armA rc=$rcA armB rc=$rcB; results in runs/, log $LOG"
say "NIGHT DONE armA=$rcA armB=$rcB"
