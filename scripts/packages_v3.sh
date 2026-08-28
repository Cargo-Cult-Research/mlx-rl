#!/bin/bash
# lifecycle: one-off (archive when the packages row is in the lab book)
# Packages row, third attempt — the reward is repriced and the oracle is gone.
#
# What changed from v2 (2026-08-24, after reading the rollouts):
#
#  * pypi_lookup is NOT offered. It is a tool no deployed assistant has, and
#    it shared self._pypi with the grader — "checked it" and "cannot be marked
#    wrong" were the same event, so every tool-calling answer scored +1 by
#    construction. The grader still uses _pypi; the policy gets web_search,
#    which is what it would actually have.
#  * wrong_penalty 3 -> 9. At P=3 the break-even is p = P/(1+P) = 0.75, i.e.
#    "guess whenever you are more than 75% sure", and the base model is 88.7%
#    correct — so guessing always won and the tools collapsed to 0.13 by step
#    40. At P=9 the break-even is 0.90 and guessing is worth -0.13 in
#    expectation, below abstaining.
#  * call_cost 0.15 per round, capped at 0.8, so the order is
#    correct-no-tools 1.00 > correct-with-tools 0.85..0.40 > abstain 0 > invent -9.
#    A flat +1 was flat in tool count and gave nothing to argue against burning
#    the whole budget every time.
#
# Comparability with the trivia and papers rows is deliberately broken here;
# that was Urs's call on 2026-08-24 ("compatibility with old runs that are
# broken doesn't matter").
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
TK='{"domain":"packages","situation":"single","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl","wrong_penalty":9.0,"call_cost":0.15}'
CELLS='trivia:single,papers:single'
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

say "waiting for the papers run to finish"
while pgrep -f "mlx_rl.train.*20260824-papers-v1" >/dev/null; do sleep 60; done
say "papers is done"

# --- smoke gate: one step, then look before betting a night on it ---------
SMOKE=runs/smoke/packages-v3-$(date +%Y%m%d-%H%M%S)
say "smoke: 1 step, 1 prompt x 8 -> $SMOKE"
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty --task-kwargs "$TK" \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 1 --batch-prompts 1 --group-size 8 --group-stage1 8 \
  --update-adv-frac 0.5 --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 8 \
  --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 999 --eval-n 0 --checkpoint-every 999 --seed 0 \
  --lease-wait 1800 --required-gb 62 --out "$SMOKE" 2>&1

# hard checks. Any failure means DO NOT spend the night.
S="$SMOKE/samples.jsonl"
[ -s "$S" ] || { say "ABORT: smoke produced no samples"; exit 1; }
uv run python - "$S" <<'PY' || exit 1
import json, sys
r = json.loads(open(sys.argv[1]).readline())
comps = r["completions"]
bad = [c for c in comps for t in (c.get("tool_calls") or []) if t.get("name") == "pypi_lookup"]
assert not bad, "ABORT: pypi_lookup was still reachable by the policy"
rw = [c["reward"] for c in comps]
assert any(x != rw[0] for x in rw) or len(set(rw)) == 1, "sanity"
print("smoke ok: %d answers, rewards %s" % (len(comps), sorted(set(round(x,2) for x in rw))))
print("tool calls seen: %s" % sorted({t.get("name") for c in comps for t in (c.get("tool_calls") or []) if t.get("name")}))
PY
say "smoke passed -> $SMOKE (raw rollouts are in $S)"

OUT=runs/curve/20260825-packages-v3
say "training packages, repriced, 60 steps -> $OUT"
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty --task-kwargs "$TK" \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 60 --batch-prompts 12 --group-size 8 --group-stage1 4 --stage1-skip saturated \
  --update-adv-frac 0.5 --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 48 \
  --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 10 --eval-n 160 --eval-cells "$CELLS" --eval-cells-n 64 \
  --checkpoint-every 5 --seed 0 --lease-wait 1800 --required-gb 62 \
  --out "$OUT" 2>&1
uv run python scripts/promote_adapter.py "$OUT" --out "$OUT/promoted" 2>&1 || say "nothing to promote"
say "PACKAGES V3 DONE -> $OUT"
