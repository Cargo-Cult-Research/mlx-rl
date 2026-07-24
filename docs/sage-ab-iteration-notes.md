# SAGE-RL A/B — iteration notes (2026-07-11)

Terminated the correct-SAGE vs vanilla A/B mid-run to fix a throughput problem.
This is what we learned and what the next iteration should carry forward.

## The headline: the "slow backward" was a swap cliff, not compute

Directly benchmarked one backward pass of the exact training path (qwen36 +
rank-16 LoRA, 12 layers), varying sequence length. Measured, not guessed:

| total seq len | fwd | fwd+bwd | 3-pass/seq | peak mem | swap |
|--------------:|----:|--------:|-----------:|---------:|-----:|
| 512  | 1.2s | 2.7s  | 5.0s  | 35.9 GB | none |
| 1024 | 2.3s | 5.8s  | 10.5s | 52.1 GB | none |
| 1792 | 4.0s | 26.1s | 34.1s | 76.5 GB | 5.8 GB ← spills |
| 3072 | 22s  | 180s  | 225s  | 117 GB  | 16 GB ← thrash |

- Compute is ~linear in length **while it fits** (512→1024 doubles both).
- Peak activation memory grows ~+16-25 GB per 512 tokens on top of the 22 GB
  weights (O(L²) attention in the 10 full-attn layers accelerates it).
- Around **L≈1792 the peak crosses physical RAM and macOS pages to SSD**; time
  then goes super-linear (5.8→26→180s). That paging — not SAGE, not a loop bug —
  is what turned steps into 40 min–2.2 h, and why `update_s` was so *variable*
  (swap contention is nondeterministic: 2441 vs 4532s for the same seq count).

The killed run's `mean_len` had grown to ~1700 with individuals >2000, i.e.
squarely in the swap regime, ×16-24 sequences ×3 passes/step.

## Traps that cost us time (don't fall for these again)

1. **`peak_gb` in metrics.jsonl is bogus as a value.** It's
   `mx.get_peak_memory()` **never reset in the loop** — a monotonic allocation
   high-water (identical `166.12` every step). Not resident memory. A genuine
   154 GB on a 96 GB box would panic; the run lived 4.7 h. Real signal = system
   swap (`vm.swapusage` / `psutil.swap_memory`).
2. **asitop leaks.** The Apple-Silicon monitor (root-owned, needs sudo to kill)
   had leaked to ~18 GB over 4.5 h, lowering the swap cliff from ~2300 to ~1792
   tokens. Worth uninstalling. Any "mystery resident GB" — check for it.
3. **`96 GiB`, not 103 GB.** `hw.memsize = 103079215104` bytes = 96 GiB. Divide
   by 1024³, not 1e9.
4. **metrics.jsonl / samples.jsonl are opened in append mode** (`train.py`),
   so successive runs into the same `--out` dir accumulate. Analysis must slice
   from the *last* `step0` baseline. (Candidate cleanup: truncate on start, or
   stamp a run id.)

## Fixes landed this session

- **Swap watchdog** (`memory.SwapGuard`, wired into `train.py`): daemon thread
  samples swap every 3 s; if it grows >`swap_guard_margin_gb` (default 3) above
  the run-start baseline, prints a banner and `os._exit(137)`. Fail loud, not
  slow. CLI: `--swap-guard-margin` (0 disables). Lease is stale-reaped so a hard
  exit still restores the host server.
- **Length cap = memory cap**: `ab_sage_v2.sh` now `--max-new-tokens 1024`
  (peak ~52 GB, swap-free). The 16-24 sequence count (batch_prompts×group_size)
  is unchanged — we capped length, not the group.
- Verified: 40 unit tests pass; guard arms/aborts; new-run baseline eval
  mean_len dropped 1837 → 953; swap stayed flat at 1.8 GB.

## Still open for the next iteration

1. **`think_len` accounting anomaly (investigate first).** In the killed run's
   traces the SAGE `think_len` was **bimodal**: `35,42,68,184,297,384,384,
   1416,1480,2069`. The `384,384` pair looks like a cap being hit, and
   `2069 > max_new_tokens 2048` should be impossible — smells like an
   off-by-something in `_sage_batched`'s think-length bookkeeping (re-prefill /
   chunk accounting). With the 1024 cap, any reported `think_len ≥ 1024` in the
   next run is a confirmed bug. Check before trusting SAGE length stats.
2. **The 0.92 average `think_frac` hid that bimodality.** It is *suggestive*
   that reasoning stayed in the think block (no prose relocation → no obvious
   reward hack), and `frac_correct` climbed 0.50→0.79 with `reward_sage`
   competitive — but this is NOT confirmed. Re-run clean at 1024 and eyeball the
   per-rollout traces (deep-dive artifact pattern) before declaring anything.
3. **`old_lp` pass is redundant when `epochs_per_batch=1`.** `update_policy`
   runs 3 forward passes/seq (old_lp, ref_lp, fwd+bwd); with a single epoch the
   ratio is provably 1 (train.py comment), so the old_lp forward is dead work —
   removing it cuts ~1/3 of forward cost. Do it as an isolated change so it
   doesn't confound the A/B.
4. **Greedy eval loops on this model** → keep eval bounded (the 1024 cap now
   also bounds it; a low eval temperature would be cleaner than greedy).

## Where the A/B stands

Not completed. The correct-SAGE decoder (faithful to arXiv 2602.08354) and the
batched beam (~3.9× on this MoE) are in and unit-tested. The clean, fast,
swap-guarded A/B (correct SAGE vs vanilla, capped 1024) is ready to relaunch via
`scripts/ab_sage_v2.sh` once the machine is free again — but resolve open item
(1) first so we're not measuring a buggy length signal.

---

## RESOLVED (2026-07-11 evening session) — open items 1 & 2, plus a grader hole

**Open item 1 (think_len anomaly): both halves explained, both real.**
- The `384,384` pair + the all-576 step-2 group are **not from the killed run
  at all** — samples.jsonl lines 0-1 are an earlier aborted attempt (old
  384-cap decoder) appended into the same file. The "bimodal" list
  `35,42,68,184,297,384,384,1416,1480,2069` is exactly the union of the two
  runs' dumps — trap (4) in action. Fixed structurally: `_rotate()` in
  train.py moves old config/metrics/samples aside on every run start.
- The `2069 > max_new_tokens 2048` is a REAL bug, not bookkeeping:
  `_sage_batched`'s reasoning loop was bounded only by
  `max_reasoning_steps × step_tokens` (= 3072), never by `max_new_tokens` —
  and the answer phase then had zero budget left (`while len(tok) <
  max_new_tokens` was false on entry). Fixed: reasoning is hard-capped at
  `max_new_tokens - sage_answer_reserve` (default 256) in both beam paths;
  a completion can no longer exceed the cap, and the answer always has room.

**Open item 2 upgraded to a grader hole (worse than suspected).** Step-3
arithmetic: 3 of 6 sampled rollouts hit the 2048 cap INSIDE an unclosed
`<think>` and were still graded reward 1.0 — the arithmetic grader takes the
last `<answer>` match anywhere, and the models draft tags inside the CoT. So
`frac_correct` 0.50→0.79 was partly cap-hitting false passes, and the
"vanilla tag-in-ramble" hack was being actively REWARDED. Fixed at the
harness level (all tasks at once): rollouts are graded on the **visible
reply** — text after the final think-close marker; unclosed think = empty
reply = reward 0 (`_visible_reply` in train.py, same for eval). Replayed the
recorded step-3 group through the new grader: completions 3/4/5 (unclosed)
and 6 (think 2069, zero visible tokens) flip 1.0 → 0.0; the two honest
passes (1, 7) keep 1.0. Honest step-3 rate: 2/8, not 6/8.

**New monitoring** (so the next bug class is caught at step 1, not post-hoc):
- metrics: `frac_think_closed`, `frac_len_capped`, real per-step `peak_gb`
  (reset each step, GiB), `swap_gb` growth, `ts`; samples: `finish`, ts.
- train-loop tripwires print `[BUG]` on think_len > cap or reward-without-
  close (both now structurally impossible — the print is the regression alarm).
- **Live dashboard**: `scripts/dashboard.py` (stdlib HTTP on :8377, tailnet
  reachable on the local network). Metric curves,
  server-side anomaly scan (budget breach / grader leak / group truncation /
  swap / dead groups), per-rollout trace browser with think-vs-visible-reply
  split, run-log tail. Auto-refreshes every 5 s.
- Regression tests: `tests/test_grading.py` (the distilled step-3 shapes +
  SAGE budget invariant on the tiny model, batched and unbatched).

Relaunch = `scripts/ab_sage_v2.sh` → fresh dirs `runs/sage3-codemix` /
`runs/vanilla3-codemix` (v2 dirs left untouched as the record of the bug).
NB the old runs' numbers (sage1 collapse verdict, the vanilla "tag-in-ramble"
control verdict, sage2's frac_correct climb) were all measured under the leaky
grader — treat them as decoder-behavior evidence only, not reward evidence.
(2026-07-11 late: the leaky-grader dirs were moved to
`runs/archive-leaky-grader-20260711/` with a README so nobody cites them by
accident; the nothink runs — control*/convergence/probe — were unaffected by
the leak and stay in place.)

## v4 follow-up: the 1024 cap vs the paper (2026-07-11 night)

**Yes, 1024 deviates from the paper.** SAGE's termination is supposed to be
confidence-driven (the top-h Φ gate), with the budget as a distant backstop.
The v3 live data shows our budget is BINDING, not backstop: on arithmetic and
code the SAGE think length pins at exactly 769 (= budget+forced close) every
single group — the gate never fires within 768 tokens on hard tasks (it does
fire naturally on toolformat: 49–278). So v3 measures "forced-close + answer
reserve discipline", not the paper's confidence-gated stopping. The killed
2048-cap run saw natural closes at think 184/1416/1480/2069 → on 5-op
arithmetic at temp 1.0 the gate seems to want ~1.4–2k tokens.

**But htop under-reports the constraint.** During the generation phase
(roughly half of each step's wall-clock) resident memory is only ~25 GB —
that's what "half the memory empty" in htop is. The binding transient is the
backward pass: the new honest per-step `peak_gb` reads **64–68 GiB at cap
1024** (96 GiB physical), so real headroom is ~22–26 GiB, not ~48. Measured
backward scaling (07-11 bench): 52 GB @1024 → 76.5 GB @1792 — but that 1792
point had asitop's ~18 GB leak resident (asitop now uninstalled), so the
clean cliff is nearer ~2300 total tokens.

**v4 plan (after the v3 A/B finishes):**
1. One-off clean backward probe at total seq 1536/2048/2304 (asitop gone) to
   re-locate the cliff — 20 min, off the training path.
2. Raise to **cap 1536** (think budget 1280, reserve 256) if the probe
   confirms ~75–82 GiB step peak: inside RAM with the swap guard as the net.
   Expect step time ~2× (the 3-pass cost is superlinear in seq len);
   16 steps ≈ 4–5 h/arm — still an overnight.
3. Cap 2048 only if the probe says it fits; do NOT trust the old 1792
   datapoint (asitop-contaminated, pessimistic).
4. Cheap orthogonal win: **eval at a higher cap than training** (eval is
   generation-only — no backward memory). `--eval-max-new-tokens 2048` would
   measure whether learned termination generalizes past the training wall.
   Needs a small config addition.

---

## v3 verdict + the cache-creep root cause (2026-07-12)

**Arm B (vanilla3) completed 16/16 — the first honest post-grader-fix
learning signal on this box:** eval_reward 0.0 → 0.0 (step 8) → **0.125**
(step 16); eval_think_closed 0.125 → **0.5**; eval_mean_len flat ~960 (no
instant-guess collapse). Caveats: the 8-problem eval has 0.125 resolution
(that's ONE problem flipping), and 4/16 updates were skipped on zero-variance
groups — gradient signal was sparse, largely because 50-83% of rollouts hit
the 1024 cap (zero-advantage all-capped groups).

**Arm A (sage3) was swap-guard-killed at step 8** (exit 137, swap +3.0 GB
over baseline) — so v3 never actually compared the arms. It was NOT a SAGE
memory cost: vanilla3 peaked *higher* per step (70.8 vs 67.6 GiB) and
survived. The difference was wall-clock at high water (~8 min/step SAGE beam
vs ~2.5 min vanilla) plus the real root cause: **train.py never called
`mx.clear_cache()`** — MLX's buffer cache keeps every freed allocation
resident indefinitely, so after each 65-70 GiB backward peak the process
*stayed* that big, and over ~an hour macOS started paging. Cumulative creep,
not a per-step overrun. Fixed: the trainer clears the cache after every
update.

**v4 changes** (same script, `scripts/ab_sage_v2.sh`, dirs
`runs/sage4-codemix` / `runs/vanilla4-codemix`):
- per-step `mx.clear_cache()` (the creep fix),
- cap 1024 → **2048** + `--grad-checkpoint` (probe results below),
- eval widened 8 → **32** problems and run at `--eval-max-new-tokens 3072`
  (generation-only) to measure termination generalization past the wall,
- fail LOUD reaches the human now: swap-guard kills / crashes write
  `runs/<name>/ABORTED`, the dashboard raises an error anomaly + red "✕
  died" tab badge (sage3's death was only visible as a silently stuck
  run), and `ab_sage_v2.sh` fires the notify hook on any nonzero arm exit + once on
  completion (path live-tested).

**Probe re-run (2026-07-12, `scripts/probe_backward.py`): the clean cliff is
LOWER than the old table, and grad checkpointing dissolves it.** Plain
backward of a single 1536-token sequence peaks **83.3 GiB and swaps** — the
07-11 "52 GB @1024 / cliff at 1792" numbers were optimistic (the live v3
runs' 64-68 GiB per-step peaks at ~1150 total were the honest curve). With
`mlx_lm.tuner.trainer.grad_checkpoint` on every layer class (bit-identical
gradients, verified on the tiny model):

| total seq | bwd_s | peak GiB | swap |
|----------:|------:|---------:|-----:|
| 1536 | 13.2 | 37.5 | 0 |
| 2048 | 19.4 | 44.6 | 0 |
| 2304 | 23.0 | 49.8 | 0 |
| 3072 | 39.0 | 67.9 | 0 |

So v4 trains at cap 2048 with ~50 GiB of headroom, covering qwen36's
natural think length (gate closes at 1.4-2k on 5-op arithmetic) — the
paper-fidelity deviation from the binding 1024 budget is gone.

## The oracle (before more RL): does SAGE decoding help at all?

The honesty question, 2026-07-12: at binding budgets we force-close the
model's thinking early and then TRAIN on the result — RL could be making
the model worse while the curves look fine. Before v4 runs,
`scripts/oracle_sage.py` measures the BASE model (no adapter) on the same
held-out stream the A/B eval uses (seed+100000, n=32), budget 4096
(non-binding), four decode conditions: greedy@1024 (quantifies what the old
cap cost), greedy@4096, sampled temp-1 k=4 (what vanilla rollouts see), and
SAGE m=2 tr=0.5. Grading identical to training. Decision rule: if
sage@4096 doesn't beat sampled/greedy, there is nothing for RL to distill —
stop and rethink. A SWE-bench-proper oracle via an external coding-benchmark
harness is the follow-up if this one is positive.

**Oracle RESULT (2026-07-12, `runs/oracle-sage-20260712/`): there IS a there
there.** n=32 held-out (the A/B eval stream), base qwen36, think-aware
grading:

| cond | acc | closed | mean len | capped | per-task (arith / code / toolformat) |
|------|----:|-------:|---------:|-------:|:--|
| greedy@1024 | 0.275 | 0.41 | 829 | 0.59 | 0.00 / 0.33 / 0.35 |
| greedy@4096 | 0.681 | 0.91 | 2207 | 0.19 | 0.83 / 0.78 / 0.35 |
| sampled temp-1 (k=4, n=128) | 0.608 | 0.80 | 2102 | 0.21 | 0.88 / 0.68 / 0.24 |
| **SAGE m=2 tr=0.5** | **0.806** | **1.00** | **1832** | **0.00** | **1.00 / 1.00 / 0.23** |

Reading:
1. **The old 1024 cap was devastating** — the base model gets 0.275 greedy
   at 1024 vs 0.681 at 4096 (arithmetic literally 0.00). The v3 A/B trained
   in a regime where most of the reward signal was truncation noise. The
   honesty concern, quantified.
2. **SAGE beats everything while being SHORTER**: +12.5 pts over greedy,
   +20 over the sampled distribution, at 1832 mean tokens vs ~2200, zero
   cap hits, every think block closed. On arithmetic+code it is a clean
   sweep (24/24). The RL premise — groups mixing sampled (0.61) and SAGE
   (0.81) rollouts carry an advantage gradient toward SAGE-like behavior —
   has a real target. n=32 caveat: the SAGE-vs-greedy gap is ~4 problems.
3. **Honest termination split**: the gate fired naturally on 23/32 (mean
   think 1083); 9/32 pinned at the reasoning-step bound
   (max_reasoning_steps×step_tokens = 3072 — a second, sneakier budget; the
   first summary.json called these "natural", since corrected). Those 9
   still scored 0.89 thanks to force-close + answer reserve.
4. **toolformat is flat ~0.25 in every condition** including greedy@1024 —
   not a decoding or budget problem; it's a format/capability gap. RL on
   the mixture may move it (it has group variance), but the SAGE mechanism
   is not the lever there.
5. Cost: SAGE ~44 s/completion vs ~6 s sampled (~7×) — which is exactly why
   distilling it into the plain sampling policy is worth training for.

→ v4 A/B is GO (cap 2048, grad-checkpoint, eval 32 @ 3072).

## v4 verdict (2026-07-12 afternoon) — vanilla wins at this scale

Both arms trained clean once the machine was actually exclusive (memory-lease
fix 7e7128b: takes 1+2 of sage4 died at step 1 co-resident with the restoring
22 GB host server via phantom-free `_available_gb`; take 3 ran with a
verified `FREED` line — peaks 52–57 GiB, swap flat, zero incidents).

| arm | baseline | eval@8 | eval@16 | greedy len | gen share of step |
|-----|---------:|-------:|--------:|-----------:|------------------:|
| vanilla4 | 0.431 | 0.681 | **0.775** | 1981→1134 | 21% |
| sage4 (take 3) | 0.431 | 0.588 | **0.650** | 1981→1342 | 45% |
| oracle, untrained SAGE decode | — | — | 0.806 | 1832 | — |

- **SAGE-in-the-loop lost to vanilla GRPO** (0.650 vs 0.775, ~1.4σ at n=32)
  despite SAGE members outscoring sampled members inside nearly every group
  (e.g. step 8: 1.0 vs 0.39). Distillation pressure was present; it converted
  into *less* greedy-eval gain than plain GRPO. Suspects, in order: 34% of
  rollouts truncated at cap 2048 (the sampled contrast is distorted), 16
  steps × 3 prompts is tiny, off-policy ratio-1 injection may fight the
  group baseline at G=8 (2 SAGE members shift the mean all members are
  scored against).
- **RFCS (now implemented, incl. post-hoc)**: both arms sit at ~0.37–0.45 —
  the model finds the answer well before mid-think and keeps going; the
  paper's redundancy claim reproduces in our traces. Neither arm moved it
  visibly in 16 steps.
- Honest conclusion: at this compute shape the extra 7× SAGE rollout cost
  buys nothing that vanilla GRPO doesn't get cheaper. SAGE's proven value
  today is as a *decoder* (0.806 untrained) — now servable as the
  `qwen36-sagedecode` backend. SAGE-RL gets one more shot at v5 scale
  (cap 3072, ~24h, math-heavy mix) or is shelved with the oracle as its
  legacy.

## Efficient-RL levers (2026-07-12, from docs/efficient-rl-edge-reading.md)

v4 measurements corrected the reading list's premise: on this box with
grad-checkpoint, **the update dominates** (vanilla 79% of wall-clock; sage
55%) — and **45–48% of all generation fed zero-advantage groups** (two
whole steps had 0 active groups). Landed, all flag-gated, defaults off:

- `--group-stage1 N` + `--stage1-skip saturated|uniform` (GRESO 2506.02177 /
  AERO 2602.14338 spirit, adapted within-step since our prompts rarely
  recur): sample N members first, abandon groups whose stage-1 rewards are
  already decided before paying for the rest + SAGE beams. `saturated`
  (default) abandons only uniform-at-1.0 groups — an all-wrong stage-1 can
  still yield signal. Stage-1 grades are cached (code = subprocess) and
  reused.
- `--update-adv-frac f` (AERO selective rejection / 2504.20834 spirit):
  skip the backward for rollouts with |adv| < f × group max. The loss
  denominator stays the FULL active batch — truncation, not reweighting.
  Attacks the dominant cost directly.
- `gen_nll` metric (QeRL 2510.11696 observable): mean NLL of sampled tokens
  = the exploration/entropy proxy. We already ARE the QeRL configuration
  (4-bit policy rollouts + LoRA updates); now we can watch the noise.
  QeRL's adaptive-noise-injection component remains a future experiment.
- Also new: `math` task (DeepScaleR-Preview-Dataset, 25,333 numerically
  verifiable problems + 200 held out) — the paper's actual domain, for v5.

v5 plan: A/B the levers on vanilla GRPO first (they must not change the
learning outcome, only the bill), then the last SAGE-RL attempt at cap 3072.
