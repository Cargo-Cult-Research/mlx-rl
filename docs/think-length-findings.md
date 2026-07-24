# Think-length distributions: vanilla vs SAGE (2026-07-15)

**Question (after v8 refuted the truncation hypothesis):** how long does
the model *want* to think, per task — and does SAGE decoding compress it into
a trainable window? Decision rubric: if SAGE takes the median from 20k→10k we
are delusional; 6k→3k maybe close; 3k→2k already there.

**Instrument:** `scripts/think_length.py` — the v8 eval stream (mixture
math .35 / code .35 / arith .3, seed+100000, the same 32 problems behind
v8's eval_correct 0.656), base model, non-binding budgets. think_len is exact
for every condition (index of `</think>` = token 248069); the 07-12 oracle
data could not answer this (vanilla think_len never populated; SAGE
right-censored at 3073 = its 64×48 step budget). Runs:
`runs/think-length-20260715` (greedy/sampled k=4/SAGE m=2 tr=0.5 @8192,
SAGE step budget raised to 167 steps) and `runs/think-length-16k-math`
(greedy + sampled k=2 @16384, math slice only).

## Headline table — think-length quantiles @8192

| cond | task | n | med | p25 | p75 | p90 | max | censored | acc |
|---|---|---|---|---|---|---|---|---|---|
| sampled | arithmetic | 48 | 1328 | 970 | 1807 | 2563 | 4309 | 0% | 1.000 |
| sampled | code | 40 | 1693 | 920 | 3261 | 6512 | 8192 | 8% | 0.925 |
| sampled | math | 40 | **8192** | 8192 | 8192 | 8192 | 8192 | **82%** | 0.100 |
| sage | arithmetic | 12 | 1119 | 634 | 1299 | 1335 | 1717 | 0% | 1.000 |
| sage | code | 10 | 1663 | 669 | 1720 | 4058 | 4058 | 0%* | 1.000 |
| sage | math | 10 | **7921** | 5437 | 7921 | 7921 | 7921 | 0%* | 0.400 |

\* SAGE "0% censored" is cosmetic: it force-appends `</think>` at its think
ceiling (7921 = 165×48+1 here), which is censoring in disguise — every math
7921 is a forced close.

**Would fit under a training cap** (uncensored think < cap):

| cond | task | <2048 | <4096 | <8192 |
|---|---|---|---|---|
| sampled | arithmetic | 85% | 96% | 100% |
| sampled | code | 65% | 80% | 92% |
| sampled | math | 0% | **8%** | 18% |
| sage | arithmetic | 100% | 100% | 100% |
| sage | code | 80% | **100%** | 100% |
| sage | math | 10% | 20% | 100%* |

## De-censoring math @16384

Even at 16k, **half the math problems never close thinking, in any of 3
attempts each** (greedy + 2 sampled). Per problem (think/reward, `>` = never
closed; # = position in the 32-problem eval stream):

| # | greedy@16k | sampled@16k ×2 | sage@8k (07-15 run) |
|---|---|---|---|
| 1 | >16384/0 | >16384/0 · >16384/0 | 7921F/0 |
| 6 | >16384/0 | >16384/0 · >16384/0 | 7921F/0 |
| 10 | 3131/1 | >16384/0 · 14936/1 | **5437/1** |
| 11 | >16384/0 | >16384/0 · >16384/0 | 7921F/0 |
| 16 | 7691/1 | 7538/1 · 11390/1 | **3714/1** |
| 24 | >16384/0 | >16384/0 · >16384/0 | 7921F/0 |
| 27 | 1877/1* | 2281/1 · 2626/1 | **1629/1** |
| 28 | >16384/0 | >16384/0 · >16384/0 | 7921F/0 |
| 31 | 13855/1 | >16384/0 · >16384/0 | 7921F/0 |
| 32 | 5823/1 | 9834/1 · 10045/1 | **7365/1** |

(F = forced close at the SAGE ceiling; *greedy@8k value, the 16k rerun of
#27 closed at 2266.)

The math task is **bimodal**: a *solvable stratum* (10, 16, 27, 31, 32 —
closes at 2.3k–13.9k, vanilla median ≈ 7.5–10k) and a *black-hole stratum*
(1, 6, 11, 24, 28 — beyond 16k or never; the model likely loops). Greedy@16k
accuracy is 0.500 — capability saturates at half this set regardless of
budget.

## SAGE compression, measured per problem (solvable stratum)

- #16: vanilla 7.5–11.4k → SAGE **3.7k** (2–3×)
- #10: vanilla 3.1–14.9k → SAGE **5.4k**
- #27: vanilla 2.3–2.6k → SAGE **1.6k** (~1.5×)
- #32: vanilla 5.8–10.0k → SAGE **7.4k**
- #31: vanilla solves once at 13.9k → SAGE fails at its 7.9k ceiling

On solvable math SAGE compresses roughly **1.5–2×** and turns acc 0.10→0.40
at 8k. By the rubric that is the "6k→3k maybe close" regime — but only for
the stratum the model can solve at all. Code is already "3k→2k": SAGE
compresses the p90 tail 6.5k→4.1k, 100% fits a 4096 cap, acc 1.0.

## Conclusions (with the dashboard rollup on v8)

1. **The v8 mixture is bimodal by construction.** Dashboard anomaly rollup on
   v8 samples: math groups 75% whole-group-truncated and 75% zero-variance
   all-FAIL; arithmetic 76% zero-variance all-PASS; code (20%/34%) carries
   nearly all gradient signal. The think-length data explains why: math at
   this difficulty cannot fit any affordable cap (8% under 4096), arithmetic
   is too easy to have variance.
2. **No cap fixes math as-is.** The lever is task difficulty, not budget:
   graded math whose think distribution straddles ~2–4k (where code lives,
   and where the pass boundary generates variance), plus detection/resampling
   of black-hole prompts (5/10 of current math eval problems train nothing at
   any cap).
3. **SAGE-as-teacher is plausible but ceiling-bound**: real 1.5–2×
   compression on solvable math, tail compression + perfect acc on code —
   worth considering for distillation *after* the task difficulty is fixed,
   not before (its ceiling loses problems vanilla can reach, cf. #31).
4. Probe cost: 8192 all-tasks run 72 min + 16k math slice 18 min, both under
   the shared memory lease (the host server stayed up).
