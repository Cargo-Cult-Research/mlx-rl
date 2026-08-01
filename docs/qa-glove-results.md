# Calibrated honesty in the deployment register — glove program results

*2026-07-31. Runs: qa-full-20260726 (flagship), qa-chatmix-full-20260730,
qa-gloveA-20260731 (+ aborted qa-gloveA-20260730), qa-gloveB-20260730,
qa-binding-A200-20260731. Arm C in flight. All seed 0, single runs.*

## Claim

A 35B local model can be RL-trained (GRPO + LoRA, one M3 Ultra) to act on
its own uncertainty **in free conversation** — hedging on unknowable
questions, keeping answers on known ones, and dropping the
asserted-nonexistence failure mode — provided the abstain affordance ships
as a system prompt (**the glove**) and the chat curriculum is
unknown-heavy. Trained on trivia only, the behavior transfers to arXiv
author/year questions in chat: post-cutoff hedging 0.03 → 0.78–0.88 while
famous-paper answering stays 0.95–1.00.

## The path (each negative forced the next design)

1. **Flagship (tag-only)**: wrong 0.30→0.095, precision 70%→87%. Required
   symmetric oracle injection (all-abstain is an absorbing state under
   zero-variance group dropping) and a dead-run watchdog.
2. **Transfer split**: the tag adapter abstains 0.96/0.86 on post-cutoff
   papers *in the tag frame*, 0.01 in chat. Capability transfers across
   domains; register doesn't.
3. **Naive frame mixture (no glove)**: flat. Sampled chat declines 4/356 →
   3/335 between run halves — a ~1% propensity gets no policy-gradient
   mass. Sparse ignition, not a broken mechanism (judge 219/219 correct on
   injected declines; specimen groups show correct gradient direction).
4. **Glove control (no training)**: unknown-hedging 0.04→0.33–0.79, but
   uncalibrated (0.17 hedge on knowns) and famous-paper fabrication
   untouched. Instruction = affordance + exploration floor, not
   calibration.
5. **Factorial**: B (glove, tag-tuned bands) ≈ flat in-run; only
   constructed-unknowable buckets move. A (glove, chat bands
   0.15/0.35/0.50) learns: answered 0.93→0.715 (best ckpt 160), wrong
   →0.10, denial →0.0. **Glove = affordance; unknown-heavy chat
   curriculum = learning.**

## Headline grid (free chat, hedge+denial; probes k=4)

| bucket | base | +glove | B | A-200 | A-160 | A-200 no-glove |
|---|---|---|---|---|---|---|
| chat-unknown | 0.04 | 0.33 | 0.25 | 0.42 | 0.71 | 0.00 |
| real-obscure | 0.00 | 0.08 | 0.08 | 0.50 | 0.58 | 0.00 |
| fictional-people | 0.20 | 0.56 | 0.82 | 0.97 | 0.88 | 0.11 |
| papers-post | 0.50 | 0.79 | 0.96 | 0.88 | 0.85 | 0.30 |
| chat-known (cost) | 0.04 | 0.17 | 0.08 | 0.29 | 0.46 | 0.00 |

arXiv recall (chat frame): A-200 hedges 0.78 (authors) / 0.88 (year) on
post-cutoff papers vs base 0.03/0.03; famous correct 1.00/0.95 kept.
Confident-wrong: real-obscure 0.67→0.17, fictional-people 0.80→0.03.
Denial (asserted nonexistence) on fictional-people: 0.01.

**Register binding**: glove-OFF probes of both A checkpoints sit at base
everywhere. The adapter+glove pair is the artifact; the adapter alone is
inert (and cannot leak hedging into contexts that didn't opt in).

**Checkpoint dial**: A-160 hedges harder on unknowns AND knowns (0.71 /
0.46), A-200's late drift rebalances (0.42 / 0.29). The global
propensity dial moves with training; per-item separation is real but
partial.

## Binding correlation (held-out, n=200, k=8, judge-graded)

decline@k vs base pass@8 on questions never trained and ~absent from the
calib file: **pearson −0.44; decline 0.09 on pass≥0.8 vs 0.41 on pass=0;
AUROC(flags pass=0) 0.71** — a lower bound (alias-grading false negatives
on both axes; e.g. a "pass=0" item whose policy answer "Frank and Joe
Hardy" is simply ungraded-correct). The policy reads per-item uncertainty;
band memorization is excluded by construction. Kadavath-style OOD
calibration collapse did not occur at this (modest) distribution shift.

## Method constants

Qwen3.6-35B-A3B 4-bit (22.09 GB), LoRA r16 s20 last-12-layers (4.28M
params). Per step 8×8=64 completions (1 injected oracle/group), micro 4,
192 max tokens, temp 1.0, thinking off. lr 3e-6, KL 0.01 vs frozen base,
200 steps ≈ 12.8k completions ≈ 3–4 h. Reward +1/0/−3 (threshold 0.75);
chat frames judged by cached Opus commitment parser (answer/abstain/denial;
denial = −3; cache 23.8k verdicts). Data: TriviaQA 138,384 (500 held out);
2,000-question pass@8 calib → bands 997/468/535; tag mix 0.65/0.25/0.10
(EV-balanced), A chat mix 0.15/0.35/0.50. Guards: swap-guard (12 GB margin
for judge-heavy runs — the per-step judge subprocess ratchets stale swap
~50 MB/step without thrashing), inactive-window abort, memlease
displacement of the serving stack.

## Arm C (known-side pressure) — the deliverable (added 2026-08-01)

Single variable vs A: chat bands 0.35/0.35/0.30. C-200 keeps A's
fabrication protection (fictional-people 0.95, papers-post 0.92, arXiv
post-year 0.88, famous 1.00/0.95) while cutting known-hedging 0.29->0.21
and recovering chat-known correct to 0.46 (base level). Binding improves:
pearson -0.497, AUROC 0.738, decline 0.021 known vs 0.347 unknown — a 16x
per-item separation (A-200: 4.6x). Cost: real-obscure hedging 0.33 (A:
0.50), post-cutoff-author 0.65 (A: 0.78). **qa-gloveC-200-20260731 +
HONESTY_SYSTEM is the current deliverable pair.**

**Seed-1 replication (2026-08-01): REPLICATED.** Fabrication buckets
within noise (fictional-people 0.91, papers-post 0.90, famous 1.00/1.00),
known-side cost even lower (chat-known h+d 0.08, correct 0.50). Binding:
pearson -0.467, AUROC 0.738 (identical), known 0.013 vs unknown 0.291 =
22x separation. Seed-noise floor visible on n=12 buckets (multi-unknown
0.42->0.58) and post-cutoff-author hedging (0.65->0.46) — report as such.

## Open
- Famous-paper summarization-without-disclaimer untouched (0.95) — a
  chapter-1 detector target, not a reward-shape target.
- c=1 (TruthRL threshold) arm; capability regression gates (coding
  slices); multi-seed replication; alias-grading noise floor.
