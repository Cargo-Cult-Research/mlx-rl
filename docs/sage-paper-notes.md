# SAGE paper notes — arXiv 2602.08354

**Self-Aware Guided Efficient Reasoning.** The spec this repo's implementation
is checked against. Verbatim quotes marked with "…".

## Thesis

Reasoning models reach the correct answer *early* in the chain, then keep
thinking redundantly. Measured by **RFCS = Ratio of First-Correct-Step** = the
step index where the correct answer first appears / total reasoning steps; RFCS
≪ 1 for >50% of correct answers. "LRMs implicitly know the appropriate time to
stop thinking, [but] this capability is obscured by current sampling." SAGE is a
decoding paradigm that surfaces the early, confident stop.

## SAGE decoding — the exact algorithm

- **Reasoning step** = segment delimited by `"\n\n"` (double newline). The beam
  operates over *steps*, not tokens.
- **Confidence Φ** = length-normalized cumulative log-prob of the whole chain:
  `Φ(y≤k) = (1/k) Σ_{i=1..k} log π_θ(y_i | y_<i, x)` — i.e. **mean per-token
  log-prob**. Ranking is by Φ, NOT marginal single-token prob.
- **Beam update (step-wise)**: keep top-`m` candidate chains. Each step, for each
  candidate, **sample `2m` full reasoning steps** (random sampling, temp = top-p
  = 1.0) → `2m²` expanded candidates → rank by Φ → keep top-`m`.
- **Termination (top-h acceptance)**: a candidate whose step ends with `</think>`
  is accepted as a completion when `</think>` is **within the top-h** ranked
  candidates. **Tolerance ratio `TR = h / (2m)`**. In practice "when Φ is present,
  `</think>` consistently ranks **first** within the candidate set at the moment
  it appears" — Φ ranking "prevents **low-confidence early termination**".
  Collect completions until `|O| ≥ r`, then stop.
- **Budget**: `T_max` = max reasoning **steps** (token budgets in experiments are
  large: 10k–32,768). **No small token cap.**
- **Length-collapse warning**: using marginal `ϕ` (single-token prob) instead of
  cumulative `Φ` "suffers a rapid degradation in accuracy that closely tracks the
  sharp decline in response length." **Φ, not ϕ, is what prevents collapse.**

## SAGE-RL

- `SAGE(m,r)` produces `r` of the `G` group members; the other `G−r` are ordinary
  random samples. Typical: **SAGE(2,2)**, G=8 → 2 SAGE + 6 sampled.
- **Reward is unchanged RLVR** (GRPO/GSPO). "The sole difference between SAGE-RL
  and RLVR lies in the rollout phase." **No length reward, no length penalty.**
- Efficiency emerges because the policy imitates the genuinely-short-correct
  chains SAGE surfaces: entropy ↓, response length ↓, **RFCS ↑** (stop right after
  the answer). Accuracy is *maintained or improved* while length drops.

## Defaults (experiments)

m=2, r=2, TR ∈ {0.5, 0.75, 1.0} (stable), temp = top-p = 1.0, `\n\n` step delim.
Models: DS-1.5B, DeepScaleR, DS-7B, Qwen3-8B. Benchmarks: MATH-500, AIME 24/25,
AMC23, OlympiadBench, Minerva (math only). Headline: MATH-500 +1.6% acc /
−1967 tok; AIME24 +3.7% / −5057 tok. Token-efficiency +71–111%.

**SAGE-RL operating point (from the paper):** RL **training budget 8,192
tokens**; evaluation reported at
**32,768**. G=8 with SAGE(2,2) (2 SAGE + 6 sampled). No curriculum or
difficulty filtering mentioned. Baseline (pre-RL) response lengths for
DS-1.5B: **MATH-500 ≈ 4,882 tok → 2,921 after SAGE-GSPO; AIME25 ≈ 11,669 →
7,167** — i.e. ~1.6× compression, matching our measured 1.5–2× on solvable
math. Implication for us: the paper never demands thinking fit 4,096 —
even its easiest benchmark baseline (4.9k) would blow our 4096 cap; our
uniform-DeepScaleR sampling is AIME/Olympiad-heavy relative to the MATH-500
regime the headline numbers come from.

## Implementation pitfalls — plausible shortcuts that break the algorithm

| aspect | paper | plausible shortcut | consequence |
|---|---|---|---|
| granularity | step-wise (`\n\n`) | token-wise beam | different search |
| stop rule | `</think>` in **top-h** by Φ ("ranks first") | eager tolerance stop (`best_stop.Φ ≥ cutoff − tol`) | **low-confidence early stop** — the exact thing top-h prevents |
| budget | `T_max` reasoning steps (~10k+ tok) | small hard think-token cap | forced truncation (think_len median pinned at the cap) |
| Φ | cumulative mean log-prob | marginal single-token prob | wrong ranking signal |

These are not hypothetical: an implementation with the first three shortcuts
produces SAGE members that are short by **premature/forced stopping**, not
genuine confidence. RL then reinforces them (correct on easy tasks) and the
policy learns to slam `</think>` shut and relocate its CoT into visible prose
— degrading downstream behavior. The shipped decoder is the faithful version:

Faithful `SAGE(m, tr)`: step-wise `\n\n` beam, `2m` step-expansions/candidate,
Φ ranking, top-h `</think>` acceptance (h = round(TR·2m)), `T_max` *step* budget
with a large token safety ceiling (not a small think cap). Log **RFCS**. Reward
stays pure RLVR (no length term) to match the paper — the faithful decoder
removes the relocation incentive by itself.
