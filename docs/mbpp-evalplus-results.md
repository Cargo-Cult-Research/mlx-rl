# MBPP → EvalPlus — what a comparable code-RL number costs

*2026-08-12/15. Runs: code-conv (50-step, aborted line), code-conv2
(collapsed), code-conv3 (flagship of the internal-split era), kodcode-run1,
kodcode-run3, kodcode-run4, coder15-run1, coder15-run3 (full epoch;
coder15-run2 died at step 69 to a swap-guard false positive), plus the
qwen36 baseline. All seed 0, single runs, Qwen2.5-0.5B-Instruct-4bit unless stated.*

## Claim

GRPO + LoRA moves MBPP pass@1 on a small local model, and the movement is
real — but almost none of it was **comparable to any published number**, and
most of what finally *was* comparable turned out to be repair of a prompt
pathology rather than coding capability. Getting to an honest claim required
three corrections in sequence: fix the eval set (contamination), fix the
harness (EvalPlus), and then fix the *subject model* (prompt sensitivity).

After all three, the claim that survives: **Qwen2.5-Coder-1.5B-4bit goes
0.672 → 0.730 MBPP and 0.571 → 0.624 MBPP+ under the official EvalPlus
harness** (paired McNemar p = 0.003 / 0.008), trained only on prompt dialects
the benchmark does not use, with the artefact chosen by a rule fixed in
advance. Everything before that is method — and the record of what a
comparable number costs.

Scale context, kept next to it so the result is not oversold: an **untrained
qwen36 (Qwen3.6-35B-A3B-4bit) scores 0.870** on the same benchmark. Model
choice dominates everything RL bought at 1.5B.

## The path (each negative forced the next design)

1. **Internal-split success (`code-conv3`)**: 0.237 → 0.413 plateau on the
   repo's own 80-problem holdout. Real learning, useless for comparison.
2. **Contamination**: the shipped sanitized-MBPP 427 contains 257 problems
   from the canonical MBPP *test* split (ids 11–510). The seeded shuffle put
   **209 of them in the train pool**. Any leaderboard comparison was dead on
   arrival.
3. **EvalPlus**: rebuilt around `evalplus/mbppplus` (378 tasks, the exact set
   behind the leaderboard) as a strictly held-out eval, with a disjoint
   KodCode train pool. First uncontaminated baseline: **0.283**.
4. **The harness disagreed**: our 0.392 artifact scored **0.061** under the
   real EvalPlus harness. Cause was not scoring but *prompting* — EvalPlus
   wraps the problem in a ```` ```python ```` fence (the **fenced** prompt,
   vs our **bare** one: same instruction and docstring, no fence — the two
   differ by that one fence alone), and the 0.5B echoes the prompt back
   instead of answering (98% of responses contained no code).
5. **Format repair ≠ capability**: retraining on the fenced prompt took the
   official score 0.000 → 0.310, but the cross-format matrix showed the
   trained model's fenced score (0.265) merely *equalled the base model's
   unfenced score* (0.283). We had bought back what the format was
   suppressing, and nothing else.
6. **Mixing dialects worked** (`kodcode-run4`, official **0.365 / 0.299**),
   but training on the evaluation harness's own template is benchmark
   adaptation, and the counterfactual was measured, not hypothetical.
7. **Sensitivity sweep**: the pathology is not universal — it is idiosyncratic
   to particular checkpoints. Four of six models are format-neutral. So the
   fix is to *change the subject model*, not to argue about the prompt.
8. **Clean result** (`coder15-run1`): on Qwen2.5-Coder-1.5B, which has a zero
   format gap, trained on non-eval dialects only — **0.672 → 0.709 MBPP,
   0.571 → 0.601 MBPP+**, official harness both sides, p = 0.044 paired.
9. **Full epoch** (`coder15-run3`): sampling without replacement over the
   whole 7601-problem pool — **0.730 / 0.624**, p = 0.003 / 0.008. Both
   columns significant. Cost: 16.1 h for ~+2 points, and the eval curve was
   flat from step 350, so ~600 steps bought nothing measurable.

## Era 1 — the internal-split numbers (main branch, `code` task)

Sanitized MBPP 427, seeded 80/347 split, reward = all hidden asserts pass.

| run | config | baseline | result |
|---|---|---|---|
| code-conv | 50 steps, kl 0.01, eval n=32 | 0.281 | peak 0.344, **final 0.219** — no significant gain; policy collapsed to ~50-token one-liners |
| code-conv2 | 150 steps, **kl 0.05** | 0.238 | **destroyed at step 16** (gen_nll 0.77 → 3.0, number-babble); unrecoverable because zero-variance batches skipped the update entirely |
| code-conv3 | 150 steps, kl 0.02, lr 5e-6 | 0.237 | **0.413 plateau** (best 0.463, final 0.412) |

Two fixes came out of this era, one of which did not survive review.
`code-conv2`'s death produced a **KL-rescue update** (a zero-variance batch
still applied the KL term, pulling a degenerated policy back toward base);
it fired 13× in `code-conv3` with no blowups, but was **removed in PR
review** on the argument that it medicates a dying run instead of letting it
fail loudly — and on an all-pass batch it is quiet *un*-learning. Its
replacement is the dead-run watchdog (`abort_inactive_window`, now on by
default at 40): degeneration aborts with a clear error instead of being
silently treated. The lr-anneal knob from the same era was likewise removed
after its negative result (this doc is where both findings live). And the flip analysis on
`code-conv3` showed the flat plateau was hiding ±35-problem churn every 20
steps — the policy was trading marginal solutions, not stalling. Averaging
late checkpoints ("soup") recovered part of that: 0.392 vs 0.362 best single.

**Why none of it counts**: 209 of the 257 canonical-test problems in our
data were trainable. See step 2 above.

## Era 2 — EvalPlus, and the prompt discovery

New task (`kodcode`): train on KodCode, eval on the 378 EvalPlus tasks with
exact-coverage cycling so `--eval-n 378` is the whole benchmark, every time.

### Training data — KodCode

The contamination problem in Era 1 was structural: we were training and
evaluating on two slices of the same 427 problems. The fix is a training
corpus with no MBPP in it at all.

[**KodCode**](https://arxiv.org/abs/2503.02951) (Xu et al. 2025) is a
synthetic coding corpus of 447K question/solution/test triplets where
**every triplet is execution-verified** — the authors ran each reference
solution against its own tests and kept only what passed — and the corpus is
**decontaminated against MBPP and HumanEval** by construction. That second
property is what makes the EvalPlus eval trustworthy: no eval problem can
leak in through the training pool. We use the `KodCode-Light-RL-10K` split
(10K problems curated by the authors for RL), which ships `gpt_difficulty`
labels and 12 topical subsets.

Practical notes:

- **Grading**: KodCode tests are pytest-style (`from solution import f`), so
  the candidate is written to `solution.py` and graded by pytest's exit code
  in the same Seatbelt sandbox as the MBPP asserts. Sampling 30 reference
  solutions through our runner passed 30/30 on both paths — a check that the
  harness, not just the data, is sound.
- **Excluded subsets**: `Package` and `Docs` need non-stdlib imports, which
  the no-network sandbox cannot satisfy. Every candidate there fails for
  environment reasons, poisoning the reward with always-zero groups.
- **Licence**: CC BY-NC 4.0 — **non-commercial**. Fine for this research;
  relevant if anything downstream is ever commercialised. (MBPP itself is
  CC BY 4.0; the mlx-rl code is MIT.)

**Difficulty calibration matters more than pool size.** GRPO gets gradient
only from groups with reward variance; a problem all 8 samples solve and a
problem none solve are equally useless. `scripts/kodcode_calibrate.py` probes
per-problem pass@8 at temperature 1.0 through the real sandboxed reward. On
the 0.5B, 300 KodCode-easy problems gave:

| band | share |
|---|---|
| saturated (8/8) | 0% |
| **learnable (1–7 of 8)** | **39%** |
| unsolved (0/8) | 61% |

By subset, `Filter`/`Prefill` were 53% learnable while
`Algorithm`/`Data_Structure`/`Leetcode`/`Code_Contests` were 85–100%
unsolved — dead weight at that model size. Restricting to Filter+Prefill
raised the expected active-group fraction from 39% to 53%, and the runs
sustained 2.0–2.5 active groups of 4 throughout. The same reasoning is why
`coder15-run1` *widens* the pool: bands are per-model, and a pool tuned for a
0.5B is partly saturated for a model 2.4× stronger.

The two prompts differ by one code fence:

| | prompt body |
|---|---|
| **bare** (ours) | instruction + blank line + `"""docstring + first assert"""` |
| **fenced** (EvalPlus official) | instruction + `` ```python `` + same docstring + `` ``` `` |

Cross-format matrix (our harness, 378 tasks, greedy):

| adapter | bare | fenced |
|---|---|---|
| base | 0.283 | **0.000** |
| souplate1 (trained bare) | 0.392 | 0.032 |
| souplate3 (trained fenced) | 0.283 | 0.265 |

Read the last row against the first: training on the fenced format bought
0.000 → 0.265 on that format while leaving bare performance exactly at base.
The official +31 points were **format repair, not capability**. Mixing three
training dialects (`kodcode-run4`) did add real capability on top —
official **0.365 / 0.299 (MBPP / MBPP+)** vs base 0.000 / 0.000 — and its
eval curve was still climbing at step 150.

Every headline number above is a **souplate** — an average of late
checkpoints; see [Checkpoint soups](#checkpoint-soups-souplates) for what
they are, what each one scored, and why composition turned out not to matter.

## Era 3 — prompt-sensitivity sweep (the reason we changed models)

EvalPlus-378, greedy pass@1, both prompts, identical harness:

| model | fenced (official) | bare | gap |
|---|---|---|---|
| Qwen3.6-35B-A3B-4bit (qwen36, thinking off) | **0.870** | not run | — |
| Qwen2.5-Coder-1.5B-Instruct-4bit | 0.667 | 0.667 | **0.000** |
| Qwen2.5-1.5B-Instruct-4bit | 0.563 | 0.574 | −0.011 |
| Llama-3.2-3B-Instruct-4bit | 0.537 | 0.566 | −0.029 |
| gemma-2-2b-it-4bit | 0.468 | 0.508 | −0.040 |
| Qwen2.5-3B-Instruct-4bit | 0.333 | 0.640 | **−0.307** |
| Qwen2.5-0.5B-Instruct-4bit | 0.000 | 0.283 | **−0.283** |

**Format brittleness is not a scale effect.** Same size, different family:
Qwen2.5-3B is destroyed (−0.307) where Llama-3.2-3B is fine (−0.029). Same
family, different size: 1.5B is fine, 3B is not, 0.5B is worst. Four of six
models across three families are neutral within decode noise. This is
[Sclar et al. 2024](https://arxiv.org/abs/2310.11324) ("format performance
correlates only weakly between models") in its sharpest form; their
recommendation to report a *range* across formats rather than a single
number is why this table has two columns.

Sobering side note: **Qwen2.5-1.5B scores 0.563 untrained**, and Coder-1.5B
0.667 — both far above the 0.365 our best-trained 0.5B artifact reached. At
0.5B the dominant measurable signal was brittleness, not coding ability.

## Era 4 — the honest attempt (`coder15-run1`)

**Subject:** Qwen2.5-Coder-1.5B-Instruct-4bit — highest official score in the
sweep, zero measured format gap.

**Trained bare.** `train_formats: "bare,instruct"` — the EvalPlus template is
deliberately *excluded* from training. The model has no format deficit to
repair, so it does not need the template, and excluding it removes the
benchmark-adaptation objection by construction rather than by argument. Two
non-eval dialects are mixed because `kodcode-run4` showed single-dialect
training overfits to its phrasing.

**Pool:** KodCode easy+medium, all subsets except `Package`/`Docs`. Widened
from run4's easy/Filter+Prefill, which was calibrated for a model ~2.4×
weaker and would be partly saturated here — a group where all eight samples
pass carries no gradient, exactly like one where none do. Active-group count
is monitored from step 1; re-calibrating with
`scripts/kodcode_calibrate.py --model <coder>` is the fallback if the band is
wrong.

**Constants:** 150 steps, 4 prompts × group 8, 640-token budget, kl 0.02,
lr 5e-6, LoRA r16 all layers, full-378 eval every 25 steps, seed 0.

```sh
.venv/bin/mlx-rl-train \
  --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
  --task kodcode \
  --task-kwargs '{"difficulties": "easy,medium", "train_formats": "bare,instruct"}' \
  --steps 150 --batch-prompts 4 --group-size 8 \
  --max-new-tokens 640 --kl-coef 0.02 --lr 5e-6 \
  --eval-every 25 --eval-n 378 \
  --out runs/coder15-run1 2>&1 | tee runs/coder15-run1.log
```

`--eval-n 378` plus the task's exact-coverage cycling means every eval is the
*whole* benchmark, not a resample — eval points are comparable to each other
and to the sweep baseline. Step 0 reproduced 0.6667 exactly, confirming the
adapter starts at identity.

**Baseline to beat: 0.667 official.** Any gain is unambiguously capability.

### Result (150 steps, 132 min)

| step | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|---|
| eval | 0.6667 | 0.6561 | 0.6799 | 0.6905 | 0.6931 | 0.6905 | **0.7090** |
| (n) | 252 | 248 | 257 | 261 | 262 | 261 | **268** |

Six of seven evals above baseline, ending on the high-water mark. Train-side
rose throughout (0.431 → 0.597 by fifths) with 2.2–2.6 groups active
start to finish — the widened pool never saturated — and only 4 rescue steps
fired.

**Official EvalPlus harness, both models, verified base control:**

| | base Coder-1.5B | souplate5 | Δ |
|---|---|---|---|
| MBPP | 0.672 (254/378) | **0.709 (268/378)** | +3.7 pts |
| MBPP+ | 0.571 (216/378) | **0.601 (227/378)** | +3.0 pts |

Because both were measured through the same harness on the same tasks, the
paired test is the right one. On MBPP: 240 problems both solve, **28 only
souplate5 solves, 14 only base solves** — McNemar exact **p = 0.044**. MBPP+
moves the same way (26 vs 15 discordant, +11) but at p = 0.12 is not
significant alone.

**This is the first clean result in the line.** The eval prompt never appeared
in training, so none of it is format repair; the baseline was 0.672, not a
broken 0.000, so there were no easy points to reclaim; both numbers come from
the official harness with a control verified to actually serve base weights;
and the gain survives MBPP+'s ~5× larger test suites, so it is not an artifact
of the three original asserts.

Caveats worth keeping attached: one run at one seed, MBPP+ directional but
not significant, and **14 problems regressed** — RL bought 28 and gave back
14.

### Churn — a different learning dynamic

Flips between consecutive checkpoints (EvalPlus-378, fenced):

| transition | gained | lost | churn | net | total |
|---|---|---|---|---|---|
| base → 20 | 8 | 14 | 22 | −6 | 246 |
| 20 → 40 | 16 | 10 | 26 | +6 | 252 |
| 40 → 60 | 9 | 4 | 13 | +5 | 257 |
| 60 → 80 | 14 | 11 | 25 | +3 | 260 |
| 80 → 100 | 4 | 4 | 8 | 0 | 260 |
| 100 → 120 | 9 | 5 | 14 | +4 | 264 |
| 120 → 140 | 4 | 4 | 8 | 0 | 264 |
| 140 → 150 | 5 | 1 | 6 | +4 | 268 |

Two things differ from the 0.5B line. **Churn is lower** — 6–26 problems move
per window against ~35 for `code-conv3` — and it *decays* as training
proceeds (22–26 early, 6–8 late) instead of staying flat. And the flips are
**asymmetric**, +69/−53 overall, where the 0.5B runs traded evenly and netted
nothing (one window was exactly +35/−35). That is accumulation with noise on
top rather than a random walk across marginal solutions.

The first transition is negative (−6): the early adapter is briefly worse
than base, matching the step-25 dip in the eval curve.

Union across the 8 checkpoints is 284/378 (0.751) against a best single of
268 — a 16-problem gap, proportionally far smaller than the 0.5B's (178 union
vs ~130 single). 224 problems pass at *every* checkpoint.

That correctly predicts what souping does here, which is **nothing, or
slightly worse than nothing**: souplate5 scores 265 in our harness against
step 150's 268, captures 265 of the 284-problem union, and solves **zero**
problems that no ingredient solved. Compare souplate1 on the 0.5B, which beat
its best ingredient by 8 and found 4 brand-new. Low churn leaves averaging
nothing to average over — the same lesson the lr-anneal run taught in
reverse. It also means this run converged rather than wandered.

(souplate5 was still the artifact sent through the official harness, at
0.709/0.601. Its 0.709 there and step-150's 0.709 in our harness are
different measurements that coincide, not the same number.)

## Era 5 — the full epoch (`coder15-run3`)

Same recipe as `coder15-run1` with three changes: `sampling="epoch"` (without
replacement — run1's 600 draws-with-replacement touched only 581 of 7901
problems), a 300-problem validation holdout reserved BEFORE training
(`val_frac: 300`), and `--batch-prompts 8` over the full easy+medium pool.
951 steps = one pass over all 7601 training problems. 16.1 h wall.

| step | 0 | 25 | 100 | 300 | 350 | 600 | 800 | 950 |
|---|---|---|---|---|---|---|---|---|
| eval | 0.667 | 0.704 | 0.709 | 0.722 | 0.735 | **0.738** | 0.728 | 0.720 |

Fast rise to ~0.70 by step 25, a second climb to ~0.73 around step 350, then
**flat for 600 steps** — everything from 350 on sits in 0.720–0.738, a
6-problem band. Active groups decayed 4.45 → 2.88 of 8 as the policy
saturated the pool: the binding constraint is problem difficulty, not data
volume. A harder pool, not more steps, is the next lever.

**Official EvalPlus, souplate6 (pre-specified last-three rule):**

| | base | run1 souplate5 | **run3 souplate6** |
|---|---|---|---|
| MBPP | 0.672 (254) | 0.709 (268) | **0.730 (276)** |
| MBPP+ | 0.571 (216) | 0.601 (227) | **0.624 (236)** |

Paired vs base: MBPP +22 (37 gained / 15 lost, McNemar **p = 0.0032**);
MBPP+ +20 (36 / 16, **p = 0.0078**). Both columns individually significant —
run1's MBPP+ was only directional (p = 0.12). The oracle checkpoint (step
600) would score ~0.738 in-loop, so the honest selection rule cost ~1 point.

**Scale context (measured, not estimated): untrained qwen36
(Qwen3.6-35B-A3B-4bit, thinking off) scores 0.870 (329/378)** on the same
benchmark, same prompt, no format pathology. Fourteen points above our best
trained 1.5B. If the goal is a strong local coding model, model choice
dominates; the 1.5B remains the right *subject* for studying transfer
because it has headroom the 35B mostly lacks.

### Validation-based selection failed here — a negative result worth keeping

Run3 had what run1 lacked: a holdout reserved before training. Scored every
100 steps:

| step | 100 | 300 | 500 | 600 | 800 | 951 |
|---|---|---|---|---|---|---|
| val (KodCode, n=300) | 0.630 | 0.673 | 0.660 | 0.687 | 0.693 | **0.707** |
| test (EvalPlus-378) | 0.709 | 0.722 | 0.733 | **0.738** | 0.728 | 0.717 |

**Validation rises monotonically while test plateaus and drifts down**
(r = 0.387 over 10 checkpoints — below the 0.632 significance threshold, and
picking the test-WORST of the late checkpoints). The mechanism is ordinary
overfitting: the holdout is drawn from the *training* distribution, so it
keeps improving as the model fits KodCode harder, precisely while transfer
stops improving. Source-side validation stays the honest protocol (it uses no
target information), but on this run it was near-useless and late-run
anti-correlated. Both honest procedures still agreed on the tail (souplate6
had the top val score), which is why the headline survived — but treat
source-val selection as weak evidence, not a guarantee.

## Checkpoint soups ("souplates")

A **souplate** is the uniform per-tensor average of several LoRA checkpoints
from one run's trajectory — a [model soup](https://arxiv.org/abs/2203.05482)
over training time rather than over hyperparameters. The name is just
*soup* + *late*: we average the late checkpoints, typically steps 80–150.

The motivation is empirical. Flip analysis on `code-conv3` showed a flat eval
curve hiding ±35-problem churn every 20 steps: the policy keeps trading
marginal solutions rather than accumulating them. Across three snapshots,
178 distinct problems passed at least once while any single snapshot realised
only ~130 — averaging is the cheapest way to try to bank the union.

Building one is four lines; the result is an ordinary mlx-lm adapter dir:

```python
import mlx.core as mx, json
from pathlib import Path
ws  = [mx.load(f"runs/<run>/adapters/adapter-{s:05d}.safetensors")
       for s in (80, 100, 120, 140, 150)]
avg = {k: sum(w[k] for w in ws) / len(ws) for k in ws[0]}
out = Path("runs/<run>/soups/souplate-adapter"); out.mkdir(parents=True)
mx.save_safetensors(str(out / "adapters.safetensors"), avg)
(out / "adapter_config.json").write_text(json.dumps(
    {"fine_tune_type": "lora", "num_layers": -1,
     "lora_parameters": {"rank": 16, "scale": 20.0,
                         "dropout": 0.0, "keys": None}}))
```

### What we built and what it scored

| souplate | run (train dialect) | ingredients | our harness | official EvalPlus |
|---|---|---|---|---|
| souplate1 | run1 (instruct) | 80–150 | **0.392** *bare* | 0.061 / 0.053 |
| souplate2 | run2 (instruct, lr-annealed) | 80–150 | 0.376 *bare* | not run |
| souplate3 | run3 (fenced) | 80–150 | 0.265 *fenced* | 0.310 / 0.262 |
| souplate4 | run4 (mixed) | 80–150 | 0.339 *fenced* | 0.365 / 0.299 |
| souplate5 | coder15-run1 (bare+instruct) | 120–150 | 0.701 *fenced* | 0.709 / 0.601 |
| souplate6 | coder15-run3 (bare+instruct, full epoch) | 920–951 | 0.710 *val* | **0.730 / 0.624** |

Three findings, all of which argue *against* treating soup as a free win:

1. **Soup needs diverse ingredients.** souplate1 beat its best single
   checkpoint by 8 problems and solved 4 nothing else solved. souplate2 —
   same recipe but from the lr-annealed run, whose late checkpoints are
   near-duplicates — scored *lower* (0.376 vs 0.392) despite that run being
   more stable. souplate5, from a low-churn run, scored *below* its best
   ingredient (265 vs 268) and found **zero** brand-new problems. The churn
   soup feeds on is the same churn a well-behaved run doesn't have, so soup
   pays off exactly when training is misbehaving.
2. **Composition barely matters.** Ablating all eight cumulative windows on
   run4 (each row folding in one older checkpoint) spanned 128–139 problems
   against a standard error of 9.3 — roughly one SE end to end, with
   non-monotone marginal effects (adding step 100 cost 9 problems; adding
   step 40 gained 4). The shipped 80–150 window was in fact one of the weaker
   picks; the best was {120, 140, 150} at 139.
3. **So don't search for the maximum.** Picking the best of eight windows
   scored on the same 378 problems is eval-set selection, the same error as
   fitting the prompt. Use a pre-specified rule — *last three checkpoints* —
   and validate it on the next run, not the one that chose it.

## Choosing the artifact — checkpoint selection

Standard practice is to pick the checkpoint by validation score. We did not:
every number above uses a **pre-specified rule** (last three checkpoints,
souped). The reason is that our only eval *was* the test benchmark, and
choosing a checkpoint by its EvalPlus score then reporting that score is
selection-on-test — the same error as training on the eval prompt. Absent a
validation set, a rule fixed in advance spends no test information and is the
least-bad option, but it is a workaround for a missing piece, not a design.

**Which validation set, though?** MBPP problems outside the EvalPlus 378
exist (49 in the sanitized set, 596 in full MBPP) and are same-distribution
as the test set, which is what makes validation predictive. But this is a
**transfer** setup — train KodCode, test MBPP — so selecting with
MBPP-distributed data is target-domain (oracle) model selection, well known
and criticised in the domain-adaptation literature. It would weaken the claim
from "KodCode training transfers to MBPP" to "the best-on-MBPP checkpoint,
chosen using MBPP". The conservative choice is a **held-out slice of the
training distribution**, so `val_frac` carves a seeded holdout out of the
KodCode pool before training (`val_frac >= 1` is an absolute count).

### Does source-side selection actually work?

Tested retroactively on `coder15-run1` with 300 KodCode problems it provably
never sampled, scored by the same sandboxed pytest reward:

| step | source val (n=300) | test (EvalPlus-378) |
|---|---|---|
| 20 | 188 = 0.627 | 246 = 0.651 |
| 40 | 191 = 0.637 | 252 = 0.667 |
| 60 | 190 = 0.633 | 257 = 0.680 |
| 80 | 196 = 0.653 | 260 = 0.688 |
| 100 | 186 = 0.620 | 260 = 0.688 |
| 120 | 192 = 0.640 | 264 = 0.698 |
| **140** | **207 = 0.690** | 264 = 0.698 |
| 150 | 203 = 0.677 | **268 = 0.709** |

Source-val selects **step 140 → test 0.698**. The oracle would take step 150
at 0.709, so **honest selection costs 4 problems (~1.1 points)** — the price
of the conservative protocol, worth reporting rather than hiding.
Correlation is r = 0.643 across the 8 checkpoints: moderately predictive, but
with n = 8 the critical value is 0.707, so *not* statistically significant.
Claim no more than "moderately predictive, unproven".

**The reassuring part**: all three honest procedures land within ~1 point —
source-val 0.698, pre-specified soup 0.709 official, last-checkpoint 0.709 —
against a +3.7-point gain over base. The headline does not depend on the
selection rule, which is a stronger statement than any single number.
(**Run3 weakens this**: with 6× more training, val and test diverged —
r = 0.387, val picking the test-worst late checkpoint. See Era 5.)

> **Recovering what a run actually trained on.** `samples.jsonl` logs only
> the FIRST prompt's group per step (see the comment at its write site), so
> it holds 150 records for a 150-step × 4-prompt run. Using it as the trained-on
> set undercounts 4×: it suggested 149 distinct problems where the truth was
> **581**. Replay the training stream instead — `random.Random(cfg.seed)` is
> consumed only by `task.sample()` when `sage_r`/`inject_r`/`group_stage1`
> are 0 — and verify against the log (150/150 first-prompts matched). Getting
> this wrong leaks trained-on problems into the validation set.

## Reproducing an official EvalPlus number

Our in-loop eval is a proxy; the leaderboard-comparable number comes from
EvalPlus's own harness driven against a served adapter. EvalPlus is **not** a
project dependency (it pulls a large tree), so install it separately:

```sh
uv venv /tmp/evalplus-venv && uv pip install --python /tmp/evalplus-venv/bin/python evalplus
make patch-venv          # REQUIRED: unpatched mlx_lm.server silently ignores --adapter-path
```

Serve the adapter, then generate and grade:

```sh
# 1. serve (leave running)
.venv/bin/python -m mlx_lm.server \
  --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
  --adapter-path runs/<run>/soups/souplate-adapter --port 8080

# 2. generate 378 greedy samples through the official prompt+backend
OPENAI_API_KEY=dummy /tmp/evalplus-venv/bin/evalplus.codegen \
  default_model mbpp --backend openai \
  --base_url http://127.0.0.1:8080/v1 --greedy

# 3. grade (both columns: MBPP base tests, MBPP+ augmented tests)
EVALPLUS_MAX_MEMORY_BYTES=-1 /tmp/evalplus-venv/bin/evalplus.evaluate \
  mbpp --samples evalplus_results/mbpp/default_model_openai_temp_0.0.jsonl
```

Four traps, each of which silently produces a wrong number rather than an
error:

- **Request the model as `default_model`.** mlx-lm keys the CLI adapter under
  that name; asking for the real repo id resolves to a no-adapter entry and
  serves **base weights** — a very easy way to "validate" the wrong model.
  Confirm before trusting a run: pick a problem the adapter passes and the
  base fails, and check the served response.
- **`EVALPLUS_MAX_MEMORY_BYTES=-1` on Darwin**, or `reliability_guard` hits
  the same `setrlimit` rejection as our sandbox and every task scores 0.000.
- **Delete stale `*_eval_results.json`** before re-grading; otherwise
  `evalplus.evaluate` prompts interactively to overwrite and dies on EOF in a
  non-interactive shell.
- **`--greedy` fixes temperature 0 and n=1**; anything else is not the
  leaderboard protocol.

## Method constants and gotchas

- **Sandbox**: candidate code runs under `sandbox-exec` (network denied,
  writes confined to a per-candidate tempdir) plus rlimits and a scrubbed
  env. Memory cannot be capped — Darwin rejects `RLIMIT_DATA`/`RLIMIT_AS`
  outright — so blowups are bounded only by the 8s timeout.
- **In-loop eval reads ~1–4.5 points low** vs the official harness (souplate3
  0.265→0.310, souplate4 0.339→0.365): our extraction is last-fenced-block,
  theirs is AST-based. Treat in-loop numbers as a conservative proxy.
- **Two mlx-lm 0.31.3 bugs** are patched in `patches/` (run
  `make patch-venv` after `uv sync`): `server.py` silently drops
  `--adapter-path` (serves base weights while looking fine), and `gemma2.py`
  cannot batch (its 5-D GQA scores can't broadcast against a 4-D mask —
  works at batch 1, fails above).
- **EvalPlus on Darwin** needs `EVALPLUS_MAX_MEMORY_BYTES=-1`; its
  `reliability_guard` hits the same `setrlimit` rejection.
- **Eval completions are not logged** by `evaluate()` — inspecting what the
  model actually wrote requires reloading a checkpoint.
- **`samples.jsonl` logs one prompt's group per step**, not the whole batch;
  see the checkpoint-selection section for how to recover the true
  trained-on set by replaying the rng.
- **Sampling is with replacement by default.** 150 steps × 4 prompts touched
  581 of 7901 problems; full coverage would need ~n·ln(n) ≈ 71k draws.
  `sampling="epoch"` draws without replacement so N draws cover N distinct
  problems.
- **Piping the trainer through `tee` hides crashes** — a shell pipeline
  reports tee's exit status, so a MemoryGuardError arrives as "exit 0". Use
  `set -o pipefail` or redirect.

## Open threads

1. ~~Finish `coder15-run1`; souplate and validate officially.~~ Done:
   0.709 / 0.601 vs base 0.672 / 0.571, McNemar p = 0.044 on MBPP.
   Next: a second seed, since this is one run.
2. ~~One full epoch over the 7601-problem pool.~~ Done (`coder15-run3`,
   after run2 died to a swap-guard false positive at step 69): 0.730/0.624,
   both significant. Verdict on the data question: 13× more distinct
   problems bought ~+2 points and plateaued at step 350 — difficulty, not
   volume, is the constraint now. Next: re-calibrate the pool against the
   trained model (drop saturated problems, add `hard`), and a second seed.
3. Held-out-format control: train a model on `bare,instruct` and check
   whether fenced performance moves. Separates general robustness from
   template fitting for the 0.5B line retroactively.
4. Graded rewards (assertion-fail vs crash vs no-code) to soften the
   all-or-nothing cliff that makes marginal problems churn.
5. Report both mlx-lm bugs upstream.
