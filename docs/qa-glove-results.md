# Teaching a 35B model to say "I don't know" — results

*2026-07-31, with additions through 2026-08-17. Runs: qa-full-20260726,
qa-chatmix-full-20260730, qa-gloveA-20260731 (plus an aborted
qa-gloveA-20260730), qa-gloveB-20260730, qa-binding-A200-20260731, and arm C.
Seed 0 unless a section says otherwise.*

*Rewritten in plain English 2026-08-17; no number changed. Where the old
vocabulary survives it is a filename or a config key — see
[glossary.md](glossary.md). Artifacts: [artifacts.md](artifacts.md).*

## What we claim

A 35B model running on one Mac Studio can be trained, with reinforcement
learning on a small LoRA adapter, to act on its own uncertainty **in ordinary
conversation**: declining questions it cannot know, still answering the ones it
does know, and dropping the habit of insisting that unfamiliar things do not
exist.

Two conditions turn out to be necessary. The model must be told, in its system
prompt, that declining is acceptable — we ship that prompt with the adapter as
a pair. And the conversational half of its training must be weighted toward
questions it cannot answer, or there is nothing for the training signal to
grab.

The result transfers. Training used trivia questions only; the behaviour shows
up on arXiv papers published after the model's knowledge cutoff, where the
declining rate goes from 0.03 to 0.78–0.88 while answers about famous papers
stay correct at 0.95–1.00.

## How we got there — each failure forced the next design

1. **Tagged format, first working run.** The model answers in `<answer>` tags
   or replies `<abstain/>`. Wrong answers fell 0.30 → 0.095, precision rose
   70% → 87%. Two things were needed to make it train at all: injecting the
   correct decision as a demonstration in both directions (see below), and a
   watchdog that kills a run whose samples have gone flat.

2. **It doesn't transfer to conversation.** That same adapter declines on
   post-cutoff papers 0.96 (authors) / 0.86 (year) **when asked in the tagged
   format**, and 0.01 when asked the same thing in conversation. The
   *capability* crosses domains. The *behaviour* does not cross prompt formats.
   This is the finding the rest of the project is built on.

3. **Mixing conversation into training, with no honesty prompt: flat.**
   Sampled declines in conversation went from 4-of-356 to 3-of-335 between the
   first and second half of the run. A ~1% tendency gets essentially no
   policy-gradient mass. The mechanism was fine — the judge scored 219 of 219 injected
   declines correctly, and hand-inspected groups showed the gradient pointing
   the right way. There was simply almost nothing to reinforce.

4. **The honesty prompt alone, no training.** Declining on unknowable
   questions rose from 0.04 to 0.33–0.79. But it was uncalibrated: 0.17 decline
   rate on questions the model *does* know, and fabrication about famous papers
   was untouched. So the prompt supplies permission and raises the base rate;
   it does not supply calibration.

5. **The two-factor experiment.** Arm B (honesty prompt, conversation mix tuned
   like the tagged format) stayed flat during the run — only artificially
   unanswerable questions moved. Arm A (honesty prompt, conversation mix
   0.15/0.35/0.50 weighted toward unknowns) learned: answered rate 0.93 → 0.715
   at its best checkpoint (160), wrong answers → 0.10, denials → 0.0.

   **The prompt gives permission; the unknown-weighted conversation mix is what
   makes it learn.** Neither alone is enough.

## Main results (ordinary conversation, "declined or denied", 4 samples/question)

| question bucket | base | + prompt only | B | A-200 | A-160 | A-200, prompt removed |
|---|---|---|---|---|---|---|
| unanswerable-by-construction | 0.04 | 0.33 | 0.25 | 0.42 | 0.71 | 0.00 |
| real but obscure | 0.00 | 0.08 | 0.08 | 0.50 | 0.58 | 0.00 |
| fictional people | 0.20 | 0.56 | 0.82 | 0.97 | 0.88 | 0.11 |
| post-cutoff papers | 0.50 | 0.79 | 0.96 | 0.88 | 0.85 | 0.30 |
| questions it knows (cost) | 0.04 | 0.17 | 0.08 | 0.29 | 0.46 | 0.00 |

The last row is the price, not the prize: declining a question you could have
answered is a cost, and it rises alongside the benefit.

**Asked directly who wrote a paper**, A-200 declines 0.78 (authors) / 0.88
(year) on post-cutoff papers against a base rate of 0.03/0.03, while keeping
famous-paper answers correct at 1.00/0.95. Confidently-wrong answers fell from
0.67 to 0.17 on real-but-obscure questions and from 0.80 to 0.03 on fictional
people. Flat denial that a fictional person exists: 0.01.

**The behaviour is bound to the prompt.** Probe either A checkpoint with the
honesty prompt removed and every bucket sits at base-model levels. The adapter
and the prompt are one artifact; the adapter by itself does nothing — which
also means it cannot make the model cagey in contexts that never opted in.

**Later is not simply better.** A-160 declines more on both unknowns *and*
knowns (0.71 / 0.46); by step 200 the run has drifted back to a more
answer-happy point (0.42 / 0.29). Training moves an overall willingness-to-
answer setting; the per-question discrimination underneath it is real but
partial. Pick the checkpoint that matches what you are deploying.

## Does it track what it actually knows? (held-out, n=200, 8 samples, judge-graded)

The worry with any of this is that the model just declines more, everywhere,
and the bucket averages hide it. So: on questions never trained on and
essentially absent from the calibration file, compare how often the trained
model declines against how often the base model gets that question right.

**Correlation −0.44; declines 0.09 on questions the base model gets right at
least 80% of the time, 0.41 on ones it never gets right; AUROC 0.71.** All
three are lower
bounds — our grader misses correct answers phrased unusually, on both axes
(one "failed" item had the policy answering "Frank and Joe Hardy", which is
simply right and ungraded). It is reading per-question uncertainty, and it
cannot be memorising the confidence groups, which are excluded by
construction. The out-of-distribution calibration collapse reported by Kadavath
et al. did not appear at this (modest) distribution shift.

## How it was trained

Qwen3.6-35B-A3B at 4-bit (22.09 GB). LoRA rank 16, scale 20, last 12 layers —
4.28M trainable parameters. Each step draws 8 questions × 8 samples = 64
completions, one of which is a demonstration of the correct decision; 4 at a
time through the backward pass; 192 tokens max; temperature 1.0; thinking off.
Learning rate 3e-6, KL penalty 0.01 against the frozen base. 200 steps ≈ 12.8k
completions ≈ 3–4 hours.

Reward is +1 correct, 0 declined, −3 wrong; that ratio sets the calibration
target, since a reward-maximising model answers exactly when it believes it is
right more than 75% of the time. Conversational replies have no tags to parse,
so a larger model grades what the reply committed to — answer, decline, or
denial — with denial scored as a wrong answer. Those verdicts are cached;
23.8k of them at time of writing, 48,227 in the [released
bundle](artifacts.md).

Data is TriviaQA, 138,384 questions with 500 held out. A 2,000-question
calibration probe sorted them into 997 reliably-right / 468 sometimes / 535
reliably-wrong. The tagged half of training draws 0.65/0.25/0.10 from those
groups; arm A's conversational half draws 0.15/0.35/0.50.

**Why a demonstration is injected into every group, in both directions.** If
you only ever demonstrate declining, all-declining becomes a state the run
cannot escape: every sample in the group agrees, the variance is zero, GRPO
drops the group, and no gradient ever pulls it back. Injecting the *correct*
decision instead — the gold answer when the model reliably knows it, a decline
when it doesn't — makes both collapse directions self-correcting.

Guards: a swap-space watchdog (12 GB margin on judge-heavy runs, because the
judge subprocess ratchets stale swap by ~50 MB/step without actually
thrashing), an abort-if-flat window, and the machine-wide memory lease.

## Arm C — buying the known side back (2026-08-01)

One variable changed from A: the conversation mix goes to 0.35/0.35/0.30,
putting a third of the weight back on questions the model knows.

C-200 keeps essentially all of A's protection against fabrication — fictional
people 0.95, post-cutoff papers 0.92, post-cutoff paper years 0.88, famous
papers 1.00/0.95 — while cutting the cost on known questions from 0.29 to 0.21
and restoring correct answers on known questions to 0.46, which is the base
rate. Discrimination improves too: correlation −0.497, AUROC 0.738, declining
0.021 on knowns versus 0.347 on unknowns — a 16× separation, against A-200's
4.6×.

It is not free: real-but-obscure declining drops to 0.33 (A: 0.50) and
post-cutoff authors to 0.65 (A: 0.78).

**`qa-gloveC-200-20260731` plus the honesty prompt is the deliverable.**

**Second seed (2026-08-01): replicated.** Fabrication buckets land within
noise — fictional people 0.91, post-cutoff papers 0.90, famous 1.00/1.00 — and
the cost on known questions is lower still (0.08 declined, 0.50 correct).
Discrimination: correlation −0.467, AUROC 0.738, identical to seed 0; 0.013 on
knowns versus 0.291 on unknowns, a 22× separation. The seed-to-seed noise floor
is visible on the small buckets (n=12: 0.42 → 0.58) and on post-cutoff author
declining (0.65 → 0.46). Quote those with the spread, not as point estimates.

**A third seed, independently run (2026-08-17).** Reproduced on a revised
trainer by an outside collaborator: 0.71 (authors) / 0.94 (year) against the
original 0.78/0.88, with famous-paper answering 0.93–1.00 and confident
fabrication 1.00 → 0.16. See `docs/qa-reproduction.md`.

## Milder penalty, c=1 (2026-08-01)

Arm C's recipe with the wrong-answer penalty at 1.0 instead of 3.0, which moves
the implied calibration threshold from 75% to 50%. It did what the arithmetic
predicts: the whole operating point slides toward answering. Declining drops on
every bucket — real-but-obscure 0.33 → 0.08, post-cutoff authors 0.65 → 0.44,
post-cutoff years 0.88 → 0.68, fictional people 0.95 → 0.82 — *and* the cost on
knowns drops with it (0.17 declined, 0.00 on multi-part known questions, 0.67
correct). Famous papers 1.00/0.975. Discrimination survives: 0.014 on knowns
versus 0.182 on unknowns, 13×, AUROC 0.70.

So the penalty selects where you sit on the coverage-versus-caution curve. It
does not change how well calibrated you are along it. C-200 with the prompt
removed was confirmed inert here too, at or below base on every bucket.

## Does it cost anything elsewhere? (2026-08-01) — no

The pair was run through an agentic coding benchmark (OpenCode on SWE-bench
lv-72), same serving stack both sides, adapter-plus-prompt the only variable,
with the prompt injected at the serving proxy so training and deployment see
the same thing. **Base 45/72, C-200 45/72 — identical.** The pre-registered
bar was "no worse than 8 tasks down". Calibrated declining is free on agentic
coding. Evidence in bench-coding `results/oc-qwen36-{mlxlm,c200}-gate/`.

## How much of this is the prompt, and how much is the training?

Added 2026-08-17. This doc originally reported base → trained and skipped the
middle. The number existed — measured 2026-07-30, written down in
`uplift-over-prompt.md` — and should have been in this table from the start.
Asked directly who wrote a post-cutoff paper:

| | declines, authors | declines, year |
|---|---|---|
| base model | 0.03 | 0.03 |
| **base + honesty prompt, no training** | **0.26** | **0.46** |
| A-200 + prompt | 0.78 | 0.88 |
| C-200 + prompt | 0.65 | 0.88 |

The independent reproduction re-measured the middle row at 0.28 / 0.47 without
having seen it — within one or two replies per cell — and it now serves as one
of that reproduction's matched anchors.

The honesty prompt does real work on its own: roughly a third of the final
declining rate on authors, half on years. Training adds 0.43 or more on top of
it on every bucket, and adds the thing the prompt cannot: per-question
discrimination. The prompt alone declines 0.17 on questions the model knows,
with no correlation to what it actually knows; the trained pair declines 0.02
on those, at AUROC 0.74. Both halves are load-bearing, and the claim should
always be stated against the prompt, not against the bare base model.

## Open

- Summarising a famous paper without any disclaimer is untouched at 0.95. That
  is a job for a detector, not for reward shaping.
- Our grader's tolerance for unusual phrasing sets a noise floor under every
  number here. More seeds on the marginal buckets.
