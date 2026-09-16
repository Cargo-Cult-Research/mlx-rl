# Teaching a 35B model to say "I don't know" — results

## What this shows

**Qwen3.6-35B-A3B** at 4-bit, on a single 96 GB Apple Silicon machine, can be
trained with reinforcement learning on a small LoRA adapter to act on its own
uncertainty **in ordinary conversation**: declining questions it cannot know,
still answering the ones it does know, and dropping the habit of insisting
that unfamiliar things do not exist.

Two conditions are necessary. The model must be told, in its system prompt,
that declining is acceptable — that prompt ships with the adapter as a pair.
And the conversational half of its training must be weighted toward questions
it cannot answer, or there is nothing for the training signal to grab.

The result transfers. Training used trivia questions only; the behaviour shows
up on arXiv papers published after the model's knowledge cutoff, where the
declining rate goes from 0.03 to 0.78–0.88 while answers about famous papers
stay correct at 0.95–1.00.

Seed 0 unless a section says otherwise.

## Main results (ordinary conversation, "declined or denied", 4 samples/question)

| question bucket | base | + prompt only | B | A-200 | A-160 | A-200, prompt removed |
|---|---|---|---|---|---|---|
| unanswerable-by-construction | 0.04 | 0.33 | 0.25 | 0.42 | 0.71 | 0.00 |
| real but obscure | 0.00 | 0.08 | 0.08 | 0.50 | 0.58 | 0.00 |
| fictional people | 0.20 | 0.56 | 0.82 | 0.97 | 0.88 | 0.11 |
| post-cutoff papers | 0.50 | 0.79 | 0.96 | 0.88 | 0.85 | 0.30 |
| questions it knows (cost) | 0.04 | 0.17 | 0.08 | 0.29 | 0.46 | 0.00 |

Arm A trains a conversational mix weighted 0.15/0.35/0.50 toward questions the
model does not know; arm B uses the tagged format's mix and stays flat. The
last row is the price, not the prize: declining a question you could have
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

On questions never trained on and essentially absent from the calibration
file, compare how often the trained model declines against how often the base
model gets that question right.

**Correlation −0.44; declines 0.09 on questions the base model gets right at
least 80% of the time, 0.41 on ones it never gets right; AUROC 0.71.** All
three are lower bounds — the grader misses correct answers phrased unusually,
on both axes (one "failed" item had the policy answering "Frank and Joe
Hardy", which is simply right and ungraded). It is reading per-question
uncertainty, and it cannot be memorising the confidence groups, which are
excluded by construction. The out-of-distribution calibration collapse
reported by Kadavath et al. did not appear at this (modest) distribution
shift.

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
23.8k of them at the time of these runs.

Data is TriviaQA, 138,384 questions with 500 held out. A 2,000-question
calibration probe sorted them into 997 reliably-right / 468 sometimes / 535
reliably-wrong, and ships as `data/labels/trivia-pass@8-qwen36.jsonl`. The
tagged half of training draws 0.65/0.25/0.10 from those groups; arm A's
conversational half draws 0.15/0.35/0.50.

**Why a demonstration is injected into every group, in both directions.** If
you only ever demonstrate declining, all-declining becomes a state the run
cannot escape: every sample in the group agrees, the variance is zero, GRPO
drops the group, and no gradient ever pulls it back. Injecting the *correct*
decision instead — the gold answer when the model reliably knows it, a decline
when it doesn't — makes both collapse directions self-correcting.

Guards: a swap-space watchdog (12 GB margin on judge-heavy runs, because the
judge subprocess ratchets stale swap by ~50 MB/step without actually
thrashing) and an abort-if-flat window.

## Arm C — buying the known side back

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

**C-200 plus the honesty prompt is the deliverable of this arm.**

**Second seed: replicated.** Fabrication buckets land within noise —
fictional people 0.91, post-cutoff papers 0.90, famous 1.00/1.00 — and the
cost on known questions is lower still (0.08 declined, 0.50 correct).
Discrimination: correlation −0.467, AUROC 0.738, identical to seed 0; 0.013 on
knowns versus 0.291 on unknowns, a 22× separation. The seed-to-seed noise
floor is visible on the small buckets (n=12: 0.42 → 0.58) and on post-cutoff
author declining (0.65 → 0.46). Quote those with the spread, not as point
estimates.

**Third seed, reproduced independently on a revised trainer:** 0.71 (authors)
/ 0.94 (year) against the original 0.78/0.88, with famous-paper answering
0.93–1.00 and confident fabrication 1.00 → 0.16.

## Milder penalty, c=1

Arm C's recipe with the wrong-answer penalty at 1.0 instead of 3.0, which
moves the implied calibration threshold from 75% to 50%. It did what the
arithmetic predicts: the whole operating point slides toward answering.
Declining drops on every bucket — real-but-obscure 0.33 → 0.08, post-cutoff
authors 0.65 → 0.44, post-cutoff years 0.88 → 0.68, fictional people 0.95 →
0.82 — *and* the cost on knowns drops with it (0.17 declined, 0.00 on
multi-part known questions, 0.67 correct). Famous papers 1.00/0.975.
Discrimination survives: 0.014 on knowns versus 0.182 on unknowns, 13×,
AUROC 0.70.

So the penalty selects where you sit on the coverage-versus-caution curve. It
does not change how well calibrated you are along it. C-200 with the prompt
removed was confirmed inert here too, at or below base on every bucket.

## Does it cost anything elsewhere? — no

The pair was run through an agentic coding benchmark (OpenCode on SWE-bench
lv-72), same serving stack both sides, adapter-plus-prompt the only variable,
with the prompt injected at the serving proxy so training and deployment see
the same thing. **Base 45/72, C-200 45/72 — identical.** The pre-registered
bar was "no worse than 8 tasks down". Calibrated declining is free on agentic
coding.

## How much is the prompt, and how much is the training?

Asked directly who wrote a post-cutoff paper:

| | declines, authors | declines, year |
|---|---|---|
| base model | 0.03 | 0.03 |
| **base + honesty prompt, no training** | **0.26** | **0.46** |
| A-200 + prompt | 0.78 | 0.88 |
| C-200 + prompt | 0.65 | 0.88 |

An independent reproduction re-measured the middle row at 0.28 / 0.47 —
within one or two replies per cell.

The honesty prompt does real work on its own: roughly a third of the final
declining rate on authors, half on years. Training adds 0.43 or more on top of
it on every bucket, and adds the thing the prompt cannot: per-question
discrimination. The prompt alone declines 0.17 on questions the model knows,
with no correlation to what it actually knows; the trained pair declines 0.02
on those, at AUROC 0.74. Both halves are load-bearing, and the claim is
stated against the prompt, not against the bare base model. Per-bucket
prompt-only versus prompt-plus-adapter rows:
[uplift-over-prompt.md](uplift-over-prompt.md).

## What tool rounds and multi-turn add

The same question with a real check in the loop: the policy is offered
`web_search` and `fetch_url`, the system prompt also states today's date, and
tool rounds happen *inside* the rollout, so the tokens a tool call and its
result contribute are part of the episode the advantage is computed over. A
multi-turn variant gives every group member its own 3-turn transcript and
grades each turn.

Method constants for every row below: 64 held-out questions, real web tools,
judge-graded episode reward (+1 correct, 0 decline, −3 wrong answer or flat
denial; declining after a search that came back empty is +1), and the
served-agent behaviour at the tool-call limit — inject "tool call limit
reached, answer with what you have" and grade the reply. Four adapters: two
single-turn (one trained against a frozen snapshot index, one against live
search), one multi-turn, one trained in the thinking register.

| serving cell | prompt-only | snapshot-trained | live-trained | multi-turn | thinking-arm |
|---|---|---|---|---|---|
| thinking off · tools · 1 turn | 0.84 (fict 0.60) | 0.96 (fict 1.00) | 0.86 (fict 0.20) | **0.97** (fict 1.00) | 0.97 (fict 1.00) |
| thinking on · tools · 1 turn | 0.54 (fict 0.20) | 0.92 | 0.67 | 0.59 (fict −0.20) | **0.99** |
| thinking on · tools · 3 turns | 0.57 | 0.66 | 0.61 | 0.78 | **0.79** |

Training adds **+0.12** in the thinking-off single-turn cell, nearly all of it
on the declining side (fictional titles 1.00 against the base's 0.60), and
adds most where serving is hardest: **+0.45** with thinking on at one turn
(0.54 → 0.99), **+0.22** with thinking on at three turns (0.57 → 0.79).
Calibration trained in the thinking-off register does not carry into the
thinking register, which is why the thinking arm exists.

**Out-of-family** (n=236 episodes per arm, two independent judges, agreement
0.82–1.00):

| family (n) | prompt-only | snapshot-trained | multi-turn | thinking-arm |
|---|---|---|---|---|
| all | −0.93 | 0.29 | **0.45** | 0.33 |
| PopQA (96) | −1.65 | −0.45 | −0.21 | −0.40 |
| invented people (40) | −0.07 | 0.70 | 0.80 | 0.50 |
| invented events (40) | −1.23 | 0.88 | 1.00 | 1.00 |
| papers, post-cutoff (40) | −0.62 | 0.88 | 0.93 | 0.95 |
| papers, famous (20) | 0.80 | 0.71 | 0.90 | 0.90 |

Checking before answering generalises to distributions the training never saw
in any form. The correctness ceiling does not move with it: PopQA stays
negative because the adapters call the tool on every item (1.00) and still
answer wrong 0.24 of the time, at −3 each.

**Asserted-future-year falsification** (n=64 per arm per condition). Told a
paper is from a year after the stated today, the prompt-only base denies it
exists one time in five (reward −0.11, calls the tool 0.64); every adapter
searches every time (1.00), never denies (0.00), and scores 0.80–0.88. With
an asserted year at or before today all arms score 0.92–0.94.

**These adapters are not register-bound**, unlike C-200. With the honesty
sentence removed and only the date line left (n=128 per arm), single-turn
reward is 0.95 / 0.98 / 0.98 against 0.96 / 0.97 / 0.97 with it, and the tool
is still called 1.00. There is no prompt-level opt-out for them.

**Capability gate:** SWE-bench lv-72 through the OpenCode harness, paired in
the adapter's own register (thinking off), **base 52/72 against multi-turn
adapter 51/72** — one instance apart.

**Against the trivia-only adapter**, same yardstick, real tools, tool-limit
message:

| test | C-200 (trivia-trained) | multi-turn tool adapter |
|---|---|---|
| single question | 0.73 (invented titles −0.11, calls the tool 0.70) | **0.88** (invented 1.00, calls 1.00) |
| three turns | 0.18 (0.66 → 0.09 → −0.22) | **0.75** (0.91 → 0.74 → 0.59) |
| out-of-family (PopQA, invented, papers probe) | −0.73 | **0.33** |

## Reproducing

Both calibration files ship, so the probes below run without a re-measurement.
Chat-format buckets (base, prompt-only, and adapter arms of the main table):

```sh
# prompt-only control
uv run python scripts/qa_chat_probe.py --calib data/labels/trivia-pass@8-qwen36.jsonl \
    --system honesty --k 4 --out runs/qa-chat-prompt-only

# the artifact: same prompt, adapter loaded
uv run python scripts/qa_chat_probe.py --calib data/labels/trivia-pass@8-qwen36.jsonl \
    --system honesty --adapter <adapter-dir> --k 4 --out runs/qa-chat-c200
```

Training the honesty task on one domain and scoring the held-out one every
eval:

```sh
.venv/bin/mlx-rl-train --profile qwen36 --task honesty \
  --task-kwargs '{"domain": "trivia", "judge_backend": "local"}' \
  --eval-cells papers --eval-every 20 --eval-n 32 \
  --steps 200 --batch-prompts 8 --group-size 8 --inject-r 1 \
  --max-tool-rounds 2 --lora-layers 12 --lr 3e-6 --kl-coef 0.01 \
  --out runs/honesty-trivia
```

Scoring adapters against the base across both domains — the table shape used
for the tool rows above:

```sh
uv run python scripts/matrix_eval.py --cells papers,trivia \
    --arm base= --arm tools=<adapter-dir> --n 32 --k 2
```

The three-turn and serving-grid harnesses that produced the tool rows are not
shipped in this repo; `matrix_eval.py` reproduces the single-turn cells.
