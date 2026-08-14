# MBPP → EvalPlus — what a comparable code-RL number costs

*2026-08-12/13. Runs: code-conv (50-step, aborted line), code-conv2
(collapsed), code-conv3 (flagship of the internal-split era), kodcode-run1,
kodcode-run3, kodcode-run4, coder15-run1 (in flight). All seed 0, single
runs, Qwen2.5-0.5B-Instruct-4bit unless stated.*

## Claim

GRPO + LoRA moves MBPP pass@1 on a 0.5B local model, and the movement is
real — but almost none of it was **comparable to any published number**, and
most of what finally *was* comparable turned out to be repair of a prompt
pathology rather than coding capability. Getting to an honest claim required
three corrections in sequence: fix the eval set (contamination), fix the
harness (EvalPlus), and then fix the *subject model* (prompt sensitivity).
The current run is the first configured so that a gain, if it appears, means
what it says.

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
   wraps the problem in a ```` ```python ```` fence, and the 0.5B echoes the
   prompt back instead of answering (98% of responses contained no code).
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

## Era 1 — the internal-split numbers (main branch, `code` task)

Sanitized MBPP 427, seeded 80/347 split, reward = all hidden asserts pass.

| run | config | baseline | result |
|---|---|---|---|
| code-conv | 50 steps, kl 0.01, eval n=32 | 0.281 | peak 0.344, **final 0.219** — no significant gain; policy collapsed to ~50-token one-liners |
| code-conv2 | 150 steps, **kl 0.05** | 0.238 | **destroyed at step 16** (gen_nll 0.77 → 3.0, number-babble); unrecoverable because zero-variance batches skipped the update entirely |
| code-conv3 | 150 steps, kl 0.02, lr 5e-6 | 0.237 | **0.413 plateau** (best 0.463, final 0.412) |

Two durable fixes came out of this era. `code-conv2`'s death produced the
**KL-rescue update** (a zero-advantage batch now still applies the KL term,
so a degenerated policy is pulled back toward base instead of freezing); it
fired 13× in `code-conv3` with no blowups. And the flip analysis on
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
| Qwen2.5-Coder-1.5B-Instruct-4bit | **0.667** | 0.667 | **0.000** |
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

## Era 4 — the honest attempt (`coder15-run1`, in flight)

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

### Progress (step 50/150, in flight)

| step | eval (fenced 378) | mean len |
|---|---|---|
| 0 | 0.6667 (252) | 69 |
| 25 | 0.6561 (248) | 60 |
| 50 | **0.6799 (257)** | 47 |

First point above baseline, but +1.3 points against a ±2.4-point standard
error is not yet a result. The dip-then-recover shape matches `kodcode-run4`,
which was also flat-to-down through step 75 before climbing.

The train side is clearer than the eval: on-policy pass rate rises
0.402 → 0.455 → 0.512 across thirds, with 2.5/4 groups still active — the
widened easy+medium pool has not saturated, and the policy is learning its
training distribution. KL sits at 0.046, an order of magnitude below the
0.5B runs.

**This is the run's actual question**: those train-side gains are on `bare`
and `instruct` prompts, and the eval is `fenced`. Improvement shows up only
if it generalises across prompt dialect rather than sharpening one phrasing.
That is a strictly harder target than every earlier run — no format
pathology to repair, no easy points, and an out-of-distribution measurement
by construction — and it is the price of a capability claim that needs no
asterisk.

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
| souplate4 | run4 (mixed) | 80–150 | 0.339 *fenced* | **0.365 / 0.299** |

Three findings, all of which argue *against* treating soup as a free win:

1. **Soup needs diverse ingredients.** souplate1 beat its best single
   checkpoint by 8 problems. souplate2 — same recipe but from the
   lr-annealed run, whose late checkpoints are near-duplicates — scored
   *lower* (0.376 vs 0.392) despite the annealed run being more stable. The
   churn the anneal removed was also the diversity the soup fed on.
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

## Open threads

1. Finish `coder15-run1` (step 50/150 at time of writing, 0.680 vs 0.667
   baseline); souplate the last three checkpoints and validate through the
   official harness.
2. Held-out-format control: train a model on `bare,instruct` and check
   whether fenced performance moves. Separates general robustness from
   template fitting for the 0.5B line retroactively.
3. Graded rewards (assertion-fail vs crash vs no-code) to soften the
   all-or-nothing cliff that makes marginal problems churn.
4. Report both mlx-lm bugs upstream.
