# MBPP → EvalPlus — a leaderboard-comparable code-RL number

## Claim

**Qwen2.5-Coder-1.5B-Instruct-4bit goes 0.672 → 0.730 MBPP and 0.571 → 0.624
MBPP+ under the official EvalPlus harness** (paired McNemar p = 0.003 / 0.008),
trained by GRPO + LoRA only on prompt dialects the benchmark does not use,
with the artifact chosen by a rule fixed in advance. Scale context, so the
result is not oversold: an untrained **Qwen3.6-35B-A3B** 4-bit scores
**0.870** on the same benchmark. All numbers are seed 0, single runs.

## Training data — KodCode

Training on MBPP and evaluating on MBPP does not produce a comparable number:
the shipped sanitized-MBPP 427 contains 257 problems from the canonical MBPP
*test* split, 209 of which land in the train pool under the seeded shuffle.
The fix is a training corpus with no MBPP in it at all.

[**KodCode**](https://arxiv.org/abs/2503.02951) (Xu et al. 2025) is 447K
question/solution/test triplets where **every triplet is execution-verified**
and the corpus is **decontaminated against MBPP and HumanEval** by
construction — which is what makes the EvalPlus eval trustworthy. Training
uses the `KodCode-Light-RL-10K` split. Tests are pytest-style (`from solution
import f`), so the candidate is written to `solution.py` and graded by
pytest's exit code in the same Seatbelt sandbox as the MBPP asserts; 30
reference solutions sampled through the runner passed 30/30. The `Package` and
`Docs` subsets are excluded: they need non-stdlib imports the no-network
sandbox cannot satisfy, so every candidate fails for environment reasons.
⚠️ **Licence: CC BY-NC 4.0 — non-commercial.** (MBPP itself is CC BY 4.0; the
mlx-rl code is MIT.)

**Difficulty calibration matters more than pool size.** GRPO gets gradient
only from groups with reward variance. `scripts/kodcode_calibrate.py` probes
per-problem pass@8 at temperature 1.0 through the real sandboxed reward: on
Qwen2.5-0.5B-Instruct-4bit, 300 KodCode-easy problems gave 0% saturated (8/8),
**39% learnable (1–7 of 8)**, 61% unsolved; `Filter`/`Prefill` were 53%
learnable against 85–100% unsolved for
`Algorithm`/`Data_Structure`/`Leetcode`/`Code_Contests`. Bands are per-model —
a pool tuned for a 0.5B is partly saturated for a model 2.4× stronger.

## Prompt format is a measured variable

The two prompts differ by one code fence: **bare** (this repo) is instruction
+ blank line + `"""docstring + first assert"""`; **fenced** (EvalPlus
official) wraps the same docstring in ```` ```python ```` … ```` ``` ````.

Cross-format matrix on Qwen2.5-0.5B-Instruct-4bit (in-repo harness, 378 tasks,
greedy), and the same two prompts across model families (EvalPlus-378, greedy
pass@1, identical harness):

| adapter | bare | fenced |
|---|---|---|
| base | 0.283 | **0.000** |
| souplate1 (trained bare) | 0.392 | 0.032 |
| souplate3 (trained fenced) | 0.283 | **0.265** |

| model | fenced (official) | bare | gap |
|---|---|---|---|
| Qwen3.6-35B-A3B-4bit (thinking off) | **0.870** | not run | — |
| Qwen2.5-Coder-1.5B-Instruct-4bit | 0.667 | 0.667 | **0.000** |
| Qwen2.5-1.5B-Instruct-4bit | 0.563 | 0.574 | −0.011 |
| Llama-3.2-3B-Instruct-4bit | 0.537 | 0.566 | −0.029 |
| Qwen2.5-3B-Instruct-4bit | 0.333 | 0.640 | **−0.307** |
| Qwen2.5-0.5B-Instruct-4bit | 0.000 | 0.283 | **−0.283** |

Training on the fenced format bought the 0.5B 0.000 → 0.265 on that format
while leaving bare performance exactly at base: format repair, not capability.
And brittleness is not a scale effect — Qwen2.5-3B is destroyed (−0.307) where
Llama-3.2-3B is fine (−0.029); three of five models across three families are
neutral within decode noise. This is [Sclar et
al. 2024](https://arxiv.org/abs/2310.11324) in its sharpest form, and their
recommendation to report a *range* across formats is why the second table has
two columns. The subject model below was picked for its zero format gap, and
the EvalPlus template is excluded from its training set.

## The clean run (`coder15-run1`)

150 steps, 4 prompts × group 8, 640-token budget, kl 0.02, lr 5e-6, LoRA r16
all layers, KodCode easy+medium minus `Package`/`Docs`, full-378 eval every 25
steps, seed 0.

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

`--eval-n 378` plus the task's exact-coverage cycling makes every eval the
*whole* benchmark, not a resample. Step 0 reproduced 0.6667 exactly.

| step | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|---|
| eval | 0.6667 | 0.6561 | 0.6799 | 0.6905 | 0.6931 | 0.6905 | **0.7090** |
| (n) | 252 | 248 | 257 | 261 | 262 | 261 | **268** |

| official EvalPlus | base Coder-1.5B | souplate5 | Δ |
|---|---|---|---|
| MBPP | 0.672 (254/378) | **0.709 (268/378)** | +3.7 pts |
| MBPP+ | 0.571 (216/378) | **0.601 (227/378)** | +3.0 pts |

Both sides went through the same harness on the same tasks, so the paired test
is the right one. On MBPP: 240 problems both solve, **28 only souplate5
solves, 14 only base solves** — McNemar exact **p = 0.044**. MBPP+ moves the
same way (26 vs 15 discordant, +11) but at p = 0.12 is not significant alone.
Caveats worth keeping attached: one run at one seed, and 14 problems
regressed. Flips are low and decaying (6–26 problems per 20-step window) and
asymmetric, +69/−53 overall; the union across the 8 checkpoints is 284/378
against a best single of 268, with 224 problems passing at every checkpoint.

## Full epoch (`coder15-run3`)

Same recipe, three changes: `sampling="epoch"` (without replacement — run1's
600 draws touched only 581 of 7901 problems), a 300-problem validation holdout
reserved before training (`val_frac: 300`), and `--batch-prompts 8`. 951 steps
= one pass over all 7601 training problems, 16.1 h wall.

| step | 0 | 25 | 100 | 300 | 350 | 600 | 800 | 950 |
|---|---|---|---|---|---|---|---|---|
| eval | 0.667 | 0.704 | 0.709 | 0.722 | 0.735 | **0.738** | 0.728 | 0.720 |

| official EvalPlus | base | run1 souplate5 | **run3 souplate6** |
|---|---|---|---|
| MBPP | 0.672 (254) | 0.709 (268) | **0.730 (276)** |
| MBPP+ | 0.571 (216) | 0.601 (227) | **0.624 (236)** |

Paired against base: MBPP +22 (37 gained / 15 lost, McNemar **p = 0.0032**);
MBPP+ +20 (36 / 16, **p = 0.0078**) — both columns individually significant.
Everything from step 350 on sits in a 6-problem band, and active groups decay
4.45 → 2.88 of 8 as the policy saturates the pool: the binding constraint is
problem difficulty, not data volume. The oracle checkpoint (step 600) would
score ~0.738 in-loop, so the pre-specified selection rule cost ~1 point.

**Validation-based selection failed on this run.**

| step | 100 | 300 | 500 | 600 | 800 | 951 |
|---|---|---|---|---|---|---|
| val (KodCode, n=300) | 0.630 | 0.673 | 0.660 | 0.687 | 0.693 | **0.707** |
| test (EvalPlus-378) | 0.709 | 0.722 | 0.733 | **0.738** | 0.728 | 0.717 |

Validation rises monotonically while test plateaus and drifts down (r = 0.387
over 10 checkpoints, below the 0.632 significance threshold, and picking the
test-worst late checkpoint). The holdout is drawn from the *training*
distribution, so it keeps improving as the model fits KodCode harder, exactly
while transfer stops improving. On `coder15-run1` the same protocol was
moderately predictive (r = 0.643 over 8 checkpoints, critical value 0.707) and
selected step 140 → test 0.698 against an oracle 0.709.

## Choosing the artifact

Every number above uses a **pre-specified rule** — last three checkpoints,
souped — rather than the best validation score, because the only eval
available *was* the test benchmark and choosing a checkpoint by its EvalPlus
score then reporting that score is selection-on-test. MBPP problems outside
the EvalPlus 378 exist (49 in the sanitized set, 596 in full MBPP), but
selecting with MBPP-distributed data in a transfer setup is target-domain
(oracle) model selection: it would weaken the claim from "KodCode training
transfers to MBPP" to "the best-on-MBPP checkpoint, chosen using MBPP". The
conservative choice is a held-out slice of the *training* distribution, which
is what `val_frac` carves out of the KodCode pool before training
(`val_frac >= 1` is an absolute count). On run1 all three honest procedures
land within ~1 point — source-val 0.698, pre-specified soup 0.709 official,
last-checkpoint 0.709 — against a +3.7-point gain.

A **souplate** is the uniform per-tensor average of several LoRA checkpoints
from one run's trajectory — a [model soup](https://arxiv.org/abs/2203.05482)
over training time rather than over hyperparameters (*soup* + *late*).
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

Soup is not a free win. It needs diverse ingredients: souplate1 (0.5B) beat
its best single checkpoint by 8 problems and solved 4 nothing else solved,
while souplate5, from a low-churn run, scored *below* its best ingredient (265
vs 268) and found zero new problems. Composition barely matters — ablating all
eight cumulative windows on the 0.5B mixed run spanned 128–139 problems
against a standard error of 9.3. And searching those windows for the maximum
is eval-set selection, the same error as fitting the prompt.

> **Recovering what a run actually trained on.** `samples.jsonl` logs only the
> FIRST prompt's group per step, so it holds 150 records for a 150-step ×
> 4-prompt run. Using it as the trained-on set undercounts 4×: it suggested
> 149 distinct problems where the truth was **581**. Replay the training
> stream instead — `random.Random(cfg.seed)` is consumed only by
> `task.sample()` when `sage_r`/`inject_r`/`group_stage1` are 0 — and verify
> against the log. Getting this wrong leaks trained-on problems into the
> validation set.

## Reproducing an official EvalPlus number

The in-loop eval is a proxy; the leaderboard-comparable number comes from
EvalPlus's own harness driven against a served adapter. EvalPlus is **not** a
project dependency (it pulls a large tree), so install it separately:

```sh
uv venv /tmp/evalplus-venv && uv pip install --python /tmp/evalplus-venv/bin/python evalplus
make patch-venv          # REQUIRED: unpatched mlx_lm.server silently ignores --adapter-path
```

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

## Gotchas — each one silently produces a wrong number

- **Request the model as `default_model`.** mlx-lm keys the CLI adapter under
  that name; asking for the real repo id resolves to a no-adapter entry and
  serves **base weights**. Confirm by picking a problem the adapter passes and
  the base fails, then checking the served response.
- **`EVALPLUS_MAX_MEMORY_BYTES=-1` on Darwin**, or `reliability_guard` hits
  the same `setrlimit` rejection as the sandbox and every task scores 0.000.
- **Delete stale `*_eval_results.json`** before re-grading, or
  `evalplus.evaluate` prompts to overwrite and dies on EOF in a
  non-interactive shell.
- **`--greedy` fixes temperature 0 and n=1**; anything else is not the
  leaderboard protocol.
- ⚠️ **Sandbox**: candidate code runs under `sandbox-exec` (network denied,
  writes confined to a per-candidate tempdir) plus rlimits and a scrubbed env.
  Memory cannot be capped — Darwin rejects `RLIMIT_DATA`/`RLIMIT_AS` — so
  blowups are bounded only by the 8 s timeout.
- **In-loop eval reads ~1–4.5 points low** vs the official harness (souplate3
  0.265 → 0.310, souplate4 0.339 → 0.365): extraction here is
  last-fenced-block, theirs is AST-based.
- **Three mlx-lm 0.31.3 bugs** are patched in `patches/` (run
  `make patch-venv` after `uv sync`): `server.py` silently drops
  `--adapter-path`, `gemma2.py` cannot batch, and batched generation crashes
  when only some requests carry logits processors.
- **Eval completions are not logged** by `evaluate()`; inspecting what the
  model wrote requires reloading a checkpoint.
- **Sampling is with replacement by default.** 150 steps × 4 prompts touched
  581 of 7901 problems; `sampling="epoch"` draws without replacement so N
  draws cover N distinct problems.
- **Piping the trainer through `tee` hides crashes** — the pipeline reports
  tee's exit status, so a MemoryGuardError arrives as "exit 0". Use
  `set -o pipefail` or redirect.
