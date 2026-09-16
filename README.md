# mlx-rl

GRPO-style RL fine-tuning of LoRA adapters on Apple Silicon (MLX), tuned for a
single large-memory Mac (developed on a 96 GB machine).

**Why this exists:** RL with verifiable rewards is rollout-dominated — an
episode carries ~1 bit of information (the reward), so adapter updates are
cheap and small (low-rank LoRA suffices) while almost all compute goes into
sampling. That is exactly the shape of workload an inference-strong /
training-weak machine is good at.

## Design

- **Algorithm:** GRPO — sample a group of G completions per prompt, use the
  group mean as the baseline (no value network), clipped PPO-style surrogate,
  k3 KL penalty to the frozen base model.
- **Reference policy for free:** LoRA adapters have a multiplicative `scale`;
  zeroing it turns the model back into the frozen base. Reference logprobs
  cost one forward pass, not a second copy of the weights.
- **In-process rollouts** — no server, one copy of the weights in memory.
  `engine.py` batches them: the group prompt is prefilled once and each
  member gets a copy-on-write KV-cache clone (5.6× wall-clock, table below).
- **Verifiable rewards only:** a `Task` supplies prompts and a programmatic
  reward (`src/mlx_rl/tasks/`). No reward models.
- **SAGE-RL hybrid rollouts** (arXiv 2602.08354): optionally generate r of
  the G group members with SAGE confidence-guided decoding — see the
  dedicated section below.
- **Optional small-batch kernel:** a group of G=8–16 rollouts decodes at
  8–16 rows, where MLX's 4-bit matmul re-streams weights or pads to a 32-row
  tile. Point `MLX_RL_QMM_SMALL` at a module supplying 8/16-row Metal tiles
  and the engine applies them around generator steps only — the kernel has no
  gradient, so the update pass never sees it. Set the variable empty to
  disable.
- **Optional memory lease:** if something else memory-hungry shares the box
  (a local inference server, say), point `MLX_RL_MEMLEASE_CMD` at an external
  coordinator command; the trainer calls it to make room before loading and
  hands the lease back on release, including on crash if the command is
  PID-aware (`machine.py` has the CLI). Unset, runs proceed unmanaged. Either
  way the in-process guard (`memory.py`) is the backstop: it refuses runs that
  do not fit, and a swap watchdog aborts a run that starts paging to disk.

### Invariants

1. **Zero-variance groups are dropped, and signal-free steps are skipped.**
   With every advantage zero, the residual gradient is fp16 kernel noise, and
   Adam rescales any nonzero gradient to a full-size step.
2. **`old_lp` is recomputed teacher-forced at update time**, not carried over
   from generation. The KV-cached incremental forward drifts from the padded
   batch forward by up to 0.125 nats/token at fp16, which would inject
   spurious importance ratios.
3. **Rollout temperature is 1.0 for training**, so the recorded logprobs are
   the model's own distribution. SAGE members are the deliberate exception:
   off-policy demonstrations that enter the surrogate at ratio 1, which
   invariant 2 is what makes legal.
4. **LoRA depth is the update-memory knob.** Backward retains activations from
   the deepest adapted layer down to the loss. On Qwen3.6-35B-A3B at 576-token
   sequences, adapters on all 40 layers peak at 85.5 GB and the last 12 layers
   peak at 49.4 GB. Use `--lora-layers 12` on big models.

## Install

Requires macOS on Apple Silicon and Python ≥ 3.12. With [uv](https://docs.astral.sh/uv/):

```sh
uv sync                            # creates .venv and installs everything
```

or with pip:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The default model (`tiny` profile, Qwen2.5-0.5B-Instruct) downloads from the
Hugging Face Hub on first use. The big-model profiles (`qwen36` =
Qwen3.6-35B-A3B, `qwen38` = Qwen3.8-27B) point at **local** MLX 4-bit model
directories under `MLX_RL_MODELS_DIR` (default `~/models/mlx`) — convert or
download those yourself first, e.g. with `mlx_lm.convert`. See
`src/mlx_rl/profiles.py`.

## Quickstart

```sh
uv run pytest                      # fast, no model needed

# Three steps on the default 0.5B model (downloads ~300 MB on first use):
uv run mlx-rl-train --steps 3 --batch-prompts 2 --group-size 4 --out runs/smoke

# Multi-hour run, in its own session so it outlives the terminal:
uv run python scripts/launch_detached.py --log runs/myrun.log -- \
  .venv/bin/mlx-rl-train --steps 200 --out runs/myrun
```

Each run directory gets `config.json`, `metrics.jsonl` (per-step reward / KL /
lengths / throughput / peak memory, plus periodic held-out greedy evals),
`samples.jsonl` (raw completion groups), and `adapters/` checkpoints.

## SAGE-RL (arXiv 2602.08354)

**Premise:** reasoning models already "know" when to stop thinking — rank
candidate reasoning chains by Φ = length-normalized cumulative logprob (mean
per-token logprob of the whole chain) and the end-of-thinking token scores
near the top long before greedy decoding would pick it. **SAGE** is a
step-wise beam over reasoning steps (`\n\n`-delimited): keep the top-m
candidate chains by Φ; each iteration expands every candidate with 2m sampled
steps and accepts a step ending in `</think>` when it ranks **within the top-h
by Φ**, where `h = round(TR · 2m)` (`--sage-tr`, the paper's tolerance ratio).
**SAGE-RL** injects r SAGE-decoded members into each GRPO group of G; the
verifier rewards their short-correct chains and the group-relative advantage
teaches the policy to make that its default. At inference you run plain
sampling, no beam.

Implementation notes (`engine.py::sage_completion`):

- Beams fork the KV cache with the same copy-on-write clone as group prompt
  sharing. The default batched variant runs all 2m² step expansions as rows
  of one batched forward per token — ~4× over the per-beam reference (kept as
  `batched=False`); sparse-MoE expert routing caps the win below a dense
  model's, see [docs/memory-and-compute-anatomy.md](docs/memory-and-compute-anatomy.md).
- Steps are sampled at `--sage-think-temp` (default 1.0, the paper's setting),
  so the r SAGE members of a group are not byte-identical; Φ ranking always
  uses the true (untempered) logprobs.
- Reasoning is hard-bounded at `max_new_tokens − --sage-answer-reserve`; if
  the gate has not fired by then (or by `--sage-max-steps`), the end-of-think
  token is force-committed, so the answer phase always has room and the
  completion never exceeds `--max-new-tokens`.
- The end-of-think token comes from the model profile (`think_end`):
  Qwen3.6-35B-A3B `</think>` = 248069, with thinking mode switched on
  automatically when `--sage-r > 0`. Gating on a turn-terminal token instead
  (`--think-end`) is supported; the completion then ends at the commit.

```sh
# Qwen3.6-35B-A3B thinking mode, 2 of 8 group members SAGE-decoded
.venv/bin/mlx-rl-train --profile qwen36 --steps 16 --batch-prompts 3 \
  --group-size 8 --sage-r 2 --sage-m 2 --sage-tr 0.5 \
  --micro-batch 1 --max-new-tokens 1024 --sage-answer-reserve 256 \
  --lora-layers 12 --no-normalize-std \
  --task-kwargs '{"n_operands": 7, "max_operand": 9999, "format_reward": 0.0}' \
  --eval-every 4 --eval-n 16 --out runs/sage-qwen36
```

Hybrid-run metrics add `mean_len`, `mean_len_sage` / `mean_len_sampled`,
`mean_think_len`, `reward_sage` / `reward_sampled`, and evals add
`eval_mean_len` — the deployment metric (greedy, no beam).

### Grading thinking-mode completions

A reward that greps for answer tags anywhere is collectable from inside the
think block, so:

- Thinking-mode completions are graded **only on the visible reply after the
  final think-close marker** (`_visible_reply` in `train.py`); an unclosed
  think block is reward 0.
- Runtime tripwires print a loud `[BUG]` line on positive reward with an
  unclosed think, and on SAGE budget breaches.
- `--length-penalty` is correctness-gated and counts **total** tokens, so
  moving reasoning out of `<think>` into visible prose buys nothing.
- `samples.jsonl` gets the first prompt's whole completion group every step.
  Read them: reward hacks show up there, not in the curves.

## Does it learn?

One run answers that, on a model that downloads itself, so the claim is
checkable without any local weights.

**The task.** `arithmetic` asks for a two-operand sum or difference:
*"Compute 47 + 82. Think step by step, then end your reply with the final
integer wrapped in answer tags, like `<answer>7</answer>`."* Reward is 1.0 for
the correct integer in the last answer tag, 0.2 for well-formed tags around a
wrong value, 0 otherwise. Operands up to 99 leave Qwen2.5-0.5B-Instruct
mediocre at the start, which is the point: a group whose members disagree is a
group with gradient in it.

```sh
uv run mlx-rl-train --steps 50 --batch-prompts 4 --group-size 8 \
  --max-new-tokens 80 --eval-every 10 --eval-n 32 \
  --task-kwargs '{"n_operands": 2, "max_operand": 99}' --out runs/convergence
```

Held-out greedy accuracy, `eval_correct` in `runs/convergence/metrics.jsonl`,
32 problems the policy never trains on, 4-bit Qwen2.5-0.5B-Instruct, 1.8 GB
peak:

| step | 0 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| held-out accuracy | 0.56 | 0.88 | 0.91 | 0.84 | 0.81 | 0.78 |

It learns, it peaks at step 20, and it gives some of that back. Both halves
matter. Step 10 clear of step 0 is the smoke test: if that gap is not there,
something is wrong before the task is. And the last checkpoint is not the best
one, which is what `promote_adapter.py --step` is for — it takes the newest
checkpoint unless you name the one the curve prefers.

**At scale**, on models that need local MLX 4-bit weights: Qwen3.6-35B-A3B
with thinking off goes from 0.00 to 1.00 held-out on 7-operand arithmetic
within 10 steps at a 384-token budget; a 180-step mixture run over math, code
and arithmetic at a 2560-token cap moves `eval_correct` 0.56 → 0.66, with the
plateau from about step 40.

**Batched engine** (`engine.py`), 4 prompts × group 8 versus one-at-a-time,
rollout only, Qwen3.6-35B-A3B: 249 tok/s against 44 tok/s, a 5.6× speedup, at
24.6 GB peak. Training the same model peaks at 63.2 GB for a 96-token budget
and 85.5 GB for 384, with LoRA on every layer.

## Adapter lifecycle & regression validation

Run outputs under `runs/` are disposable (gitignored). An adapter that earned
a name gets **promoted to an adapter library** (defaults to
`~/models/adapters/<name>/`, override with `MLX_RL_ADAPTERS_DIR`):

```sh
uv run python scripts/promote_adapter.py runs/myrun --name sage-arith
```

This writes the adapter in **mlx-lm's native adapter format** — directly
consumable by `mlx_lm.server --adapter-path` and `mlx_lm.load(adapter_path=...)`
— plus a `MANIFEST.md` with full provenance (base model, run config, eval
trajectory, mlx-rl commit) and a regression checklist.

**RL on task X must not silently cost capability on task Y.** These adapters
deliberately change thinking behaviour, which could move agentic coding either
way, so promotion is step one. Before an adapter is served for real, walk it
through tiers of increasing cost, each against an already-measured base-model
baseline. Serve it with `mlx_lm.server --adapter-path <dir>` and point an
external benchmark harness at it:

| tier | what | cost |
|---|---|---|
| 0 | in-repo off-task check: `toolformat` canonical rate with the adapter loaded | minutes, in-process |
| 1 | single-shot coding, e.g. an EvalPlus HumanEval slice | ~30 min against the served adapter |
| 2 | agentic, e.g. a small SWE-bench Verified slice | hours, Docker-scored |

Measure the base model on the same server first, so a tier-1/2 delta is
attributable to the adapter and not the serving stack. Record results by
ticking the manifest checklist with numbers and run-dir pointers.

## Tasks

Nine tasks ship, all with programmatic rewards (`--task <name>`). For the
corpora behind them — provenance, licensing, and the measured per-problem
difficulty atlases that drive curriculum bands — see [DATASETS.md](DATASETS.md).

- **`arithmetic`** — toy multi-operand integer arithmetic with difficulty
  knobs (`n_operands`, `max_operand`). The reward reads the last answer-tag
  match, so thinking-mode drafts do not confuse it.
- **`math`** — competition math from
  [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
  (MIT; fetched from the HF Hub on first use, filtered to ~25k numerically
  verifiable answers, fixed held-out split). Reward: the last `\boxed{}` value
  matches the reference exactly.
- **`code`** — sanitized MBPP (427 problems, shipped in `data/` — see
  [data/README.md](data/README.md) for provenance and license). Reward: the
  model's function passes the hidden asserts. Candidate code runs under macOS
  `sandbox-exec` by default (network denied, writes confined to its temp dir)
  with rlimits on CPU, file size, fds and procs, and a scrubbed env.
  `--task_kwargs '{"sandbox": false}'` disables the Seatbelt layer — ⚠️
  candidate code then runs with your user's filesystem and network access.
  Memory is not capped either way (Darwin rejects `RLIMIT_DATA`); the 8 s
  timeout bounds blowups. For untrusted prompts or third-party models, use a
  container or VM.
- **`kodcode`** — the leaderboard-comparable coding task. Trains on
  KodCode-Light-RL-10K (execution-verified, decontaminated against
  MBPP/HumanEval by its authors) and evaluates on `evalplus/mbppplus` — the
  exact 378 tasks behind the EvalPlus leaderboard, never trained on.
  `eval_sample` cycles in dataset order rather than drawing with replacement,
  so `--eval-n 378` is exactly one full pass over the benchmark, and the eval
  prompt byte-matches EvalPlus's own chat backend (a test asserts it). Same
  Seatbelt sandbox as `code`. ⚠️ KodCode is **CC BY-NC 4.0**
  (non-commercial); it is fetched from the HF cache, never redistributed here.
  Results and method:
  [docs/mbpp-evalplus-results.md](docs/mbpp-evalplus-results.md).
- **`deepcoder`** — competition programming (TACO / SYNTHETIC-1 / pre-cutoff
  LiveCodeBench, via
  [agentica-org/DeepCoder-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)),
  filtered to stdin/stdout problems for one unambiguous judge: 18,983 train /
  175 test, fetched by `scripts/fetch_deepcoder.py`. Reward: the emitted
  program matches every stored test case. Supports a difficulty curriculum via
  `labels_file=` with `min_pass=`/`max_pass=`. This is the corpus with
  headroom — Qwen3.6-35B-A3B scores 0.52 pass@3 where MBPP is saturated at
  0.97 — and it needs a ≥32k token cap to measure honestly
  ([DATASETS.md](DATASETS.md)). ⚠️ **Unlike `code`, this executes
  model-generated code in a plain subprocess, with your user's filesystem and
  network access. Use a container or VM.**
- **`honesty`** — check before you answer, decline when the check comes back
  empty. Two domains: `papers` (arXiv author/year questions over a frozen
  metadata snapshot, answered from real search results captured once and
  frozen) and `trivia` (TriviaQA with alias gold and live web tools). Every
  item carries a regime the reward can see — does the base model know it
  (measured pass rate, `data/labels/`), is it published on or before the
  stated date, is it real at all. Reward: a correct answer +1, less the cost
  of a search it did not need; a decline 0, or +1 when declining is right and
  an empty search backs it up; a wrong answer or a flat denial −penalty.
  Train on one domain and score the other with `--eval-cells`: the held-out
  subject is what separates learning to check from memorising one domain's
  surface. Commitment is judge-graded (`--judge-backend local` runs the judge
  on the resident base model). Results:
  [docs/qa-glove-results.md](docs/qa-glove-results.md).
- **`qa_abstain`** — the tagged form of the same question: answer a short
  factual question in `<answer>` tags or reply `<abstain/>`, graded against
  alias gold with no judge. Reward: correct +1, abstain 0, wrong or malformed
  −penalty — the penalty sets the implied confidence threshold (default 3.0 →
  answer iff p(correct) > 0.75). TriviaQA (Apache-2.0, ~138k) to train, PopQA
  (MIT) as the out-of-distribution transfer eval. `scripts/qa_calibrate.py`
  probes per-question pass@k so a `calib_file` and `band_mix` curriculum keeps
  decision variance inside GRPO groups. Use `--inject-r 1`: the base policy
  almost never samples `<abstain/>`, and GRPO cannot reinforce what is never
  sampled, so injection supplies the off-policy demonstration that invariant 2
  makes legal. The injected member is the per-question calibration oracle
  (gold answer on high-pass-rate questions, `<abstain/>` otherwise); keep it
  symmetric so both collapse directions stay self-correcting, and pair it with
  `--abort-inactive-window 30`. Prior work and positioning:
  [docs/qa-abstain-related-work.md](docs/qa-abstain-related-work.md).
- **`toolformat`** — canonical tool-call format plus tool and argument
  correctness; doubles as a format regression detector for adapters.
- **`mixture`** — samples a weighted mix of the above per example (e.g.
  `{"weights": {"math": 0.35, "code": 0.35, "arithmetic": 0.3}}`), so the
  policy is not shaped by a single distribution.

### Adding a task

Implement `sample(rng) -> Example` and `reward(example, completion) ->
RewardResult` in `src/mlx_rl/tasks/`, decorate with `@register`, import it in
`tasks/__init__.py`. Rewards must be verifiable: computed, not judged.

## Scaling notes (96 GB unified memory)

- Tiny models (≤1B): run anywhere; a configured lease leaves any co-resident
  server up.
- Qwen3.6-35B-A3B / Qwen3.8-27B class (~16–22 GB 4-bit): a configured lease
  displaces and restores a co-resident server automatically. Use
  `--lora-layers 12` (invariant 4) and `--micro-batch 1` for ≥384-token
  budgets.
- Past ~1536-token budgets, `--grad-checkpoint` (recompute instead of
  retaining activations) and the serial GatedDeltaNet backward (`gdn_serial`,
  on by default for this model class) are what make it fit: with them an
  8192-token backward costs less than a 2560-token one did stock. The measured
  memory anatomy is in
  [docs/memory-and-compute-anatomy.md](docs/memory-and-compute-anatomy.md).

`mlx-rl-train --help` documents the full CLI, including the efficiency levers
(`--group-stage1`/`--stage1-skip`, `--update-adv-frac`, `--token-subset-frac`)
and the length-shaping reward knobs (`--length-penalty`, `--length-budget`).

## Docs

[DATASETS.md](DATASETS.md) — what corpora the tasks draw on, plus the measured
difficulty atlases (MBPP pass@5 across three temperatures; the DeepCoder pilot
and the token cap it needs).

- [memory-and-compute-anatomy.md](docs/memory-and-compute-anatomy.md) — where
  backward memory goes on a hybrid-attention MoE; the GDN-scan root cause and
  the serial-scan fix (34.3 → 2.37 GiB per layer @4096).
- [sage-paper-notes.md](docs/sage-paper-notes.md) — close reading of the SAGE
  paper and the exact algorithm this repo implements.
- [qa-glove-results.md](docs/qa-glove-results.md) — the honesty program:
  teaching a 35B model to decline questions it cannot answer, in ordinary
  conversation, and what that costs on the ones it can.
- [qa-abstain-related-work.md](docs/qa-abstain-related-work.md) — prior work
  on abstention and calibration training, and where these tasks sit in it.
- [mbpp-evalplus-results.md](docs/mbpp-evalplus-results.md) — the code line's
  training and EvalPlus-comparable evaluation.
- [uplift-over-prompt.md](docs/uplift-over-prompt.md) — what the training adds
  over prompting alone.

## License

MIT — see [LICENSE](LICENSE). The MBPP dataset in `data/` is CC BY 4.0 from
Google Research and is **not** covered by the MIT license — see
[data/README.md](data/README.md). Datasets the tasks fetch at runtime carry
their own terms and are never redistributed here; note KodCode
(`kodcode` task) is **CC BY-NC 4.0**, so results from it are fine for
research but not for commercial use.
