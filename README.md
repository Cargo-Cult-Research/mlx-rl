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
  member gets a copy-on-write KV-cache clone (5.6–8.2× wall-clock speedup,
  table below).
- **Verifiable rewards only:** a `Task` supplies prompts and a programmatic
  reward (see `src/mlx_rl/tasks/`). No reward models.
- **SAGE-RL hybrid rollouts** (arXiv 2602.08354): optionally generate r of
  the G group members with SAGE confidence-guided decoding — see the
  dedicated section below.
- **Optional memory lease:** if you run something else memory-hungry on the
  same box (e.g. a local inference server), point `MLX_RL_MEMLEASE_CMD` at an
  external coordinator command and the trainer will call it to make room before
  loading and hand it back on release — even on crash, if your command is
  PID-aware (see `machine.py` for the exact CLI). This is **off by default**;
  with nothing configured, runs proceed unmanaged. Either way the in-process
  guard (`memory.py`) is the final backstop — it refuses runs that don't fit
  and a swap watchdog hard-aborts a run that starts paging to disk.

### Hard-won correctness details (do not regress these)

1. **Zero-variance groups are dropped, and signal-free steps are skipped.**
   With all advantages zero, the residual "gradient" is fp16 kernel noise —
   and Adam rescales any nonzero gradient to a full-size step. Updating on
   noise is a destructive random walk (measured: format-following collapsed
   0.875 → 0.0 in 3 steps before this fix).
2. **old_lp is recomputed teacher-forced at update time**, not taken from
   generation. The KV-cached incremental forward drifts from the padded batch
   forward by up to 0.125 nats/token at fp16; trusting it injects spurious
   importance ratios.
3. **Masks are applied inside `exp()`** in the objective; garbage logprobs at
   padded positions otherwise overflow to `inf * 0 = nan`.
4. **Rollout temperature is 1.0 for training** — recorded logprobs are the
   model's own distribution; a different sampling temperature would be
   uncorrected off-policy sampling. (SAGE members are the deliberate
   exception: off-policy injected demonstrations that enter the surrogate at
   ratio 1 because old_lp is recomputed — invariant 2 is what makes them
   legal.)
5. **LoRA depth is the update-memory knob.** Backward retains activations
   from the *deepest adapted layer* to the loss, so adapters on all 40 qwen36
   layers peak at 85.5 GB (swap death on 96 GB) while the last 12 layers peak
   at 49.4 GB for the same 576-token sequences. Prefer `--lora-layers 12` for
   big models; go deeper only when the task demonstrably needs it.

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

The default model (`tiny` profile, Qwen2.5-0.5B) downloads from the Hugging
Face Hub on first use. The big-model profiles (`qwen36`, `gemma26`) point at
**local** MLX 4-bit model directories under `MLX_RL_MODELS_DIR` (default
`~/models/mlx`) — convert or download those yourself first (e.g. with
`mlx_lm.convert`); see `src/mlx_rl/profiles.py`.

## Quickstart

```sh
uv run pytest                      # fast, no model needed

# Smoke (0.5B, downloads ~300 MB on first use):
uv run mlx-rl-train --steps 3 --batch-prompts 2 --group-size 4 --out runs/smoke

# Convergence demo on the toy task:
uv run mlx-rl-train --steps 50 --batch-prompts 4 --group-size 8 \
  --task-kwargs '{"n_operands": 2, "max_operand": 99}' --out runs/convergence
```

For long unattended runs launch `.venv/bin/mlx-rl-train` directly instead of
`uv run` (a stale uv environment lock has been observed to stall start-up; the
venv entry point has no such dependency).

Each run directory gets `config.json`, `metrics.jsonl` (per-step reward /
KL / lengths / throughput / peak memory, plus periodic held-out greedy
evals), `samples.jsonl` (raw completion groups — always look at these, not
just the curves), and `adapters/` checkpoints.

## SAGE-RL (arXiv 2602.08354)

**Premise:** reasoning models already "know" when to stop thinking — rank
candidate reasoning chains by Φ = length-normalized cumulative logprob (mean
per-token logprob of the whole chain) and the end-of-thinking token scores
near the top long before greedy decoding would pick it (locally, one more
sentence of reasoning always wins). **SAGE** is a step-wise beam over
reasoning steps (`\n\n`-delimited): keep the top-m candidate chains by Φ;
each iteration expands every candidate with 2m sampled steps and accepts a
step ending in `</think>` when it ranks **within the top-h by Φ**, where
`h = round(TR · 2m)` (`--sage-tr`, the paper's tolerance ratio) — a
confidence gate against low-confidence early stops. **SAGE-RL** injects r
SAGE-decoded members into each GRPO group of G; the verifier rewards their
short-correct chains and the group-relative advantage teaches the policy to
make that its default — at inference you run plain sampling, no beam.

Implementation notes (`engine.py::sage_completion`):

- Beams fork the KV cache with the same copy-on-write clone as group prompt
  sharing. The default batched variant runs all 2m² step expansions as rows
  of one batched forward per token — ~4× over the per-beam reference (kept
  as `batched=False`); sparse-MoE expert routing caps the win below a dense
  model's, see `docs/memory-and-compute-anatomy.md`.
- Steps are sampled at `--sage-think-temp` (default 1.0, the paper's
  setting), so the r SAGE members of a group are not byte-identical; Φ
  ranking always uses the true (untempered) logprobs.
- Reasoning is hard-bounded at `max_new_tokens − --sage-answer-reserve`; if
  the gate hasn't fired by then (or by `--sage-max-steps`), the end-of-think
  token is force-committed, so the answer phase always has room and the
  completion can never exceed `--max-new-tokens`.
- The end-of-think token comes from the model profile (`think_end`): qwen36
  `</think>` = 248069 (thinking mode is switched on automatically when
  `--sage-r > 0`); gemma26 `<channel|>` = 101. Gating on a turn-terminal
  token (e.g. gemma's `<turn|>` = 106, via `--think-end`) is supported: the
  completion then ends at the commit.

```sh
# qwen36 thinking mode, 2 of 8 group members SAGE-decoded
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

### Reward design: "answer appears in tags" is not "answers and stops"

GRPO will find any hole in a reward. The canonical hole on thinking-mode
tasks: a reward that greps for answer tags *anywhere* lets the policy
**draft the tags inside the think block and ramble to the token ceiling** —
reward collected, rumination fully intact, zero terminating completions. On
the reward curve it looks like a triumph; a vanilla-GRPO arm once "won" an
A/B exactly this way, with more measured reward than the SAGE-RL arm and
not a single completion that actually stopped.

Defenses built into this trainer:

- Thinking-mode completions are graded **only on the visible reply after the
  final think-close marker** (`_visible_reply` in `train.py`); an unclosed
  think block is reward 0.
- Runtime tripwires print a loud `[BUG]` line on positive reward with an
  unclosed think (grader leak) and on SAGE budget breaches.
- The optional length penalty (`--length-penalty`) is correctness-gated and
  counts **total** tokens, so relocating reasoning out of `<think>` into
  visible prose buys nothing.
- `samples.jsonl` gets the first prompt's whole completion group every step.
  Every reward hack this project caught was found by reading samples —
  none by curves. Look at your samples.

**Model applicability:** gemma26 is currently *not* a SAGE-RL patient — under
in-process mlx-lm it degenerates at temp 1 (broken arithmetic, character
runs) and its stop-token confidence never enters the top-m in 1000+ tokens.
That is a serving-stack root cause to fix, not a task knob.

## Validation results (96 GB M3 Ultra)

| run | model | result |
|---|---|---|
| toy convergence | Qwen2.5-0.5B (1.7 GB peak) | held-out greedy 0.56 → 0.88 by step 10 |
| positive control | qwen36, no thinking, 7-operand arithmetic, 384 budget | **0.00 → 1.00 by step 10** (all-40-layer LoRA peaked 85.5 GB — use `--lora-layers 12`) |
| 180-step mixture run | qwen36, math+code+arithmetic, cap 2560 | eval_correct 0.56 → 0.66, plateau from ~step 40; see `docs/memory-and-compute-anatomy.md` |

**Batched engine (`engine.py`), 4 prompts × group 8, vs one-at-a-time**
(rollout-only):

| Model | Batched wall | Sequential | Speedup | Peak (rollout) | Peak (training, all-layer LoRA) |
|---|---|---|---|---|---|
| qwen36 35B-A3B | 249 tok/s | 44 tok/s | 5.6× | 24.6 GB | 63.2 GB @96-tok budget (85.5 GB @384) |
| gemma26 26B-A4B | 480 tok/s | 58 tok/s | 8.2× | 18.7 GB | 32.8 GB |

LoRA backward is validated on both architectures — qwen36 exercises the
hybrid GatedDeltaNet path, gemma26 the sliding-window path — but see the
gemma26 generation caveat in the SAGE-RL section before training it.

## Adapter lifecycle & regression validation

Run outputs under `runs/` are disposable (gitignored). An adapter that earned
a name gets **promoted to an adapter library** (defaults to
`~/models/adapters/<name>/`):

```sh
uv run python -m mlx_rl.promote runs/myrun --name sage-arith
```

(The library location can be overridden with `MLX_RL_ADAPTERS_DIR`.)

This writes the adapter in **mlx-lm's native adapter format** — directly
consumable by `mlx_lm.server --adapter-path` and `mlx_lm.load(adapter_path=...)`
— plus a `MANIFEST.md` with full provenance (base model, run config, eval
trajectory, mlx-rl commit) and a regression checklist.

**RL on task X must not silently cost capability on task Y.** The KL leash
and shallow LoRA make catastrophic forgetting unlikely, but this family of
adapters *deliberately changes thinking behavior* — exactly the kind of
change that could move agentic coding either way. So promotion is only step
one; before an adapter is served for real, walk it through a few tiers of
increasing cost, each against an already-measured base-model baseline. Serve
the promoted adapter with `mlx_lm.server --adapter-path <dir>` and point an
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

Five tasks ship, all with programmatic rewards (`--task <name>`):

- **`arithmetic`** — toy multi-operand integer arithmetic with difficulty
  knobs (`n_operands`, `max_operand`). The reward reads the LAST answer-tag
  match so thinking-mode drafts don't confuse it.
- **`math`** — competition math from
  [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
  (MIT; fetched from the HF Hub on first use, filtered to ~25k numerically
  verifiable answers, fixed held-out split). Reward: last `\boxed{}` value
  matches the reference exactly.
- **`code`** — sanitized MBPP (427 problems, shipped in `data/` — see
  [data/README.md](data/README.md) for provenance/license). Reward: the
  model's function passes the hidden asserts. ⚠️ **This executes
  model-generated code in a plain subprocess — NOT a sandbox.** It runs with
  your user's filesystem and network access; use a container/VM if that
  matters to you.
- **`toolformat`** — canonical tool-call format + tool/arg correctness;
  doubles as a format regression detector for adapters.
- **`mixture`** — samples a weighted mix of the above per example (e.g.
  `{"weights": {"math": 0.35, "code": 0.35, "arithmetic": 0.3}}`), so the
  policy isn't shaped by a single distribution.

### Adding a task

Implement `sample(rng) -> Example` and `reward(example, completion) ->
RewardResult` in `src/mlx_rl/tasks/`, decorate with `@register`, import it in
`tasks/__init__.py`. Rewards must be verifiable (computed, not judged).

## Scaling notes (96 GB unified memory)

- Tiny models (≤1B): run anywhere; if a lease command is configured it leaves
  any co-resident server up.
- qwen36 / gemma26 class (~16–22 GB 4-bit): a configured lease displaces and
  restores a co-resident server automatically. Use `--lora-layers 12` (see
  correctness detail 5) and `--micro-batch 1` for ≥384-token budgets.
- For sequence budgets past ~1536, `--grad-checkpoint` (recompute instead of
  retaining activations) and the serial GatedDeltaNet backward (`gdn_serial`,
  on by default for qwen36-class models) are what make it fit: the full
  memory anatomy, measured, is in
  [docs/memory-and-compute-anatomy.md](docs/memory-and-compute-anatomy.md) —
  with them, an 8192-token backward costs less than a 2560-token one did
  stock.
- gpt-oss-120b (~59 GB): out of reach for training on 96 GB; don't try.

`mlx-rl-train --help` documents the full CLI, including the
efficiency levers (`--group-stage1`/`--stage1-skip`, `--update-adv-frac`,
`--token-subset-frac`) and the length-shaping reward knobs
(`--length-penalty`, `--length-budget`).

## Docs & scripts

Technical notes in [docs/](docs/):

- [memory-and-compute-anatomy.md](docs/memory-and-compute-anatomy.md) — where
  backward memory actually goes on a hybrid-attention MoE; the GDN-scan root
  cause and the serial-scan fix (34.3 → 2.37 GiB per layer @4096).
- [sage-paper-notes.md](docs/sage-paper-notes.md) — close reading of the SAGE
  paper and the exact algorithm this repo implements.

Standalone instruments in [scripts/](scripts/): `probe_backward.py`
(memory-vs-length probe), `anatomy_gdn.py` / `anatomy_sched.py` (per-layer
GDN measurements behind the serial-scan fix), `bench_rollout.py` (batched vs
sequential rollout), `oracle_sage.py` / `think_length.py` /
`math_calibrate.py` (decode-quality and dataset-difficulty probes),
`dashboard.py` (stdlib live run dashboard over `runs/`), `sage_server.py`
(OpenAI-compatible server that decodes with SAGE). Each has a docstring with
usage.

## License

MIT — see [LICENSE](LICENSE). The MBPP dataset in `data/` is CC BY 4.0 from
Google Research and is **not** covered by the MIT license — see
[data/README.md](data/README.md).
