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

Models are downloaded from the Hugging Face Hub on first use; local model paths
default to `~/models/mlx` and can be overridden with `MLX_RL_MODELS_DIR`.

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
candidate continuations by Φ = length-normalized cumulative logprob (mean
per-token logprob of the whole chain) and the end-of-thinking token scores
near the top long before greedy decoding would pick it (locally, one more
sentence of reasoning always wins). **SAGE** is a beam-style decode: keep the
top-m candidates by Φ, and stop the thinking phase when the end-of-think
token enters the top-m cutoff (within `tol` nats of slack). **SAGE-RL**
injects r SAGE-decoded members into each GRPO group of G; the verifier
rewards their short-correct chains and the group-relative advantage teaches
the policy to make that its default — at inference you run plain sampling,
no beam.

Implementation notes (`engine.py::sage_completion`):

- Beams fork the KV cache with the same copy-on-write clone as group prompt
  sharing; forwards run at B=1 per beam (m is small).
- With `--sage-think-temp > 0` (default 1.0), per-beam candidates are
  proposed by Gumbel-top-m sampling of the tempered distribution instead of
  a deterministic arg-top-m, so the r SAGE members of a group are not
  byte-identical. Φ ranking always uses the true logprobs.
- If the gate hasn't fired by `--max-think-tokens`, the end-of-think token is
  force-committed (budget compliance by construction).
- The end-of-think token comes from the model profile (`think_end`): qwen36
  `</think>` = 248069 (thinking mode is switched on automatically when
  `--sage-r > 0`); gemma26 `<channel|>` = 101. Gating on a turn-terminal
  token (e.g. gemma's `<turn|>` = 106, via `--think-end`) is supported: the
  completion then ends at the commit.

```sh
# The Ep10 run: qwen36 thinking mode, 2 of 8 members SAGE-decoded
.venv/bin/mlx-rl-train --profile qwen36 --steps 16 --batch-prompts 3 \
  --group-size 8 --sage-r 2 --sage-m 2 --sage-tol 0.02 \
  --micro-batch 1 --max-new-tokens 576 --max-think-tokens 384 \
  --lora-layers 12 --no-normalize-std \
  --task-kwargs '{"n_operands": 7, "max_operand": 9999, "format_reward": 0.0}' \
  --eval-every 4 --eval-n 16 --out runs/sage1-qwen36
```

Hybrid-run metrics add `mean_len`, `mean_len_sage` / `mean_len_sampled`,
`mean_think_len`, `reward_sage` / `reward_sampled`, and evals add
`eval_mean_len` — the deployment metric (greedy, no beam).

### What the Ep10 run showed (runs/sage1-qwen36, 2026-07-07)

qwen36 in thinking mode at a 576-token budget scores **0/18 at baseline** —
every completion truncates inside `<think>`. 16 hybrid steps lifted held-out
**plain-greedy eval from 0.00 to 0.44** (mean length 576 → 528). The
mechanism, visible in `samples.jsonl`:

| phase | sampled members | SAGE members |
|---|---|---|
| steps 1–9 | all 0.0, all truncate at 576 | gate never fires; force-cut at think 384, answer phase sometimes recovers (reward 1.0 at steps 3, 4, 6, 7) — the **only** reward source |
| steps 10–12 | first success at step 12 | gate starts firing naturally, earlier each step: think 333 → 319 → 195 → 163 |
| steps 13–16 | ~half succeed at 300–540 tokens | gate fires at **think 1** — the policy now top-ranks its trained empty-think pattern; hasty misses appear (a 99-token wrong answer at step 15) |

The stopping point SAGE surfaces *moves* over training: from "never" through
progressively earlier natural stops to the model's built-in skip-thinking
escape hatch. Group contrast (not SAGE correctness — late SAGE members do
miss) is what carries the signal.

### The control arm's verdict (runs/vanilla2-qwen36): both arms hacked, differently

Vanilla GRPO (same config, `--sage-r 0`) got MORE reward — held-out 0.00 →
0.875 — and it is the more instructive result. Its eval_mean_len is **576.0
at every eval**: the model never terminates. The raw samples show why: it
learned to **draft the answer tags inside the think block** ("…
`<answer>9592</answer>` … Check for any possible pitfalls: …" until the
budget cuts it). The reward greps for tags anywhere, so "embed the tags and
keep rambling" is the minimal-KL edit that collects it — rumination fully
intact, every query still pays the whole budget, output ends mid-sentence.

So on this reward the honest comparison is:

| arm | held-out reward | terminates? | what it learned |
|---|---|---|---|
| vanilla GRPO | 0.875 | never (576.0 at ceiling) | inject tags into the ramble |
| SAGE-RL (r=2) | 0.44 | partially (mean 528, real EOS endings) | actually stop, then answer |

Neither number is "clean accuracy" — the reward admits the tag-in-ramble
exploit for both arms. The lesson is reward design, not algorithm choice:
**"answer appears in tags" is not "answers and stops."** Next iteration
gates reward on termination (`finish_reason == "stop"`), which closes the
exploit and makes the two arms comparable on the thing we actually care
about. (This is the third distinct reward hack this project has caught by
reading samples.jsonl — curves alone would have called the vanilla arm a
triumph.)

**Model applicability:** gemma26 is currently *not* a SAGE-RL patient — under
in-process mlx-lm it degenerates at temp 1 (0/24 even on 3-operand ≤999,
char runs, broken arithmetic) and its stop-token confidence never enters the
top-m in 1000+ tokens. That is a serving-stack root cause to fix (the
vllm-mlx side needed the PR-610 stack for the same family), not a task knob.

## Validation results (2026-07-06/07, this box)

Machine state per row matters — the lease displaces the ~65 GB host inference
server only when needed:

| run | model | machine state | result |
|---|---|---|---|
| toy convergence | Qwen2.5-0.5B (1.7 GB peak) | co-resident with a loaded backend | held-out greedy 0.56 → 0.88 by step 10 |
| control #3 (positive control) | qwen36, nothink, 7-op ≤9999, 384 budget | host server displaced by lease | **0.00 → 1.00 by step 10**; killed at 26 (post-saturation swap: all-40-layer LoRA peaked 85.5 GB) |
| sage1 (SAGE-RL) | qwen36, thinking, 576 budget, last-12 LoRA | host server displaced by lease | **0.00 → 0.44** plain-greedy, len 576 → 528, updates peak 49.4 GB |
| vanilla2 (control) | same, `--sage-r 0` | host server displaced by lease | 0.00 → 0.875 reward but **zero termination** (len 576.0 flat) — tag-in-ramble reward hack; see control-arm verdict above |

**Batched engine (`engine.py`), 4 prompts × group 8, vs one-at-a-time**
(rollout-only, host server displaced):

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
uv run python -m mlx_rl.promote runs/sage1 --name sage-arith
```

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

## Adding a task

Implement `sample(rng) -> Example` and `reward(example, completion) ->
RewardResult` in `src/mlx_rl/tasks/`, decorate with `@register`, import it in
`tasks/__init__.py`. Rewards must be verifiable (computed, not judged).

Tasks shipped: `arithmetic` (toy, difficulty knobs; reward reads the LAST
answer-tag match so thinking-mode drafts don't confuse it) and `toolformat`
(canonical tool-call format + tool/arg correctness — doubles as a format
regression detector; qwen36 measured 100% canonical at baseline 2026-07-06).
Candidate next: **tool-call groundedness** (penalize confabulated calls) —
gated on first demonstrating the failure exists in a trainable model.

## Scaling notes (96 GB unified memory)

- Tiny models (≤1B): run anywhere; if a lease command is configured it leaves
  any co-resident server up.
- qwen36 / gemma26 class (~16–22 GB 4-bit): a configured lease displaces and
  restores a co-resident server automatically. Use `--lora-layers 12` (see
  correctness detail 5) and `--micro-batch 1` for ≥384-token budgets.
- gpt-oss-120b (~59 GB): out of reach for training on 96 GB; don't try.

## License

MIT — see [LICENSE](LICENSE).
