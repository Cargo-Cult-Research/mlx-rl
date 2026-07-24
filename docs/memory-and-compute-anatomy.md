# Backward memory & compute anatomy of the GRPO loop (2026-07-13)

Written after an interactive session deriving where the memory and compute
actually go in the qwen36 (Qwen3.6-35B-A3B-4bit) GRPO/LoRA loop, then measuring
it with `scripts/probe_backward.py`. This corrects two wrong mental models and
sizes the next experiment. Source of truth for the numbers is the probe; the
derivation is here to explain their shape.

## The model (from config.json, `text_config`)

- hidden `h = 2048`, **40 layers** = 30 GatedDeltaNet (linear-attn) + 10
  full-attn (every 4th).
- attention: 16 heads × head_dim 256 (Q is *wider* than the residual: 16·256 =
  4096), KV 2 heads (GQA), `attn_output_gate=True`.
- MoE: **8-of-256** experts, expert d_ff **512**, shared expert 512. The expert
  path is **gathered, not dense** (`mlx_lm/models/switch_layers.py` `SwitchGLU`
  → `_gather_sort` → only the 8 active experts are computed).
- vocab **248,320** (huge — this matters, see below). activations bf16 (2 B),
  SSM state fp32.
- training: LoRA rank 16 on the last 12 layers → **4.28 M trainable params**;
  `micro_batch = 1`.

## Two mental models that are wrong

1. **"Batch of 16/24 in the backward."** No — `micro_batch = 1`
   (`train.py:289`). The 24 rollouts (3 prompts × 8) go through the backward
   **one sequence at a time** with gradient accumulation. The only axes in the
   backward are sequence length and depth; there is no batch axis.

2. **"Memory ≈ constant floor + tiny linear term."** (What I claimed mid-session
   before measuring — wrong.) The probe shows it is **superlinear** in sequence
   length. See below.

Also worth stating because it removes a scary term: **attention is flash-style**
(MLX `sdpa`), so the `[heads, S, S]` scores are never *stored* — no quadratic
*activation-storage* term. But the backward still *recomputes* attention, so
there is a quadratic *transient* term in the backward (the growing slope below).

## Where the memory goes (structure)

- **Weights**: 22.1 GB resident the whole run (4-bit).
- **Optimizer + gradient**: ~34 MB *each*. Negligible — we train 4.3 M params.
  The memory is NOT the update; it's the **activation/transient graph of a full
  backward through the frozen 35B trunk** (grads flow through all 40 layers to
  reach the LoRA params).
- **The 248 K-vocab slab**: `model(inp)` → logits `[1, S, 248320]`. At S=2048
  that's 2048·248320·2 B = **1.0 GB just for logits**, ~same again for the
  softmax in the cross-entropy VJP, plus the dequantized LM head (~1 GB). ~3 GB
  that a "hidden-size" account misses entirely, and it scales with S.
- **Dequant transients**: every frozen 4-bit matmul's VJP forms `Wᵀ·dy`,
  dequantizing to bf16 on the fly; `GatherQMM::vjp` (the MoE path, guarded at
  `models.py:56`) is the heavy one. MLX's lazy graph holds several layers' worth
  live before `mx.eval`, and its allocator keeps freed buffers resident
  (`train.py:490`) — this is the slow swap-creep source.

## Measured: peak GiB vs sequence length (probe_backward.py, 2026-07-13)

micro_batch 1, rank-16 LoRA last-12, one fwd + one fwd/bwd per length,
`mx.clear_cache()` between — the exact training path. Peak includes the 22 GB
weights.

| L (tokens) | **peak, grad-ckpt ON** | swap | peak, ckpt OFF | swap |
|-----------:|-----------------------:|-----:|---------------:|-----:|
| 1024 | — | — | 61.6 | 0 |
| 1536 | **37.5** | 0 | **83.4** | +10.3 (swapping) |
| 2048 | **44.6** | 0 | — | — |
| 2304 | **49.8** | 0 | — | — |
| 2560 | **55.5** | +0.7 | — | — |
| 3072 | **67.9** | +0.7 | — | — |

Readings:
- **Checkpointing is load-bearing**: 83.4 → 37.5 GiB at 1536 (−46 GB, −55%).
  Without it we swap at 1536; with it we fit to 3072 in a *single-seq* probe.
- **The curve is superlinear.** Non-weight memory (ckpt): 15.5 GB @1536 → 45.9
  GB @3072 — ~3× for 2× tokens. The extra-slope is the attention-backward
  recompute (O(S²) transient across the 10 full-attn layers). `bwd_s` confirms
  it in compute: 13.3 → 39.4 s (~3×) over the same range, while `fwd_s` is ~2×
  (linear). **So a 10× longer budget is much worse than 10× cost.**
- **Probe fits ≠ live fits.** The probe is one sequence. Live training adds the
  `old_lp`/`ref_lp` numpy arrays, the un-freed rollout/gen KV, three forward
  passes, and hours of allocator creep. That gap is why v5 ARM1 (vanilla) *just*
  completed 32 steps at cap 2304 (~50 GiB probe) while ARM2 (+levers, a second
  gen phase) swap-died at step 26 at the same cap. **The honest live ceiling for
  this workload is ~cap 2304, and 2048 is the safe-for-unattended number.**

## Where the memory goes (compute) — the "97%" was a phase ratio, not a 30×

Backward ≈ 2× forward FLOPs; grad-ckpt adds one recompute forward → ~3× a
forward. There is no 30× anywhere. The "update is ~79% of wall-clock" is a
**phase ratio**:
- Generation (all 24 completions): memory-bandwidth-bound decode, ~380 tok/s.
  ~36 K tokens ÷ 380 ≈ **~95 s/step**.
- Update: **six full-sequence, compute-bound passes** at micro_batch 1 —
  `old_lp` + `ref_lp` + policy fwd + backward(2×) + ckpt recompute — over 24
  rollouts × ~1900 tok. ≈ **290–640 s/step** (matches the logged `upd`).

So the update dominates because it pays 6 full-sequence passes at low
utilization (mb=1, MoE-gather) against one bandwidth-efficient generation pass —
not because any backward is disproportionate.

## The knob tension the levers can't resolve

"Raise the reasoning budget" (paper fidelity / less hard-math truncation) and
"drop to 2048" (memory safety) are the **same knob** (`max_new_tokens`) pulled
opposite ways — mutually exclusive by construction. Given the measured curve, on
*this* model you cannot have both a long budget and headroom. To afford a longer
budget you must buy the memory back elsewhere:

**Where the 2× can come from, ranked:**
1. **Token-subset backprop** (reading-list #1, S-GRPO / T-SPMO,
   arXiv:2504.20834). Backprop only 30–50 % of completion tokens → cuts the
   activation graph **and** the O(S²) attention-backward **and** the update FLOPs
   ~proportionally. Cheapest integration (choose which positions enter the
   surrogate in `grpo_objective`/`build_training_arrays`), and it hits exactly
   the dominant cost. **Highest expected value.**
2. **Smaller/dense model — Gemma 4 26B.** ~15 GB weights, dense MLP (no
   gather-VJP transient), fewer layers → the whole curve shifts down; real
   headroom for long-context RL. Cost: different/weaker model + gemma serving
   quirks. Legitimate if long context is the actual goal.
3. **Heavier quantization** (3-bit experts): drops weights ~5 GB but quality
   risk on an already-4-bit model; marginal. Low priority.
4. LoRA depth is already the tuned knob (last-12 halves the update peak);
   fewer layers trades learning for memory.

## Recommended next experiments

**RESULT — v7 `runs/v7-fewer-long` (completed 2026-07-14, 180 steps, exit 0):**
last-4 LoRA rank-8, cap 2560, vanilla, seed 0, math/code/arith mix. Ran
flawlessly: peak 55.3 GiB, swap growth −0.25 GiB (flat), ~6.2 min/step, ~18.6 h.

| metric | baseline | final | note |
|---|---|---|---|
| eval_reward / eval_correct | 0.562 | **0.656** (peaked 0.688 @152–176) | **real correctness gain**, +3–4 of 32 |
| eval_think_closed | 0.594 | 0.656 | closes `<think>` more |
| eval_format | 0.312 | 0.406 | cleaner formatting |
| eval_mean_len | 2170 | 1955 | more concise (greedy) |
| eval_rfcs | 0.328 | 0.443 | reaches the answer earlier |
| eval_task_{math,code,arith} | .312/.312/.375 | .312/.312/.375 | **task MIX fractions, not accuracy** — constant by construction |

Findings:
- **Correctness DID move** (0.562→0.656, holding 0.62–0.69 from step 40 on). The
  earlier "frozen eval_task_* ⇒ RL can't move accuracy" read was WRONG:
  `mixture.py:53` sets `task_{name}=1.0` per example "so metrics show the mix
  proportions" — those three fields are the eval-set composition (12/32, 10/32,
  10/32), fixed by seed, never accuracy. The real signal is `eval_correct`, and
  it rose in v7 (0.56→0.66) and v5 (0.56→0.72).
- **Plateau is real, not undertraining**: correctness saturated by ~step 40 and
  held for 140 more steps. For this task/model, ~40–50 steps ≈ 180 steps.
- **No reward-hacking**: eval got *both* more correct and more concise, KL small
  (≤0.012) throughout — the control-arm hack pattern did not appear at rank-8.
- fewer-layers + low-rank did NOT cost correctness vs the richer v5 configs —
  consistent with "capacity isn't the bottleneck; the signal is."

**NEXT (needs a little code, keep a human in the loop):**
- **Token-subset backprop** (lever #1 above). The one change that buys the 2×.
  Validate it's loss-equivalent on a short run before trusting it (show the
  spec/code mapping to a reviewer first).

## IMPLEMENTED 2026-07-14: token-subset backprop (S-GRPO)

Motivating v7 data (why this is the run): **51.5% of rollouts hit the 2560
cap** (56% late-run), and **79/180 steps had `active_groups == 0`** — 44% of
steps produced no gradient at all. Dead steps average 60% cap-rate vs 45% on
live steps: all-truncated groups → uniform near-zero reward → zero variance →
skipped. Roughly half the run's compute bought no learning signal, and the
problems that still fail are exactly the ones that truncate.

**Selection rule = uniform random over completion tokens, per row, exact
counts.** Principled, not just simple: the advantage is one scalar per
sequence (`advantages[:, None]` in `grpo_objective`), so the objective
carries NO per-token credit signal — a uniform subset with the denominator
scaled by `frac` is an unbiased estimator of the full-token gradient
(dropout-over-the-token-axis), and its variance cost is small because
within-sequence terms share the advantage. It is also the only rule with a
ground truth to validate against (frac=1.0 must reproduce the dense loss) —
entropy/"forking-token" selection (2506.01939-style) is deliberately arm-2:
it confounds bias-by-design with the memory change. Paper context
(2504.20834): under LoRA, full-token GRPO barely beat base while a 30–50%
subset lifted SVAMP 46→70+ — subsetting plausibly regularizes gradient
interference from boilerplate tokens on a low-rank adapter.

**Spec → code mapping:**
- `grpo.py: subsample_token_mask(mask, frac, rng)` — per-row
  `max(1, round(frac·n))` without replacement (never empties a row;
  inclusion prob deviates from frac by ≤ 1/(2n) — negligible at rollout
  lengths). Pure numpy, unit-tested.
- `models.py: selective_logprobs(model, inp, tgt_sel, sel_idx)` — trunk on
  the FULL sequence (attention needs every position), gather hidden states
  at `sel_idx` BEFORE the vocab projection → logits **[B, K, 248320]**
  instead of [B, L, 248320]. Numerically identical per position (softmax is
  row-wise).
- `rollout.py: gather_selected(...)` — compresses a microbatch slice to its
  mask==1 positions (right-padded; padding gated to exactly zero by the
  existing mask-in-exponent trick).
- `train.py: update_policy` — subsamples the mask, scales `denom` by frac
  (NOT the realized count — preserves the "pruned terms contribute zero"
  semantics and keeps pg/kl on the full-token scale), routes microbatches
  through the selective head. `config.token_subset_frac`: 0 = off,
  1.0 = selective-all (equivalence setting), (0,1) = S-GRPO.
- `scripts/probe_backward.py --token-subset-frac` — memory probe of the
  selective path.

**Tests (`tests/test_token_subset.py`, 5, all green + 73 existing):** exact
per-row counts & subset-of-mask; frac=1 identity; gather padding masked;
dense vs selective loss AND grads equal at full selection on a structural
mini-model; 400-draw subset-loss mean within 4 SEM of the dense loss.

**HONESTY CORRECTION to lever #1's claim above:** token-subset does NOT
automatically cut the O(S²) attention backward — the forward still runs the
full sequence, and one layer below the top the cotangent is dense across all
positions regardless of which loss rows are zero. What it provably cuts is
the vocab slab (logits + CE-softmax VJP, linear in S) and the top-of-graph
loss tensors. The 12→8-layer null result already showed the superlinear
term's source is not settled — the probe arbitrates, again. If more is
needed later: L39's outputs at unselected positions are consumed by nothing
(it is the last layer), so its attention could run queries-gathered —
O(f·S²) exactly — at the cost of real surgery inside the qwen3_next block.
The learning-efficiency result (paper) justifies the arm even if the new cap
lands at 3k rather than 4k.

**MEASURED 2026-07-14 (probe, last-4 rank-8 ckpt, frac 0.35): the memory
claim is DEAD, the peak doesn't move at all.**

| L | dense peak GiB | subset-0.35 peak GiB |
|---:|---:|---:|
| 2560 | 50.0 | 50.0 |
| 3072 | 58.8 | 58.8 |
| 3584 | 71.5 | 71.5 |
| 4096 | — | 86.0 (swap +0.3) |

Identical to the tenth of a GiB at every overlapping length. (Engagement is
not in doubt: the selective path's gathered arrays are shape-incompatible
with the dense code path, so running at all proves it took the selective
branch.) Conclusion: **the backward's high-water mark is inside the trunk
(checkpoint-segment recompute + dequant transients), not the vocab-head/loss
phase** — the ~1.3–2.6 GB logits slab is real but sits entirely below the
peak. Lever #1's "buys the 2×" is refuted; the memory ledger now reads:
LoRA depth (null at long S), token-subset (null), leaving the trunk itself —
smaller/dense model (Gemma-26B) or real kernel surgery — as the only
remaining sequence-memory levers. What survives of token-subset is the
paper's actual headline: **better LEARNING under LoRA at the same memory**
(SVAMP 46→70 with 30–50% subsets) — a science arm, not a memory lever.
Corollary from the same table: the honest cap ladder is unchanged
(2560 ≈ 50, 3072 ≈ 59, 3584 ≈ 71.5 GiB probe-peak; v7 measured live ≈
probe + ~5 GiB), so modest truncation relief comes from raising the cap
toward 3072 on the existing curve, not from subsetting.
- **Improve eval RESOLUTION (corrected from the earlier "eval is broken" claim).**
  The eval is not broken — `eval_correct` tracks correctness and it moves. But
  (a) it logs only AGGREGATE correctness + task *composition*, never per-task
  accuracy, so we're blind to which task drives the gain; add a real per-task
  correctness metric (`correct` split by `meta["_task"]`). (b) `eval_n=32` gives
  1/32 granularity — the 0.56→0.66 gain is only ~3 problems; raise to 128+ for
  the plateau to be statistically legible.

## MEASURED 2026-07-13: layer count is a SHORT-sequence lever only

Probed `--lora-layers {4,8,12}` × long lengths (rank 8, grad-ckpt). Peak GiB:

| L (total) | last-12 | last-8 | last-4 |
|----------:|--------:|-------:|-------:|
| 2560 | 55.5 | ~53 | **50.0** |
| 3072 | 67.9 | 67.7 | **58.8** |
| 3584 | — | — | **71.5** |
| 4096 | — | 97.6 (swap) | — |

- **12→8 saves ~nothing at long S** (67.9→67.7 @3072). 8→4 saves ~9 GB. The
  README's big layer-count savings (85→49 GB) were at **576 tokens** — the
  depth lever has strong leverage only when the per-adapted-layer backward is a
  big fraction of the total, i.e. short sequences.
- At the sequences we actually want, memory is **sequence-dominated** (O(S²)
  attention + 40-layer forward + the 248 K vocab slab), none of which the LoRA
  depth touches. So **no layer count unlocks 4096-length training on 96 GB.**
  The safe single-step ceiling (~50–55 GB, matching the clean ARM1 that ran swap-
  flat) caps total L at ~2600–2900 → **cap ~2560 regardless of layers.**
- Consequence: to train at *materially* longer sequences you must attack the
  sequence terms, not the depth: chunked cross-entropy (kills the ~3 GB vocab
  slab), token-subset backprop, or a smaller/dense model (Gemma-26B). Fewer
  layers is still worth doing (last-4 = the clean boundary AND where memory
  finally drops) — but for capacity/regularization + a modest length bump, not
  as the long-context unlock.

**Parameter-efficiency reframe (memory is set by DEPTH, not param count):**
LoRA's memory cost is the backward *depth* — an adapter at layer N forces the
backward to retain the activation graph of every layer from the top down to N
(README: all-40 = 85.5 GB vs last-12 = 49.4 GB, monotonic in depth). The
adapters themselves are 34 MB. So there is one expensive axis (depth) and two
cheap ones (rank: `Ax` is `[S,r]`, ~0.8%/matrix of the `[S,2048]` we already
retain; width: extra thin keys hang off already-taped activations for free).
Memory-optimal recipe for max trainable-params/GB:
1. Fix depth K by budget, then **pack every thin projection at higher rank into
   those K top layers** — capacity moves off the expensive axis onto the cheap
   ones, same activation memory.
2. **Do not adapt the 256-way MoE experts** — the `GatherQMM` VJP dequant is the
   one per-adapter cost that actually bites (thin q/k/v/o/GDN projections don't).
3. **Land the depth cutoff just above a full-attention layer** (D=32 or 36, not
   28): the 3 full-attn layers (31/35/39) drive the superlinear O(S²) term;
   D=28→32 crosses 2 full-attn instead of 3 for only 4 GDN layers of lost
   adaptation. *Hypothesis — probe it.*
Caveat: memory-optimal ≠ learning-optimal (depth-spread often helps quality;
early-layer adaptation is unavoidably expensive). **Concrete follow-up: an
equal-parameter sweep — rank-64 top-4-all-keys vs rank-16 top-12 — comparing
eval reward AND peak GB.** If shallow-wide matches on reward at lower peak,
that's the headroom for longer sequences or the token-subset lever.

## ROOT CAUSE FOUND & FIXED IN ISOLATION 2026-07-14: the GDN training scan

After the token-subset null, we named the pattern ("strong memory claims,
walked back when the probe disagrees — instrument and find where it actually
goes"). Instrumented per-layer (`scripts/anatomy_gdn.py`, single synthetic
layer with real config dims — no weights, no memory lease needed) and found it:

**`GatedDeltaNet.__call__` dispatches `use_kernel=not self.training`** — in
training mode every GDN layer abandons the fused Metal kernel (no VJP) for
`gated_delta_ops`: a Python loop over ALL T timesteps whose per-step fp32
recurrent state is [B, 32, 128, 128] = **2.10 MB/token** (the residual
stream is 4 KB/token — 500×). Under mx.grad every step's ~4 state-sized
temporaries stay live for the backward:

| S | one GDN layer bwd | one full-attn layer bwd |
|---:|---:|---:|
| 1024 | 8.8 GiB | 1.0 |
| 2048 | 17.4 GiB | 1.4 |
| 4096 | 34.3 GiB | 3.1 |

**The scan is 12× attention.** This — not O(S²) attention, not the vocab
slab — is the sequence-memory wall. It explains every anomaly at once:
last-12→8 null (peak = the GDN scan window, saturated at both depths),
8→4 = −9 GB (one fewer scan overlapping), token-subset null (trunk-side),
and the "superlinear" curve (a huge linear term, 8.5 GB/1k-tokens/layer,
stacked across overlapping layer backwards).

**The fix (validated in isolation, `make_serial_scan` in anatomy_gdn.py):**
scan as an `mx.custom_function` — forward runs the FUSED KERNEL (training
forward = serving forward, and faster); backward is a custom VJP that walks
chunk segments in reverse, recomputing each segment on the fly. Two
mechanism subtleties, both measured (`scripts/anatomy_sched.py`):
1. The naive segmented vjp graph is ALREADY cheap (1.4 GiB) when only one
   grad output is consumed — but inside the layer all five input grads are
   live outputs, and the executor materializes one concat tree at a time,
   so every segment's recomputed states stay resident for the later trees —
   sum, not max (17.4 GiB, unchanged from stock).
2. Fix: `mx.depends` each segment's recompute on ALL of the previous
   segment's grads (the full bundle, not just the state cotangent) → each
   segment retires completely before the next recompute starts.

| S | stock GDN layer | serial (chunk 64) | grads |
|---:|---:|---:|---|
| 1024 | 8.8 | **1.17** | ≡ chunked-exact (1.5e-8) |
| 2048 | 17.4 | **1.44** | ≡ (7.5e-9) |
| 4096 | 34.3 | **2.37** (14.5×) | ≡ (3.7e-9) |

Backward 2–3.5× faster too (kernel forward + fewer live buffers). ~Flat in
S. Composes with the outer per-layer mx.checkpoint. Forward |dy| vs ops
path = 1.6e-2 bf16 — this is the SAME kernel the serving path uses, so
training-forward now matches inference exactly (arguably a consistency fix;
old_lp/ref_lp/policy all shift together).

**Revised lever ranking (measured, not guessed):**
1. **Integrate the serial scan into training** (patch point:
   `q35.gated_delta_update`, e.g. in `load_policy`) → full-model probe at
   4096/6144/8192. Extrapolation from the ledger: last-4 at 4096 ≈ 22
   (weights) + 3×~2.4 (GDN) + 3 (attn) + slab ≈ **~35 GiB vs 86 measured
   stock** — but extrapolations have burned us three times; PROBE IT.
2. Once the scan stops dominating, the vocab slab RESURFACES as the next
   term — chunked-CE / token-subset become real levers again (they stack).
3. MoE GatherQMM dequant arm still unmeasured (`anatomy_gdn.py --moe`).
4. Upstream candidate: training-mode GDN memory is an mlx-lm-wide problem
   (every qwen3.5/3.6-family LoRA fine-tune pays it).

**RISKIER (research note only, not tonight):**
- Experience replay (reading-list #4): reuse each expensive rollout over k
  updates. Legal here because `old_lp` is recomputed teacher-forced. Biggest
  structural win for an inference-strong/train-weak box, but needs a buffer +
  staleness handling — too much new surface for an unattended run.
- GRESO-proper cross-step prompt filter (vs the within-step `uniform` skip that
  discarded ~44 % of v5 steps and may throw away learnable all-wrong groups).
- Gemma-26B long-context arm.

## INTEGRATED & PROBED FULL-MODEL 2026-07-14: the wall is down

The serial scan is now `src/mlx_rl/gdn_serial.py`, installed by default
(`gdn_serial: bool = True`, `--no-gdn-serial` to opt out; `gdn_chunk: 64`).
It patches `mlx_lm.models.qwen3_5.gated_delta_update` (qwen3_5_moe reuses
that module); masked calls delegate to stock; GQA repeat lives outside the
custom_function so autodiff owns its VJP. Tests: `tests/test_gdn_serial.py`
(fwd + all-five-primal grads vs stock ops, mask delegation, install
idempotence). Commit f23e979.

**MoE arm (anatomy --moe, real 4-bit MoE mlp):** GatherQMM adds only
~0.4 GiB to the layer backward (stock 34.7 / serial 2.56 GiB @4096). No
hidden MoE term; the scan explanation stands with the production mlp.

**Full-model probe (probe_backward.py, last-4 rank-8 + grad-checkpoint,
micro_batch 1, serial ON):**

| L | fwd_s | bwd_s | peak GiB | swap | stock (2026-07-13) |
|---:|---:|---:|---:|---:|---|
| 4096 | 2.0 | 6.6 | **25.4** | +0.0 | 86.0 GiB, 36.9 s |
| 6144 | 3.2 | 10.2 | **29.3** | +0.0 | (unrunnable) |
| 8192 | 4.4 | 13.9 | **33.6** | +0.0 | (unrunnable) |

3.4× less memory and 5.6× faster backward at 4096; ~1.9 GiB per extra 1k
tokens; **8192 tokens now costs less than 2560 did stock** (v7 per-step
peaks were ~52 GiB). Forward scoring passes (old_lp/ref_lp) are ~5× faster
too (kernel prefill). The extrapolation (~35 GiB @4096) was PESSIMISTIC by
10 GiB — first time an estimate erred in our favor; measured is measured.

**On-model grad equivalence (--grad-check, L=1024, real weights, identical
inputs):** serial-vs-stock LoRA grads cosine **0.998132**, max rel dev
6.1e-2, loss delta +1.1% — the bf16 kernel-vs-ops forward difference
propagating, i.e. noise-level agreement in the trainable direction. The
8-step serial-vs-stock train-loop pair (runs/equiv-*) is the last gate
before the long run.

**Train-loop equivalence (runs/equiv-serial vs equiv-stock, 8 steps, seed 0,
cap 1024): PASSED.** Same trajectory shape (identical dead steps 2–3,
per-step rewards within one sample, kl 0→0.0008 vs 0→0.0010; final eval
0.375 vs 0.50 at n=8 = one problem, noise) — and the serial arm's updates
ran **3.6× faster** (22–46 s vs 80–161 s) with generation **27% faster**
(455 vs 357 tok/s: training-mode prefill now hits the fused kernel).

## Upstream landscape (checked 2026-07-14): don't file a third PR

mlx-lm main is NOT fixed, but two open PRs already target exactly this:
**#1217** (SudarkinV, since April: Metal backward VJP kernel +
chunk-checkpointed Python fallback) and **#1389** (tsato081, since June:
chunk-parallel gated UT/WY formulation — the flash-linear-attention
algorithm — per-chunk mx.checkpoint, GQA/mask/carried-state support). Both
active (updated 2026-07-12), cross-verified by each other's authors plus a
third party (fblissjr: 27B QLoRA 117→39 GB, 3× throughput, four-way grad
agreement ≤2.6e-7).

Head-to-head on our anatomy harness (scripts/pr1389/, one training-mode GDN
layer, real qwen36 dims):

| S | gdn_serial (ours) | PR #1389 | stock |
|---:|---|---|---|
| 1024 | 0.94 GiB / 0.33 s | 0.90 GiB / 0.09 s | 8.8 GiB / 0.57 s |
| 2048 | 1.19 GiB / 0.68 s | 1.56 GiB / 0.19 s | 17.2 GiB / 1.63 s |
| 4096 | **2.10 GiB** / 1.38 s | 3.18 GiB / **0.39 s** | 34.1 GiB / 4.84 s |

Grads byte-identical between the two fixes (both rel ~3e-3 vs stock = the
recompute-noise class). Theirs: ~3.5× faster backward (parallel matmuls
beat sequential recompute). Ours: lower memory at long S with a flatter
slope (their in-graph checkpoint chunks retain more — the same executor
retention family anatomy_sched.py isolated, milder because their per-chunk
transients are small).

**Decisions (2026-07-14 night):** (1) no third PR — redundant with two
well-served active PRs; (2) v8 runs on gdn_serial (fully gated today;
#1389 unvalidated end-to-end here, speed edge worth only ~15–20% of step
time); (3) adopt #1389 when it merges upstream (deletes our custom code);
(4) possible contribution = M3-Ultra/35B benchmark + the anatomy_sched
retention mechanism as review comments on their threads — ONLY with explicit
maintainer go-ahead (publishing).

## EPILOGUE 2026-07-15: v8 closes the arc — the wall was real, the plateau wasn't behind it

v8 (runs/v8-cap4096: cap 4096, 120 steps, last-4 rank-8 vanilla mixture,
serial scan) completed 120/120 in **8.1 h, avg step 219 s, peak 26.0 GiB,
swap 0.0** — v7 needed 372 s/step and 52+ GiB at a 1.6× smaller cap. The
serial scan is production-validated end to end.

The science answer is a clean negative: doubling the cap did NOT lift the
learning plateau. Completions grew to fill the room (mean_len 2485; capped
rate 51.5%→34%, zero-gradient steps 44%→40%), and eval_correct bounced
0.53–0.69 trendlessly — final 0.656, v7's exact ceiling. The bottleneck is
the LEARNING SIGNAL (40% of steps still have zero reward variance within
every group), not the sequence budget this document was written to buy.
Sequence-memory levers are now closed as an explanation; the next levers
(task-difficulty curriculum so groups straddle the pass boundary, larger G,
dead-prompt resampling) are run-design decisions.

## Operational note

The trainer takes the **exclusive** memory lease and frees the host inference
server (offline) for the whole run; it restores on release. ⚠️ Known failure
mode: restore-on-release has left the server written-active-but-disabled+down
before. When this run ends, verify the server is actually back with a real
completion probe, not just the status files.
