# Backward memory & compute anatomy of the GRPO loop

Where the memory and compute actually go when GRPO/LoRA-training a hybrid
linear-attention MoE (qwen36 = Qwen3.6-35B-A3B-4bit) on 96 GB of unified
memory — derived first, then measured with `scripts/probe_backward.py` and
the per-layer instruments `scripts/anatomy_gdn.py` / `scripts/anatomy_sched.py`.
The probes are the source of truth for every number here; the derivations
explain their shape.

**TL;DR**

- The sequence-memory wall in training mode is the **GatedDeltaNet scan**,
  not attention and not the vocab head: 34.3 GiB backward for ONE GDN layer
  at 4096 tokens. Fixed by `src/mlx_rl/gdn_serial.py` (custom-VJP serial
  scan, on by default): 2.37 GiB per layer, and the **full model backward at
  8192 tokens fits in 33.6 GiB** — less than 2560 tokens cost stock.
- **Gradient checkpointing is load-bearing** (83.4 → 37.5 GiB at 1536
  tokens). LoRA *depth* and token-subset backprop are **not** long-sequence
  memory levers — both measured null at long S (details below).
- The update phase dominates wall-clock as a **phase ratio** (six
  full-sequence, compute-bound passes at micro_batch 1 vs one
  bandwidth-efficient generation pass), not because any backward is
  disproportionate.

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
- reference training config: LoRA rank 16 on the last 12 layers → **4.28 M
  trainable params**; `micro_batch = 1`.

## Two mental models that are wrong

1. **"Batch of 16/24 in the backward."** No — `micro_batch = 1`: the 24
   rollouts (3 prompts × 8) go through the backward **one sequence at a
   time** with gradient accumulation. The only axes in the backward are
   sequence length and depth; there is no batch axis.

2. **"Memory ≈ constant floor + tiny linear term."** The probe shows it is
   **superlinear** in sequence length — and the reason turned out to be a
   huge *linear* term stacked per layer, not the quadratic one (see the GDN
   section).

Also worth stating because it removes a scary term: **attention is flash-style**
(MLX `sdpa`), so the `[heads, S, S]` scores are never *stored* — no quadratic
*activation-storage* term. But the backward still *recomputes* attention, so
there is a quadratic *transient* term in the backward.

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
  (the trainer calls `mx.clear_cache()` at every phase boundary for this
  reason) — otherwise the process stays at its high-water mark and macOS
  eventually starts paging.

## Measured: peak GiB vs sequence length (probe_backward.py)

micro_batch 1, rank-16 LoRA last-12, one fwd + one fwd/bwd per length,
`mx.clear_cache()` between — the exact training path, **stock GDN scan**
(the shipped default `gdn_serial` moves the whole curve, see below). Peak
includes the 22 GB weights.

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
  Without it a 96 GB box swaps at 1536; with it a *single-seq* probe fits to
  3072.
- **The curve is superlinear** on the stock path. Non-weight memory (ckpt):
  15.5 GB @1536 → 45.9 GB @3072 — ~3× for 2× tokens.
- **Probe fits ≠ live fits.** The probe is one sequence. Live training adds
  the `old_lp`/`ref_lp` numpy arrays, the un-freed rollout/gen KV, three
  forward passes, and hours of allocator creep — measured live peaks run
  **~5 GiB above** the probe at the same cap, and an arm that probed "just
  fits" can still swap-die mid-run. Budget accordingly.

## Where the compute goes — the update dominates as a phase ratio

Backward ≈ 2× forward FLOPs; grad-ckpt adds one recompute forward → ~3× a
forward. There is no 30× anywhere. "The update is ~79% of wall-clock" is a
**phase ratio**:
- Generation (all 24 completions): memory-bandwidth-bound decode, ~380 tok/s.
  ~36 K tokens ÷ 380 ≈ **~95 s/step**.
- Update: **six full-sequence, compute-bound passes** at micro_batch 1 —
  `old_lp` + `ref_lp` + policy fwd + backward(2×) + ckpt recompute — over 24
  rollouts × ~1900 tok. ≈ **290–640 s/step**.

So the update dominates because it pays 6 full-sequence passes at low
utilization (mb=1, MoE-gather) against one bandwidth-efficient generation pass —
not because any backward is disproportionate.

### Aside: batching decode on a sparse MoE undershoots the dense estimate

The batched SAGE beam (`engine.py::_sage_batched`) runs all 2m² step
expansions as rows of one forward per token. On a dense model, decode is
weight-bandwidth-bound, so B rows would cost ~B=1 wall-time (~8× at B=8).
On this sparse MoE it measures **~3.9× at B=8**: a batch of 8 tokens routes
to the *union* of their experts (~57 of 256), so expert weights are
re-loaded, not amortized. Measured decode (64 tok, fixed): per-row tok/s
falls 88.7 (B1) → 43.5 (B8) → 27.7 (B16); aggregate speedup 1.0× → 3.9× →
5.0×. The win comes from the dense share (30/40 GDN layers + full-attn +
router + shared expert + unembed) amortizing, plus GPU utilization vs
latency-bound B=1 — diminishing returns, so don't raise the beam width
expecting linear gains.

## ROOT CAUSE: the GDN training scan (12× attention per layer)

**`GatedDeltaNet.__call__` dispatches `use_kernel=not self.training`** — in
training mode every GDN layer abandons the fused Metal kernel (no VJP) for
`gated_delta_ops`: a Python loop over ALL T timesteps whose per-step fp32
recurrent state is [B, 32, 128, 128] = **2.10 MB/token** (the residual
stream is 4 KB/token — 500×). Under mx.grad every step's ~4 state-sized
temporaries stay live for the backward. Isolated per-layer
(`scripts/anatomy_gdn.py`, single synthetic layer with real config dims — no
weights needed):

| S | one GDN layer bwd | one full-attn layer bwd |
|---:|---:|---:|
| 1024 | 8.8 GiB | 1.0 |
| 2048 | 17.4 GiB | 1.4 |
| 4096 | 34.3 GiB | 3.1 |

**The scan is 12× attention.** This — not O(S²) attention, not the vocab
slab — is the sequence-memory wall. It explains every otherwise-confusing
measurement at once: why reducing LoRA depth 12→8 saves ~nothing while 8→4
saves ~9 GB (the peak is the GDN scan window, saturated at both larger
depths; one fewer scan overlapping at 4), why token-subset backprop doesn't
move the peak (trunk-side, below the head), and why the curve looks
"superlinear" (a huge linear term, 8.5 GB/1k-tokens/layer, stacked across
overlapping layer backwards).

**The MoE path is exonerated:** with the real 4-bit MoE mlp in the layer
(`anatomy_gdn.py --moe`), GatherQMM adds only ~0.4 GiB to the layer backward.
No hidden MoE term; the scan explanation stands.

## The fix: serial GDN scan (`src/mlx_rl/gdn_serial.py`, on by default)

The scan becomes an `mx.custom_function`: the forward runs the **fused Metal
kernel** (training forward = serving forward, and faster); the backward is a
hand-written VJP that walks the scan in `chunk`-sized segments in REVERSE,
recomputing each segment's forward-for-VJP on the fly from kernel-computed
boundary states.

Two mechanism subtleties, both isolated in `scripts/anatomy_sched.py`:
1. A naive segmented vjp graph is ALREADY cheap (1.4 GiB) when only one
   grad output is consumed — but inside the layer all five input grads are
   live outputs, and MLX's executor materializes one output concat tree at a
   time, so every segment's recomputed states stay resident for the later
   trees — the live set becomes the SUM over segments, not the max
   (17.4 GiB, unchanged from stock).
2. Fix: `mx.depends` each segment's recompute on ALL of the previous
   segment's grads (the full bundle, not just the state cotangent) → each
   segment retires completely before the next recompute starts. Live set =
   one segment.

| S | stock GDN layer | serial (chunk 64) | grads |
|---:|---:|---:|---|
| 1024 | 8.8 | **1.17** | ≡ chunked-exact (1.5e-8) |
| 2048 | 17.4 | **1.44** | ≡ (7.5e-9) |
| 4096 | 34.3 | **2.37** (14.5×) | ≡ (3.7e-9) |

Backward 2–3.5× faster too (kernel forward + fewer live buffers). ~Flat in
S. Composes with the outer per-layer mx.checkpoint. Forward |dy| vs ops
path = 1.6e-2 bf16 — the SAME kernel the serving path uses, so
training-forward now matches inference exactly (arguably a consistency fix;
old_lp/ref_lp/policy all shift together).

Integration: it patches `mlx_lm.models.qwen3_5.gated_delta_update`
(qwen3_5_moe reuses that module); masked calls delegate to stock; GQA repeat
lives outside the custom_function so autodiff owns its VJP. `gdn_serial:
bool = True` in the config, `--no-gdn-serial` to opt out, `--gdn-chunk` for
the segment length. Tests: `tests/test_gdn_serial.py` (fwd + all-five-primal
grads vs stock ops, mask delegation, install idempotence).

**Full-model probe with the fix** (probe_backward.py, last-4 rank-8 +
grad-checkpoint, micro_batch 1, serial ON):

| L | fwd_s | bwd_s | peak GiB | swap | stock |
|---:|---:|---:|---:|---:|---|
| 4096 | 2.0 | 6.6 | **25.4** | +0.0 | 86.0 GiB, 36.9 s |
| 6144 | 3.2 | 10.2 | **29.3** | +0.0 | (unrunnable) |
| 8192 | 4.4 | 13.9 | **33.6** | +0.0 | (unrunnable) |

3.4× less memory and 5.6× faster backward at 4096; ~1.9 GiB per extra 1k
tokens; **8192 tokens now costs less than 2560 did stock**. Forward scoring
passes (old_lp/ref_lp) are ~5× faster too (kernel prefill).

**Equivalence, verified at three levels:** per-layer grads vs stock ops
≤1.5e-8 (chunked-exact); on-model LoRA grads (real weights, identical
inputs) cosine 0.998, loss delta +1.1% — the bf16 kernel-vs-ops forward
difference propagating, i.e. noise-level agreement in the trainable
direction; and an 8-step serial-vs-stock training pair (seed 0, cap 1024)
with the same trajectory shape and rewards within one sample — while the
serial arm's updates ran **3.6× faster** (22–46 s vs 80–161 s) with
generation **27% faster** (455 vs 357 tok/s: training-mode prefill now hits
the fused kernel). End-to-end, a 120-step cap-4096 mixture run completed at
**avg step 219 s, peak 26.0 GiB, swap 0.0** — a 1.6× longer cap at half the
memory and 1.7× the speed of the best stock-scan configuration.

## Null results (measured, kept honest)

**LoRA depth is a SHORT-sequence lever only.** Probed `--lora-layers
{4,8,12}` × long lengths (rank 8, grad-ckpt, stock scan). Peak GiB:

| L (total) | last-12 | last-8 | last-4 |
|----------:|--------:|-------:|-------:|
| 2560 | 55.5 | ~53 | **50.0** |
| 3072 | 67.9 | 67.7 | **58.8** |
| 3584 | — | — | **71.5** |
| 4096 | — | 97.6 (swap) | — |

12→8 saves ~nothing at long S; 8→4 saves ~9 GB. The README's big
layer-count savings (85→49 GB) were at **576 tokens** — the depth lever has
strong leverage only when the per-adapted-layer backward is a big fraction
of the total, i.e. short sequences. At long S, memory is sequence-dominated
(the GDN scan above), which LoRA depth doesn't touch.

**Token-subset backprop does not move the peak.** The S-GRPO-style
selective path (`--token-subset-frac`; see below) was hypothesized to cut
the peak by shrinking the vocab-head/loss phase. Measured (probe, last-4
rank-8 ckpt, frac 0.35): dense 50.0/58.8/71.5 GiB at 2560/3072/3584 vs
subset **50.0/58.8/71.5** — identical to the tenth of a GiB. (Engagement is
not in doubt: the selective path's gathered arrays are shape-incompatible
with the dense code path, so running at all proves it took the selective
branch.) Conclusion: the backward's high-water mark is inside the trunk
(checkpoint-segment recompute + scan/dequant transients), not the
vocab-head/loss phase — the ~1.3–2.6 GB logits slab is real but sits
entirely below the peak. What survives of token-subset is the paper's
actual headline: **better LEARNING under LoRA at the same memory** (SVAMP
46→70 with 30–50% subsets) — a science lever, not a memory lever.

## Token-subset backprop (S-GRPO) — design notes

**Selection rule = uniform random over completion tokens, per row, exact
counts.** Principled, not just simple: the advantage is one scalar per
sequence (`advantages[:, None]` in `grpo_objective`), so the objective
carries NO per-token credit signal — a uniform subset with the denominator
scaled by `frac` is an unbiased estimator of the full-token gradient
(dropout-over-the-token-axis), and its variance cost is small because
within-sequence terms share the advantage. It is also the only rule with a
ground truth to validate against (frac=1.0 must reproduce the dense loss) —
entropy/"forking-token" selection (2506.01939-style) deliberately confounds
bias-by-design with the memory change. Paper context (2504.20834): under
LoRA, full-token GRPO barely beat base while a 30–50% subset lifted SVAMP
46→70+ — subsetting plausibly regularizes gradient interference from
boilerplate tokens on a low-rank adapter.

**Spec → code mapping:**
- `grpo.py: subsample_token_mask(mask, frac, rng)` — per-row
  `max(1, round(frac·n))` without replacement (never empties a row;
  inclusion prob deviates from frac by ≤ 1/(2n)). Pure numpy, unit-tested.
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
  through the selective head.

Tests (`tests/test_token_subset.py`): exact per-row counts & subset-of-mask;
frac=1 identity; gather padding masked; dense vs selective loss AND grads
equal at full selection; subset-loss mean within 4 SEM of the dense loss
over 400 draws.

## Does a longer budget buy learning? (measured: no, on this task mix)

With the memory wall down, cap 4096 was compared against cap 2560 on the
math/code/arithmetic mixture (120 vs 180 steps, last-4 rank-8): completions
grew to fill the room (mean length +~500 tokens; cap-hit rate 51.5%→34%),
and held-out `eval_correct` plateaued at the same **0.656** in both — with
~40% of steps still producing zero reward variance within every group
(no gradient). The bottleneck on this task mix is the **learning signal**,
not the sequence budget: the next levers are run-design (task-difficulty
curriculum so groups straddle the pass boundary, larger G, dead-prompt
resampling), not more memory.

Two metric gotchas for anyone reading their own `metrics.jsonl` here:
`eval_task_<name>` fields are the eval-set task *composition* (mixture.py
tags each example's task with weight 1.0), never per-task accuracy — the
correctness signal is `eval_correct`; and at `eval_n=32` a 0.56→0.66 move is
~3 problems, so raise `--eval-n` to 128+ before reading plateaus.

## Upstream landscape (checked 2026-07-14)

mlx-lm main pays the stock GDN training-scan cost for every qwen3.5/3.6
LoRA fine-tune, but two open PRs already target exactly this:
[#1217](https://github.com/ml-explore/mlx-lm/pull/1217) (Metal backward VJP
kernel + chunk-checkpointed Python fallback) and
[#1389](https://github.com/ml-explore/mlx-lm/pull/1389) (chunk-parallel
gated UT/WY formulation — the flash-linear-attention algorithm — per-chunk
mx.checkpoint, GQA/mask/carried-state support). Both active and
cross-verified by third parties (e.g. 27B QLoRA 117→39 GB, 3× throughput,
four-way grad agreement ≤2.6e-7).

Head-to-head on our anatomy harness (`scripts/pr1389/`, one training-mode
GDN layer, real qwen36 dims):

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
transients are small). Status: no third PR filed (redundant with two active
ones); `gdn_serial` ships here until one merges upstream, at which point it
can be deleted in favor of the upstream fix.

## Operational note

If you run the trainer with an external memory-lease command
(`MLX_RL_MEMLEASE_CMD`) that pauses another service for the duration of the
run: after release, verify the service is actually back with a real
completion probe, not just its status files.
