# SAGE beam batching — plan + upstream investigation (2026-07-11)

## Why: SAGE decoding is the slow part

`sage_completion` runs its `2m²` step-expansions as separate **B=1** forwards
(one token at a time). That's the dominant cost of SAGE members in RL rollouts
(~40–120s each on qwen36) and of the planned SWE-bench SAGE-decoding oracle
(8× per agent turn). Sampled rollouts are already batched (`BatchGenerator`,
~391 tok/s aggregate) and reuse the shared-prompt KV — SAGE is the outlier.

## Upstream question: should the batching be a PR? **No.**

- **mlx_lm 0.31.3 already ships the batched-beam primitives.** `BatchKVCache`
  has `merge` (fork/combine caches, left-padded for ragged candidate lengths),
  `filter(batch_indices)` (in-place prune to the surviving rows), and
  `extract(idx)` (pull one row out as a `KVCache` for the answer phase). That is
  exactly the fork / prune / select a step-wise beam needs. Our batching is pure
  **application code on top of existing mlx_lm APIs** — nothing to add upstream.
- **SAGE decoding itself is a paper-port** (arXiv 2602.08354; no released
  reference impl, PyTorch-oriented). mlx_lm deliberately carries only standard
  samplers, not paper-specific decode strategies, so a "step-wise confidence
  beam" helper wouldn't belong there either. It stays in mlx-rl.

Conclusion: implement in `engine.py`, no upstream PR.

## The qwen36 wrinkle (why it's non-trivial, but bounded)

qwen36 is a **hybrid GatedDeltaNet / full-attention MoE**: FA layers use
`KVCache` (→ `BatchKVCache.merge`), but GDN layers use recurrent-state
`ArraysCache` (no `BatchKVCache` equivalent). The model already forwards at
batch B for the sampled rollouts (B=18–24 through the GDN/FA stack), so the
layers DO support a batch dim — we just need to fork/stack both cache kinds:
- FA layers: `BatchKVCache.merge([cand_cache]*2m …)`.
- GDN layers: stack the per-candidate `ArraysCache` state arrays along batch 0.
Then one batched forward per token replaces the `2m²` sequential ones.

## STATUS: DONE (2026-07-11)

Implemented `_sage_batched` in `engine.py`; `sage_completion(batched=True)` is the
default. **Correctness parity** vs the unbatched reference on qwen36 arithmetic
(all ✓). **Speedup: ~3.9× at B=8, NOT the ~8× a dense model gives** — this is a
sparse MoE (35B-A3B), so a batch of 8 tokens routes to the *union* of their
experts (~57 of 256), and those expert weights are re-loaded, not amortized.
Measured decode (64 tok, fixed): per-row tok/s falls 88.7 (B1) → 43.5 (B8) → 27.7
(B16); aggregate 1.0× → 3.9× → 5.0×. The win comes only from the DENSE share
(30/40 GDN layers + FA + router + shared expert + unembed) amortizing, plus GPU
utilization vs latency-bound B=1. Diminishing returns → don't raise `m` (→ larger
B) expecting linear gains. (An earlier "~5×" claim was wrong: it compared total
wall-time when the batched runs stopped earlier, on a dense-bandwidth assumption.)
Two implementation decisions:
- **Fixed `step_tokens` chunks** (default 48) instead of `\n\n`-delimited steps,
  so the batch is rectangular; `</think>` is still detected mid-chunk. (The
  unbatched reference keeps `\n\n` steps and is retained as `batched=False`.)
- **GDN `ArraysCache` can't be trimmed** (recurrent state), so instead of
  trimming the winner's over-generated cache we **re-prefill** its clean token
  sequence once (`_answer_from`). `merge`=fork, `filter(idxs)`=replicate+prune.
Tests: 47 pass incl. batched end-to-end + hybrid-rollout on the tiny model.

## Original plan (queued: A/B run → THIS → oracle)

1. Add `_expand_batched(model, beam, …)` using `BatchKVCache` for FA + stacked
   state for GDN; replaces the `for cand: for _ in range(2m): _grow_step` loop.
   Keep the current unbatched `_grow_step` as the reference path, selectable by a
   `batched=True/False` flag on `sage_completion`.
2. **Parity verify on GPU** (needs the model, so after the A/B frees it): with a
   fixed seed, batched and unbatched must produce the same expansions /
   acceptance and matching Φ; measure the speedup (expect several×).
3. Only then use batched decoding for the SWE-bench oracle. If parity is off,
   the oracle falls back to the (correct) unbatched decoder.

The acceptance logic (`_accept`, top-h) is unchanged and already unit-tested —
batching only changes HOW the expansions are generated, not the SAGE rule.
