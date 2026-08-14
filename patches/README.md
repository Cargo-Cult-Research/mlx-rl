# patches/

Local fixes to **mlx-lm 0.31.3** (pinned in `pyproject.toml`), which this
project depends on for generation and serving. The files they patch live in
`.venv/lib/python3.12/site-packages/mlx_lm/` — build output, not source
control — so **`uv sync` or any venv rebuild silently reverts them**.

```sh
./scripts/apply_patches.sh     # idempotent; run after uv sync
```

The script refuses to run if the installed mlx-lm is not 0.31.3: these are
line-level patches and an upgrade needs each one re-verified (or dropped, if
upstream has fixed it).

## mlx-lm-0.31.3-models-gemma2.patch

**Symptom:** any batched generation with a `gemma-2-*` model dies with
`ValueError: [broadcast_shapes] Shapes (B,1,L,S) and (B,n_kv,repeats,L,S)
cannot be broadcast`. Batch size 1 works, so the bug only appears once you
batch — i.e. in every rollout and eval this repo does.

**Cause:** under grouped-query attention (`repeats > 1`) `Attention.__call__`
reshapes queries to `(B, n_kv_heads, repeats, L, head_dim)` and expands
keys/values, so `scores` is 5-D. Masks arrive 4-D, `(B, 1, L, S)`.
Right-aligned broadcasting pads the mask to `(1, B, 1, L, S)`, lining `B` up
against `n_kv_heads` — which coincidentally succeeds when `B == 1` (the
padded leading 1 broadcasts against anything) and raises for any real batch.

**Fix:** insert the head-group axis so the mask matches the 5-D layout,
handling both the usual `(B, 1, L, S)` mask and a per-head
`(B, n_heads, L, S)` one.

**Verified:** with the patch, batched greedy output is byte-identical to the
previously-working unbatched path across 6 prompts (a mis-applied mask
changes attention, so identical tokens are strong evidence of correctness);
`group_size=8, temp=1.0` yields 8 distinct completions per group; batch-32
runs clean. Only `gemma2.py` is touched, so no other architecture is
affected.

## mlx-lm-0.31.3-server.patch

**Symptom:** `mlx_lm.server --adapter-path ...` serves the **base model** —
the adapter is silently ignored on every request. Nothing errors; scores just
come back at base-model level, which is a very easy way to "validate" the
wrong weights.

**Cause:** `ModelProvider.load()` remaps `model_path` through `_model_map`
*before* looking the adapter up in `_adapter_map`. The adapter map is keyed by
the original name (`"default_model"`), so after the remap the lookup always
misses.

**Fix:** do the adapter lookup before remapping the model path.

**Note:** request the model as `"default_model"` to get the CLI-specified
adapter. Requesting it by its real repo id resolves to a no-adapter entry and
serves the base model — which is a useful way to A/B base vs adapter from one
server, but a trap if you assume the name is cosmetic.

## Upstream

Neither is reported upstream yet (<https://github.com/ml-explore/mlx-lm>).
Both are self-contained with easy reproducers: gemma2 needs any `gemma-2-*`
model and `batch >= 2`; the server one needs any `--adapter-path` plus a
request whose behaviour you can distinguish from the base model.
