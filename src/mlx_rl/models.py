"""Model loading, LoRA attachment, reference-policy switching, checkpoints."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_load
from mlx_lm.tuner.utils import linear_to_lora_layers

from .config import DEFAULT_LORA_KEYS, LoraConfig
from .memory import assert_fits, estimate_run_gb, model_disk_gb


def resolve_model_path(model_id: str) -> Path:
    """Local path passes through; a HF repo id is downloaded (cache-first)
    so the memory guard can size the weights BEFORE anything is loaded."""
    p = Path(model_id).expanduser()
    if p.exists():
        return p
    return Path(snapshot_download(model_id))


class VLMTextPolicy(nn.Module):
    """An mlx-vlm multimodal model driven through the mlx-lm text-model
    contract the rest of this trainer speaks: `model(inputs, cache=...)` ->
    bare logits, `.layers`, `.make_cache()`, `.language_model` nesting.

    Text rollouts call straight into the language model (which self-computes
    its per-layer embeddings from token ids); the frozen towers ride along
    in the module tree so Phase-2 multimodal prefill is a forward-path
    change, not a reload. Adapter weights saved from this wrapper carry a
    leading `vlm.` on their keys.
    """

    def __init__(self, vlm: nn.Module):
        super().__init__()
        self.vlm = vlm

    @property
    def language_model(self):
        return self.vlm.language_model

    @property
    def layers(self):
        return self.vlm.language_model.layers

    def make_cache(self):
        return self.vlm.language_model.make_cache()

    def __call__(self, inputs: mx.array, cache=None, mask=None):
        out = self.vlm.language_model(inputs, cache=cache, mask=mask)
        return out.logits if hasattr(out, "logits") else out


def load_policy(
    model_id: str,
    lora: LoraConfig,
    headroom_gb: float = 4.0,
    grad_checkpoint: bool = False,
    required_gb: float = 0.0,
    vlm: bool = False,
):
    """Load base weights, freeze them, attach trainable LoRA adapters.

    Returns (model, tokenizer, info). Raises MemoryGuardError instead of
    loading a model that would not fit next to what is already resident.

    required_gb > 0 overrides the worst-case estimator — for workloads far
    from its calibration regime (e.g. 1-token rollouts). The claim is
    checked, not trusted: assert_fits still gates the load and the SwapGuard
    hard-aborts the run if the real peak proves the override wrong.

    vlm=True loads a multimodal checkpoint via mlx-vlm and wraps it in
    VLMTextPolicy; the tokenizer still comes from mlx-lm's loader so the
    trainer sees its usual TokenizerWrapper (eos_token_ids etc.).
    """
    path = resolve_model_path(model_id)
    weights_gb = model_disk_gb(path)
    assert_fits(required_gb or estimate_run_gb(weights_gb, headroom_gb))

    if vlm:
        from mlx_lm.utils import load_tokenizer
        from mlx_vlm import load as vlm_load

        inner, _processor = vlm_load(str(path))
        # mlx-vlm gemma4's audio tower stores AudioRelativePositionEmbedding
        # as a private `_rel_pos` attribute; those instances end up without
        # nn.Module's bookkeeping attrs and crash freeze(). Repair the
        # bookkeeping — they're frozen inference-only modules either way.
        for _, mod in inner.named_modules():
            if not hasattr(mod, "_no_grad"):
                object.__setattr__(mod, "_no_grad", set())
            if not hasattr(mod, "_training"):
                object.__setattr__(mod, "_training", True)
        model = VLMTextPolicy(inner)
        # Multimodal consumers (audio/vision prompts) need the processor for
        # feature extraction; plain attribute, invisible to the param tree.
        object.__setattr__(model, "processor", _processor)
        tokenizer = load_tokenizer(path)
    else:
        model, tokenizer = mlx_load(str(path))
    model.freeze()
    num_layers = lora.num_layers if lora.num_layers > 0 else len(model.layers)
    linear_to_lora_layers(
        model,
        num_layers,
        {
            "rank": lora.rank,
            "scale": lora.scale,
            "dropout": lora.dropout,
            "keys": lora.keys or DEFAULT_LORA_KEYS,
        },
    )
    # Training mode matters beyond dropout: switch_layers only stop-grads the
    # expert-routing indices when self.training is set, and without that the
    # backward pass dies in GatherQMM::vjp ("gradient wrt the indices").
    model.train()
    if grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint as _ckpt

        # _ckpt patches type(layer).__call__ — once per distinct layer class
        # (hybrid models like qwen3_5_moe mix GatedDeltaNet and full-attn
        # blocks), never twice, or the wrapper nests.
        seen: set[type] = set()
        for layer in model.layers:
            if type(layer) not in seen:
                _ckpt(layer)
                seen.add(type(layer))
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    info = {
        "model_path": str(path),
        "weights_gb": round(weights_gb, 2),
        "trainable_params": n_trainable,
        "lora_layers": num_layers,
        "grad_checkpoint": grad_checkpoint,
    }
    return model, tokenizer, info


def selective_logprobs(
    model, inp: mx.array, tgt_sel: mx.array, sel_idx: mx.array
) -> mx.array:
    """Per-token logprobs at selected positions only.

    Runs the trunk on the full sequence (attention needs every position),
    then gathers hidden states at `sel_idx` [B, K] BEFORE the vocab
    projection — so the 248k-vocab logits/softmax slab is [B, K, V] instead
    of [B, L, V]. `tgt_sel` [B, K] must be pre-gathered with the same
    indices. Numerically identical per position to the dense path (softmax
    is row-wise); only the head's input shrinks.
    """
    # VLM-style wrappers (qwen3_5/qwen3_5_moe) nest the CausalLM one level
    # down as .language_model; plain models (qwen3_next) ARE the CausalLM.
    lm = getattr(model, "language_model", model)
    h = lm.model(inp)  # [B, L, hidden] — full-seq trunk, incl. final norm
    h = mx.take_along_axis(h, sel_idx[..., None], axis=1)  # [B, K, hidden]
    if hasattr(lm, "logits_from_hidden"):
        # mlx-vlm gemma4 exposes the full head (tied embeddings + softcap)
        logits = lm.logits_from_hidden(h)
        return -nn.losses.cross_entropy(logits, tgt_sel, reduction="none")
    if getattr(lm.args, "tie_word_embeddings", False):
        logits = lm.model.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)
    # mlx-lm gemma4_text applies final-logit softcapping inline in __call__;
    # skipping it here shifted candidate logprobs by up to 28 nats on E4B
    # (caught 2026-08-03 by the text-vs-vlm A/B rig).
    softcap = getattr(lm, "final_logit_softcapping", None)
    if softcap:
        logits = mx.tanh(logits / softcap) * softcap
    return -nn.losses.cross_entropy(logits, tgt_sel, reduction="none")


def lora_modules(model):
    return [m for _, m in model.named_modules() if hasattr(m, "lora_a") and hasattr(m, "scale")]


@contextmanager
def adapters_disabled(model):
    """Zero every adapter's multiplicative scale: the forward pass becomes the
    frozen base model, which is exactly the GRPO reference policy — no second
    copy of the weights in memory."""
    mods = lora_modules(model)
    saved = [m.scale for m in mods]
    for m in mods:
        m.scale = 0.0
    try:
        yield
    finally:
        for m, s in zip(mods, saved):
            m.scale = s


def save_adapter(model, out_dir: str | Path, lora: LoraConfig, model_id: str, step: int) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"adapter-{step:05d}.safetensors"
    mx.save_safetensors(str(fname), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "rank": lora.rank,
                "scale": lora.scale,
                "dropout": lora.dropout,
                "num_layers": lora.num_layers,
                "keys": lora.keys or DEFAULT_LORA_KEYS,
            },
            indent=2,
        )
        + "\n"
    )
    return fname
