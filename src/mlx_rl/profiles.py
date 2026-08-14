"""Model profiles: the per-model facts the trainer needs.

Two first-class citizens (both MoE, both trainable within the ~30 GB weight
ceiling a 96 GB machine affords): qwen36 and gemma26. Building for both
keeps the engine honest — qwen36 exercises the hybrid
GatedDeltaNet/ArraysCache path, gemma26 the sliding-window RotatingKVCache
path.

NOTE: the qwen36/gemma26 profiles point at LOCAL model directories under
MLX_RL_MODELS_DIR (default ~/models/mlx) — they expect you to have converted
or downloaded MLX 4-bit weights there yourself (e.g. with mlx_lm.convert).
Only the `tiny` profile references a Hugging Face repo id that downloads
automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Local model weights live under this directory (a real dir or a tree of
# symlinks into the HF cache). Override with MLX_RL_MODELS_DIR; defaults to
# ~/models/mlx, which expands to the same location the profiles historically
# hardcoded.
MODELS_DIR = os.environ.get("MLX_RL_MODELS_DIR", os.path.expanduser("~/models/mlx"))


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    lora_keys: tuple[str, ...] | None = None  # None = config.DEFAULT_LORA_KEYS
    chat_kwargs: dict = field(default_factory=dict)
    extra_eos: tuple[int, ...] = ()
    # SAGE (arXiv 2602.08354) needs the end-of-thinking token id and the
    # chat kwargs that put the template into thinking mode.
    think_end: int | None = None
    think_chat_kwargs: dict = field(default_factory=dict)
    # Multimodal checkpoint loaded via mlx-vlm (VLMTextPolicy wrapper);
    # text-only rollouts, towers frozen in the tree for Phase-2 audio.
    vlm: bool = False


_ATTN = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)

PROFILES: dict[str, ModelProfile] = {
    # Test rig: fast, dense, plain KVCache.
    "tiny": ModelProfile(
        name="tiny",
        model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    ),
    # Qwen 3.6 35B-A3B (qwen3_5_moe): hybrid linear attention — most layers
    # GatedDeltaNet (ArraysCache), every Nth full attention. Default q_proj
    # keys match nothing on the linear layers; target both kinds.
    "qwen36": ModelProfile(
        name="qwen36",
        model=os.path.join(MODELS_DIR, "Qwen3.6-35B-A3B-4bit"),
        lora_keys=_ATTN
        + (
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ),
        chat_kwargs={"enable_thinking": False},
        # thinking-on template pre-opens <think>\n; </think> = 248069.
        think_end=248069,
        think_chat_kwargs={"enable_thinking": True},
    ),
    # Qwen3.8-27B (qwen3_5, released 2026-08-14): the DENSE sibling of the
    # qwen36 profile above — same hybrid 3:1 GatedDeltaNet:full-attention
    # layout (64 layers, full_attention_interval=4), so the same two families
    # of LoRA keys apply. Checkpoint is multimodal; mlx-lm drops the vision
    # tower on load, so this is the text model only.
    "qwen38": ModelProfile(
        name="qwen38",
        model=os.path.join(MODELS_DIR, "Qwen3.8-27B-4bit"),
        lora_keys=_ATTN
        + (
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ),
        chat_kwargs={"enable_thinking": False},
        think_chat_kwargs={"enable_thinking": True},
    ),
    # Gemma 4 26B-A4B (gemma4): sliding-window RotatingKVCache + periodic
    # full attention; standard proj names. extra_eos carries the config EOS
    # list {1, 106, 50} — lose it and generation never stops (<turn|> runaway).
    # ALWAYS thinks: the mlx conversion's chat template ignores
    # enable_thinking (and thinking-off gemma26 is the known-pathological
    # serving mode anyway), and it ruminates — budget max_new_tokens >= 768
    # for arithmetic-class tasks or every completion truncates at reward 0.
    "gemma26": ModelProfile(
        name="gemma26",
        model=os.path.join(MODELS_DIR, "gemma-4-26b-a4b-it-4bit"),
        lora_keys=_ATTN,
        extra_eos=(1, 106, 50),
        # Thought channel closes with <channel|> = 101 (default template
        # already thinks; no extra chat kwargs needed).
        think_end=101,
    ),
    # gemma-4-E4B, the ears-project model (gemma4 E-series LM: altup,
    # per-layer embeddings, 18 KV-shared layers — those have no k/v_proj of
    # their own, attn-key LoRA simply skips them there; plus the audio/vision
    # towers, frozen). Loaded via mlx-vlm so Phase-2 audio RL is a
    # forward-path change, not an infra change.
    "e4b": ModelProfile(
        name="e4b",
        model=os.path.join(MODELS_DIR, "gemma-4-E4B-it-8bit"),
        lora_keys=_ATTN,
        extra_eos=(1, 106, 50),
        vlm=True,
    ),
    # Text-only extraction of the same weights (ears/tools/
    # convert_e4b_text.py), verified bit-identical to the vlm language model
    # 2026-08-03 (max |dlogp| = 0.0). Reference/control rig: pure mlx-lm
    # path, no towers, no mlx-vlm involvement.
    "e4b-text": ModelProfile(
        name="e4b-text",
        model=os.path.join(MODELS_DIR, "gemma-4-E4B-it-8bit-text"),
        lora_keys=_ATTN,
        extra_eos=(1, 106, 50),
    ),
}


def get_profile(name: str) -> ModelProfile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile {name!r}; available: {sorted(PROFILES)}")
    return PROFILES[name]
