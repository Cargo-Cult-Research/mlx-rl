"""Model profiles: the per-model facts the trainer needs.

Two first-class citizens (both MoE, both trainable within the ~30 GB weight
ceiling this box measured, both potential daily drivers): qwen36 and gemma26.
Building for both from the start keeps the engine honest — qwen36 exercises
the hybrid GatedDeltaNet/ArraysCache path, gemma26 the sliding-window
RotatingKVCache path.
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
    # Gemma 4 26B-A4B (gemma4): sliding-window RotatingKVCache + periodic
    # full attention; standard proj names. extra_eos carries the config EOS
    # list {1, 106, 50} — the PR-610 lesson: lose it and <turn|> runs away.
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
}


def get_profile(name: str) -> ModelProfile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile {name!r}; available: {sorted(PROFILES)}")
    return PROFILES[name]
