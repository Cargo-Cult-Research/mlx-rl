"""Run configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Attention projections exist under these names in every architecture we
# target (qwen, gemma, llama families). MoE expert MLPs are deliberately not
# defaulted in; add them via `keys` when experimenting.
DEFAULT_LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]


@dataclass
class LoraConfig:
    rank: int = 16
    scale: float = 20.0
    dropout: float = 0.0
    num_layers: int = -1  # -1 = every transformer block
    keys: list[str] | None = None  # None = DEFAULT_LORA_KEYS


@dataclass
class TrainConfig:
    model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    task: str = "arithmetic"
    task_kwargs: dict = field(default_factory=dict)
    # Extra chat-template kwargs, e.g. {"enable_thinking": False} for Qwen3.x
    chat_kwargs: dict = field(default_factory=dict)
    profile: str | None = None  # named ModelProfile that supplied the defaults
    extra_eos: tuple = ()  # additional stop token ids (gemma: config EOS list)
    share_prompt: bool = True  # prefill group prompt once, fork KV per member
    manage_machine: bool = True  # take the host memory lease for the run
    lease_wait_s: float = 0.0  # how long to wait if another agent holds it
    rollout_batch_size: int = 64  # continuous-batching completion batch cap
    steps: int = 100
    batch_prompts: int = 4  # prompts per optimizer step
    group_size: int = 8  # completions per prompt (the G in GRPO)
    # Two-stage rollouts (GRESO 2506.02177 / AERO 2602.14338 spirit): sample
    # group_stage1 members first and abandon the group — no further members,
    # no SAGE beams — when stage-1 is already decided (see stage1_skip).
    # v4 measured 45-48% of generation spent on zero-advantage groups. 0 = off.
    group_stage1: int = 0
    stage1_skip: str = "saturated"  # saturated: uniform AND >=0.99 | uniform: any all-equal
    # Selective-rejection update (AERO / token-efficient GRPO 2504.20834):
    # skip the backward for rollouts with |advantage| < frac * group max.
    # The loss denominator stays the FULL active batch, so retained terms
    # keep their original weight (truncation, not reweighting). 0 = off.
    update_adv_frac: float = 0.0
    # Token-subset backprop (S-GRPO, arXiv:2504.20834): compute the surrogate
    # (and the vocab-head logits) on a uniform random fraction of each
    # rollout's completion tokens; denominator scaled to keep the estimator
    # unbiased. 0 = off (dense path). 1.0 = selective path over ALL
    # completion tokens — same loss as dense, but the head skips prompt/pad
    # positions (equivalence-test setting).
    token_subset_frac: float = 0.0
    micro_batch: int = 4  # sequences per backward pass (grad accumulation)
    epochs_per_batch: int = 1  # >1 makes the PPO clip active
    max_new_tokens: int = 128
    # Sampling logprobs are the model's own (temp=1) distribution; a different
    # rollout temperature introduces an off-policy mismatch the ratio does not
    # correct for. Keep 1.0 for training runs.
    temperature: float = 1.0
    lr: float = 1e-5
    kl_coef: float = 0.01
    clip_eps: float = 0.2
    normalize_std: bool = True  # GRPO-style; False = mean-only (Dr. GRPO)
    # SAGE-RL (arXiv 2602.08354) hybrid rollouts: sage_r of every group's
    # members are generated with SAGE confidence-guided decoding (short,
    # decisive chains), the rest with ordinary sampling. Requires think_end
    # (from the model profile). 0 disables.
    sage_r: int = 0
    sage_m: int = 2  # beam width (exploration width); paper default 2
    # Tolerance ratio TR = h/(2m): </think> accepted when it lands in the top-h
    # by Φ. Paper stable over {0.5, 0.75, 1.0}; 0.5 = strictest confidence gate.
    sage_tr: float = 0.5
    sage_max_reasoning_steps: int = 64  # T_max: reasoning-step budget
    sage_max_step_tokens: int = 512     # per-step token cap (unbatched path)
    sage_step_tokens: int = 48          # fixed step-chunk size (batched path, ~5x)
    sage_think_temperature: float = 1.0  # step-sampling temp (paper: 1.0)
    # Answer-phase token reserve: SAGE reasoning is capped at
    # max_new_tokens - sage_answer_reserve so the answer phase always has
    # room. Without it the reasoning loop could fill (or overshoot — an
    # observed think_len 2069 > cap 2048 bug) the whole budget and emit
    # ZERO visible answer tokens, which grades 0 under think-aware reward.
    sage_answer_reserve: int = 256
    # Off-policy demonstration injection: the last inject_r members of every
    # group are replaced by the task's `injected_completion(example)` text
    # (e.g. qa_abstain injects "<abstain/>"). Solves the exploration gap when
    # a rewarded behavior has ~zero probability under the base policy — GRPO
    # cannot reinforce what is never sampled. Legal for the same reason SAGE
    # members are: old_lp is recomputed teacher-forced, so injected members
    # enter the clipped surrogate at ratio 1 (correctness detail 2).
    inject_r: int = 0
    think_end: int | None = None  # end-of-thinking token id (profile)
    # Correctness-gated total-length efficiency (anti reasoning-relocation
    # hack): final = base * (1 - length_penalty * min(1, tokens/budget)).
    # Length only helps when the answer is CORRECT (wrong stays 0, so it can't be
    # gamed by truncation-guessing) and it counts TOTAL tokens (so moving reasoning
    # out of <think> into prose buys nothing — a hack RL reliably finds).
    length_penalty: float = 0.0  # 0 = off
    length_budget: int = 0       # 0 = use max_new_tokens
    # Gradient checkpointing (mlx_lm.tuner.trainer.grad_checkpoint): recompute
    # layer forwards during backward instead of retaining activations.
    # Numerically identical; trades ~one extra forward through the adapted
    # layers for the activation mass that puts the plain backward at 83 GiB
    # for a single 1536-token sequence (scripts/probe_backward.py) — the
    # enabler for caps past 1024 on 96 GiB.
    grad_checkpoint: bool = False
    # Serial GDN scan (gdn_serial.py): custom-VJP GatedDeltaNet backward,
    # memory bounded at ~one chunk of scan states instead of 2.1 MB x S per
    # GDN layer — the root-cause fix for the sequence-memory wall
    # (34.3 -> 2.37 GiB per layer @4096 in isolation). Also makes training
    # forwards (old_lp/ref_lp scoring, prefill) use the fused kernel.
    gdn_serial: bool = True
    gdn_chunk: int = 64  # scan segment length (memory floor vs recompute count)
    max_grad_norm: float = 1.0  # 0 disables clipping
    eval_every: int = 10
    eval_n: int = 32
    # Eval-only length cap (0 = max_new_tokens). Eval is generation-only —
    # no backward pass, no activation peak — so it may safely exceed the
    # training cap; running it higher measures whether learned termination
    # generalizes past the training-time budget wall.
    eval_max_new_tokens: int = 0
    checkpoint_every: int = 20
    seed: int = 0
    activation_headroom_gb: float = 4.0
    # Swap watchdog: hard-abort if system swap grows this many GB above the
    # baseline captured at run start (a backward spilling to SSD makes a step
    # 10-100x slower — fail loud, not slow). 0 disables. See memory.SwapGuard.
    swap_guard_margin_gb: float = 3.0
    lora: LoraConfig = field(default_factory=LoraConfig)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
