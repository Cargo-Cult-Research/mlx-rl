# Compute-Efficient RL on the Edge — reading queue for mlx-rl

Curated 2026-07-12 for the mlx-rl project (GRPO/LoRA RL on one M3 Ultra, 96 GB
unified). The thesis of mlx-rl is that RL-with-verifiable-rewards is
**rollout-dominated** — an episode carries ~1 bit (the reward), so updates are
cheap (low-rank LoRA) and nearly all compute is sampling. That makes an
inference-strong / training-weak Apple box a good fit. Every paper below buys
back compute on exactly that axis: fewer/cheaper rollouts, smaller updates, or
lower-precision math — no second GPU, no reward model, no weight replica.

The anchor is a paper we **already featured**, QeRL (2510.11696,
<https://arxiv.org/abs/2510.11696>): quantization noise *raises policy entropy*,
so it works as free exploration during RL — "noise as feature, not bug." That is
the NVIDIA/NVFP4 thread you flagged. These five extend it along the other cost
axes, ordered most-on-thesis first.

---

## 1. Token-Efficient RL for LLM Reasoning
Alan Lee, Harry Tong — <https://arxiv.org/abs/2504.20834>

**Gist.** Two critic-free variants of GRPO — S-GRPO and T-SPMO — that compute the
policy-gradient update on only a **small, informative subset of the generated
tokens** instead of every token. Explicitly built to stay compatible with LoRA
fine-tuning. On Qwen2-1.5B it lifts SVAMP accuracy from 46% to over 70% and works
on multi-digit multiplication, all under tight memory.

**Why it fits this machine.** This is the closest paper to what mlx-rl already
is: LoRA + critic-free group RL. The finding that *full-token* GRPO under LoRA
barely beats the base model, but a 30–50% token subset does, is directly
actionable — the backward pass retains activations per updated token, so
subsetting tokens is a lever on the same "LoRA depth is the update-memory knob"
budget we already track. Cheapest possible integration: change which token
positions contribute to the surrogate.

## 2. Act Only When It Pays: Efficient RL via Selective Rollouts (GRESO)
Haizhong Zheng, Yang Zhou, Brian R. Bartoldson et al. — <https://arxiv.org/abs/2506.02177>

**Gist.** A prompt's *informativeness* is temporally consistent: a prompt that
yields all-same-reward (zero-advantage) rollouts this epoch tends to stay that
way. GRESO is a lightweight online **pre-rollout filter** that predicts and skips
those prompts before spending any sampling on them, using reward training
dynamics. Result: up to **2.4× rollout / 2.0× total** wall-clock speedup with no
accuracy loss.

**Why it fits this machine.** mlx-rl already *drops* zero-variance groups
**after** paying to generate them (correctness rule #1). GRESO says don't pay in
the first place. On a single box where sampling is the whole cost, skipping dead
prompts up front is the highest-leverage change available and needs no kernel
work — just a cheap predictor over the reward history we already log.

## 3. Train Less, Learn More: Adaptive Efficient Rollout Optimization (AERO)
Zhi Zhang, Zhen Han, Costas Mavromatis et al. — <https://arxiv.org/abs/2602.14338>

**Gist.** Attacks the same zero-advantage waste from the *group* side: an
adaptive rollout strategy plus **selective rejection** to prune rollouts, and a
**Bayesian posterior** over per-prompt success probability to keep allocating
samples away from "dead zones" (groups that will collapse to zero advantage).
~48% less total training compute and ~45% less wall-clock per step, matching or
beating GRPO.

**Why it fits this machine.** Complements GRESO (prompt-level skip) with
group-level sample allocation — decide *how many* of the G members to draw per
prompt instead of a fixed G. The copy-on-write KV-cache group engine in mlx-rl
makes variable group size cheap to implement, and a Bayesian posterior over
reward is tiny state. This is the natural upgrade to the fixed-G sampler.

## 4. Efficient RL Training for LLMs with Experience Replay
Charles Arnal, Vivien Cabannes, Taco Cohen et al. — <https://arxiv.org/abs/2604.08706>

**Gist.** Reuse stored rollouts across multiple updates via a **replay buffer**,
rather than sampling fresh on-policy data every step. Formalizes buffer design as
a trade-off between staleness-induced variance, sample diversity, and generation
cost, and shows strict on-policy sampling is **suboptimal when generation is
expensive** — a well-designed buffer cuts inference compute without hurting final
performance or collapsing entropy.

**Why it fits this machine.** This is the deepest structural win for an
inference-strong/training-weak box: amortize each expensive rollout over several
gradient steps. mlx-rl is already positioned to do this legally — correctness
rule #2 recomputes `old_lp` teacher-forced at update time, which is exactly the
importance-ratio anchor an off-policy replay update needs (the same invariant
that already lets SAGE off-policy members enter the surrogate at ratio 1). Turns
"one sample = one update" into "one sample = k updates."

## 5. Quartet II: Accurate LLM Pre-Training in NVFP4
Andrei Panferov, Erik Schultheis, Soroush Tabesh et al. — <https://arxiv.org/abs/2601.22813>

**Gist.** The training-numerics side of the NVFP4 story. Introduces **MS-EDEN**,
a quantizer with >2× lower quantization error than stochastic rounding, giving
**unbiased NVFP4 gradients** for fully-quantized pre-training; up to **4.2×
speedup over BF16** validated to 1.9B params.

**Why it's here — and the honest caveat.** This is the *counterpoint* to QeRL,
not more of it. QeRL wants quantization noise (it aids RL exploration); Quartet II
wants to *remove the bias* in that noise so pre-training gradients stay
unbiased. Reading them together frames the real question for mlx-rl: which parts
of the loop tolerate — or benefit from — 4-bit noise (rollouts, exploration) vs.
which need clean gradients (the LoRA update). Caveat: it's pre-training, not RL,
and NVFP4 is an NVIDIA tensor-core format; on Apple the transferable idea is the
*error/bias analysis of low-bit training*, which maps onto MLX's own low-bit
paths, not the FP4 kernels themselves.

---

## Bonus / adjacent (not queued, worth a look)
- **ReQAT** — 4-bit FP QAT that recovers full-precision *reasoning* accuracy: <https://arxiv.org/abs/2606.15682>
- **Rollout-Level Advantage-Prioritized Experience Replay for GRPO** — prioritized replay variant of #4: <https://arxiv.org/abs/2606.04560>
- **QeRL (already featured)** — the noise-as-exploration anchor: <https://arxiv.org/abs/2510.11696>

## The through-line for mlx-rl
Three orthogonal cost axes, all runnable on one box:
1. **Cheaper updates** — subset the tokens you backprop (#1).
2. **Cheaper rollouts** — skip dead prompts (#2), allocate samples adaptively (#3), reuse samples off-policy (#4).
3. **Cheaper math** — low-bit training, and knowing where its noise helps vs. hurts (#5, plus the featured QeRL).

Highest expected value to try first: **#2 (GRESO)** and **#4 (replay)** — both
target sampling, which is ~all of mlx-rl's wall-clock, and both reuse invariants
(zero-variance detection; teacher-forced `old_lp`) the codebase already has.
