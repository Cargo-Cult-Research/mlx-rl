"""GRPO training loop and CLI.

Usage:
    uv run mlx-rl-train --steps 30 --out runs/smoke
    uv run mlx-rl-train --model <hf-repo-or-path> --task arithmetic ...
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import random
import time
from collections import deque
from dataclasses import asdict, fields, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from . import machine
from .config import LoraConfig, TrainConfig
from .engine import (Completion, Episode, Segment, _tap, rollout_episodes,
                     rollout_groups, sage_completion)
from .grpo import (
    active_groups,
    group_advantages,
    grpo_objective,
    subsample_token_mask,
    token_logprobs,
)
from .memory import SwapGuard, swap_used_gb, write_abort_marker
from .models import adapters_disabled, load_policy, save_adapter, selective_logprobs
from .profiles import get_profile
from .rfcs import rfcs
from .rollout import (
    Rollout,
    build_training_arrays,
    encode_prompt,
    gather_selected,
    score_logprobs,
    tool_response_ids,
)
from .tasks import get_task
from .tasks.base import Example
from .toolcall import parse_tool_call


def _sample_batched(model, tokenizer, examples, cfg: TrainConfig, group_size, temperature, task=None):
    """Batched rollout over examples; returns (per-example completion groups,
    prompt token lists, BatchStats)."""
    chat_kwargs = {**getattr(task, "chat_template_kwargs", {}), **cfg.chat_kwargs}
    prompts = [encode_prompt(tokenizer, ex.messages, **chat_kwargs, **ex.chat_kwargs)
               for ex in examples]
    groups, stats = rollout_groups(
        model,
        tokenizer,
        prompts,
        group_size,
        cfg.max_new_tokens,
        temperature,
        extra_eos=tuple(cfg.extra_eos),
        share_prompt=cfg.share_prompt,
        completion_batch_size=cfg.rollout_batch_size,
    )
    return groups, prompts, stats


def _step_delim_ids(tokenizer) -> set[int]:
    """Token ids that end a reasoning step (paper delimits steps by '\\n\\n').
    Cached on the tokenizer; the fallback vocab scan runs once, only if '\\n\\n'
    is not a single token."""
    cached = getattr(tokenizer, "_sage_step_delim", None)
    if cached is not None:
        return cached
    enc = tokenizer.encode("\n\n", add_special_tokens=False)
    if len(enc) == 1:
        ids = set(enc)
    else:
        ids = set()
        try:
            for i in set(tokenizer.get_vocab().values()):
                if tokenizer.decode([i]).endswith("\n\n"):
                    ids.add(i)
        except Exception:
            pass
        ids = ids or set(enc[-1:])
    tokenizer._sage_step_delim = ids
    return ids


def _completion_text(tokenizer, comp) -> str:
    toks = comp.tokens
    if comp.finish_reason == "stop" and toks:
        toks = toks[:-1]  # drop the terminating EOS from the visible text
    return tokenizer.decode(toks)


def _think_close_marker(tokenizer, cfg: TrainConfig, task) -> str | None:
    """The decoded end-of-thinking marker when this run generates in thinking
    mode, else None (grade the full text)."""
    chat_kwargs = {**getattr(task, "chat_template_kwargs", {}), **cfg.chat_kwargs}
    if cfg.think_end is None or not chat_kwargs.get("enable_thinking"):
        return None
    return tokenizer.decode([cfg.think_end])


def _visible_reply(text: str, think_close: str | None) -> tuple[str, bool]:
    """What a user of the model would actually see, and whether the think
    block closed. Thinking-mode completions are graded ONLY on the text after
    the FINAL think-close marker: rollouts that hit the token cap inside an
    unclosed <think> can otherwise grade as passes off draft `<answer>` tags
    inside the CoT — a reward-hack vector (ramble forever, drop tags in the
    ramble). Unclosed think = no visible reply = reward 0."""
    if not think_close:
        return text, True
    if think_close in text:
        return text.rsplit(think_close, 1)[1], True
    return "", False


def _stage1_dead(rewards: list[float], mode: str) -> bool:
    """Is a stage-1 group already decided? 'saturated' abandons only groups
    uniform at ~max reward (safe: an all-wrong stage-1 can still yield signal
    from later members or a SAGE rescue); 'uniform' abandons ANY all-equal
    stage-1 group (GRESO-aggressive: all-same-reward prompts tend to stay
    that way). Judged on base rewards, before the length gate."""
    if max(rewards) - min(rewards) >= 1e-6:
        return False
    return mode == "uniform" or min(rewards) >= 0.99


def _grade_batch(task, pairs):
    """Grade [(example, visible_text), ...] -> [RewardResult, ...], through
    task.batch_reward when the task defines it (one judge call per batch for
    judge-graded frames) and per-item reward() otherwise."""
    br = getattr(task, "batch_reward", None)
    if br is not None:
        return br([ex for ex, _ in pairs], [v for _, v in pairs])
    return [task.reward(ex, v) for ex, v in pairs]


def _two_stage(examples, cfg: TrainConfig, n_sampled: int, sample, grade, total_of):
    """Two-stage rollouts (cfg.group_stage1 > 0), shared by the completion and
    episode collectors: sample group_stage1 members, grade them, abandon the
    groups that are already decided (_stage1_dead), then sample the rest for
    the survivors only. Abandoned groups' rewards are kept on the stats so
    reward_mean_all stays honest -- survivors-only means lie downward as the
    policy improves, because stage1_skip=saturated removes exactly the
    all-good groups (the 08-21 curve printed -3.000 on a step where 11/12
    groups were skipped as saturated; true mean ~ +0.4).

    sample(examples, n) -> (groups, prompts, stats); grade(pairs) grades
    [(example, member), ...] into the caller's cache; total_of(member) reads
    the base reward back. Returns (examples, groups, prompts, stats, skipped).
    """
    if not (0 < cfg.group_stage1 < n_sampled):
        groups, prompts, stats = sample(examples, n_sampled)
        return examples, groups, prompts, stats, 0
    g1 = cfg.group_stage1
    groups, prompts, stats = sample(examples, g1)
    grade([(ex, m) for ex, group in zip(examples, groups) for m in group])
    live = [i for i, group in enumerate(groups)
            if not _stage1_dead([total_of(m) for m in group], cfg.stage1_skip)]
    live_set = set(live)
    skipped_rewards = [total_of(m) for i, group in enumerate(groups)
                       if i not in live_set for m in group]
    skipped = len(examples) - len(live)
    examples = [examples[i] for i in live]
    groups = [groups[i] for i in live]
    prompts = [prompts[i] for i in live]
    if examples:
        mx.clear_cache()  # stage-1 KV is dead weight under stage 2
        groups2, _, stats2 = sample(examples, n_sampled - g1)
        for group, extra in zip(groups, groups2):
            group.extend(extra)
        stats = stats2  # tok/s of the larger phase; a metric, not a ledger
    stats.stage1_skipped_rewards = skipped_rewards
    return examples, groups, prompts, stats, skipped


def collect_rollouts(model, tokenizer, examples, cfg: TrainConfig, task):
    """Sample group_size completions per example and score them.
    Returns (rollouts, stats, groups_skipped).

    SAGE-RL hybrid rollouts (cfg.sage_r > 0): the last sage_r members of
    every group are generated with SAGE confidence-guided decoding instead
    of ordinary sampling — short, decisive chains sitting next to the
    exploratory ones, so correctness rewards can select for efficiency.
    SAGE members are off-policy relative to temp-1 sampling; since old_lp is
    recomputed teacher-forced, they enter the surrogate at ratio 1 like any
    injected demonstration.

    Two-stage rollouts (cfg.group_stage1 > 0): sample that many members
    first; groups whose stage-1 rewards are already decided (_stage1_dead)
    are abandoned before the remaining members and SAGE beams are paid for —
    they'd be dropped as zero-variance at update time anyway.
    """
    sage_r = cfg.sage_r if cfg.think_end is not None else 0
    inject_r = cfg.inject_r
    n_sampled = cfg.group_size - sage_r - inject_r
    think_close = _think_close_marker(tokenizer, cfg, task)
    # id(Completion) -> (Completion, RewardResult). Judge/code grading is
    # costly, so nothing is graded twice. The cache holds the Completion
    # itself: abandoned stage-1 groups are otherwise garbage-collected, and a
    # stage-2 member reusing the freed id would inherit a stale reward.
    graded: dict[int, tuple[Completion, object]] = {}

    def _grade(pairs):
        results = _grade_batch(task, [
            (ex, _visible_reply(_completion_text(tokenizer, comp),
                                think_close)[0]) for ex, comp in pairs])
        for (_, comp), res in zip(pairs, results):
            graded[id(comp)] = (comp, res)

    examples, groups, prompts, stats, skipped = _two_stage(
        examples, cfg, n_sampled,
        lambda exs, n: _sample_batched(model, tokenizer, exs, cfg, n, cfg.temperature, task),
        _grade, lambda comp: graded[id(comp)][1].total)
    if sage_r and examples:
        # the sampled rollouts' KV buffers are dead now; don't run the SAGE
        # beams on top of them (the two phases stack in memory otherwise)
        mx.clear_cache()
        eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
        eos |= set(cfg.extra_eos)
        step_delim = _step_delim_ids(tokenizer)
        for prompt, group in zip(prompts, groups):
            for _ in range(sage_r):
                group.append(
                    sage_completion(
                        model,
                        list(prompt),
                        cfg.think_end,
                        eos=eos,
                        step_delim=step_delim,
                        m=cfg.sage_m,
                        tr=cfg.sage_tr,
                        max_new_tokens=cfg.max_new_tokens,
                        max_reasoning_steps=cfg.sage_max_reasoning_steps,
                        max_step_tokens=cfg.sage_max_step_tokens,
                        step_tokens=cfg.sage_step_tokens,
                        think_temperature=cfg.sage_think_temperature,
                        answer_temperature=cfg.temperature,
                        answer_reserve=cfg.sage_answer_reserve,
                    )
                )
    if inject_r and examples:
        # Off-policy demonstrations at fixed group slots (see config.inject_r).
        # Tokens end with EOS so grading/packing treat them like sampled stops.
        eos_id = next(iter(sorted(tokenizer.eos_token_ids)))
        for ex, group in zip(examples, groups):
            text = task.injected_completion(ex)
            toks = tokenizer.encode(text, add_special_tokens=False) + [eos_id]
            for _ in range(inject_r):
                group.append(Completion(list(toks), [0.0] * len(toks), "stop"))
    # Grade everything not covered by stage 1 in one batch (one judge call
    # per rollout batch for judge-graded frames).
    pending = [(ex, comp) for ex, group in zip(examples, groups)
               for comp in group if id(comp) not in graded]
    if pending:
        _grade(pending)
    rollouts: list[Rollout] = []
    for ex, prompt, group in zip(examples, prompts, groups):
        for gi, comp in enumerate(group):
            text = _completion_text(tokenizer, comp)
            visible, closed = _visible_reply(text, think_close)
            _, res = graded[id(comp)]
            # Correctness-gated total-length efficiency (kills the
            # reasoning-relocation hack) + a relocation monitor for BOTH arms.
            ntok = len(comp.tokens)
            budget = cfg.length_budget or cfg.max_new_tokens
            len_norm = min(1.0, ntok / max(1, budget))
            marker = think_close or "</think>"
            think_chars = text.index(marker) if marker in text else len(text)
            parts = dict(res.parts)
            parts["base_reward"] = round(res.total, 4)
            parts["len_norm"] = round(len_norm, 4)
            parts["think_frac"] = round(think_chars / max(1, len(text)), 4)
            parts["think_closed"] = float(closed)
            # RFCS (paper's mechanism metric): only for correct completions
            # with a closed think and a numeric reference answer.
            if (parts.get("correct") == 1.0 and closed and marker in text
                    and "answer" in ex.meta):
                r = rfcs(text.split(marker, 1)[0], str(ex.meta["answer"]))
                if r is not None:
                    parts["rfcs"] = round(r, 4)
            # max(total, 0): scale down positive rewards only. Multiplying a
            # NEGATIVE total by (1 - lam*len_norm) shrank the penalty for
            # longer failures — a length BONUS on exactly the rambling the
            # knob exists to discourage.
            gated = res.total - cfg.length_penalty * len_norm * max(res.total, 0.0)
            # Tripwires for two previously-observed bug classes — cheap, loud.
            if comp.think_len is not None and comp.think_len > cfg.max_new_tokens:
                print(f"[BUG] think_len {comp.think_len} > max_new_tokens "
                      f"{cfg.max_new_tokens} — SAGE budget breached", flush=True)
            if gated > 0 and not closed:
                print("[BUG] positive reward on an unclosed think block — "
                      "grader leak", flush=True)
            rollouts.append(
                Rollout(
                    prompt_tokens=list(prompt),
                    completion_tokens=comp.tokens,
                    sampling_logprobs=comp.logprobs,
                    text=text,
                    reward=gated,
                    reward_parts=parts,
                    meta=ex.meta,
                    sage=n_sampled <= gi < n_sampled + sage_r,
                    injected=gi >= n_sampled + sage_r,
                    think_len=comp.think_len,
                    finish=comp.finish_reason,
                )
            )
    return rollouts, stats, skipped


def neutralize_capped_rewards(rewards: np.ndarray, capped: np.ndarray) -> int:
    """Replace each length-capped member's reward with the mean of its
    group's UNcapped members, in place. Its group-relative advantage becomes
    exactly 0 — no gradient toward or away from running out of budget — and
    the group baseline stays unbiased. (Scoring truncation as no_reply/-P fed
    a shortness gradient: 1 in 8 rewards of the 08-22 run measured the token
    budget, not the policy.) All-capped groups are left alone: zero-variance,
    dropped by active_groups anyway. Returns the number neutralized."""
    n = 0
    for g in np.flatnonzero(capped.any(axis=1)):
        alive = ~capped[g]
        if alive.any():
            rewards[g, capped[g]] = rewards[g, alive].mean()
            n += int(capped[g].sum())
    return n


def prune_advantages(upd_adv: np.ndarray, group_size: int, frac: float):
    """Advantage pruning, per SIGN, then re-centered.

    A single |adv| threshold is sign-biased on asymmetric reward scales
    ({+1,0,-3}): a 7-good/1-bad group normalizes to [+0.38 x7, -2.65] and
    kept only the -2.65 — every majority-good group became pure suppression
    and good completions were never reinforced (the 08-22 run updated on
    17/96 sequences this way). Thresholding within each sign keeps both
    sides; the re-center removes the residual so kept advantages still sum
    to ~0 per group. Returns (kept_advantages, keep_mask, n_pruned)."""
    adv2d = upd_adv.reshape(-1, group_size)
    a2 = np.abs(adv2d)
    pos, neg = adv2d > 0, adv2d < 0
    pos_max = np.where(pos, a2, 0.0).max(axis=1, keepdims=True)
    neg_max = np.where(neg, a2, 0.0).max(axis=1, keepdims=True)
    keep2d = ((pos & (a2 >= frac * pos_max))
              | (neg & (a2 >= frac * neg_max)))
    kept_n = np.maximum(keep2d.sum(axis=1, keepdims=True), 1)
    kept_mean = (adv2d * keep2d).sum(axis=1, keepdims=True) / kept_n
    adv2d = adv2d - kept_mean
    keep = keep2d.reshape(-1)
    return adv2d.reshape(-1)[keep], keep, int((~keep).sum())


def update_policy(model, optimizer, loss_and_grad, rollouts, advantages, cfg,
                  pad_id, denom_tokens: float | None = None,
                  subset_rng: np.random.Generator | None = None):
    """One (or more) clipped-PG epochs over the rollout batch, microbatched
    with gradient accumulation. Returns (pg_mean, kl_mean, pre-clip grad norm).

    denom_tokens: pass the PRE-pruning completion-token count when the batch
    was thinned by update_adv_frac, so dropped terms count as zero instead of
    inflating the survivors' weight."""
    inp, tgt, mask, _ = build_training_arrays(rollouts, pad_id)
    denom = denom_tokens if denom_tokens else float(mask.sum())
    if 0.0 < cfg.token_subset_frac < 1.0:
        # Uniform token subset; denom scaled by the SAME frac (not the
        # realized count) so pruned/rounded terms keep contributing zero
        # rather than reweighting survivors — the estimator stays unbiased
        # and pg/kl metrics stay on the full-token scale.
        mask = subsample_token_mask(
            mask, cfg.token_subset_frac,
            subset_rng if subset_rng is not None
            else np.random.default_rng(cfg.seed))
        denom *= cfg.token_subset_frac

    # Recompute old_lp teacher-forced instead of trusting generation-time
    # logprobs: the KV-cached incremental forward drifts from the padded
    # batch forward at fp16 (measured up to 0.125 nats/token), which would
    # inject spurious importance ratios. Recomputed, epoch 1 has ratio == 1
    # exactly (pure REINFORCE with baseline).
    old_chunks, ref_chunks = [], []
    for lo in range(0, len(rollouts), cfg.micro_batch):
        hi = lo + cfg.micro_batch
        old_chunks.append(score_logprobs(model, inp[lo:hi], tgt[lo:hi]))
    old_lp = np.concatenate(old_chunks, axis=0)

    # Reference (base-model) logprobs: same weights with adapter scales zeroed.
    with adapters_disabled(model):
        for lo in range(0, len(rollouts), cfg.micro_batch):
            hi = lo + cfg.micro_batch
            ref_chunks.append(score_logprobs(model, inp[lo:hi], tgt[lo:hi]))
    ref_lp = np.concatenate(ref_chunks, axis=0)

    for _ in range(cfg.epochs_per_batch):
        acc = None
        pg_total, kl_total = 0.0, 0.0
        for lo in range(0, len(rollouts), cfg.micro_batch):
            hi = lo + cfg.micro_batch
            if cfg.token_subset_frac > 0:
                # Selective path: the head/loss only ever sees the selected
                # positions — the [*, L, 248k] logits slab shrinks to
                # [*, K, 248k] (K = this microbatch's selection width).
                sel_idx, sel_mask, tgt_s, old_s, ref_s = gather_selected(
                    mask, tgt, old_lp, ref_lp, lo, hi)
                (loss, pg_sum, kl_sum), grads = loss_and_grad(
                    model,
                    mx.array(inp[lo:hi]),
                    mx.array(tgt_s),
                    mx.array(sel_mask),
                    mx.array(old_s),
                    mx.array(ref_s),
                    mx.array(advantages[lo:hi]),
                    denom,
                    mx.array(sel_idx),
                )
            else:
                (loss, pg_sum, kl_sum), grads = loss_and_grad(
                    model,
                    mx.array(inp[lo:hi]),
                    mx.array(tgt[lo:hi]),
                    mx.array(mask[lo:hi]),
                    mx.array(old_lp[lo:hi]),
                    mx.array(ref_lp[lo:hi]),
                    mx.array(advantages[lo:hi]),
                    denom,
                )
            acc = grads if acc is None else tree_map(mx.add, acc, grads)
            mx.eval(acc)
            pg_total += float(pg_sum)
            kl_total += float(kl_sum)
        gnorm = float("nan")
        if cfg.max_grad_norm > 0:
            # clip_grad_norm hands back the PRE-clip norm and this used to
            # discard it. Without it there is no way to tell a healthy step
            # from one that is clipped flat every time -- and if every step
            # clips, the learning rate sets only direction, not size.
            acc, gn = optim.clip_grad_norm(acc, cfg.max_grad_norm)
            gnorm = float(gn)
        optimizer.update(model, acc)
        mx.eval(model.parameters(), optimizer.state)
    return pg_total / denom, kl_total / denom, gnorm




def _eval_set(task, cfg: TrainConfig) -> tuple[TrainConfig, list[Example]]:
    """The held-out examples every eval flavour scores: a fixed seed disjoint
    from the training stream, the task's eval split when it has one, and the
    eval-only length cap applied to the config."""
    if cfg.eval_max_new_tokens:
        cfg = replace(cfg, max_new_tokens=cfg.eval_max_new_tokens)
    rng = random.Random(cfg.seed + 100_000)
    esample = getattr(task, "eval_sample", task.sample)
    return cfg, [esample(rng) for _ in range(cfg.eval_n)]


def _mean_parts(results, prefix: str = "eval_") -> dict:
    """Mean of every reward part over a result list, absent parts as 0."""
    keys = sorted({k for r in results for k in r.parts})
    return {f"{prefix}{k}": float(np.mean([r.parts.get(k, 0.0) for r in results])) for k in keys}


def evaluate(model, tokenizer, task, cfg: TrainConfig):
    """Greedy decode on a fixed held-out set (batched); returns mean reward + rates."""
    if int(getattr(task, "turns", 1)) > 1:
        return evaluate_multiturn(model, tokenizer, task, cfg)
    if getattr(task, "tools", None):
        return evaluate_episodes(model, tokenizer, task, cfg)
    cfg, examples = _eval_set(task, cfg)
    groups, _, _ = _sample_batched(model, tokenizer, examples, cfg, 1, 0.0, task)
    think_close = _think_close_marker(tokenizer, cfg, task)
    visibles = [_visible_reply(_completion_text(tokenizer, group[0]),
                               think_close) for group in groups]
    results = _grade_batch(task, [(ex, vis) for ex, (vis, _)
                                  in zip(examples, visibles)])
    closed_rates = []
    rfcs_vals = []
    for ex, group, (visible, closed), res in zip(examples, groups,
                                                 visibles, results):
        text = _completion_text(tokenizer, group[0])
        closed_rates.append(float(closed))
        if (res.parts.get("correct") == 1.0 and closed and think_close
                and think_close in text and "answer" in ex.meta):
            r = rfcs(text.split(think_close, 1)[0], str(ex.meta["answer"]))
            if r is not None:
                rfcs_vals.append(r)
    out = {
        "eval_think_closed": float(np.mean(closed_rates)),
        "eval_reward": float(np.mean([r.total for r in results])),
        # THE SAGE-RL deployment metric: greedy single-pass length. The whole
        # point is that after training, plain decoding is already concise.
        "eval_mean_len": float(np.mean([len(g[0].tokens) for g in groups])),
    }
    if rfcs_vals:
        out["eval_rfcs"] = float(np.mean(rfcs_vals))
        out["eval_rfcs_n"] = len(rfcs_vals)
    return {**out, **_mean_parts(results)}


def train(cfg: TrainConfig, out_dir: str | Path) -> Path:
    from .preflight import preflight
    preflight(cfg)  # dies loudly BEFORE weights load or the lease is taken
    holder = None
    if cfg.manage_machine:
        from .memory import estimate_run_gb, model_disk_gb
        from .models import resolve_model_path

        required = cfg.required_gb or estimate_run_gb(
            model_disk_gb(resolve_model_path(cfg.model)), cfg.activation_headroom_gb
        )
        holder = machine.acquire(
            required, wait_s=cfg.lease_wait_s, block=cfg.lease_block,
            note=f"train {cfg.task} on {cfg.model}"
        )
    try:
        return _train(cfg, out_dir)
    except BaseException as e:
        # Loud in-band death: the swap guard's os._exit writes its own marker;
        # this covers every other crash class (OOM guard, decode bugs, ^C).
        write_abort_marker(Path(out_dir), f"{type(e).__name__}: {e}")
        raise
    finally:
        machine.release(holder)


def _rotate(path: Path) -> None:
    """One run per file: move a previous run's file aside instead of
    appending. Mixed-run jsonl turns analysis into archaeology (an apparent
    'bimodal' metric can turn out to span two different runs)."""
    if path.exists() and path.stat().st_size:
        stamp = time.strftime(
            "%Y%m%d-%H%M%S", time.localtime(path.stat().st_mtime))
        path.rename(path.with_name(f"{path.stem}.{stamp}{path.suffix}"))


def _resume_dir(out: Path) -> Path:
    return out / "resume"


def save_resume(out: Path, step: int, optimizer, rng, subset_rng, task,
                activity_window, keep: int) -> None:
    """Everything needed to continue this run bit-exactly, minus the adapter
    weights (already written by save_adapter at the same step).

    Optimizer moments are ~2x the adapter, so only the newest `keep` are
    retained; older snapshots are pruned and their steps stop being resumable.
    """
    d = _resume_dir(out)
    d.mkdir(parents=True, exist_ok=True)
    mx.eval(optimizer.state)
    mx.save_safetensors(str(d / f"opt-{step:05d}.safetensors"),
                        dict(tree_flatten(optimizer.state)))
    get_state = getattr(task, "get_state", None)
    (d / f"state-{step:05d}.pkl").write_bytes(pickle.dumps({
        "step": step,
        "rng": rng.getstate(),
        "subset_rng": subset_rng.bit_generator.state,
        "task": get_state() if get_state else None,
        "activity_window": list(activity_window),
    }))
    # Prune oldest first; zero-padded step names sort chronologically. Drop
    # the .pkl only after its .safetensors so an interrupted prune never
    # leaves a state file pointing at missing optimizer moments.
    for pattern in ("opt-*.safetensors", "state-*.pkl"):
        stale = sorted(d.glob(pattern))[:-keep] if keep > 0 else []
        for f in stale:
            f.unlink(missing_ok=True)


def load_resume(src: Path, model, optimizer, rng, subset_rng, task):
    """Restore the newest resumable checkpoint from a PREVIOUS run directory.
    Read-only: nothing is written to src. Returns (step, extra)."""
    out = src
    d = _resume_dir(src)
    states = sorted(d.glob("state-*.pkl")) if d.exists() else []
    if not states:
        raise SystemExit(
            f"--resume: no resume state under {d}. Only checkpoints saved "
            "with keep_resume >= 1 are resumable, and older ones are pruned.")
    st = pickle.loads(states[-1].read_bytes())
    step = st["step"]
    adapter = out / "adapters" / f"adapter-{step:05d}.safetensors"
    opt_file = d / f"opt-{step:05d}.safetensors"
    for f in (adapter, opt_file):
        if not f.exists():
            raise SystemExit(f"--resume: {f} missing; cannot resume step {step}")
    model.load_weights(str(adapter), strict=False)
    optimizer.state = tree_unflatten(list(mx.load(str(opt_file)).items()))
    rng.setstate(st["rng"])
    subset_rng.bit_generator.state = st["subset_rng"]
    set_state = getattr(task, "set_state", None)
    if set_state and st["task"] is not None:
        set_state(st["task"])
    return step, st


def _train(cfg: TrainConfig, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "metrics.jsonl", "samples.jsonl"):
        _rotate(out / name)
    (out / "ABORTED").unlink(missing_ok=True)  # fresh run, fresh verdict
    cfg.save(out / "config.json")
    metrics_f = (out / "metrics.jsonl").open("a")
    samples_f = (out / "samples.jsonl").open("a")

    mx.random.seed(cfg.seed)
    rng = random.Random(cfg.seed)
    subset_rng = np.random.default_rng(cfg.seed)  # token-subset draws
    task = get_task(cfg.task, **cfg.task_kwargs)
    eval_cells = build_eval_cells(cfg)
    if cfg.inject_r and not (hasattr(task, "injected_completion")
                             or hasattr(task, "injected_episode")):
        raise SystemExit(
            f"--inject-r needs task {cfg.task!r} to define injected_completion()")

    if cfg.gdn_serial:
        from . import gdn_serial

        gdn_serial.install(cfg.gdn_chunk)

    model, tokenizer, info = load_policy(
        cfg.model, cfg.lora, cfg.activation_headroom_gb,
        grad_checkpoint=cfg.grad_checkpoint, required_gb=cfg.required_gb,
    )
    print(f"loaded {cfg.model}: {info}")
    # Let a judge_backend=local judge generate on THIS model with adapters
    # disabled (== the frozen base) instead of loading a second ~19 GB copy.
    from .judge_local import register_resident_model
    from .models import resolve_model_path
    register_resident_model(model, tokenizer, resolve_model_path(cfg.model))
    if cfg.init_adapter:
        apath = Path(cfg.init_adapter).expanduser()
        wfile = apath / "adapters.safetensors" if apath.is_dir() else apath
        n_before = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        weights = dict(mx.load(str(wfile)).items())
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        n_after = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        assert n_before == n_after
        matched = sum(1 for k, _ in tree_flatten(model.trainable_parameters()) if k in weights)
        print(f"init adapter {wfile}: {matched}/{len(weights)} tensors matched trainable params",
              flush=True)
        if matched == 0:
            raise RuntimeError("init adapter matched no trainable parameters — LoRA config mismatch?")
    pad_id = next(iter(sorted(tokenizer.eos_token_ids)))

    # Fail loud, not slow: hard-abort if a backward spills to swap.
    swap_guard = SwapGuard(
        margin_gb=cfg.swap_guard_margin_gb,
        rate_mb_s=cfg.swap_rate_mb_s,
        rate_samples=cfg.swap_rate_samples,
        abort_marker=out / "ABORTED",
    ).start()

    # Dead-run watchdog (config.abort_inactive_window): a collapsed policy
    # produces near-zero active groups indefinitely — steps still burn full
    # rollout compute but carry no gradient. Windowed rate, not consecutive
    # zeros: collapsed runs still emit stray active groups.
    activity_window: deque[int] = deque(maxlen=cfg.abort_inactive_window or 1)

    def watch_activity(n_active: int) -> None:
        if not cfg.abort_inactive_window:
            return
        activity_window.append(n_active)
        if len(activity_window) < cfg.abort_inactive_window:
            return
        floor = 0.02 * cfg.abort_inactive_window * cfg.batch_prompts
        if sum(activity_window) < floor:
            msg = (f"policy collapse: {sum(activity_window)} active groups in "
                   f"the last {cfg.abort_inactive_window} steps "
                   f"(< 2% of {cfg.abort_inactive_window * cfg.batch_prompts})")
            (out / "ABORTED").write_text(msg + "\n")
            swap_guard.stop()
            raise SystemExit(f"ABORTED — {msg}")

    optimizer = optim.Adam(learning_rate=cfg.lr)

    def loss_fn(model, inp, tgt, mask, old_lp, ref_lp, adv, denom,
                sel_idx=None):
        cur_lp = (token_logprobs(model(inp), tgt) if sel_idx is None
                  else selective_logprobs(model, inp, tgt, sel_idx))
        return grpo_objective(
            cur_lp, old_lp, ref_lp, adv, mask, denom, cfg.clip_eps, cfg.kl_coef
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    start_step, baseline = 1, None
    if cfg.resume_from:
        src = Path(cfg.resume_from).expanduser()
        if src.resolve() == out.resolve():
            raise SystemExit(
                "--resume-from must differ from --out: a resumed run writes a "
                "NEW directory so the source run's record stays intact")
        # A resumed run silently continuing under different hyperparameters
        # poisons every comparison made with it. Diff the configs; ignore the
        # fields a legitimate resume changes. Deliberate drift (e.g. an lr
        # sweep continuing a warmup) opts in via MLX_RL_ALLOW_RESUME_DRIFT=1.
        src_cfg_f = src / "config.json"
        if src_cfg_f.exists() and not os.environ.get("MLX_RL_ALLOW_RESUME_DRIFT"):
            a = json.loads(src_cfg_f.read_text())
            b = json.loads((out / "config.json").read_text())
            ignore = {"resume_from", "steps", "keep_resume", "lease_wait_s",
                      "manage_machine", "lease_block", "required_gb"}
            drift = {k: (a.get(k), b.get(k)) for k in (set(a) | set(b))
                     if k not in ignore and a.get(k) != b.get(k)}
            if drift:
                raise SystemExit(
                    "RESUME REFUSED: config drift vs the source run\n"
                    + "\n".join(f"  {k}: {v0!r} -> {v1!r}"
                                for k, (v0, v1) in sorted(drift.items()))
                    + "\nSet MLX_RL_ALLOW_RESUME_DRIFT=1 if this is deliberate.")
        done, st = load_resume(src, model, optimizer, rng, subset_rng, task)
        start_step = done + 1
        activity_window.extend(st["activity_window"])
        # Baseline lives in the SOURCE run's metrics; carried over only so the
        # final line can report against it (not copied into the new metrics).
        mfile = src / "metrics.jsonl"
        if mfile.exists():
            baseline = next(
                (r for l in mfile.read_text().splitlines()
                 if (r := json.loads(l)).get("step") == 0), None)
        metrics_f.write(json.dumps({
            "step": done, "resumed_from": str(src)}) + "\n")
        metrics_f.flush()
        print(f"resumed {src} at step {done}; continuing at {start_step} "
              f"in {out}")
    else:
        baseline = {**evaluate(model, tokenizer, task, cfg),
                    **evaluate_cells(model, tokenizer, eval_cells, cfg)}
        mx.clear_cache()  # phase boundary: don't stack eval KV under step-1 gen
        print(f"step 0 baseline: {baseline}")
        metrics_f.write(json.dumps({"step": 0, **baseline}) + "\n")
        metrics_f.flush()

    for step in range(start_step, cfg.steps + 1):
        # Per-step seed rather than one seed for the whole run: MLX exposes no
        # way to read back the global RNG state, so a run-level seed cannot be
        # restored on resume. Deriving it from (seed, step) makes generation
        # reproducible at any step, whether reached by running through or by
        # resuming into it.
        mx.random.seed(cfg.seed * 1_000_003 + step)
        examples = [task.sample(rng) for _ in range(cfg.batch_prompts)]

        mx.reset_peak_memory()  # per-step peak, not a run-lifetime high-water
        t0 = time.time()
        rollouts, gen_stats, skipped1 = (
            collect_multiturn if int(getattr(task, "turns", 1)) > 1 else
            collect_episodes if getattr(task, "tools", None) else collect_rollouts
        )(model, tokenizer, examples, cfg, task)
        t_gen = time.time() - t0
        # Phase boundary: generation KV buffers are dead weight during the
        # backward. Without this the pre-update high-water is the SUM of the
        # phases — eval KV + rollout KV + SAGE beams + backward peak, easily
        # +7-8 GB, enough to push a borderline run into swap.
        mx.clear_cache()

        if not rollouts:  # every group abandoned at stage 1
            rec = {"step": step, "groups_skipped_stage1": skipped1,
                   "gen_s": round(t_gen, 2), "ts": round(time.time(), 1)}
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            print(f"step {step:4d}  all {cfg.batch_prompts} groups dead at "
                  "stage 1 — nothing to train on", flush=True)
            watch_activity(0)
            continue

        # rows = SURVIVING groups (may be < batch_prompts under group_stage1)
        rewards = np.array(
            [r.reward for r in rollouts], dtype=np.float32
        ).reshape(-1, cfg.group_size)
        n_capped_neutral = 0
        if getattr(task, "neutralize_len_capped", False):
            capped = np.array([r.finish == "length" for r in rollouts]
                              ).reshape(-1, cfg.group_size)
            n_capped_neutral = neutralize_capped_rewards(rewards, capped)
        advantages = np.array(
            group_advantages(mx.array(rewards), cfg.normalize_std)
        ).reshape(-1, cfg.group_size)
        active = np.array(active_groups(mx.array(rewards)))

        t1 = time.time()
        n_pruned = 0
        no_update = False
        if active.any():
            idx = np.flatnonzero(np.repeat(active, cfg.group_size))
            upd_rollouts = [rollouts[i] for i in idx]
            upd_adv = advantages.reshape(-1)[idx]
            # Denominator from the FULL active batch, before pruning: pruned
            # terms contribute exactly 0 instead of reweighting the survivors.
            denom_tokens = float(sum(
                (sum(r.gen_mask) if r.gen_mask is not None
                 else len(r.completion_tokens)) for r in upd_rollouts))
            if cfg.update_adv_frac > 0:
                upd_adv, keep, n_pruned = prune_advantages(
                    upd_adv, cfg.group_size, cfg.update_adv_frac)
                upd_rollouts = [r for r, k in zip(upd_rollouts, keep) if k]
            pg, kl, gnorm = update_policy(
                model,
                optimizer,
                loss_and_grad,
                upd_rollouts,
                upd_adv,
                cfg,
                pad_id,
                denom_tokens=denom_tokens,
                subset_rng=subset_rng,
            )
        else:
            # No reward spread anywhere: skip the update. Deliberately NOT
            # medicated (an earlier KL-only "rescue" update was removed in
            # review) — a degenerating run should die loudly via the
            # dead-run watchdog (abort_inactive_window, on by default), not
            # be silently pulled back toward base.
            pg, kl, gnorm = 0.0, 0.0, float("nan")
            no_update = True
        t_upd = time.time() - t1
        # Tell the swap guard how long that took. Paging traffic alone is not
        # evidence of trouble — it killed a healthy 200-step run on transients
        # from a nearly-full swap file — so the rate detector aborts only when
        # step times confirm the run is being hurt.
        swap_guard.note_step(t_gen + t_upd)
        # Return the backward peak's buffers to the OS. MLX's buffer cache
        # keeps freed allocations resident indefinitely, so after each 65-70
        # GiB update the process stays that large; over ~an hour macOS starts
        # paging (slow swap creep).
        mx.clear_cache()

        # The batch mean INCLUDING groups abandoned at stage 1. reward_mean
        # (survivors only) lies downward as the policy improves, because
        # stage1_skip=saturated removes exactly the all-good groups: the
        # 08-21 overnight run printed -3.000 on a step where 11/12 groups
        # were skipped as saturated (true mean ~ +0.4).
        skip_rw = list(getattr(gen_stats, "stage1_skipped_rewards", []) or [])
        rec = {
            "step": step,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_mean_all": float(np.mean(
                np.concatenate([rewards.reshape(-1),
                                np.asarray(skip_rw, dtype=np.float32)])
                if skip_rw else rewards.reshape(-1))),
            **{
                f"frac_{key}": float(
                    np.mean([r.reward_parts.get(key, 0.0) for r in rollouts])
                )
                # rfcs is defined only on correct+numeric completions — a
                # zero-default mean over everything would be meaningless
                for key in sorted({k for r in rollouts for k in r.reward_parts})
                if key != "rfcs"
            },
            "active_groups": int(active.sum()),
            "mean_len": float(np.mean([len(r.completion_tokens) for r in rollouts])),
            "pg": pg,
            "kl": kl,
            # Hyper-parameter sanity, all of it previously invisible:
            #   grad_norm  pre-clip. Persistently above max_grad_norm means
            #              every step is clipped and lr sets direction only.
            #   n_seqs     sequences that actually reached the update. The
            #              first legs ran on ~3 -- the plotted curve was noise
            #              by construction, not by bad luck.
            #   adv_std    spread of the advantages being learned from; ~0
            #              means the group agreed and there is nothing to learn.
            "grad_norm": None if gnorm != gnorm else round(gnorm, 4),
            "n_seqs": int(len(upd_rollouts)) if active.sum() else 0,
            "adv_std": (round(float(np.std(upd_adv)), 4) if active.sum() else 0.0),
            "gen_tok_s": round(gen_stats.generation_tps, 1),
            "prompt_tok_s": round(gen_stats.prompt_tps, 1),
            "gen_s": round(t_gen, 2),
            "update_s": round(t_upd, 2),
            # Per-step MLX allocation peak in GiB (reset each step — the old
            # never-reset value was a useless run-lifetime constant).
            "peak_gb": round(mx.get_peak_memory() / 1024**3, 2),
            # Real pressure signal: system swap growth since the guard armed.
            "swap_gb": round(swap_used_gb() - swap_guard.baseline_gb, 2),
            "frac_len_capped": float(
                np.mean([r.finish == "length" for r in rollouts])
            ),
            # pg/kl are 0.0 on a skipped step, indistinguishable from a
            # measured zero without this flag.
            "no_update": no_update,
            "ts": round(time.time(), 1),
        }
        if n_capped_neutral:
            rec["len_capped_neutralized"] = n_capped_neutral
        rfcs_vals = [r.reward_parts["rfcs"] for r in rollouts
                     if "rfcs" in r.reward_parts]
        if rfcs_vals:
            rec["rfcs_mean"] = float(np.mean(rfcs_vals))
            rec["rfcs_n"] = len(rfcs_vals)
        if cfg.group_stage1:
            rec["groups_skipped_stage1"] = skipped1
        web = getattr(task, "web", None)
        if web is not None:  # live-tool volume: cache hits vs live calls, errors
            rec.update({f"web_{k}": v for k, v in web.stats.items()})
            rec.update({f"web_{k}": v for k, v in getattr(task, "tool_stats", {}).items()})
        if cfg.update_adv_frac:
            rec["rollouts_pruned"] = n_pruned
        # QeRL observable (2510.11696): mean NLL of the sampled tokens under
        # the (4-bit) policy — the exploration/entropy proxy. Quantization
        # noise raises it; watch it fall as the policy sharpens.
        # Averaged over GENERATED tokens only: injected tool-response tokens
        # carry logprob 0.0 (engine fills them as context), and their share
        # grows over training — unmasked, they diluted the 08-21 curve's NLL
        # ~2x, overstating the sharpening.
        nll = []
        for r in rollouts:
            if r.sage or r.injected or not r.sampling_logprobs:
                continue
            lps = (r.sampling_logprobs if r.gen_mask is None else
                   [lp for lp, m in zip(r.sampling_logprobs, r.gen_mask) if m])
            if lps:
                nll.append(-float(np.mean(lps)))
        if nll:
            rec["gen_nll"] = round(float(np.mean(nll)), 4)
        if cfg.inject_r:
            inj = [r.reward for r in rollouts if r.injected]
            if inj:
                rec["reward_injected"] = float(np.mean(inj))
        if cfg.sage_r:
            sage_rs = [r for r in rollouts if r.sage]
            samp_rs = [r for r in rollouts if not r.sage and not r.injected]
            rec["mean_len_sage"] = float(
                np.mean([len(r.completion_tokens) for r in sage_rs])
            )
            rec["mean_len_sampled"] = float(
                np.mean([len(r.completion_tokens) for r in samp_rs])
            )
            rec["mean_think_len"] = float(
                np.mean([r.think_len for r in sage_rs if r.think_len is not None])
            )
            rec["reward_sage"] = float(np.mean([r.reward for r in sage_rs]))
            rec["reward_sampled"] = float(np.mean([r.reward for r in samp_rs]))
        # Checkpoint BEFORE eval at the same step: eval is the likeliest
        # crash site (biggest single batch of the loop), and with both on the
        # same cadence the old order left a first-eval crash with zero
        # checkpoints, losing every step of compute the run had done.
        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            save_adapter(model, out / "adapters", cfg.lora, cfg.model, step)
            if cfg.keep_resume:
                save_resume(out, step, optimizer, rng, subset_rng, task,
                            activity_window, cfg.keep_resume)
        if cfg.eval_every and step % cfg.eval_every == 0:
            rec.update(evaluate(model, tokenizer, task, cfg))
            rec.update(evaluate_cells(model, tokenizer, eval_cells, cfg))
            mx.clear_cache()  # phase boundary: eval KV vs next step's gen
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        print(
            f"step {step:4d}  reward {rec['reward_mean']:.3f}±{rec['reward_std']:.3f}  "
            f"correct {rec.get('frac_correct', 0.0):.2f}  len {rec['mean_len']:.0f}  "
            f"thinkfrac {rec.get('frac_think_frac', 0.0):.2f}  "
            f"kl {kl:.4f}  gen {rec['gen_tok_s']} tok/s  upd {rec['update_s']}s"
            + (f"  sage len {rec['mean_len_sage']:.0f}/r {rec['reward_sage']:.2f}"
               if cfg.sage_r else "")
            + (f"  EVAL reward {rec['eval_reward']:.3f} correct {rec.get('eval_correct', 0.0):.2f} len {rec['eval_mean_len']:.0f}"
               if "eval_reward" in rec else "")
        )

        # Raw samples: the first prompt's whole group, so reward spread is visible.
        group = rollouts[: cfg.group_size]
        # Same scoreboard to the live dashboard (dash.), one amber note per
        # step next to the streamed episodes; rl-dash. shows the full group.
        _tap().note(
            f"step {step}: reward {rec['reward_mean']:+.2f}±{rec['reward_std']:.2f} "
            f"active {int(active.sum())}/{len(rewards)} len {rec['mean_len']:.0f} "
            + " ".join(f"{k[5:]} {rec[k]:.2f}" for k in (
                "frac_correct", "frac_called", "frac_abstain",
                "frac_denial", "frac_no_reply", "frac_len_capped") if k in rec)
            + f" | first group: {' '.join(f'{r.reward:+.1f}' for r in group)}"
            + f" | gen {rec['gen_s']:.0f}s upd {rec['update_s']:.0f}s")
        samples_f.write(
            json.dumps(
                {
                    "step": step,
                    "ts": round(time.time(), 1),
                    "meta": group[0].meta,
                    "completions": [
                        {
                            "reward": r.reward,
                            "parts": r.reward_parts,
                            "sage": r.sage,
                            "injected": r.injected,
                            "think_len": r.think_len,
                            "len": len(r.completion_tokens),
                            "finish": r.finish,
                            "text": r.text,
                            **({"tool_calls": r.tool_calls} if r.tool_calls else {}),
                        }
                        for r in group
                    ],
                }
            )
            + "\n"
        )
        samples_f.flush()
        watch_activity(int(active.sum()))

    swap_guard.stop()
    final = evaluate(model, tokenizer, task, cfg)
    print(f"final eval: {final}  (baseline was {baseline})")
    metrics_f.write(json.dumps({"step": cfg.steps, "final": True, **final}) + "\n")
    save_adapter(model, out / "adapters", cfg.lora, cfg.model, cfg.steps)
    metrics_f.close()
    samples_f.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="GRPO LoRA training on MLX")
    d = TrainConfig()
    p.add_argument("--profile", default=None, help="named model profile: tiny | qwen36 | qwen38")
    p.add_argument("--model", default=None, help="model path/repo (overrides profile)")
    p.add_argument("--task", default=d.task)
    p.add_argument("--task-kwargs", default="{}", help='JSON, e.g. \'{"max_operand": 99}\'')
    p.add_argument("--chat-kwargs", default="{}", help='JSON chat-template kwargs, e.g. \'{"enable_thinking": false}\'')
    p.add_argument("--lora-keys", default=None, help="comma-separated module keys for LoRA")
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--batch-prompts", type=int, default=d.batch_prompts)
    p.add_argument("--group-size", type=int, default=d.group_size)
    p.add_argument("--group-stage1", type=int, default=d.group_stage1,
                   help="two-stage rollouts: sample this many members first, "
                        "abandon the group when stage-1 is decided (see "
                        "--stage1-skip); 0 = off")
    p.add_argument("--stage1-skip", default=d.stage1_skip,
                   choices=["saturated", "uniform"],
                   help="abandon rule: saturated = all-equal AND reward>=0.99; "
                        "uniform = any all-equal stage-1 group")
    p.add_argument("--update-adv-frac", type=float, default=d.update_adv_frac,
                   help="skip the backward for rollouts with |adv| < frac * "
                        "group max (denominator stays full-batch); 0 = off")
    p.add_argument("--token-subset-frac", type=float, default=d.token_subset_frac,
                   help="S-GRPO token-subset backprop: compute the surrogate on "
                        "this fraction of each rollout's completion tokens "
                        "(unbiased, denom-scaled); 0 = off, 1.0 = selective "
                        "path over all tokens")
    p.add_argument("--micro-batch", type=int, default=d.micro_batch)
    p.add_argument("--epochs-per-batch", type=int, default=d.epochs_per_batch)
    p.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--max-tool-rounds", type=int, default=d.max_tool_rounds,
                   help="tool tasks: call->response rounds per episode; a "
                        "breach ends the episode and is scored (tool_cap)")
    p.add_argument("--max-episode-tokens", type=int, default=d.max_episode_tokens,
                   help="tool tasks: generated-token budget per episode "
                        "(0 = max_new_tokens per round)")
    p.add_argument("--init-adapter", default=d.init_adapter,
                   help="start from this adapter dir/file (promoted checkpoint) "
                        "instead of zero LoRA")
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--kl-coef", type=float, default=d.kl_coef)
    p.add_argument("--clip-eps", type=float, default=d.clip_eps)
    p.add_argument("--no-normalize-std", dest="normalize_std", action="store_false")
    p.add_argument("--sage-r", type=int, default=d.sage_r,
                   help="SAGE-decoded members per group (0 = vanilla GRPO)")
    p.add_argument("--inject-r", type=int, default=d.inject_r,
                   help="off-policy demonstration members per group, from the "
                        "task's injected_completion() (0 = off)")
    p.add_argument("--abort-inactive-window", type=int,
                   default=d.abort_inactive_window,
                   help="abort when <2%% of groups were active over this many "
                        "steps — the policy has collapsed to zero variance "
                        "(0 = off)")
    p.add_argument("--sage-m", type=int, default=d.sage_m, help="SAGE beam/exploration width")
    p.add_argument("--sage-tr", type=float, default=d.sage_tr,
                   help="tolerance ratio TR=h/(2m): </think> accepted in top-h by Φ")
    p.add_argument("--sage-max-steps", dest="sage_max_reasoning_steps", type=int,
                   default=d.sage_max_reasoning_steps,
                   help="reasoning-step budget T_max")
    p.add_argument("--sage-max-step-tokens", type=int, default=d.sage_max_step_tokens,
                   help="per-reasoning-step token safety cap")
    p.add_argument("--sage-think-temp", dest="sage_think_temperature", type=float,
                   default=d.sage_think_temperature,
                   help="step-sampling temperature (paper: 1.0)")
    p.add_argument("--sage-answer-reserve", type=int, default=d.sage_answer_reserve,
                   help="tokens reserved for the answer phase (reasoning capped "
                        "at max_new_tokens minus this)")
    p.add_argument("--think-end", type=int, default=None,
                   help="end-of-thinking token id (default: from profile)")
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--eval-n", type=int, default=d.eval_n)
    p.add_argument("--eval-cells", default=d.eval_cells,
                   help="extra held-out subjects scored every eval, e.g. "
                        "'papers,trivia' (never trained on)")
    p.add_argument("--eval-cells-n", type=int, default=d.eval_cells_n,
                   help="items per extra subject (0 = --eval-n)")
    p.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every)
    p.add_argument("--resume-from", default=d.resume_from, metavar="DIR",
                   help="continue a previous run dir's newest resumable "
                        "checkpoint; writes to --out, never touches DIR")
    p.add_argument("--keep-resume", type=int, default=d.keep_resume,
                   help="optimizer snapshots to retain (0 disables resume "
                        "saving; 1 = only the latest checkpoint is resumable)")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--rank", type=int, default=d.lora.rank)
    p.add_argument("--lora-scale", type=float, default=d.lora.scale)
    p.add_argument("--lora-layers", type=int, default=d.lora.num_layers)
    p.add_argument("--no-share-prompt", dest="share_prompt", action="store_false")
    p.add_argument("--no-manage-machine", dest="manage_machine", action="store_false")
    p.add_argument("--lease-wait", dest="lease_wait_s", type=float, default=d.lease_wait_s,
                   help="seconds to wait for the machine lease")
    p.add_argument("--lease-block", default=d.lease_block,
                   choices=["exclusive", "experiments"],
                   help="lease block to request: experiments coexists with "
                        "whatever else holds memory (use it for runs that fit "
                        "alongside)")
    p.add_argument("--required-gb", type=float, default=d.required_gb,
                   help="override the worst-case run-size estimate (0 = "
                        "estimator); guard + swap watchdog still enforce it")
    p.add_argument("--rollout-batch-size", type=int, default=d.rollout_batch_size)
    p.add_argument("--length-penalty", type=float, default=d.length_penalty,
                   help="correctness-gated total-length penalty λ (0 = off)")
    p.add_argument("--length-budget", type=int, default=d.length_budget,
                   help="token budget for length normalisation (0 = max_new_tokens)")
    p.add_argument("--activation-headroom", dest="activation_headroom_gb", type=float,
                   default=d.activation_headroom_gb,
                   help="GB added to the memory-guard estimate (default 4)")
    p.add_argument("--swap-rate-mb-s", type=float, default=d.swap_rate_mb_s,
                   help="abort on sustained paging at/above this rate; "
                        "0 disables the rate detector")
    p.add_argument("--swap-guard-margin", dest="swap_guard_margin_gb", type=float,
                   default=d.swap_guard_margin_gb,
                   help="hard-abort if swap grows this many GB above baseline (0 = off)")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute layer forwards in backward instead of "
                        "retaining activations (memory for compute; enables "
                        "caps past 1024 on 96 GiB)")
    p.add_argument("--no-gdn-serial", dest="gdn_serial", action="store_false",
                   help="disable the serial GDN-scan backward (stock "
                        "gated_delta_ops path: 2.1 MB x S per GDN layer)")
    p.add_argument("--gdn-chunk", type=int, default=d.gdn_chunk,
                   help="serial-scan segment length")
    p.add_argument("--eval-max-new-tokens", type=int, default=d.eval_max_new_tokens,
                   help="eval-only length cap (0 = max_new_tokens); eval is "
                        "generation-only, so it may exceed the training cap")
    p.add_argument("--out", default=None, help="run dir (default runs/<ts>-<task>)")
    a = p.parse_args()

    prof = get_profile(a.profile) if a.profile else None
    model = a.model or (prof.model if prof else d.model)
    lora_keys = (
        a.lora_keys.split(",")
        if a.lora_keys
        else (list(prof.lora_keys) if prof and prof.lora_keys else None)
    )
    base_kwargs = dict(prof.chat_kwargs) if prof else {}
    think_end = a.think_end if a.think_end is not None else (prof.think_end if prof else None)
    if a.sage_r + a.inject_r >= a.group_size:
        p.error("--sage-r plus --inject-r must leave at least one "
                "ordinarily-sampled group member")
    if a.sage_r > 0:
        if think_end is None:
            p.error("--sage-r needs an end-of-thinking token: use a profile with think_end or pass --think-end")
        if prof:
            base_kwargs.update(prof.think_chat_kwargs)  # SAGE trains the thinking policy
    chat_kwargs = {**base_kwargs, **json.loads(a.chat_kwargs)}

    # Every flag whose dest names a TrainConfig field flows straight in; the
    # rest (profile-derived, JSON, LoRA) are assembled explicitly below.
    explicit = {"model", "task_kwargs", "chat_kwargs", "think_end"}
    field_names = {f.name for f in fields(TrainConfig)} - explicit
    cfg = TrainConfig(
        **{k: v for k, v in vars(a).items() if k in field_names},
        model=model,
        extra_eos=tuple(prof.extra_eos) if prof else (),
        task_kwargs=json.loads(a.task_kwargs),
        chat_kwargs=chat_kwargs,
        think_end=think_end,
        lora=LoraConfig(
            rank=a.rank,
            scale=a.lora_scale,
            num_layers=a.lora_layers,
            keys=lora_keys,
        ),
    )
    out = a.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}-{cfg.task}"
    print(json.dumps(asdict(cfg), indent=2))
    code = 0
    try:
        train(cfg, out)
    except BaseException:  # noqa: BLE001 — print it ourselves, then hard-exit
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        # Hard exit: live-tool libraries (ddgs) run non-daemon threads that
        # can hang inside their HTTP client and block interpreter shutdown —
        # a "finished" trainer then lingers holding ~30 GB of Metal buffers
        # and the next run OOMs (2026-08-16). Everything durable (adapters,
        # metrics, lease release) has already happened by here.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)



# ---------------------------------------------------------------------------
# Tool-using tasks: segmented episodes
# ---------------------------------------------------------------------------

def _tool_stop_ids(tokenizer) -> tuple[int, ...]:
    ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError("</tool_call> is not a single token on this "
                           f"tokenizer ({ids}); tool rounds need a 1-token stop")
    return (ids[0],)


def _sample_episodes(model, tokenizer, examples, cfg: TrainConfig, task,
                     group_size, temperature):
    """Batched episode rollout: the task's tools are offered via the chat
    template, calls are executed by task.run_tool() and the rendered
    response is spliced onto the row's KV cache (engine.rollout_episodes).
    Returns (per-example Episode groups, prompt token lists, BatchStats)."""
    chat_kwargs = {**getattr(task, "chat_template_kwargs", {}), **cfg.chat_kwargs}
    prompts = [encode_prompt(tokenizer, ex.messages, **chat_kwargs, **ex.chat_kwargs)
               for ex in examples]
    think_close = _think_close_marker(tokenizer, cfg, task)

    def on_tool(pi, gi, ep, seg_text):
        ex = examples[pi]
        # A call drafted inside an unclosed think block is not a call.
        visible, closed = _visible_reply(seg_text, think_close)
        call = parse_tool_call(visible) if closed else None
        if call is None:
            text = ("Malformed tool call: no <function=...> block found inside "
                    "<tool_call>. Emit the call again in the specified format.")
            rec = {"name": None, "args": {}, "ok": False, "hits": 0,
                   "result": text}
        else:
            name, args = call
            r = task.run_tool(name, args, ex)
            text = r.text
            rec = {"name": name, "args": args, "result": text, **r.meta}
        ep.tool_calls.append(rec)
        return tool_response_ids(tokenizer, text, **chat_kwargs, **ex.chat_kwargs)

    CAP_MSG = ("Error: tool call limit reached — no more tool calls are available "
               "in this conversation. Answer the user now with what you have, or "
               "say plainly what you could not find.")

    def on_cap(pi, gi, ep):
        ex = examples[pi]
        ep.tool_calls.append({"name": None, "args": {}, "ok": False, "hits": 0,
                              "capped": True, "result": CAP_MSG})
        return tool_response_ids(tokenizer, CAP_MSG, **chat_kwargs, **ex.chat_kwargs)

    groups, stats = rollout_episodes(
        model, tokenizer, prompts, group_size, cfg.max_new_tokens, temperature,
        on_tool=on_tool, on_cap=on_cap, tool_stop_ids=_tool_stop_ids(tokenizer),
        max_tool_rounds=cfg.max_tool_rounds,
        max_episode_tokens=cfg.max_episode_tokens or None,
        extra_eos=tuple(cfg.extra_eos), share_prompt=cfg.share_prompt,
        completion_batch_size=cfg.rollout_batch_size,
    )
    return groups, prompts, stats


def _episode_record(tokenizer, ep, think_close) -> dict:
    """What the task grades: the visible text of the LAST generated segment
    (after the final think-close), the tool trace, and how it ended. An
    episode that ended on a tool call (cap) or mid-generation has no reply."""
    last = ep.last_generated
    visible = ""
    if last is not None and last.finish_reason == "stop":
        toks = last.tokens[:-1]  # drop EOS
        visible, _ = _visible_reply(tokenizer.decode(toks), think_close)
    return {"visible": visible, "tool_calls": list(ep.tool_calls),
            "finish": ep.finish_reason, "rounds": ep.rounds}


def _injected_episode(tokenizer, task, ex, cfg, chat_kwargs):
    """Build an Episode from the task's oracle segments (text, generated)."""
    eos_id = next(iter(sorted(tokenizer.eos_token_ids)))
    ep = Episode()
    segs = task.injected_episode(ex)
    for k, (text, generated) in enumerate(segs):
        if generated:
            toks = tokenizer.encode(text, add_special_tokens=False)
            last = k == len(segs) - 1
            if last:
                toks = toks + [eos_id]
            ep.segments.append(Segment(tokens=toks, logprobs=[0.0] * len(toks),
                                       generated=True,
                                       finish_reason="stop" if last else "tool"))
        else:
            toks = tool_response_ids(tokenizer, text, **chat_kwargs, **ex.chat_kwargs)
            ep.segments.append(Segment(tokens=toks, logprobs=[0.0] * len(toks),
                                       generated=False))
            ep.rounds += 1
            call = parse_tool_call(segs[k - 1][0]) if k else None
            if call:
                r = task.run_tool(call[0], call[1], ex)
                ep.tool_calls.append({"name": call[0], "args": call[1],
                                      "result": r.text, **r.meta})
    ep.finish_reason = "stop"
    return ep


def collect_episodes(model, tokenizer, examples, cfg: TrainConfig, task):
    """Episode counterpart of collect_rollouts for tasks that own tools.
    Returns (rollouts, stats, groups_skipped) — one Rollout per episode with
    gen_mask marking generated vs injected tokens."""
    inject_r = cfg.inject_r
    n_sampled = cfg.group_size - inject_r
    think_close = _think_close_marker(tokenizer, cfg, task)
    chat_kwargs = {**getattr(task, "chat_template_kwargs", {}), **cfg.chat_kwargs}
    # id(Episode) -> (Episode, record, result); holds the Episode so a freed
    # id can never alias a stale grade (see collect_rollouts).
    graded: dict[int, tuple[Episode, dict, object]] = {}

    def _grade(pairs):
        recs = [_episode_record(tokenizer, ep, think_close) for _, ep in pairs]
        ress = task.episode_reward([ex for ex, _ in pairs], recs)
        for (_, ep), rec, res in zip(pairs, recs, ress):
            graded[id(ep)] = (ep, rec, res)

    examples, groups, prompts, stats, skipped = _two_stage(
        examples, cfg, n_sampled,
        lambda exs, n: _sample_episodes(model, tokenizer, exs, cfg, task, n, cfg.temperature),
        _grade, lambda ep: graded[id(ep)][2].total)
    if inject_r:
        for ex, group in zip(examples, groups):
            for _ in range(inject_r):
                group.append(_injected_episode(tokenizer, task, ex, cfg, chat_kwargs))
    pending = [(ex, ep) for ex, group in zip(examples, groups) for ep in group
               if id(ep) not in graded]
    if pending:
        _grade(pending)
    rollouts: list[Rollout] = []
    for ex, prompt, group in zip(examples, prompts, groups):
        for gi, ep in enumerate(group):
            _, rec, res = graded[id(ep)]
            toks = ep.completion_tokens
            parts = dict(res.parts)
            parts["base_reward"] = round(res.total, 4)
            parts["gen_tokens"] = ep.gen_count
            parts["tool_cap"] = float(ep.finish_reason == "tool_cap")
            parts["capped"] = float(ep.capped)
            rollouts.append(Rollout(
                prompt_tokens=list(prompt),
                completion_tokens=toks,
                sampling_logprobs=ep.sampling_logprobs,
                text=tokenizer.decode(toks),
                reward=res.total,
                reward_parts=parts,
                meta=ex.meta,
                injected=gi >= n_sampled,
                finish=ep.finish_reason,
                gen_mask=ep.gen_mask,
                tool_calls=[{k: v for k, v in c.items() if k != "result"}
                            for c in ep.tool_calls],
            ))
    return rollouts, stats, skipped


def build_eval_cells(cfg: TrainConfig) -> dict:
    """Held-out subjects that are SCORED at every eval but never trained on.

    The trained subject alone cannot tell learning from memorisation: a policy
    that picks up one domain's surface quirks and one that learns to check
    before it answers look the same there, and only come apart on a subject
    the gradient never saw. Same task class and judge wiring as training --
    only the domain and its calibration file change.
    """
    if not cfg.eval_cells:
        return {}
    from .tasks.honesty import CALIB, DOMAIN_SCOPED, HonestyTask
    cells = {}
    for domain in filter(None, (s.strip() for s in cfg.eval_cells.split(","))):
        kw = dict(cfg.task_kwargs)
        for k in DOMAIN_SCOPED:             # per-domain, never carried over
            kw.pop(k, None)
        kw["domain"] = domain
        if domain in CALIB:
            kw["calib_file"] = CALIB[domain]
        cells[domain] = HonestyTask(**kw)
    return cells


def evaluate_cells(model, tokenizer, cells, cfg: TrainConfig) -> dict:
    """Score every held-out subject; keys become eval_<domain>_*.

    One subject failing (a dead tool endpoint, a judge timeout) must not lose
    the step's real metrics, so each is guarded separately.
    """
    if not cells:
        return {}
    ecfg = replace(cfg, eval_n=cfg.eval_cells_n or cfg.eval_n)
    rec = {}
    for name, task in cells.items():
        t0 = time.time()
        try:
            r = evaluate(model, tokenizer, task, ecfg)
        except Exception as e:                      # noqa: BLE001 - see docstring
            print(f"   [eval:{name}] skipped: {type(e).__name__}: {e}", flush=True)
            continue
        for k, v in r.items():
            rec[f"eval_{name}_" + (k[5:] if k.startswith("eval_") else k)] = v
        rec[f"eval_{name}_s"] = round(time.time() - t0, 1)
        print(f"   [eval:{name}] reward {r.get('eval_reward', float('nan')):+.3f} "
              f"correct {r.get('eval_correct', 0.0):.2f} "
              f"called {r.get('eval_called', 0.0):.2f}  ({rec[f'eval_{name}_s']}s)", flush=True)
        mx.clear_cache()
    return rec


def evaluate_episodes(model, tokenizer, task, cfg: TrainConfig):
    """Greedy episode eval on the task's held-out split."""
    cfg, examples = _eval_set(task, cfg)
    # Chunked, with backoff: this used to hand the whole eval set to the
    # generator at once, so raising eval_n to get a readable curve (160 items,
    # to cut the standard error below the effect size) killed the run on a
    # Metal OOM at the first eval -- after 9 steps and before the first
    # checkpoint. Same failure the matrix eval had; same fix.
    groups, _, _ = _chunked_backoff(
        lambda exs: _sample_episodes(model, tokenizer, exs, cfg, task, 1, 0.0),
        examples, max(1, cfg.rollout_batch_size or len(examples)))
    think_close = _think_close_marker(tokenizer, cfg, task)
    records = [_episode_record(tokenizer, g[0], think_close) for g in groups]
    results = task.episode_reward(examples, records)
    # Length-capped episodes measure max_new_tokens, not the policy: they are
    # excluded from every reward/rate mean and reported as their own rate.
    # Loud by machine rule — a truncated completion must never be silently
    # graded (the -1.67-vs-+0.02 same-cell swing in the 08-21 matrix was
    # entirely this).
    is_capped = [bool(r.parts.get("len_capped")) for r in results]
    n_capped = sum(is_capped)
    if n_capped:
        print(f"   [eval] {n_capped}/{len(results)} episodes len-capped at "
              f"max_new_tokens={cfg.max_new_tokens} — excluded from eval means",
              flush=True)
    alive = [(ex, r) for ex, r, c in zip(examples, results, is_capped) if not c]
    a_results = [r for _, r in alive] or results  # all-capped: report raw
    a_examples = [ex for ex, _ in alive] or examples
    out = {
        "eval_reward": float(np.mean([r.total for r in a_results])),
        "eval_mean_len": float(np.mean([g[0].gen_count for g in groups])),
        "eval_rounds": float(np.mean([g[0].rounds for g in groups])),
        "eval_n_graded": len(alive),
    }
    out.update(_mean_parts(a_results))
    out["eval_len_capped"] = float(np.mean(is_capped))  # over ALL episodes, after the parts
    # Per-regime slices: the falsification test lives here (post vs future
    # on the same papers must differ).
    for key in ("regime", "band"):
        for val in sorted({ex.meta.get(key) for ex in a_examples} - {None}):
            sel = [r for ex, r in zip(a_examples, a_results) if ex.meta.get(key) == val]
            tag = val if key == "regime" else f"band_{val}"
            out[f"eval_{tag}_reward"] = float(np.mean([r.total for r in sel]))
            out[f"eval_{tag}_called"] = float(np.mean([r.parts.get("called", 0.0) for r in sel]))
            out[f"eval_{tag}_correct"] = float(np.mean([r.parts.get("correct", 0.0) for r in sel]))
            out[f"eval_{tag}_n"] = len(sel)
    return out


# ---------------------------------------------------------------------------
# Multi-turn (design D3): re-render, re-prefill, one training row per turn
# ---------------------------------------------------------------------------

def _episode_messages(ep, tokenizer, think_close) -> list[dict]:
    """The chat messages a finished episode adds to a transcript: assistant
    tool-call turns and their tool responses, then the final assistant
    reply. Rendered through the template on the next turn, so completed
    turns come out exactly as the template serves them (no think blocks)."""
    msgs = []
    calls = list(ep.tool_calls)
    ci = 0
    for seg in ep.segments:
        if seg.generated:
            toks = seg.tokens[:-1] if seg.finish_reason == "stop" else seg.tokens
            text, _ = _visible_reply(tokenizer.decode(toks), think_close)
            msgs.append({"role": "assistant", "content": text.strip()})
        else:
            result = calls[ci]["result"] if ci < len(calls) and "result" in calls[ci] else ""
            ci += 1
            msgs.append({"role": "tool", "content": result})
    return msgs


def _chunked_backoff(sample_fn, items, chunk):
    """Feed `items` to sample_fn in chunks, halving a chunk on Metal OOM
    instead of dying. Whole-batch submission with no backoff killed four
    runs the week of 08-18; every generation entry point goes through a
    chunked path now. Returns (groups, prompts, last_stats)."""
    out_groups, out_prompts, stats = [], [], None
    lo = 0
    while lo < len(items):
        n_try = min(chunk, len(items) - lo)
        while True:
            try:
                g, p, stats = sample_fn(items[lo:lo + n_try])
                break
            except RuntimeError as e:
                if ("Insufficient Memory" not in str(e)
                        and "out of memory" not in str(e).lower()):
                    raise
                mx.clear_cache()
                if n_try == 1:
                    raise
                n_try = max(1, n_try // 2)
                print(f"   [gen oom] retrying at {n_try} rows", flush=True)
        out_groups.extend(g)
        out_prompts.extend(p)
        lo += n_try
        mx.clear_cache()
    return out_groups, out_prompts, stats


def collect_multiturn(model, tokenizer, examples, cfg: TrainConfig, task):
    """Multi-turn rollouts for tasks that define `turns > 1` and
    `followup(example, turn, history) -> Example`.

    Turn 0 samples group_size members per example. Every member then
    carries ITS OWN transcript: its reply (and tool rounds) plus the task's
    next user message are re-rendered through the chat template and
    re-prefilled for turn 1, and so on. Each turn is one training row —
    prompt = rendered history (context, mask 0), completion = that turn's
    generation (mask 1) — graded on its own turn's gold, grouped with its
    seven siblings at the same turn (same example, same turn index) for the
    GRPO baseline. Row order is [ex0 turn0 ×G, ex1 turn0 ×G, ..., ex0
    turn1 ×G, ...] so `reshape(-1, group_size)` groups correctly.
    Not stage-1-skippable (rows share no prompt after turn 0)."""
    G = cfg.group_size
    # Episode path for any task that grades episodes (tool tasks, even when
    # served with tools=[]); string path only for plain reward() tasks.
    tools = hasattr(task, "episode_reward") or getattr(task, "tools", None)
    think_close = _think_close_marker(tokenizer, cfg, task)
    turns = int(getattr(task, "turns", 1))
    # per-row state: current Example (messages = full history so far); each
    # row owns its copy because followup() extends the history in place
    rows = [Example(messages=list(ex.messages), meta=dict(ex.meta),
                    chat_kwargs=dict(ex.chat_kwargs))
            for ex in examples for _ in range(G)]
    rollouts: list[Rollout] = []
    stats = None
    for t in range(turns):
        chunk = max(1, cfg.rollout_batch_size or len(rows))
        if tools:
            groups, prompts, stats = _chunked_backoff(
                lambda rs: _sample_episodes(model, tokenizer, rs, cfg, task,
                                            1, cfg.temperature), rows, chunk)
            eps = [g[0] for g in groups]
            recs = [_episode_record(tokenizer, ep, think_close) for ep in eps]
            results = task.episode_reward(rows, recs)
        else:
            groups, prompts, stats = _chunked_backoff(
                lambda rs: _sample_batched(model, tokenizer, rs, cfg, 1,
                                           cfg.temperature, task), rows, chunk)
            eps = []
            for g in groups:
                comp = g[0]
                ep = Episode()
                ep.segments.append(Segment(tokens=list(comp.tokens), logprobs=list(comp.logprobs),
                                           generated=True, finish_reason=comp.finish_reason))
                ep.finish_reason = comp.finish_reason
                eps.append(ep)
            results = _grade_batch(task, [
                (ex, _visible_reply(_completion_text(tokenizer, g[0]), think_close)[0])
                for ex, g in zip(rows, groups)])
        for ex, prompt, ep, res in zip(rows, prompts, eps, results):
            toks = ep.completion_tokens
            parts = dict(res.parts)
            parts["base_reward"] = round(res.total, 4)
            parts["turn"] = float(t)
            parts["gen_tokens"] = ep.gen_count
            rollouts.append(Rollout(
                prompt_tokens=list(prompt), completion_tokens=toks,
                sampling_logprobs=ep.sampling_logprobs, text=tokenizer.decode(toks),
                reward=res.total, reward_parts=parts, meta={**ex.meta, "turn": t},
                finish=ep.finish_reason, gen_mask=ep.gen_mask,
                tool_calls=[{k: v for k, v in c.items() if k != "result"} for c in ep.tool_calls],
            ))
        if t + 1 < turns:
            new_rows = []
            for ex, ep in zip(rows, eps):
                history = list(ex.messages) + _episode_messages(ep, tokenizer, think_close)
                new_rows.append(task.followup(ex, t + 1, history))
            rows = new_rows
    return rollouts, stats, 0


def evaluate_multiturn(model, tokenizer, task, cfg: TrainConfig):
    """Greedy multi-turn eval: per-turn reward and rates on the held-out
    split, so 'does the behaviour survive to turn 3+' is a number."""
    cfg, examples = _eval_set(task, cfg)
    ecfg = replace(cfg, group_size=1, temperature=0.0)
    rollouts, _, _ = collect_multiturn(model, tokenizer, examples, ecfg, task)
    out = {"eval_reward": float(np.mean([r.reward for r in rollouts])),
           "eval_mean_len": float(np.mean([len(r.completion_tokens) for r in rollouts]))}
    for key in sorted({k for r in rollouts for k in r.reward_parts}):
        out[f"eval_{key}"] = float(np.mean([r.reward_parts.get(key, 0.0) for r in rollouts]))
    for t in sorted({r.meta.get("turn") for r in rollouts}):
        sel = [r for r in rollouts if r.meta.get("turn") == t]
        out[f"eval_turn{t}_reward"] = float(np.mean([r.reward for r in sel]))
        for k in ("called", "correct", "abstain", "denial", "no_reply"):
            out[f"eval_turn{t}_{k}"] = float(np.mean([r.reward_parts.get(k, 0.0) for r in sel]))
        out[f"eval_turn{t}_n"] = len(sel)
    for regime in sorted({r.meta.get("regime") for r in rollouts} - {None}):
        sel = [r for r in rollouts if r.meta.get("regime") == regime]
        out[f"eval_{regime}_reward"] = float(np.mean([r.reward for r in sel]))
        out[f"eval_{regime}_n"] = len(sel)
    return out


if __name__ == "__main__":
    main()
