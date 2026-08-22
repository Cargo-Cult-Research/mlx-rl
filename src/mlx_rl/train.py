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
from dataclasses import asdict, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from . import machine
from .config import LoraConfig, TrainConfig
from .engine import Completion, rollout_episodes, rollout_groups, sage_completion
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
from .tasks.qa_arxiv import parse_tool_call


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
    graded: dict[int, object] = {}  # id(Completion) -> RewardResult (no double-grade)
    skipped = 0
    if 0 < cfg.group_stage1 < n_sampled:
        g1 = cfg.group_stage1
        groups, prompts, stats = _sample_batched(
            model, tokenizer, examples, cfg, g1, cfg.temperature, task
        )
        flat = [(ex, comp) for ex, group in zip(examples, groups)
                for comp in group]
        results = _grade_batch(task, [
            (ex, _visible_reply(_completion_text(tokenizer, comp),
                                think_close)[0]) for ex, comp in flat])
        for (_, comp), res in zip(flat, results):
            graded[id(comp)] = res  # judge/code grading is costly; cache it
        live = [i for i, (_, group) in enumerate(zip(examples, groups))
                if not _stage1_dead([graded[id(c)].total for c in group],
                                    cfg.stage1_skip)]
        skipped = len(examples) - len(live)
        examples = [examples[i] for i in live]
        groups = [groups[i] for i in live]
        prompts = [prompts[i] for i in live]
        if examples and n_sampled > g1:
            mx.clear_cache()  # stage-1 KV is dead weight under stage 2
            groups2, _, stats2 = _sample_batched(
                model, tokenizer, examples, cfg, n_sampled - g1,
                cfg.temperature, task
            )
            for group, extra in zip(groups, groups2):
                group.extend(extra)
            stats = stats2  # tok/s of the larger phase; a metric, not a ledger
    else:
        groups, prompts, stats = _sample_batched(
            model, tokenizer, examples, cfg, n_sampled, cfg.temperature, task
        )
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
        results = _grade_batch(task, [
            (ex, _visible_reply(_completion_text(tokenizer, comp),
                                think_close)[0]) for ex, comp in pending])
        for (_, comp), res in zip(pending, results):
            graded[id(comp)] = res
    rollouts: list[Rollout] = []
    for ex, prompt, group in zip(examples, prompts, groups):
        for gi, comp in enumerate(group):
            text = _completion_text(tokenizer, comp)
            visible, closed = _visible_reply(text, think_close)
            res = graded[id(comp)]
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
            gated = res.total * (1.0 - cfg.length_penalty * len_norm)
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


def update_policy(model, optimizer, loss_and_grad, rollouts, advantages, cfg,
                  pad_id, denom_tokens: float | None = None,
                  subset_rng: np.random.Generator | None = None):
    """One (or more) clipped-PG epochs over the rollout batch, microbatched
    with gradient accumulation. Returns (pg_mean, kl_mean) per token.

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

    pg_total, kl_total = 0.0, 0.0
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


def evaluate(model, tokenizer, task, cfg: TrainConfig):
    """Greedy decode on a fixed held-out set (batched); returns mean reward + rates."""
    if int(getattr(task, "turns", 1)) > 1:
        return evaluate_multiturn(model, tokenizer, task, cfg)
    if getattr(task, "tools", None):
        return evaluate_episodes(model, tokenizer, task, cfg)
    if cfg.eval_max_new_tokens:
        cfg = replace(cfg, max_new_tokens=cfg.eval_max_new_tokens)
    rng = random.Random(cfg.seed + 100_000)  # disjoint from training stream
    esample = getattr(task, "eval_sample", task.sample)  # held-out split, no leak
    examples = [esample(rng) for _ in range(cfg.eval_n)]
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
    for key in sorted({k for r in results for k in r.parts}):
        out[f"eval_{key}"] = float(np.mean([r.parts.get(key, 0.0) for r in results]))
    return out


def train(cfg: TrainConfig, out_dir: str | Path) -> Path:
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
        vlm=cfg.vlm_policy,
    )
    print(f"loaded {cfg.model}: {info}")
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
    # Model-graded tasks (telephone's frozen-listener reward) need the live
    # model; tasks are constructed before load, so hand it over here.
    bind = getattr(task, "bind_model", None)
    if bind is not None:
        bind(model, tokenizer)
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
        baseline = evaluate(model, tokenizer, task, cfg)
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
        advantages = np.array(
            group_advantages(mx.array(rewards), cfg.normalize_std)
        ).reshape(-1, cfg.group_size)
        active = np.array(active_groups(mx.array(rewards)))

        t1 = time.time()
        n_pruned = 0
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
                a2 = np.abs(upd_adv).reshape(-1, cfg.group_size)
                keep = (a2 >= cfg.update_adv_frac
                        * a2.max(axis=1, keepdims=True)).reshape(-1)
                n_pruned = int((~keep).sum())
                upd_rollouts = [r for r, k in zip(upd_rollouts, keep) if k]
                upd_adv = upd_adv[keep]
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

        rec = {
            "step": step,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
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
            "ts": round(time.time(), 1),
        }
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
        nll = [-float(np.mean(r.sampling_logprobs)) for r in rollouts
               if not r.sage and not r.injected and r.sampling_logprobs]
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
        if cfg.eval_every and step % cfg.eval_every == 0:
            rec.update(evaluate(model, tokenizer, task, cfg))
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
        try:
            from .engine import _tap
            _tap().note(
                f"step {step}: reward {rec['reward_mean']:+.2f}±{rec['reward_std']:.2f} "
                f"active {int(active.sum())}/{len(rewards)} len {rec['mean_len']:.0f} "
                + " ".join(f"{k[5:]} {v:.2f}" for k, v in rec.items()
                           if k.startswith("frac_") and k in (
                               "frac_correct", "frac_called", "frac_abstain",
                               "frac_denial", "frac_no_reply", "frac_len_capped"))
                + f" | first group: {' '.join(f'{r.reward:+.1f}' for r in group)}"
                + f" | gen {rec['gen_s']:.0f}s upd {rec['update_s']:.0f}s")
        except Exception:
            pass
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

        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            save_adapter(model, out / "adapters", cfg.lora, cfg.model, step)
            if cfg.keep_resume:
                save_resume(out, step, optimizer, rng, subset_rng, task,
                            activity_window, cfg.keep_resume)

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
    p.add_argument("--profile", default=None, help="named model profile: tiny | qwen36 | gemma26")
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
    p.add_argument("--sage-max-steps", type=int, default=d.sage_max_reasoning_steps,
                   help="reasoning-step budget T_max")
    p.add_argument("--sage-max-step-tokens", type=int, default=d.sage_max_step_tokens,
                   help="per-reasoning-step token safety cap")
    p.add_argument("--sage-think-temp", type=float, default=d.sage_think_temperature,
                   help="step-sampling temperature (paper: 1.0)")
    p.add_argument("--sage-answer-reserve", type=int, default=d.sage_answer_reserve,
                   help="tokens reserved for the answer phase (reasoning capped "
                        "at max_new_tokens minus this)")
    p.add_argument("--think-end", type=int, default=None,
                   help="end-of-thinking token id (default: from profile)")
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--eval-n", type=int, default=d.eval_n)
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
    p.add_argument("--lease-wait", type=float, default=d.lease_wait_s, help="seconds to wait for the machine lease")
    p.add_argument("--lease-block", default=d.lease_block,
                   choices=["exclusive", "experiments"],
                   help="memlease block: experiments coexists with the :8084 "
                        "serving slot (use for runs that fit in ~40 GB)")
    p.add_argument("--required-gb", type=float, default=d.required_gb,
                   help="override the worst-case run-size estimate (0 = "
                        "estimator); guard + swap watchdog still enforce it")
    p.add_argument("--rollout-batch-size", type=int, default=d.rollout_batch_size)
    p.add_argument("--length-penalty", type=float, default=d.length_penalty,
                   help="correctness-gated total-length penalty λ (0 = off)")
    p.add_argument("--length-budget", type=int, default=d.length_budget,
                   help="token budget for length normalisation (0 = max_new_tokens)")
    p.add_argument("--activation-headroom", type=float, default=d.activation_headroom_gb,
                   help="GB added to the memory-guard estimate (default 4)")
    p.add_argument("--swap-rate-mb-s", type=float, default=d.swap_rate_mb_s,
                   help="abort on sustained paging at/above this rate; "
                        "0 disables the rate detector")
    p.add_argument("--swap-guard-margin", type=float, default=d.swap_guard_margin_gb,
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

    cfg = TrainConfig(
        model=model,
        profile=a.profile,
        vlm_policy=prof.vlm if prof else False,
        extra_eos=tuple(prof.extra_eos) if prof else (),
        share_prompt=a.share_prompt,
        manage_machine=a.manage_machine,
        lease_wait_s=a.lease_wait,
        lease_block=a.lease_block,
        required_gb=a.required_gb,
        rollout_batch_size=a.rollout_batch_size,
        task=a.task,
        task_kwargs=json.loads(a.task_kwargs),
        chat_kwargs=chat_kwargs,
        steps=a.steps,
        batch_prompts=a.batch_prompts,
        group_size=a.group_size,
        group_stage1=a.group_stage1,
        stage1_skip=a.stage1_skip,
        update_adv_frac=a.update_adv_frac,
        token_subset_frac=a.token_subset_frac,
        micro_batch=a.micro_batch,
        max_tool_rounds=a.max_tool_rounds,
        max_episode_tokens=a.max_episode_tokens,
        init_adapter=a.init_adapter,
        epochs_per_batch=a.epochs_per_batch,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        lr=a.lr,
        kl_coef=a.kl_coef,
        clip_eps=a.clip_eps,
        normalize_std=a.normalize_std,
        sage_r=a.sage_r,
        inject_r=a.inject_r,
        abort_inactive_window=a.abort_inactive_window,
        sage_m=a.sage_m,
        sage_tr=a.sage_tr,
        sage_max_reasoning_steps=a.sage_max_steps,
        sage_max_step_tokens=a.sage_max_step_tokens,
        sage_think_temperature=a.sage_think_temp,
        sage_answer_reserve=a.sage_answer_reserve,
        think_end=think_end,
        length_penalty=a.length_penalty,
        length_budget=a.length_budget,
        activation_headroom_gb=a.activation_headroom,
        swap_guard_margin_gb=a.swap_guard_margin,
        swap_rate_mb_s=a.swap_rate_mb_s,
        eval_every=a.eval_every,
        eval_n=a.eval_n,
        eval_max_new_tokens=a.eval_max_new_tokens,
        grad_checkpoint=a.grad_checkpoint,
        gdn_serial=a.gdn_serial,
        gdn_chunk=a.gdn_chunk,
        checkpoint_every=a.checkpoint_every,
        resume_from=a.resume_from,
        keep_resume=a.keep_resume,
        seed=a.seed,
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
    from .engine import Episode, Segment
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
    graded: dict[int, tuple[dict, object]] = {}  # id(Episode) -> (record, result)
    skipped = 0

    def _grade(pairs):
        recs = [_episode_record(tokenizer, ep, think_close) for _, ep in pairs]
        ress = task.episode_reward([ex for ex, _ in pairs], recs)
        for (_, ep), rec, res in zip(pairs, recs, ress):
            graded[id(ep)] = (rec, res)

    if 0 < cfg.group_stage1 < n_sampled:
        # Two-stage: sample a few, abandon groups already decided (see
        # collect_rollouts / _stage1_dead), then sample the rest for the live ones.
        g1 = cfg.group_stage1
        groups, prompts, stats = _sample_episodes(
            model, tokenizer, examples, cfg, task, g1, cfg.temperature)
        _grade([(ex, ep) for ex, group in zip(examples, groups) for ep in group])
        live = [i for i, group in enumerate(groups)
                if not _stage1_dead([graded[id(ep)][1].total for ep in group],
                                    cfg.stage1_skip)]
        skipped = len(examples) - len(live)
        examples = [examples[i] for i in live]
        groups = [groups[i] for i in live]
        prompts = [prompts[i] for i in live]
        if examples:
            mx.clear_cache()
            groups2, _, stats2 = _sample_episodes(
                model, tokenizer, examples, cfg, task, n_sampled - g1, cfg.temperature)
            for group, extra in zip(groups, groups2):
                group.extend(extra)
            stats = stats2
    else:
        groups, prompts, stats = _sample_episodes(
            model, tokenizer, examples, cfg, task, n_sampled, cfg.temperature)
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
            rec, res = graded[id(ep)]
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


def evaluate_episodes(model, tokenizer, task, cfg: TrainConfig):
    """Greedy episode eval on the task's held-out split."""
    if cfg.eval_max_new_tokens:
        cfg = replace(cfg, max_new_tokens=cfg.eval_max_new_tokens)
    rng = random.Random(cfg.seed + 100_000)
    esample = getattr(task, "eval_sample", task.sample)
    examples = [esample(rng) for _ in range(cfg.eval_n)]
    # Chunked, with backoff: this used to hand the whole eval set to the
    # generator at once, so raising eval_n to get a readable curve (160 items,
    # to cut the standard error below the effect size) killed the run on a
    # Metal OOM at the first eval -- after 9 steps and before the first
    # checkpoint. Same failure the matrix eval had; same fix.
    chunk = max(1, cfg.rollout_batch_size or len(examples))
    groups = []
    lo = 0
    while lo < len(examples):
        n_try = min(chunk, len(examples) - lo)
        while True:
            try:
                g, _, _ = _sample_episodes(model, tokenizer, examples[lo:lo + n_try],
                                           cfg, task, 1, 0.0)
                break
            except RuntimeError as e:
                if "Insufficient Memory" not in str(e) and "out of memory" not in str(e).lower():
                    raise
                mx.clear_cache()
                if n_try == 1:
                    raise
                n_try = max(1, n_try // 2)
                print(f"   [eval oom] retrying at {n_try} prompts", flush=True)
        groups.extend(g)
        lo += n_try
        mx.clear_cache()
    think_close = _think_close_marker(tokenizer, cfg, task)
    records = [_episode_record(tokenizer, g[0], think_close) for g in groups]
    results = task.episode_reward(examples, records)
    out = {
        "eval_reward": float(np.mean([r.total for r in results])),
        "eval_mean_len": float(np.mean([g[0].gen_count for g in groups])),
        "eval_rounds": float(np.mean([g[0].rounds for g in groups])),
    }
    for key in sorted({k for r in results for k in r.parts}):
        out[f"eval_{key}"] = float(np.mean([r.parts.get(key, 0.0) for r in results]))
    # Per-regime slices: the falsification test lives here (post vs future
    # on the same papers must differ).
    for key in ("regime", "band"):
        for val in sorted({ex.meta.get(key) for ex in examples} - {None}):
            sel = [r for ex, r in zip(examples, results) if ex.meta.get(key) == val]
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
    from .engine import Episode
    G = cfg.group_size
    # Episode path for any task that grades episodes (tool tasks, even when
    # served with tools=[]); string path only for plain reward() tasks.
    tools = hasattr(task, "episode_reward") or getattr(task, "tools", None)
    think_close = _think_close_marker(tokenizer, cfg, task)
    chat_kwargs = {**getattr(task, "chat_template_kwargs", {}), **cfg.chat_kwargs}
    turns = int(getattr(task, "turns", 1))
    # per-row state: current Example (messages = full history so far)
    rows = [replace_example(ex) for ex in examples for _ in range(G)]
    rollouts: list[Rollout] = []
    stats = None
    for t in range(turns):
        if tools:
            groups, prompts, stats = _sample_episodes(
                model, tokenizer, rows, cfg, task, 1, cfg.temperature)
            eps = [g[0] for g in groups]
            recs = [_episode_record(tokenizer, ep, think_close) for ep in eps]
            results = task.episode_reward(rows, recs)
        else:
            groups, prompts, stats = _sample_batched(
                model, tokenizer, rows, cfg, 1, cfg.temperature, task)
            eps = []
            for g in groups:
                comp = g[0]
                ep = Episode()
                from .engine import Segment
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


def replace_example(ex):
    from .tasks.base import Example
    return Example(messages=list(ex.messages), meta=dict(ex.meta),
                   chat_kwargs=dict(ex.chat_kwargs))


def evaluate_multiturn(model, tokenizer, task, cfg: TrainConfig):
    """Greedy multi-turn eval: per-turn reward and rates on the held-out
    split, so 'does the behaviour survive to turn 3+' is a number."""
    if cfg.eval_max_new_tokens:
        cfg = replace(cfg, max_new_tokens=cfg.eval_max_new_tokens)
    rng = random.Random(cfg.seed + 100_000)
    esample = getattr(task, "eval_sample", task.sample)
    examples = [esample(rng) for _ in range(cfg.eval_n)]
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
