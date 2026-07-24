"""GRPO training loop and CLI.

Usage:
    uv run mlx-rl-train --steps 30 --out runs/smoke
    uv run mlx-rl-train --model <hf-repo-or-path> --task arithmetic ...
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from . import machine
from .config import LoraConfig, TrainConfig
from .engine import rollout_groups, sage_completion
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
)
from .tasks import get_task


def _sample_batched(model, tokenizer, examples, cfg: TrainConfig, group_size, temperature, task=None):
    """Batched rollout over examples; returns (per-example completion groups,
    prompt token lists, BatchStats)."""
    chat_kwargs = {**getattr(task, "chat_template_kwargs", {}), **cfg.chat_kwargs}
    prompts = [encode_prompt(tokenizer, ex.messages, **chat_kwargs) for ex in examples]
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
    n_sampled = cfg.group_size - sage_r
    think_close = _think_close_marker(tokenizer, cfg, task)
    graded: dict[int, object] = {}  # id(Completion) -> RewardResult (no double-grade)
    skipped = 0
    if 0 < cfg.group_stage1 < n_sampled:
        g1 = cfg.group_stage1
        groups, prompts, stats = _sample_batched(
            model, tokenizer, examples, cfg, g1, cfg.temperature, task
        )
        live: list[int] = []
        for i, (ex, group) in enumerate(zip(examples, groups)):
            rs = []
            for comp in group:
                visible, _ = _visible_reply(
                    _completion_text(tokenizer, comp), think_close)
                res = task.reward(ex, visible)
                graded[id(comp)] = res  # code grading = subprocess; cache it
                rs.append(res.total)
            if not _stage1_dead(rs, cfg.stage1_skip):
                live.append(i)
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
    rollouts: list[Rollout] = []
    for ex, prompt, group in zip(examples, prompts, groups):
        for gi, comp in enumerate(group):
            text = _completion_text(tokenizer, comp)
            visible, closed = _visible_reply(text, think_close)
            res = graded.get(id(comp)) or task.reward(ex, visible)
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
                    sage=gi >= n_sampled,
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
        if cfg.max_grad_norm > 0:
            acc, _ = optim.clip_grad_norm(acc, cfg.max_grad_norm)
        optimizer.update(model, acc)
        mx.eval(model.parameters(), optimizer.state)
    return pg_total / denom, kl_total / denom


def evaluate(model, tokenizer, task, cfg: TrainConfig):
    """Greedy decode on a fixed held-out set (batched); returns mean reward + rates."""
    if cfg.eval_max_new_tokens:
        cfg = replace(cfg, max_new_tokens=cfg.eval_max_new_tokens)
    rng = random.Random(cfg.seed + 100_000)  # disjoint from training stream
    esample = getattr(task, "eval_sample", task.sample)  # held-out split, no leak
    examples = [esample(rng) for _ in range(cfg.eval_n)]
    groups, _, _ = _sample_batched(model, tokenizer, examples, cfg, 1, 0.0, task)
    think_close = _think_close_marker(tokenizer, cfg, task)
    results = []
    closed_rates = []
    rfcs_vals = []
    for ex, group in zip(examples, groups):
        text = _completion_text(tokenizer, group[0])
        visible, closed = _visible_reply(text, think_close)
        res = task.reward(ex, visible)
        results.append(res)
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

        required = estimate_run_gb(
            model_disk_gb(resolve_model_path(cfg.model)), cfg.activation_headroom_gb
        )
        holder = machine.acquire(
            required, wait_s=cfg.lease_wait_s, note=f"train {cfg.task} on {cfg.model}"
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

    if cfg.gdn_serial:
        from . import gdn_serial

        gdn_serial.install(cfg.gdn_chunk)

    model, tokenizer, info = load_policy(
        cfg.model, cfg.lora, cfg.activation_headroom_gb,
        grad_checkpoint=cfg.grad_checkpoint,
    )
    print(f"loaded {cfg.model}: {info}")
    pad_id = next(iter(sorted(tokenizer.eos_token_ids)))

    # Fail loud, not slow: hard-abort if a backward spills to swap.
    swap_guard = SwapGuard(
        margin_gb=cfg.swap_guard_margin_gb, abort_marker=out / "ABORTED"
    ).start()

    optimizer = optim.Adam(learning_rate=cfg.lr)

    def loss_fn(model, inp, tgt, mask, old_lp, ref_lp, adv, denom,
                sel_idx=None):
        cur_lp = (token_logprobs(model(inp), tgt) if sel_idx is None
                  else selective_logprobs(model, inp, tgt, sel_idx))
        return grpo_objective(
            cur_lp, old_lp, ref_lp, adv, mask, denom, cfg.clip_eps, cfg.kl_coef
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    baseline = evaluate(model, tokenizer, task, cfg)
    mx.clear_cache()  # phase boundary: don't stack eval KV under step-1 gen
    print(f"step 0 baseline: {baseline}")
    metrics_f.write(json.dumps({"step": 0, **baseline}) + "\n")
    metrics_f.flush()

    for step in range(1, cfg.steps + 1):
        examples = [task.sample(rng) for _ in range(cfg.batch_prompts)]

        mx.reset_peak_memory()  # per-step peak, not a run-lifetime high-water
        t0 = time.time()
        rollouts, gen_stats, skipped1 = collect_rollouts(
            model, tokenizer, examples, cfg, task)
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
            denom_tokens = float(
                sum(len(r.completion_tokens) for r in upd_rollouts))
            if cfg.update_adv_frac > 0:
                a2 = np.abs(upd_adv).reshape(-1, cfg.group_size)
                keep = (a2 >= cfg.update_adv_frac
                        * a2.max(axis=1, keepdims=True)).reshape(-1)
                n_pruned = int((~keep).sum())
                upd_rollouts = [r for r, k in zip(upd_rollouts, keep) if k]
                upd_adv = upd_adv[keep]
            pg, kl = update_policy(
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
            pg, kl = 0.0, 0.0  # no reward spread anywhere: skip the update
        t_upd = time.time() - t1
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
        if cfg.update_adv_frac:
            rec["rollouts_pruned"] = n_pruned
        # QeRL observable (2510.11696): mean NLL of the sampled tokens under
        # the (4-bit) policy — the exploration/entropy proxy. Quantization
        # noise raises it; watch it fall as the policy sharpens.
        nll = [-float(np.mean(r.sampling_logprobs)) for r in rollouts
               if not r.sage and r.sampling_logprobs]
        if nll:
            rec["gen_nll"] = round(float(np.mean(nll)), 4)
        if cfg.sage_r:
            sage_rs = [r for r in rollouts if r.sage]
            samp_rs = [r for r in rollouts if not r.sage]
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
                            "think_len": r.think_len,
                            "len": len(r.completion_tokens),
                            "finish": r.finish,
                            "text": r.text,
                        }
                        for r in group
                    ],
                }
            )
            + "\n"
        )
        samples_f.flush()

        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            save_adapter(model, out / "adapters", cfg.lora, cfg.model, step)

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
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--kl-coef", type=float, default=d.kl_coef)
    p.add_argument("--clip-eps", type=float, default=d.clip_eps)
    p.add_argument("--no-normalize-std", dest="normalize_std", action="store_false")
    p.add_argument("--sage-r", type=int, default=d.sage_r,
                   help="SAGE-decoded members per group (0 = vanilla GRPO)")
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
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--rank", type=int, default=d.lora.rank)
    p.add_argument("--lora-scale", type=float, default=d.lora.scale)
    p.add_argument("--lora-layers", type=int, default=d.lora.num_layers)
    p.add_argument("--no-share-prompt", dest="share_prompt", action="store_false")
    p.add_argument("--no-manage-machine", dest="manage_machine", action="store_false")
    p.add_argument("--lease-wait", type=float, default=d.lease_wait_s, help="seconds to wait for the machine lease")
    p.add_argument("--rollout-batch-size", type=int, default=d.rollout_batch_size)
    p.add_argument("--length-penalty", type=float, default=d.length_penalty,
                   help="correctness-gated total-length penalty λ (0 = off)")
    p.add_argument("--length-budget", type=int, default=d.length_budget,
                   help="token budget for length normalisation (0 = max_new_tokens)")
    p.add_argument("--activation-headroom", type=float, default=d.activation_headroom_gb,
                   help="GB added to the memory-guard estimate (default 4)")
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
    if a.sage_r > 0:
        if think_end is None:
            p.error("--sage-r needs an end-of-thinking token: use a profile with think_end or pass --think-end")
        if a.sage_r >= a.group_size:
            p.error("--sage-r must leave at least one ordinarily-sampled group member")
        if prof:
            base_kwargs.update(prof.think_chat_kwargs)  # SAGE trains the thinking policy
    chat_kwargs = {**base_kwargs, **json.loads(a.chat_kwargs)}

    cfg = TrainConfig(
        model=model,
        profile=a.profile,
        extra_eos=tuple(prof.extra_eos) if prof else (),
        share_prompt=a.share_prompt,
        manage_machine=a.manage_machine,
        lease_wait_s=a.lease_wait,
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
        epochs_per_batch=a.epochs_per_batch,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        lr=a.lr,
        kl_coef=a.kl_coef,
        clip_eps=a.clip_eps,
        normalize_std=a.normalize_std,
        sage_r=a.sage_r,
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
        eval_every=a.eval_every,
        eval_n=a.eval_n,
        eval_max_new_tokens=a.eval_max_new_tokens,
        grad_checkpoint=a.grad_checkpoint,
        gdn_serial=a.gdn_serial,
        gdn_chunk=a.gdn_chunk,
        checkpoint_every=a.checkpoint_every,
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
    train(cfg, out)


if __name__ == "__main__":
    main()
