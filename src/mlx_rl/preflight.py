"""Pre-load sanity checks on a requested run configuration.

Runs BEFORE the model loads and before the memory lease is taken, so a config
that cannot work dies in under a second with a one-line reason — not forty
minutes in with a Metal OOM, and not silently (a weaker agent babysitting a
run must be able to read the failure off the first line of the log).

Failures exit(2) with a PREFLIGHT FAILED banner. Warnings print but do not
stop the run. Escape hatch for deliberate experiments:
MLX_RL_SKIP_PREFLIGHT=1 skips everything except the banner saying so.
"""

from __future__ import annotations

import os

import psutil

# Fraction of physical RAM a run may plan to use. Above this, macOS pages;
# the swap guard would kill the run later anyway — fail before loading.
_RAM_PLAN_FRACTION = 0.92


def _fail(problems: list[str]) -> None:
    print("=" * 72)
    print("PREFLIGHT FAILED — this configuration cannot run as requested:")
    for p in problems:
        print(f"  * {p}")
    print("Fix the config (or set MLX_RL_SKIP_PREFLIGHT=1 to override).")
    print("=" * 72, flush=True)
    raise SystemExit(2)


def preflight(cfg) -> None:
    from .memory import estimate_run_gb, model_disk_gb
    from .models import resolve_model_path

    if os.environ.get("MLX_RL_SKIP_PREFLIGHT"):
        print("[preflight] SKIPPED via MLX_RL_SKIP_PREFLIGHT=1", flush=True)
        return
    problems: list[str] = []
    warns: list[str] = []

    # -- memory plan, including residencies the lease estimate can't see ----
    total_gb = psutil.virtual_memory().total / 1e9
    weights_gb = judge_gb = 0.0
    try:
        weights_gb = model_disk_gb(resolve_model_path(cfg.model))
    except Exception as e:  # unresolvable model is its own loud failure later
        warns.append(f"could not size model weights ({e}); memory check skipped")
    est_gb = estimate_run_gb(weights_gb, cfg.activation_headroom_gb) if weights_gb else 0.0
    tkw = dict(getattr(cfg, "task_kwargs", None) or {})
    if tkw.get("judge_backend") == "local":
        # A local judge whose model IS the policy's base rides the resident
        # weights under adapters_disabled() — zero extra memory (the trainer
        # registers the model via judge_local.register_resident_model). Only
        # a judge on DIFFERENT weights loads a second copy, invisible to
        # estimate_run_gb, and must be counted here.
        from .profiles import DEFAULT_JUDGE_MODEL
        jpath = tkw.get("judge_model_path") or DEFAULT_JUDGE_MODEL
        try:
            if str(resolve_model_path(jpath)) != str(resolve_model_path(cfg.model)):
                judge_gb = model_disk_gb(resolve_model_path(jpath))
                est_gb += judge_gb
                warns.append(
                    f"local judge on different weights ({jpath}) — a second "
                    f"{judge_gb:.0f} GB residency; same-base judges are free")
        except Exception as e:
            warns.append(f"could not size local judge weights ({e})")
    # Hard-fail on the IRREDUCIBLE floor only (resident weights + headroom):
    # that part cannot be tuned away, so exceeding RAM there is certain
    # death (the two-60GB-models kernel-panic class). The 2.9x training-peak
    # multiplier is too coarse to hard-fail on — grad_checkpoint and small
    # LoRA ranks land far below it (measured 49.7 GB where it predicts 68) —
    # and refusing runs that fit is the failure mode available_gb() already
    # got bitten by twice. Estimate overruns warn instead.
    floor_gb = weights_gb + judge_gb + cfg.activation_headroom_gb
    if weights_gb and floor_gb > total_gb * _RAM_PLAN_FRACTION:
        problems.append(
            f"resident floor ~{floor_gb:.0f} GB (weights {weights_gb:.0f}"
            + (f" + local judge {judge_gb:.0f}" if judge_gb else "")
            + f" + headroom) exceeds {_RAM_PLAN_FRACTION:.0%} of physical "
            f"RAM ({total_gb:.0f} GB) — certain OOM, nothing to tune")
    elif est_gb and est_gb > total_gb:
        warns.append(
            f"worst-case training peak ~{est_gb:.0f} GB (x2.9 rule) exceeds "
            f"RAM ({total_gb:.0f} GB); fits only if grad_checkpoint/LoRA keep "
            "the real peak low — the swap guard is the backstop")
    if est_gb and cfg.required_gb and est_gb > cfg.required_gb:
        warns.append(
            f"--required-gb {cfg.required_gb:.0f} understates the planned peak "
            f"~{est_gb:.0f} GB (does it forget the local judge?) — the lease "
            "will clear too little room")

    # -- token budgets -------------------------------------------------------
    if cfg.max_episode_tokens and cfg.max_new_tokens > cfg.max_episode_tokens:
        problems.append(
            f"max_new_tokens {cfg.max_new_tokens} > max_episode_tokens "
            f"{cfg.max_episode_tokens}: a single round can never fit its budget")

    # -- checkpoint window ---------------------------------------------------
    if cfg.steps and cfg.checkpoint_every:
        if cfg.checkpoint_every > max(5, cfg.steps // 6):
            warns.append(
                f"checkpoint_every {cfg.checkpoint_every} of {cfg.steps} steps: "
                "at ~5 min/step a first-failure before the first checkpoint "
                "throws away every step of compute (the 08-18 night run lost "
                "10 steps exactly this way) — consider 5")
    elif cfg.steps and not cfg.checkpoint_every:
        warns.append("checkpoint_every=0: nothing is promotable until the run "
                     "SURVIVES to its final step")

    for w in warns:
        print(f"[preflight] WARNING: {w}", flush=True)
    if problems:
        _fail(problems)
