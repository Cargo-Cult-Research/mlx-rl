"""Rollout throughput benchmark: batched engine vs sequential generate_step.

Usage:
    uv run python scripts/bench_rollout.py --profile qwen36 [--seq] \
        [--prompts 4] [--group-size 8] [--max-new 128]
"""

import argparse
import random
import time

import mlx.core as mx

from mlx_rl.config import LoraConfig
from mlx_rl.engine import rollout_groups
from mlx_rl.models import load_policy
from mlx_rl.profiles import get_profile
from mlx_rl.rollout import encode_prompt, sample_completion
from mlx_rl.tasks import get_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--prompts", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seq", action="store_true", help="also run sequential baseline")
    p.add_argument("--no-share-prompt", dest="share", action="store_false")
    a = p.parse_args()

    prof = get_profile(a.profile)
    model, tok, info = load_policy(
        prof.model, LoraConfig(rank=16, keys=list(prof.lora_keys) if prof.lora_keys else None)
    )
    print(f"{prof.name}: {info}")

    task = get_task("arithmetic", n_operands=5, max_operand=999)
    rng = random.Random(7)
    prompts = [
        encode_prompt(tok, task.sample(rng).messages, **prof.chat_kwargs)
        for _ in range(a.prompts)
    ]

    # warmup (kernel compile)
    rollout_groups(model, tok, prompts[:1], 2, 16, a.temperature, extra_eos=prof.extra_eos)

    t0 = time.time()
    groups, stats = rollout_groups(
        model, tok, prompts, a.group_size, a.max_new, a.temperature,
        extra_eos=prof.extra_eos, share_prompt=a.share,
    )
    wall = time.time() - t0
    n_tok = sum(len(c.tokens) for g in groups for c in g)
    print(
        f"BATCHED  ({a.prompts}x{a.group_size}, share_prompt={a.share}): "
        f"{n_tok} tok in {wall:.1f}s = {n_tok/wall:.1f} tok/s wall | "
        f"decode {stats.generation_tps:.1f} tok/s, prefill {stats.prompt_tps:.1f} tok/s | "
        f"peak {mx.get_peak_memory()/1e9:.1f} GB"
    )
    sample = groups[0][0]
    print(f"sample (finish={sample.finish_reason}): {tok.decode(sample.tokens)[:160]!r}")

    if a.seq:
        t0 = time.time()
        n_tok = 0
        for prompt in prompts:
            for _ in range(a.group_size):
                toks, _, _ = sample_completion(model, tok, prompt, a.max_new, a.temperature)
                n_tok += len(toks)
        wall = time.time() - t0
        print(f"SEQUENTIAL: {n_tok} tok in {wall:.1f}s = {n_tok/wall:.1f} tok/s wall")


if __name__ == "__main__":
    main()
