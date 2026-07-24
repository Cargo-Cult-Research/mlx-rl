"""Batched rollout engine built on mlx_lm's BatchGenerator (continuous batching).

Two throughput optimizations over one-sequence-at-a-time generate_step:

1. **Batched decode.** All completions of a step decode together; the
   generator handles prefill/decode transitions, per-row stop tokens, and
   row retirement (continuous batching).
2. **KV reuse across a group.** A GRPO group shares one prompt, so the
   prompt is prefilled ONCE at batch=1; each member enters the generator
   with a cloned cache and a 1-token prompt (the prompt's last token).
   MLX arrays are immutable, so clones share the prompt KV physically until
   decode diverges — copy-on-write by construction, no array copies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, BatchStats
from mlx_lm.models import cache as cache_mod
from mlx_lm.sample_utils import make_sampler


@dataclass
class Completion:
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    # SAGE completions record where thinking ended (index just past the
    # end-of-thinking token); None for ordinary sampled completions.
    think_len: int | None = None


def clone_cache_list(caches: list) -> list:
    """Cheap per-sequence clone of a prompt cache.

    Shallow-copies each cache object; array attributes stay shared (safe:
    MLX slice-assignment rebinds a new array on the writing object only),
    while list attributes and nested caches are copied because those ARE
    mutated in place (ArraysCache.__setitem__, CacheList children).
    """
    out = []
    for c in caches:
        c2 = copy.copy(c)
        for name, val in vars(c2).items():
            if isinstance(val, list):
                setattr(c2, name, list(val))
        if hasattr(c2, "caches"):
            c2.caches = type(c2.caches)(clone_cache_list(list(c2.caches)))
        out.append(c2)
    return out


def prefill_cache(model, prompt_ids: list[int], prefill_step_size: int = 2048) -> list:
    """Consume prompt[:-1] into a fresh cache (chunked). The last prompt
    token is left for the generator: it becomes the 1-token 'prompt' whose
    forward produces the first completion logits."""
    c = cache_mod.make_prompt_cache(model)
    toks = mx.array(prompt_ids[:-1])
    for i in range(0, toks.size, prefill_step_size):
        model(toks[i : i + prefill_step_size][None], cache=c)
        mx.eval([x.state for x in c])
    return c


@dataclass
class _Cand:
    """A candidate reasoning chain in the step-wise beam."""
    cache: list
    pending: int                 # last committed token (forward it for the next)
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    sum_lp: float = 0.0
    ended: str | None = None     # how the last step ended: think|eos|delim|cap|budget
    n_steps: int = 0

    @property
    def phi(self) -> float:       # length-normalized cumulative log-prob (Eq. 1)
        return self.sum_lp / max(len(self.tokens), 1)


def _answer_from(model, prompt_ids, reasoning_toks, think_len, *, eos,
                 max_new_tokens, temperature, prefill_step_size=2048):
    """Fresh-prefill prompt+reasoning (ending at </think>) then sample the answer.
    Used by the batched beam: GDN recurrent caches (ArraysCache) can't be trimmed,
    so instead of trimming the winner's over-generated cache we re-prefill its
    clean token sequence once (~O(think_len))."""
    full = list(prompt_ids) + list(reasoning_toks)
    c = cache_mod.make_prompt_cache(model)
    a = mx.array(full)
    for i in range(0, a.size, prefill_step_size):
        model(a[i:i + prefill_step_size][None], cache=c)
        mx.eval([x.state for x in c])
    tok = list(reasoning_toks)
    lps = [0.0] * len(reasoning_toks)
    pend = full[-1]
    while len(tok) < max_new_tokens:
        lg = model(mx.array([[pend]]), cache=c)[0, -1].astype(mx.float32)
        lp = lg - mx.logsumexp(lg)
        t = (int(mx.random.categorical(lp / temperature)) if temperature > 0
             else int(mx.argmax(lp)))
        tok.append(t); lps.append(float(lp[t])); pend = t
        if t in eos:
            return Completion(tok, lps, "stop", think_len)
    return Completion(tok, lps, "length", think_len)


def _sage_batched(model, prompt_ids, think_end, *, eos, m, tr, max_new_tokens,
                  max_reasoning_steps, step_tokens, think_temperature,
                  answer_temperature, answer_reserve=256,
                  prefill_step_size=2048) -> Completion:
    """Batched SAGE beam: all 2m^2 expansions share one B-row forward per token
    (decode is weight-bandwidth-bound, so B rows ~= B=1 wall-time -> ~5x). Steps
    are fixed `step_tokens` chunks (rectangular batch; </think> detected mid-chunk).
    merge()=fork to batched, filter(idxs)=replicate/prune. Same top-h Φ acceptance
    as the unbatched reference (docs/sage-paper-notes.md).

    Reasoning is HARD-BOUNDED by `max_new_tokens - answer_reserve`: an
    iteration only starts if its worst-case chunk still fits, so the answer
    phase always has >= answer_reserve tokens of budget and the completion
    can never exceed max_new_tokens (sage2-codemix bug: unbounded reasoning
    produced think_len 2069 > cap 2048 with zero answer tokens)."""
    h = max(1, round(tr * 2 * m))
    think_budget = max(1, max_new_tokens - answer_reserve)
    c = cache_mod.make_prompt_cache(model)
    pre = mx.array(prompt_ids[:-1])
    for i in range(0, pre.size, prefill_step_size):
        model(pre[i:i + prefill_step_size][None], cache=c)
        mx.eval([x.state for x in c])
    cache = [type(x).merge([x]) for x in c]          # -> batched (BatchKVCache/ArraysCache)
    rows = [{"tok": [], "slp": 0.0, "pend": prompt_ids[-1]}]

    for _ in range(max_reasoning_steps):
        if max(len(r["tok"]) for r in rows) + step_tokens > think_budget:
            break  # next chunk could overrun the reasoning budget: force-close
        rep = [i for i in range(len(rows)) for _ in range(2 * m)]
        for bc in cache:
            bc.filter(rep)
        rows = [{"tok": list(rows[i]["tok"]), "slp": rows[i]["slp"],
                 "pend": rows[i]["pend"], "st": [], "sl": [], "end": None,
                 "think_at": None} for i in rep]
        B = len(rows)
        pend = mx.array([[r["pend"]] for r in rows])
        gen_len = 0
        for k in range(step_tokens):
            if all(r["end"] in ("think", "eos") for r in rows):
                break
            lg = model(pend, cache=cache)[:, -1].astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            nxt = (mx.random.categorical(lp / think_temperature)
                   if think_temperature > 0 else mx.argmax(lp, axis=-1))
            chosen = mx.take_along_axis(lp, nxt[:, None], axis=-1)[:, 0]
            mx.eval(nxt, chosen)
            nl, cl = nxt.tolist(), chosen.tolist()
            gen_len = k + 1
            for i, r in enumerate(rows):
                if r["end"] in ("think", "eos"):
                    continue
                r["st"].append(nl[i]); r["sl"].append(cl[i])
                if nl[i] == think_end:
                    r["think_at"] = len(r["st"]); r["end"] = "think"
                elif nl[i] in eos:
                    r["end"] = "eos"
            pend = nxt[:, None]
        for r in rows:
            n = r["think_at"] if r["end"] == "think" else len(r["st"])
            r["_tok"] = r["tok"] + r["st"][:n]
            r["_phi"] = (r["slp"] + sum(r["sl"][:n])) / max(len(r["_tok"]), 1)
        ranked = sorted(range(B), key=lambda i: rows[i]["_phi"], reverse=True)
        band = ranked[:h]
        acc = next((i for i in band if rows[i]["end"] == "eos"), None)
        if acc is None:
            acc = next((i for i in band if rows[i]["end"] == "think"), None)
        if acc is not None:
            win = rows[acc]["_tok"]
            return _answer_from(model, prompt_ids, win, len(win), eos=eos,
                                max_new_tokens=max_new_tokens,
                                temperature=answer_temperature,
                                prefill_step_size=prefill_step_size)
        surv = [i for i in ranked if rows[i]["end"] is None][:m] or ranked[:1]
        for bc in cache:
            bc.filter(surv)
        rows = [{"tok": rows[i]["_tok"], "slp": rows[i]["slp"] + sum(rows[i]["sl"]),
                 "pend": rows[i]["_tok"][-1] if rows[i]["_tok"] else rows[i]["pend"]}
                for i in surv]

    best = max(rows, key=lambda r: r["slp"] / max(len(r["tok"]), 1))
    reasoning = best["tok"] + [think_end]
    return _answer_from(model, prompt_ids, reasoning, len(reasoning), eos=eos,
                        max_new_tokens=max_new_tokens,
                        temperature=answer_temperature,
                        prefill_step_size=prefill_step_size)


def _accept(ranked: list["_Cand"], h: int):
    """Top-h acceptance (paper TR=h/2m): from the h highest-Φ expansions, accept
    a model-closed turn (eos) first, else the highest-Φ step that ends with
    </think>. Returns (kind, cand) or (None, None). `ranked` must be Φ-desc.
    This is the confidence gate that replaces the old eager tol-stop: a </think>
    that ranks BELOW the top-h is not accepted, preventing low-confidence stops.
    """
    band = ranked[:h]
    for c in band:
        if c.ended == "eos":
            return "eos", c
    for c in band:  # band is Φ-descending, so this is the highest-Φ think step
        if c.ended == "think":
            return "think", c
    return None, None


def _next_lp(model, pending: int, cache: list) -> mx.array:
    logits = model(mx.array([[pending]]), cache=cache)[0, -1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits)
    mx.eval(lp)
    return lp


def _grow_step(model, cand: _Cand, *, think_end: int, eos: set[int],
               step_delim: set[int], temperature: float,
               max_step_tokens: int, token_ceiling: int) -> _Cand:
    """Sample ONE full reasoning step (paper: steps are '\\n\\n'-delimited)
    off `cand`, on a cloned cache. Ranking logprobs are the TRUE (untempered)
    log-probs, so Φ stays the model's own confidence even when temperature>0."""
    cache = clone_cache_list(cand.cache)
    toks, lps, pending, ended = [], [], cand.pending, "cap"
    for _ in range(max_step_tokens):
        lp = _next_lp(model, pending, cache)
        t = (int(mx.random.categorical(lp / temperature)) if temperature > 0
             else int(mx.argmax(lp)))
        toks.append(t); lps.append(float(lp[t])); pending = t
        if t == think_end:
            ended = "think"; break
        if t in eos:
            ended = "eos"; break
        if t in step_delim:
            ended = "delim"; break
        if len(cand.tokens) + len(toks) >= token_ceiling:
            ended = "budget"; break
    return _Cand(cache=cache, pending=pending,
                 tokens=cand.tokens + toks, logprobs=cand.logprobs + lps,
                 sum_lp=cand.sum_lp + sum(lps), ended=ended,
                 n_steps=cand.n_steps + 1)


def sage_completion(
    model,
    prompt_ids: list[int],
    think_end: int,
    *,
    eos: set[int],
    step_delim: set[int],
    m: int = 2,
    tr: float = 0.5,
    max_new_tokens: int = 4096,
    max_reasoning_steps: int = 64,
    max_step_tokens: int = 512,
    think_temperature: float = 1.0,
    answer_temperature: float = 1.0,
    answer_reserve: int = 256,
    prefill_step_size: int = 2048,
    batched: bool = True,
    step_tokens: int = 48,
) -> Completion:
    """SAGE decoding — faithful to arXiv 2602.08354 (see docs/sage-paper-notes.md).

    `batched=True` (default) runs `_sage_batched` — all 2m^2 expansions in one
    B-row forward per token (~5x on qwen36). `batched=False` is the unbatched
    step-wise reference below (B=1 per expansion); both use the same top-h Φ gate.

    Step-wise beam over reasoning steps ('\\n\\n'-delimited). Each iteration:
    for each of the <=m candidates, sample 2m full steps (random sampling at
    `think_temperature`); rank all 2m^2 expansions by Φ (length-normalized
    cumulative log-prob = mean per-token log-prob); keep top-m. A step that ends
    with `</think>` is ACCEPTED as the completion when it lands within the
    **top-h** by Φ, h = round(TR * 2m) — the confidence gate that "prevents
    low-confidence early termination". Termination is confidence-driven,
    bounded by `max_reasoning_steps` and a reasoning token budget of
    `max_new_tokens - answer_reserve` (so the answer phase always has room
    and the completion can never exceed `max_new_tokens`).

    This replaces the old token-wise/eager-stop/384-cap decoder that produced the
    sage1 reasoning-relocation collapse. Answer phase: sample to EOS.
    """
    if batched:
        return _sage_batched(
            model, prompt_ids, think_end, eos=eos, m=m, tr=tr,
            max_new_tokens=max_new_tokens, max_reasoning_steps=max_reasoning_steps,
            step_tokens=step_tokens, think_temperature=think_temperature,
            answer_temperature=answer_temperature, answer_reserve=answer_reserve,
            prefill_step_size=prefill_step_size)
    h = max(1, round(tr * 2 * m))
    think_budget = max(1, max_new_tokens - answer_reserve)
    beam = [_Cand(cache=prefill_cache(model, prompt_ids, prefill_step_size),
                  pending=prompt_ids[-1])]

    for _ in range(max_reasoning_steps):
        if len(beam[0].tokens) + max_step_tokens > think_budget:
            break  # next step could overrun the reasoning budget: force-close
        # Expand: 2m sampled steps per candidate -> up to 2m^2 candidates.
        expansions: list[_Cand] = []
        for cand in beam:
            for _ in range(2 * m):
                expansions.append(_grow_step(
                    model, cand, think_end=think_end, eos=eos,
                    step_delim=step_delim, temperature=think_temperature,
                    max_step_tokens=max_step_tokens, token_ceiling=think_budget))
        expansions.sort(key=lambda c: c.phi, reverse=True)

        kind, winner = _accept(expansions, h)
        if kind == "eos":  # model closed the turn inside the accept band
            return Completion(winner.tokens, winner.logprobs, "stop",
                              len(winner.tokens))
        if kind == "think":  # confident end-of-thinking -> answer phase
            return _answer_phase(model, winner, eos=eos,
                                 max_new_tokens=max_new_tokens,
                                 temperature=answer_temperature,
                                 think_len=len(winner.tokens))
        beam = expansions[:m]
        if beam[0].ended == "budget" or len(beam[0].tokens) >= think_budget:
            break

    # Reasoning-step / token budget hit without a confident stop: force </think>
    # on the best chain, then answer.
    best = beam[0]
    best.tokens.append(think_end); best.logprobs.append(0.0)
    best.pending = think_end
    return _answer_phase(model, best, eos=eos, max_new_tokens=max_new_tokens,
                         temperature=answer_temperature, think_len=len(best.tokens))


def _answer_phase(model, cand: _Cand, *, eos: set[int], max_new_tokens: int,
                  temperature: float, think_len: int) -> Completion:
    if cand.pending in eos:  # turn already closed (think_end == eos, e.g. gemma)
        return Completion(cand.tokens, cand.logprobs, "stop", think_len)
    while len(cand.tokens) < max_new_tokens:
        lp = _next_lp(model, cand.pending, cand.cache)
        t = (int(mx.random.categorical(lp / temperature)) if temperature > 0
             else int(mx.argmax(lp)))
        cand.tokens.append(t); cand.logprobs.append(float(lp[t])); cand.pending = t
        if t in eos:
            return Completion(cand.tokens, cand.logprobs, "stop", think_len)
    return Completion(cand.tokens, cand.logprobs, "length", think_len)


def rollout_groups(
    model,
    tokenizer,
    prompts: list[list[int]],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    *,
    extra_eos: tuple[int, ...] = (),
    share_prompt: bool = True,
    completion_batch_size: int = 64,
    prefill_batch_size: int = 8,
    prefill_step_size: int = 2048,
) -> tuple[list[list[Completion]], BatchStats]:
    """Sample group_size completions for every prompt, batched.

    Returns (groups, stats): groups[i][g] is a Completion (tokens include
    the terminating EOS when finish_reason == 'stop'); stats carries
    aggregate prompt/generation tokens-per-second.
    """
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    eos |= set(extra_eos)

    gen = BatchGenerator(
        model,
        max_tokens=max_new_tokens,
        stop_tokens=[[t] for t in sorted(eos)],
        sampler=make_sampler(temp=temperature),
        completion_batch_size=max(completion_batch_size, group_size),
        prefill_batch_size=prefill_batch_size,
        prefill_step_size=prefill_step_size,
    )
    groups = [[Completion() for _ in range(group_size)] for _ in prompts]
    uid_to: dict[int, tuple[int, int]] = {}
    stats = BatchStats()
    try:
        with gen.stats(stats):
            for pi, prompt in enumerate(prompts):
                if share_prompt and len(prompt) > 1:
                    base = prefill_cache(model, prompt, prefill_step_size)
                    uids = gen.insert(
                        [[prompt[-1]] for _ in range(group_size)],
                        caches=[clone_cache_list(base) for _ in range(group_size)],
                    )
                else:
                    uids = gen.insert([list(prompt) for _ in range(group_size)])
                for gi, u in enumerate(uids):
                    uid_to[u] = (pi, gi)

            done, total = 0, len(prompts) * group_size
            while done < total:
                responses = gen.next_generated()
                if not responses:
                    break  # generator drained (shouldn't happen before total)
                for r in responses:
                    pi, gi = uid_to[r.uid]
                    comp = groups[pi][gi]
                    comp.tokens.append(int(r.token))
                    comp.logprobs.append(float(r.logprobs[r.token]))
                    if r.finish_reason is not None:
                        comp.finish_reason = r.finish_reason
                        done += 1
    finally:
        gen.close()
    return groups, stats
