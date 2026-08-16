# Tool rounds, multi-turn, and the stated date — design for review

*Status: **PLAN, NOT EXECUTED**. Written 2026-08-16 for review before any code
lands. Nothing in this document has been run. Every number quoted as "measured"
comes from a probe already in the tree or in `housekeeping/benchmarks/data/`;
everything else is a proposal.*

Target model: **qwen36** (Qwen3.6-35B-A3B). qwen38 was considered and dropped —
see the closing section.

---

## 1. What we are fixing

The shipped artifact is `qa-gloveC-200-20260731` + `HONESTY_SYSTEM` (the glove):
calibrated abstention, RL-trained with GRPO+LoRA. Its training register was
**single-turn, thinking off, 192-token horizon, no tools, no date**. Two failures
follow directly from that register.

### 1.1 The glove suppresses tool use under an asserted future year

Measured on 6 real post-cutoff arXiv papers × 3 phrasings × 3 samples (n=144,
zero truncation; probes in the 2026-08-15 session scratchpad, summary in commit
`82ae363`). Rate at which the arm calls `search_arxiv` (1.00 = always):

| phrasing | base | glove | glove + `TOOL_FIRST` clause |
|---|---|---|---|
| "the arXiv paper 'X'" | 1.00 | 1.00 | 1.00 |
| "the 2026 paper 'X'" | 0.83 | **0.06** | 0.78 |
| same, invented title | 0.67 | **0.08** | 0.42 |

This is **not** general tool suppression — the capability gate showed zero cost
on agentic coding (base 45/72, C-200 45/72). It collapses only when the question
asserts a year the model reads as future, which hands it grounds to conclude
non-existence and stop looking.

The current mitigation is a one-line `TOOL_FIRST` clause appended to the glove
only when tools are offered (`demo/app.py:96`). It works, and it is a **prompt
patch, not a fix**: "check before you decline" belongs in the policy. Retiring
this clause is the headline goal.

### 1.2 Multi-turn is untrained and unmeasured

The adapter never saw a prior turn. The demo replays turns per arm, but no
multi-turn calibration behaviour is trained or probed.

### 1.3 Adjacent, cheap, and worth fixing in the same pass

Both arms ramble past their own `</tool_call>` despite the template's explicit
"NO suffix" instruction — everything after that tag is generated blind, before
any tool result exists. **The base arm denied a real paper immediately after
calling search on it.** Stopping generation at the tag removes this
structurally.

---

## 2. Constraints — what the stack fixes for us (all verified 2026-08-16)

**Rollouts must stay in-process mlx-lm.** Not a preference: the adapter being
sampled must be the adapter being trained; the frozen reference policy comes
free by zeroing the LoRA scale; and the update recomputes `old_lp` teacher-forced
from exact token ids (correctness invariant 2). A server round-trip loses all
three, and llama.cpp cannot host LoRA training at all.

**`BatchGenerator` (mlx-lm 0.31.3) already supports resumable sequences:**

- `extract_cache(uids)` → `(kv_cache, tokens)` per sequence
- `remove(uids, return_prompt_caches=True)` → same, and frees the slot
- `insert(prompts, caches=[...], all_tokens=[...])` → resume from a cache
- `insert_segments` auto-splits so the final segment is one token

This is the same mechanism `rollout_groups` already uses for prompt sharing
(`engine.py:444`). Tool rounds need no new upstream capability.

**Stop markers are single tokens** (identical ids on both qwen36 and qwen38):

| marker | id |
|---|---|
| `</tool_call>` | 248059 |
| `</think>` | 248069 |
| `<tool_response>` | 248066 |

`</tool_call>` drops straight into the existing `rollout_groups(extra_eos=…)`
parameter. `</think>` is already the qwen36 profile's `think_end`.

**Chat template behaviour** (rendered against the real template, not assumed):

1. A tool result renders as a **`user`** message wrapping `<tool_response>` —
   there is no `tool` role in the output.
2. **Thinking is kept inside the current turn and stripped from completed
   turns.** The assistant message carrying a tool call keeps its `<think>`; the
   previous turn's reply has its think block deleted.
3. With `enable_thinking`, the generation prompt ends with `<think>\n` already
   open — the model never emits the opening tag.
4. The tools block is injected into the **system** message, *before* the glove
   text, and carries its own instruction: *"If there is no function call
   available, answer the question like normal with your current knowledge and do
   not tell the user about function calls."* Note this pushes against
   abstention; it is part of the prompt whenever tools are offered.

Finding 2 decides the architecture, and it is forced by the template rather than
chosen:

- **Tool rounds (within a turn) → splice on the KV cache.** The spliced sequence
  is identical to what a re-render would produce, because the current turn's
  thinking is retained. One prefill per episode.
- **Multi-turn (across turns) → re-render and re-prefill.** Re-rendering
  *deletes* the prior turn's thinking from the middle of the context, so no cache
  prefix survives. The trained context then matches deployment exactly.

---

## 3. Design

### D1 — Segmented rollouts (the one core trainer change)

Today `Rollout` is one prompt + one completion, and `build_training_arrays`
masks a single contiguous span (`rollout.py:73-85`). An episode becomes:

```
prompt₀ | gen₁ | ⟨tool response⟩ | gen₂ | ⟨tool response⟩ | gen₃
 mask 0  |  1   |        0        |  1   |        0        |  1
```

`Rollout` gains a list of completion spans; the mask builder becomes
segment-aware. **The invariant that must not be got wrong: injected tokens are
context, never actions.** If tool output lands under the loss mask we are
training the policy to predict arXiv.

Advantage stays GRPO-shaped: G *episodes* per prompt, one episode-level reward,
every generated token in the episode carrying that episode's advantage.

*Rejected alternative:* one training row per round sharing an advantage. It needs
no trainer change, but re-forwards the shared transcript prefix once per round in
the backward pass — and backward is exactly where memory binds (README
correctness detail 5). One row per episode = one forward.

*Consequences to price in:* group members diverge (different queries → different
tool results), so the group baseline gets noisier — keep G ≥ 8. Rows get long, so
the **row** is the budget, not the round.

### D2 — Tool rounds via cache continuation

Per round, per live episode:

1. Generate with `extra_eos = (248059,)`. Stop reason is read off the last token
   (EOS vs `</tool_call>` vs length) — `finish_reason` alone can't distinguish.
2. If it stopped on `</tool_call>`: parse the call, run it (D6), then
   `remove(uid, return_prompt_caches=True)` and re-`insert` the rendered
   `<|im_end|>\n<|im_start|>user\n<tool_response>\n…\n</tool_response><|im_end|>\n<|im_start|>assistant\n`
   block with that cache.
3. Record the injected span as mask-0, the generated span as mask-1.
4. Cap at `--max-tool-rounds` (default 2, as the demo uses). A cap breach ends
   the episode and is **scored**, not silently truncated.

### D3 — Multi-turn via re-render

Prior turns are re-rendered through `apply_chat_template` and re-prefilled. Each
turn's generated segments stay masked; everything else is context. Turn count is
a task parameter so multi-turn can be a separate arm rather than a confound.

### D4 — The stated date

Today's date goes in the system prompt, rendered **at request time**.
`serialize_proxy` already injects the glove at the serving door — it becomes a
`{today}` template, so train and deploy stay identical and the date can never go
stale. **A hardcoded date is a time bomb**: it rots silently and the failure looks
exactly like the bug being fixed.

Training must **vary** the date per example, or the policy memorises
`2026-08-16` and we have only moved the cliff. The strong version makes the date
a task variable that the reward depends on. Every arXiv question carries the
paper's real publication date, so given a stated today:

| regime | correct behaviour | verifiable by |
|---|---|---|
| paper predates the model's knowledge | answer directly | measured base pass-rate band |
| after cutoff, on or before stated-today | search finds it → answer grounded in the result (hedge if no tools offered) | tool returns hits |
| after stated-today, or fictional | search returns nothing → "I can't find it" is **correct** | tool returns empty |

This makes asserted-nonexistence a *scored* error: **denial without a search is
penalised; denial after an empty search is correct.** The policy learns the
comparison — asserted year vs. stated today, and check before declining —
instead of a year.

*Caveat to build in:* the knowledge cutoff is not a clean line, so the
regime-1/2 boundary is fuzzy. Use the measured per-question base pass rate (the
existing `scripts/qa_calibrate.py` band machinery), not an assumed cutoff date.

### D5 — The thinking arm

Most of the machinery exists: `think_chat_kwargs={"enable_thinking": True}` on
the profile, `_visible_reply` grading only after the final think-close with
unclosed-think = reward 0 (`train.py:106`), and the `[BUG]` tripwires. What is
new:

- **Horizon.** 192 tokens was viable only with thinking off. Proposed budgets:
  thinking-off ~512 tokens/round (row ≈ 3k); thinking-on ~2048/round (row ≈ 8k).
  The anatomy doc says an 8k backward is affordable with `--grad-checkpoint` +
  `gdn_serial` — **but that must be re-measured with `scripts/probe_backward.py`
  at the real row length before any long run**, not assumed.
- **Grading point.** The visible reply is the text after the **final** `</think>`
  of the **last** round.
- **Keep it a genuine arm** — separate run, thinking as the single variable, in
  the style of the A/B/C glove factorial. Not a mixed curriculum, or nothing is
  attributable.

### D6 — Tools are owned by the task, and frozen

The `Task` protocol gains:

```python
tools: list[dict]                                  # offered via the template
def run_tool(self, name, args, example) -> str     # deterministic
```

**Training must not hit the live arXiv API.** 64 completions/step × 200 steps is
~12.8k queries at a 12s timeout, blocking rollout batches, non-reproducible, and
impolite. Pre-fetch a frozen metadata snapshot covering the training question
set and serve it from a local index; keep the live API for the demo and held-out
eval only. This also makes the reward deterministic and runs resumable.

`Example` also gains per-example chat kwargs so `tools=` can vary by example
(`train.py:52` currently splats one global dict).

Reward signature: existing string-based tasks keep `reward(example, completion)`;
tool/multi-turn tasks implement an episode-aware variant receiving the segments,
the tool calls, and the final visible reply.

---

## 4. Reward hack surfaces (pre-registered)

GRPO finds every hole. Known and anticipated:

1. **Answer tags drafted inside an unclosed think block** — already defended by
   `_visible_reply` + tripwire. Keep.
2. **Tool call emitted before `</think>`.** The template pre-opens `<think>`, so
   a "call" inside the reasoning block is not a call. The reward must require it
   in the visible segment.
3. **Call the tool, ignore the result.** Grounding must be checked against the
   returned hits, not merely against "a call happened".
4. **Ramble past `</tool_call>`** — removed structurally by the stop token.
5. **Blanket search.** If searching is rewarded unconditionally the policy will
   search on everything, including questions it knows. Regime 1 must carry a
   cost for a needless call, or coverage collapses into tool-spam.
6. **All-abstain absorbing state** — the qa_abstain lesson: one-sided injection
   makes collapse gradient-free and permanent. Keep symmetric demonstrations and
   `--abort-inactive-window 30`.

`samples.jsonl` gets the first prompt's whole episode group every step. Every
reward hack this project has caught was found by reading samples, none by
curves.

---

## 5. Arms and order

Each step adds exactly one variable.

| # | arm | new variable | gate to proceed |
|---|---|---|---|
| 0 | date-in-glove plumbing + frozen arXiv snapshot | — | snapshot deterministic; `{today}` renders at the serving door |
| 1 | segmented rollouts, tools, **thinking off** | tool rounds + date | §6 criteria 1–3 |
| 2 | multi-turn | prior turns | hedging survives to turn 3+ |
| 3 | thinking on | thinking | §6 criteria 1–3 hold again |

Standard guards throughout: memlease (exclusive block), swap guard,
`--abort-inactive-window`, loud truncation reporting per the token-cap policy.

---

## 6. Success criteria — pre-registered

1. **Headline:** tool-call rate on the "2026 paper" phrasing ≥ 0.78 with the
   glove and **without** the `TOOL_FIRST` clause (currently 0.06). The clause is
   deleted, not kept as a belt.
2. **Asserted nonexistence ≤ 0.05** in both year conditions.
3. **Known-side cost** no worse than C-200: chat-known hedge+denial ≤ 0.21,
   chat-known correct ≥ 0.46.
4. **Capability gate:** SWE-bench lv-72 ≥ 37 (the same pre-registered "drop ≤ 8"
   bar the C-200 gate passed at 45/72).
5. **Multi-turn:** hedging behaviour persists at turn 3+ rather than decaying to
   base.

### The falsification test

Same question, two stated dates straddling the paper's publication date. **The
behaviour must flip.** If it does not, the policy learned a year and not the
comparison, and the reward design failed — regardless of how good the aggregate
buckets look. Run this before believing any headline number.

---

## 7. Open questions for review

1. **Frozen snapshot scope** — how many papers, and does the held-out eval use
   the live API or a second frozen slice? A frozen eval is reproducible; a live
   one is honest about deployment. Possibly both.
2. **Date range.** Plausible deployment band (± months) or deliberately wide
   (± years) to force the comparison? Wide teaches harder but drifts further
   from deployment.
3. **Needless-call cost** (hack 5) — a flat penalty, or gate it on the measured
   base pass-rate band so "needless" is defined by data rather than by assertion?
4. **Round cap.** Demo uses 2. Training at 3 costs ~50% more rollout tokens for
   a case that may be rare.

---

## 8. Why not qwen38

Considered and dropped as noise. Recorded so it is not re-litigated:

- "3.8" is branding — `config.json` says `model_type: qwen3_5`; it is the dense
  sibling of the same generation, not a successor.
- Measured rollout: qwen38 dense at 32 sequences is 193.6 t/s at **43.1 GB**
  peak (`housekeeping/benchmarks/data/qwen38-mlx-20260814.jsonl`, 2048-prompt /
  256-new), versus qwen36's 249 t/s at **24.6 GB** for the mlx-rl engine at the
  same width. Different harnesses, so directional rather than paired — but the
  memory gap is large and backward memory is what binds. `scripts/bench_rollout.py`
  would settle it if it ever matters.
- No measured capability edge: 40 vs 40 on the 51 SWE-bench instances both
  models finished, McNemar p = 1.00.
- The "qwen38-gguf serves multi-turn tools" argument **does not apply** — an
  mlx-rl adapter is mlx-lm native format and cannot be served by llama.cpp at
  any quantization. The `qwen38-mlxlm` multi-turn failure is a server bug
  (`mlx_lm.server` rejecting non-text content fragments, plus the bridge echoing
  the assistant thinking block), the same latent bug sits on `qwen36-mlxlm`, and
  the demo path (`jlens/server.py`, per-request adapter routing) bypasses it.
- Every baseline we have — the C-200 adapter, calibration bands, 23.8k cached
  judge verdicts, binding correlation, the 45/72 gate — is on qwen36.

If dense-vs-MoE is ever worth asking as an RL question in its own right, note the
`qwen38` profile is missing `think_end`, and `train.py:101` gates thinking-mode
grading on it — a thinking arm would silently lose `_visible_reply` protection.
One line (`think_end=248069`), but a quiet trap.
