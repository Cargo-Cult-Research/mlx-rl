# Datasets

Every task in this repo trains against a **verifiable reward** — a program
decides whether a completion is correct, and nothing optimizes the grader.
That constraint is what picks the corpora below: each one ships either hidden
unit tests, stored input/output cases, a numerically checkable answer, or an
alias set for exact match. There are no reward models and no LLM judges
anywhere in the training loop. The practical payoff is that a rollout's reward
is reproducible months later on a different machine, so difficulty labels
measured once stay valid, and a suspicious training curve can always be traced
back to a concrete failing test rather than to grader drift.

The corpora also span deliberately different *shapes*, because GRPO is only
informative where a group of samples disagrees. A dataset the policy always
solves and a dataset it never solves both produce zero-variance groups and
zero gradient. So the collection is stratified by difficulty rather than by
topic: MBPP sits at the saturated end (short, self-contained functions — now
useful mainly as a regression detector), DeepCoder at the hard end
(competition problems with long reasoning traces and real headroom), with
math, calibrated question-answering, and several synthetic tasks covering the
axes that neither code corpus reaches — numeric reasoning, knowing when to
abstain, output-format discipline, and emergent communication. The difficulty
sweeps documented further down exist to find, per corpus and per policy, the
band in between where the gradient actually lives.

## What ships

| Task | Corpus | Size | Reward | Source / license |
|---|---|---|---|---|
| `code` | Sanitized MBPP | 427 (347 train / 80 eval) | function passes hidden asserts | shipped in `data/`, CC BY 4.0 ([provenance](data/README.md)) |
| `deepcoder` | DeepCoder-Preview, stdin/stdout subset | 18,983 train / 175 test | program matches every stored case | fetched by `scripts/fetch_deepcoder.py` |
| `math` | DeepScaleR-Preview | ~25k verifiable of 40,315 | last `\boxed{}` matches numerically | HF Hub on first use, MIT |
| `qa_abstain` | TriviaQA `rc.nocontext` (train), PopQA (OOD eval) | ~138k / ~14k | +1 correct, 0 abstain, −penalty wrong | HF Hub on first use, Apache-2.0 / MIT |
| `arithmetic` | synthetic | unbounded | exact answer in tags | generated |
| `toolformat` | synthetic | unbounded | canonical tool-call form + right tool/args | generated |
| `mixture` | router over the above | — | delegates to the sub-task | — |

Only MBPP is redistributed in this repository. Everything else is either
generated at runtime or fetched from its upstream host on first use.

### The two code corpora are complements, not alternatives

`code` (MBPP) is 427 short problems: a docstring, a function signature, three
visible asserts, hidden asserts for grading. Solutions are tens of lines and
the reasoning is shallow. `deepcoder` is competition programming — TACO,
SYNTHETIC-1, and pre-cutoff LiveCodeBench, curated upstream so every problem's
tests were verified against a reference solution. The fetch script keeps only
stdin/stdout problems, which buys one unambiguous judge (function-call
problems, which need per-problem harness glue, are dropped; see
`data/deepcoder/FETCH-REPORT.txt` for the full accounting of what was filtered
and why). Output comparison is calibrated against upstream reference solutions
by `experimental/deepcoder_judge_check.py`: per-line rstrip, trailing blanks
dropped, then token-level comparison with 1e-6 float tolerance, because
reference answers print floats at differing precisions. Problems that accept
multiple valid orderings stay under-credited by construction — they label as
`n_pass=0`, fall outside every training band, and so cost label accuracy
without corrupting training signal.

## Why difficulty labels exist

`experimental/difficulty_sweep.py` runs a base policy over every problem in a
corpus, k samples each, and writes one JSONL row per problem with `n_pass`,
per-sample `rewards`, `lens`, and `finishes`. This is measured once and reused
forever. The `deepcoder` task consumes such a file via `labels_file=` plus
`min_pass=`/`max_pass=` to restrict *training* draws to a difficulty band
(eval draws are never filtered), so the curriculum can be stepped up as the
policy improves without re-measuring.

The band matters because of correctness detail 1 in the README: zero-variance
groups are dropped. Problems the policy always solves and problems it never
solves are both dead weight in a GRPO batch. The labels are how you buy a
batch that mostly disagrees with itself.

## Measured: MBPP difficulty atlas

Qwen3.6-35B-A3B 4-bit, k=5, cap 4096, three temperature legs under one
exclusive memory lease (`experimental/run_mbpp_sweep.sh`), 2026-08-13 → 08-14,
~166 min for the final leg. 427 problems × 3 legs = 1281 rows in
`runs/sweeps/code-pass@5.jsonl`; the same rows are the label file at
`data/labels/mbpp-pass@5-qwen36.jsonl`.

| T | split | n | pass@5 | mean pass@1 | zero-pass | at cap | median len |
|---|---|---|---|---|---|---|---|
| 1.0 | train | 347 | 0.974 | 0.876 | 9 | 33.8% | 2733 |
| 1.0 | eval | 80 | 0.975 | 0.905 | 2 | 30.8% | 2208 |
| 0.8 | train | 347 | 0.974 | 0.892 | 9 | 33.2% | 2733 |
| 0.8 | eval | 80 | 0.975 | 0.920 | 2 | 29.0% | 2206 |
| 0.6 | train | 347 | 0.968 | 0.889 | 11 | 33.6% | 2587 |
| 0.6 | eval | 80 | 0.988 | 0.945 | 1 | 25.2% | 1934 |

**Temperature barely moves pass@5** (0.972–0.974 combined across all three
legs). It moves mean pass@1 as expected — 0.881 → 0.897 → 0.900 going
1.0 → 0.8 → 0.6 — which is the usual sharpening, and confirms that training at
T=1.0 costs about two points of per-sample accuracy in exchange for the
group diversity GRPO needs.

**The train/eval split is not leaking.** Eval tracks train within noise at
every temperature, which is what the fixed seeded split in `tasks/code.py` is
supposed to guarantee.

**MBPP is saturated for this policy.** Only **4 of 427** problems failed all
15 samples across all three legs, and **3 of those 4 hit the token cap on
every sample** — they are measurement artifacts, not hard problems. That
leaves effectively *one* genuinely unsolved MBPP problem. As a curriculum
source for qwen36 this corpus is exhausted: there is no band with enough
failures to generate gradient. Its remaining value is as a **regression
detector** — a promoted adapter that drops below ~0.97 pass@5 here has broken
something.

That saturation is the direct reason DeepCoder was added.

## Measured: DeepCoder pilot

200-problem seeded sample, same base policy. Three legs, 2026-08-14.

| Leg | k | cap | pass@k | mean pass@1 | at cap |
|---|---|---|---|---|---|
| qwen36 | 5 | 4096 | 0.245 | 0.149 | **98.4%** |
| Qwen3-4B | 5 | 4096 | 0.170 | 0.137 | 86.0% |
| qwen36 | 3 | 16384 | 0.520 | 0.388 | 74.0% |

**Read the last column before the pass column.** At cap 4096, 98.4% of
qwen36's generations were still writing when the cap cut them off, and all 151
zero-pass problems had *every* sample truncated. Those two legs measured the
cap, not the models — void as capability estimates, per the project's
token-cap policy (a cap that binds on legitimate work does not lower a score,
it fabricates one).

Raising the cap to 16384 more than doubled pass@k (0.245 → 0.520) while
*lowering* k from 5 to 3, which is the signature of a measurement that was
cap-bound rather than capability-bound. It is still not settled: 74% of
generations hit 16384 too.

Length distribution at cap 16384 tells you what a settled measurement needs:

- generations that stopped naturally (n=156): median 9,994 tokens, p90 15,215, max 16,367
- generations that hit the cap: n=444
- successful generations: median 14,474, p90 16,384

Naturally-stopping traces already run to ~10k tokens with a p90 pushing the
16k ceiling, so **32,768 is the next honest cap** — and it may still bind.
DeepCoder's headroom is real, but the corpus costs roughly an order of
magnitude more decode per problem than MBPP.

## Reading these numbers

Two rules the sweeps follow, both learned the hard way:

1. **A run with truncations is a broken measurement, not a low score.** Every
   sweep row records `lens` and `finishes` per sample so `finish == "length"`
   can be counted. Any table here that quotes a pass rate also quotes the
   at-cap fraction; if the at-cap fraction is large, the pass rate is a lower
   bound on a number nobody has measured yet.
2. **Aggregate over problems, not rows.** The MBPP file holds three legs, so
   row-level aggregation silently mixes temperatures and triple-counts every
   problem. The zero-pass tail in particular looks ~3× bigger that way (34
   row-level zeros vs. 4 actually-unsolved problems).

## Open items

- **`labels_file` keeps only the last row per `task_id`.** The loader in
  `tasks/deepcoder.py` does a plain dict assignment, so pointing it at a
  multi-leg file like `data/labels/mbpp-pass@5-qwen36.jsonl` silently keeps the
  T=0.6 leg and discards T=1.0 and T=0.8 — for 170 of 427 MBPP problems (39.8%)
  the legs disagree on `n_pass`, and training rollouts sample at T=1.0, so the
  band would be built from the wrong temperature. Either filter the label file
  to one leg before use, or teach the loader to select on `temperature`.
- **`code` has no `labels_file` parameter**, so the MBPP atlas currently has no
  consumer. Given the saturation result there is little reason to add one.
- **The DeepCoder difficulty atlas does not exist yet** — only the 200-problem
  pilot does. A full sweep over 18,983 train problems at a non-binding cap is
  the prerequisite for a real curriculum band, and at ~10k+ tokens per
  generation it is a multi-day job.

## Status: the DeepCoder curriculum is PARKED (2026-08-16)

Parked deliberately, not abandoned, and **the trigger to resume is an event,
not a date**: *a SAGE-RL arm gets scheduled on a code task.*

**Why it is parked.** Nothing currently queued needs it. The near-term work is
tool-use / multi-turn / calibration on `qa_abstain`, which already has its own
difficulty machinery (`scripts/qa_calibrate.py` → `calib_file` + `band_mix`,
bands measured over 2,000 questions). The coding corpora are not on that path.

**Why it is worth resuming when SAGE comes back.** This is the strongest case
for the whole curriculum idea, and it is mechanical rather than aesthetic:
GRPO drops zero-variance groups and skips signal-free steps, so a corpus the
policy always passes contributes no gradient at all. MBPP is saturated for
qwen36 (0.97 pass@5 here; 0.870 MBPP+ under the official harness — see
`docs/mbpp-evalplus-results.md`), so it produces all-pass groups and cannot
train a SAGE arm. SAGE-RL specifically needs problems that require multi-step
reasoning *and* that the model sometimes gets right, because it learns when to
stop thinking from group-relative advantage across chains. DeepCoder is the
corpus with the headroom (qwen36 0.52 pass@3), and a difficulty band is what
turns its rollouts into signal instead of waste. Since rollouts dominate RL
cost, that is the efficiency argument for the task, not a refinement of it.

**State when parked** — nothing here needs redoing:

| artifact | state |
|---|---|
| MBPP pass@5 atlas | **done**, shipped: `data/labels/mbpp-pass@5-qwen36.jsonl` (3 temperatures — read the `labels_file` caveat above before using it) |
| DeepCoder pilots | 3 × 200 problems in `runs/sweeps/` (qwen36, qwen36-16k, qwen3-4b) |
| DeepCoder full atlas | **not started** |
| curriculum plumbing | **done**: `labels_file=` + `min_pass=`/`max_pass=` in `tasks/deepcoder.py`, `experimental/difficulty_sweep.py` |

**What resuming costs.** The sweep asks for ~60 GB and a ≥32k cap (the pilot
log shows it aborting the memory guard at 60 GB with a lens backend resident),
so it wants an exclusive memory lease and an otherwise idle box. Following the
`qa_abstain` precedent, ~2,000 labelled problems is enough for usable bands —
that is the sizing to plan against, not all 18,983.
