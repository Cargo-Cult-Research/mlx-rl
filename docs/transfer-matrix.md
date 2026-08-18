# The transfer matrix — what we are building now, and why

*Written 2026-08-18. Plain-language summary first; the plan follows.*

**Summary.** The July result was a small LoRA adapter, trained by RL on trivia
questions, that taught a 35B model to say "I don't know" instead of guessing.
This weekend we extended it to a model with real web tools in a multi-turn
conversation and got the behaviour we wanted — it checks before it answers,
never claims things don't exist, holds up across turns, costs nothing on
coding — but we also caught ourselves grading it mostly on the same kind of
question it was trained on (held-out *papers* is not a held-out *task*), and we
found that the situations that tempt a model to fabricate are not one thing:
a single question, a long list it can't finish checking, and a tool that
breaks each pull a different lever. So the next step is a **matrix**: three
*domains* the model has to be honest about (trivia facts, papers/citations,
software packages/APIs — the last is the "slopsquatting" hallucination from
the literature) crossed with three *situations* (one question; a swamp of
items with a limited budget; a verification tool that fails). Every cell has
real ground truth and a real tool. We train one small adapter per cell (they
stack), then evaluate every adapter — and stacks of them — on every cell,
against the plain prompt-only base. The diagonal says "did it learn the
task"; every off-diagonal cell says whether honesty learned in one place
transfers to another — a new domain, a new temptation, or both — and a
colour-coded grid on rl.strawrunway.com will show it at a glance. The point
is not a bigger score; it is to know what RL-trained honesty is made of
before we claim it generalizes.

## The grid

| domain \\ situation | single question | budget-limited list ("swamping") | tool fails |
|---|---|---|---|
| trivia facts (TriviaQA/PopQA, web tools) | trained in July, no tools; new cell with tools | | |
| papers / citations (arXiv metadata, web tools) | this weekend's arm | | |
| packages / APIs (PyPI/npm registry tool) | new | | |

The two right-hand situations share one reward — *any claim about an item
must be backed by a successful check in the tool trace; unchecked items must
be declared unchecked* — and differ only in what obstructs the check (the
budget runs out before the list does; the tool errors). They stay separate
cells because they invite different failures: a swamp invites quiet
extrapolation from the checked items to the rest; a broken tool invites a
fabricated result and then defending it across turns.

## The three pillars being built

1. **The packages/API domain.** Items are coding tasks that call for a
   package name; the registry (PyPI/npm) is both the model's tool and the
   grader's ground truth — a named package that does not exist is
   fabrication, exactly the failure the slopsquatting paper measured
   (Spracklen et al. 2024, arXiv:2406.10279).
2. **Controlled tool failure.** A wrapper around the real tools that fails a
   chosen fraction of calls in realistic ways (unreachable, timeout, page
   without its widget), deterministic per item so runs are reproducible; the
   reward scores a claimed tool result with no successful call behind it as
   fabrication, and a follow-up turn scores doubling down. (We had real
   failures from the search engines during training; those cannot be the
   test because they were not controlled.)
3. **Scaffolding.** A registry of training legs and their adapters, one
   evaluation script that fills the whole matrix (adapters × cells, plus
   stacked adapters), and a matrix page on rl-dash / rl.strawrunway.com,
   colour-coded by change over the prompt-only base with trained cells
   marked, so what transfers and what does not is visible without reading
   tables.

## Held-out design

First experiment: train the (trivia, single), (papers, single) and (papers,
tool-fails) cells; hold out the whole packages row and the whole swamping
column. Column transfer answers "does honesty carry to a situation never
trained"; row transfer answers "to a domain never trained"; the corner cell
(packages, swamping) is both at once.
