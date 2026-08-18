# Glossary — say it in English

This repo grew a private vocabulary. Some of it is load-bearing (it names
things that have no standard word) and some of it was just us being clever at
2 a.m. The cost showed up when an outside collaborator reproduced one of our
results and wrote the reproduction back in our dialect: the document was
correct and nearly unreadable.

Rule going forward: **prose uses the right-hand column.** The left-hand column
survives only where it is a filename, a config key, or a published artifact
name — those are identifiers, and renaming them would break the run configs and
the released adapters. When you must use one, translate it on first use.

## The words

| we said | say instead | what it actually means |
|---|---|---|
| the glove | the honesty prompt | The system prompt shipped alongside the adapter (`GLOVE.txt`, `HONESTY_SYSTEM` in code). One paragraph telling the model that saying "I don't know" is acceptable. The adapter is trained with it and only works with it. |
| register | the kind of prompt / the setting | Whether the question arrives in our tagged evaluation format or as ordinary conversation. The whole surprise of the project is that a behaviour trained in one does not appear in the other. |
| register binding | the pairing | The trained behaviour appears only when the honesty prompt is present. Without it the adapter measures at base-model behaviour everywhere. That is a feature: the adapter cannot leak caution into conversations that did not ask for it. |
| affordance | the option to decline | Whether the prompt tells the model that declining is allowed at all. |
| arm | arm | Fine — standard experiment vocabulary. One training configuration in a comparison. Keep it, but say what changed. |
| band | confidence group | Before training we ask the base model each question 8 times and sort the questions into three groups by how often it got them right: reliably right, sometimes right, reliably wrong. |
| band mix / `band_mix` | the question mix | How often training draws from each of those three groups. `0.15/0.35/0.50` means 15% reliably-right, 35% sometimes, 50% reliably-wrong. |
| unknown-heavy | weighted toward questions it doesn't know | — |
| rung | step / level of comparison | A baseline in a ladder of baselines: base model, then base + prompt, then base + prompt + training. |
| hedge | declines to answer | Says it isn't sure, or doesn't know. |
| denial | denies the thing exists | The specific failure where the model says a real paper or person does not exist because it doesn't recognise the name. Worse than a wrong answer; penalised hardest. |
| h+d | declined or denied | The two together, as a rate. |
| confidently wrong | confidently wrong | Fine. Answered, wrongly, with no hedge. |
| decline@k | decline rate over k samples | How often it declines when asked the same question k times. |
| binding (correlation) | does it track what it knows | Whether the model declines specifically on questions it gets wrong, rather than declining more across the board. Measured as correlation between its decline rate and the base model's success rate on held-out questions. |
| absorbing state | a trap the run cannot leave | Keep the term — it is precise. Once the model abstains on everything, every group of samples agrees, so there is no variance, so GRPO computes no gradient, so nothing ever pulls it back out. |
| sparse ignition | too rare to reinforce | A behaviour occurring ~1% of the time gives the policy gradient almost nothing to work with. Not a broken mechanism — an empty one. |
| propensity | tendency | — |
| EV-balanced | balanced so answering and declining pay the same | The question mix chosen so a well-calibrated model has no systematic reason to prefer either. |
| uplift | improvement over | Always name the thing being improved on — "improvement over the prompt alone" is a number; "uplift" is a mood. |
| inert | does nothing | — |
| the checkpoint dial | the checkpoint trade-off | Later checkpoints are not strictly better; they trade caution on unknowns against cost on knowns. Which one you want depends on the deployment. |
| recall frame | the "who wrote it" question | Asking directly for a paper's authors or year. |
| chat frame | the "tell me about this" question | Handing over a title and URL and asking about the paper. The same model behaves ~2× differently between the two, so always say which one a number came from. |
| the deliverable pair | the adapter and its prompt | The unit we ship. Half of it alone does nothing. |

One deliberate exception: `DATASETS.md` says **difficulty band** for the same
idea applied to problems rather than questions ("restrict training draws to a
difficulty band"). That one reads fine in context and stays.

## Artifact names you cannot rename

These are baked into published run configs, the released adapters, and other
people's reproductions. They keep the old vocabulary; explain them, don't
rewrite them.

- `qa_abstain` — the task. Answer a factual question, or decline.
- `GLOVE.txt`, `HONESTY_SYSTEM` — the honesty prompt.
- `qa-gloveA-*`, `qa-gloveB-*`, `qa-gloveC-*`, `qa-glovec1-*` — the trained
  arms. A: conversation mix weighted toward unknowns. B: conversation mix
  tuned like the tagged format. C: A, but with more weight on questions the
  model does know. c1: C with a milder wrong-answer penalty.
- `band_mix`, `chat_band_mix`, `calib_file`, `chat_frac`, `inject_r` — config
  keys.

## Why this matters beyond taste

Two concrete costs we have already paid:

1. A reproduction of our own result was written in this dialect and could not
   be read by anyone who had not read the source docs first. The finding was
   good; the write-up did not travel.
2. The results doc reported the effect as base → trained and never carried the
   middle rung — how much the honesty prompt alone achieves. Nobody hid it; the
   number was measured, written down in a different document on a different
   branch, and simply not brought forward. The jargon made the omission hard to
   see, because "the glove is the affordance, the curriculum is the learning"
   *sounds* like it accounts for the prompt. It doesn't. It took an outside
   reader re-deriving the number to put it in the table.

Compressed language hides missing baselines. That is the actual argument.
