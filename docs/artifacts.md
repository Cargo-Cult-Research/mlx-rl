# Artifacts — what we ship, and where

Training weights and run outputs are gitignored (`runs/`, `adapters/`), so for
a long time they existed on exactly one disk. That cost us a real day: an
outside reproduction of the arXiv transfer result concluded the artifacts were
lost, and rebuilt the calibration file and the judge cache from scratch. They
were not lost — they were simply never shipped anywhere.

Two destinations, deliberately separate:

| | what | where |
|---|---|---|
| **External** | the deliverable pair only: the arm-C adapter and the system prompt it is trained to work with | Hugging Face, with a model card |
| **Internal** | everything needed to re-derive or re-check the numbers: all trained arms, the calibration file, the judge cache, raw rollouts, probe outputs | a release on the private mirror `<your-org>/<your-repo>` |

The split is on purpose. The external artifact is one thing a stranger can pick
up and use. The internal bundle is the evidence chain, and it is large, raw,
and only interesting if you are checking our work.

## External — the deliverable pair

The adapter does nothing on its own. It is trained to respond to a specific
system prompt, and with that prompt absent it measures at base-model behaviour
on every probe bucket. So the unit we publish is the **pair**: adapter +
prompt. Shipping the adapter alone would be shipping half an artifact.

## Internal — the reproduction bundle

Build it with `bash scripts/bundle_artifacts.sh`, which writes two tarballs and
a checksum file to `dist/`.

```
adapters/       9 trained arms, 16 MB each (LoRA, rank 16, last 12 layers)
runs/           per training run: exact config, per-step metrics, raw rollouts
calib/          the 2,000-question calibration probe every run trained against
judge/          48,227 cached judge verdicts
probes/         the probe outputs behind every number in the results tables
```

### What each piece is for

**`adapters/`** — one directory per trained arm, each with
`adapters.safetensors`, `adapter_config.json`, and a one-line `MANIFEST.md`
recording which run and checkpoint it was promoted from. `qa-gloveC-200` also
carries `GLOVE.txt`, the system prompt it is paired with. Load one with
mlx-lm's `adapter_path`, or point a serving backend at it.

**`runs/`** — `config.json` is the complete training configuration, so a re-run
does not depend on anyone transcribing flags out of prose. (The reproduction
attempt lost two launches to exactly that: a hand-copied recipe that dropped
`--lora-layers 12 --grad-checkpoint`.) `metrics.jsonl` is per-step training
telemetry; `samples.jsonl` is the raw generations, which is where you look when
a metric moves and you want to know what the model actually said.

**`calib/`** — before training, we probe the base model 8 times on each of
2,000 trivia questions and sort them by how often it gets them right: reliably
right, sometimes right, reliably wrong. Training then samples questions from
those three groups in a chosen ratio. That file is an input to every run, and
regenerating it is a couple of hours of GPU time — hence shipping it.

**`judge/`** — grading a free-conversation reply means deciding whether the
model committed to an answer, hedged, or asserted the thing does not exist.
A larger model does that grading, and the verdicts are cached by content hash.
Ship the cache and a re-run costs no judge traffic at all for anything already
seen.

**`probes/`** — the evaluation outputs, one directory per (arm × probe). These
are the actual measurements quoted in `qa-glove-results.md`; if you want to
check a cell in a table rather than trust it, the replies and per-reply grades
are here.

## The lesson this encodes

Copying a checkpoint to `~/models/adapters` felt like promotion, but that
directory is on one machine and in no backup. An artifact is shipped when
someone else can download it. Anything we call a deliverable goes to one of the
two destinations above on the day it earns the name.
