"""MixtureTask — samples one sub-task per example by weight and routes the
reward back to it. Diverse real tasks in one RL run (code + arithmetic +
toolformat) so the policy isn't shaped by a single distribution
(single-distribution training invites degenerate shortcuts, e.g. pure
length hacks).

task_kwargs, e.g.:
  {"weights": {"code": 0.5, "arithmetic": 0.25, "toolformat": 0.25},
   "arithmetic": {"n_operands": 7, "format_reward": 0.0}}
Per-subtask kwargs are optional; each sub-task keeps its own registered reward.
"""
from __future__ import annotations

import random

from .base import Example, RewardResult, get_task, register

_DEFAULT_W = {"code": 0.5, "arithmetic": 0.25, "toolformat": 0.25}


@register
class MixtureTask:
    name = "mixture"

    def __init__(self, weights: dict | None = None, **subkwargs):
        self.weights = dict(weights or _DEFAULT_W)
        self._names = list(self.weights)
        self._w = [self.weights[n] for n in self._names]
        # instantiate each sub-task with its own optional kwargs block
        self._tasks = {n: get_task(n, **(subkwargs.get(n) or {}))
                       for n in self._names}

    def _tag(self, name: str, ex: Example) -> Example:
        ex.meta["_task"] = name
        return ex

    def _pick(self, rng: random.Random) -> str:
        return rng.choices(self._names, weights=self._w, k=1)[0]

    def sample(self, rng: random.Random) -> Example:
        n = self._pick(rng)
        return self._tag(n, self._tasks[n].sample(rng))

    def eval_sample(self, rng: random.Random) -> Example:
        n = self._pick(rng)
        t = self._tasks[n]
        s = getattr(t, "eval_sample", t.sample)
        return self._tag(n, s(rng))

    def reward(self, example: Example, completion: str) -> RewardResult:
        n = example.meta.get("_task", self._names[0])
        res = self._tasks[n].reward(example, completion)
        res.parts.setdefault("correct", res.parts.get("correct", 0.0))
        res.parts[f"task_{n}"] = 1.0        # so metrics show the mix proportions
        return res
