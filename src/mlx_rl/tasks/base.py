"""Task interface: a task supplies prompts and a verifiable reward."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Example:
    messages: list[dict]
    meta: dict = field(default_factory=dict)


@dataclass
class RewardResult:
    total: float
    parts: dict = field(default_factory=dict)


class Task(Protocol):
    name: str

    def sample(self, rng: random.Random) -> Example: ...

    def reward(self, example: Example, completion: str) -> RewardResult: ...


_REGISTRY: dict[str, type] = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_task(name: str, **kwargs) -> Task:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
