"""Resume: state round-trip, keep-k pruning, and end-to-end determinism.

The determinism test is the one that matters — it runs the real trainer twice
and asserts a resumed run produces byte-identical metrics to an uninterrupted
one. Marked integration (needs the tiny model cached); the rest are pure.
"""

import json
import pickle
import random
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_rl.train import load_resume, save_resume


class _FakeTask:
    """Minimal stand-in with sampler state, like KodCodeTask's epoch order."""

    def __init__(self):
        self.order = [3, 1, 2]
        self.epoch = 7

    def get_state(self):
        return {"order": list(self.order), "epoch": self.epoch}

    def set_state(self, s):
        self.order, self.epoch = list(s["order"]), s["epoch"]


def _optimizer():
    m = nn.Linear(4, 4)
    o = optim.Adam(learning_rate=1e-3)
    o.update(m, {k: mx.zeros(v.shape) for k, v in m.parameters().items()})
    mx.eval(o.state)
    return m, o


def test_roundtrip_restores_rng_task_and_optimizer(tmp_path):
    m, o = _optimizer()
    rng, sub = random.Random(0), np.random.default_rng(0)
    for _ in range(5):          # advance both streams
        rng.random(); sub.random()
    task = _FakeTask()
    (tmp_path / "adapters").mkdir()
    mx.save_safetensors(
        str(tmp_path / "adapters" / "adapter-00010.safetensors"),
        {"w": mx.zeros((2, 2))})
    save_resume(tmp_path, 10, o, rng, sub, task, [1, 2], keep=1)

    want_rng = rng.random()
    want_sub = sub.random()

    rng2, sub2 = random.Random(999), np.random.default_rng(999)
    task2 = _FakeTask()
    task2.order, task2.epoch = [], 0
    m2, o2 = _optimizer()
    step, st = load_resume(tmp_path, m2, o2, rng2, sub2, task2)

    assert step == 10
    assert st["activity_window"] == [1, 2]
    assert rng2.random() == want_rng          # python stream continues
    assert sub2.random() == want_sub          # numpy stream continues
    assert task2.order == [3, 1, 2] and task2.epoch == 7
    assert "step" in dict(__import__("mlx.utils", fromlist=["tree_flatten"])
                          .tree_flatten(o2.state))


def test_keep_k_prunes_oldest_and_keeps_pairs(tmp_path):
    m, o = _optimizer()
    (tmp_path / "adapters").mkdir()
    for step in (10, 20, 30, 40):
        save_resume(tmp_path, step, o, random.Random(0),
                    np.random.default_rng(0), _FakeTask(), [], keep=2)
    d = tmp_path / "resume"
    opts = sorted(p.name for p in d.glob("opt-*.safetensors"))
    states = sorted(p.name for p in d.glob("state-*.pkl"))
    assert opts == ["opt-00030.safetensors", "opt-00040.safetensors"]
    assert states == ["state-00030.pkl", "state-00040.pkl"]


def test_keep_one_is_the_default_behaviour(tmp_path):
    m, o = _optimizer()
    for step in (5, 6, 7):
        save_resume(tmp_path, step, o, random.Random(0),
                    np.random.default_rng(0), _FakeTask(), [], keep=1)
    d = tmp_path / "resume"
    assert len(list(d.glob("opt-*.safetensors"))) == 1
    assert len(list(d.glob("state-*.pkl"))) == 1


def test_resume_source_must_differ_from_out():
    from mlx_rl.config import TrainConfig
    from mlx_rl.train import _train
    cfg = TrainConfig(resume_from="runs/x", steps=1)
    with pytest.raises(SystemExit, match="must differ"):
        _train(cfg, "runs/x")


def test_resume_without_state_fails_loudly(tmp_path):
    m2, o2 = _optimizer()
    with pytest.raises(SystemExit, match="no resume state"):
        load_resume(tmp_path, m2, o2, random.Random(0),
                    np.random.default_rng(0), _FakeTask())


def test_resume_with_pruned_optimizer_fails_loudly(tmp_path):
    m, o = _optimizer()
    (tmp_path / "adapters").mkdir()
    save_resume(tmp_path, 10, o, random.Random(0), np.random.default_rng(0),
                _FakeTask(), [], keep=1)
    (tmp_path / "resume" / "opt-00010.safetensors").unlink()
    m2, o2 = _optimizer()
    with pytest.raises(SystemExit, match="missing"):
        load_resume(tmp_path, m2, o2, random.Random(0),
                    np.random.default_rng(0), _FakeTask())


@pytest.mark.integration
def test_resumed_run_matches_uninterrupted_run(tmp_path):
    """Run 6 steps straight through; separately run 3, then resume to 6.
    Steps 4-6 must match exactly — same data order, same rewards, same
    gradients. This is the whole point of saving optimizer + RNG + sampler."""
    def cmd(steps, out, *extra):
        return [sys.executable, "-m", "mlx_rl.train", "--task", "arithmetic",
                "--steps", str(steps), "--batch-prompts", "2",
                "--group-size", "4", "--max-new-tokens", "48",
                "--checkpoint-every", "3", "--eval-every", "1000",
                "--eval-n", "4", "--seed", "0", "--out", str(out), *extra]

    def run(argv):
        r = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return r

    full, part, cont = tmp_path / "full", tmp_path / "part", tmp_path / "cont"
    run(cmd(6, full))
    run(cmd(3, part))                                   # interrupt at 3
    part_before = (part / "metrics.jsonl").read_text()
    r = run(cmd(6, cont, "--resume-from", str(part)))    # continue in a NEW dir
    assert "at step 3" in r.stdout
    # the source run is a read-only record
    assert (part / "metrics.jsonl").read_text() == part_before
    assert sorted(p.name for p in (part / "adapters").glob("*.safetensors")) \
        == ["adapter-00003.safetensors"]

    def steps(p):
        rows = [json.loads(l) for l in (p / "metrics.jsonl").read_text().splitlines()]
        return {r["step"]: r for r in rows if "reward_mean" in r}

    a, b = steps(full), steps(cont)
    assert set(a) == {1, 2, 3, 4, 5, 6}
    assert set(b) == {4, 5, 6}          # the new dir holds only its own steps
    for s in (4, 5, 6):
        for k in ("reward_mean", "reward_std", "mean_len", "pg", "kl",
                  "active_groups", "gen_nll"):
            assert a[s][k] == pytest.approx(b[s][k], rel=1e-9, abs=1e-9), (
                f"step {s} field {k}: continuous {a[s][k]} != resumed {b[s][k]}")

    # assembling the segments reconstructs the full timeline
    asm = tmp_path / "asm"
    r = subprocess.run(
        [sys.executable, "scripts/assemble_run.py", str(part), str(cont),
         "--out", str(asm)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert set(steps(asm)) == {1, 2, 3, 4, 5, 6}
    prov = json.loads((asm / "provenance.json").read_text())
    assert prov["segments"][1]["resumed_at_step"] == 3
