"""Optional live tap on the rollout token streams.

Point ``MLX_RL_DASHTAP`` at a module exposing ``DashTap(src=...)`` with
``start(**meta) -> rid``, ``text(rid, s)``, ``end(rid, **meta)`` and
``note(text)``, and rollouts mirror what they generate to it as it decodes.
Unset, or on any failure to load, this is a no-op: nothing in training
depends on it.
"""

from __future__ import annotations

import importlib.util
import os
import sys


class NullTap:
    def start(self, **meta):
        return None

    def text(self, rid, s):
        pass

    def end(self, rid, **meta):
        pass

    def note(self, text):
        pass


def load_tap():
    path = os.environ.get("MLX_RL_DASHTAP", "")
    if not path or path == "0":
        return NullTap()
    try:
        spec = importlib.util.spec_from_file_location("_dashtap_lib", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        prog = os.path.basename(sys.argv[0] or "mlx-rl")
        return mod.DashTap(src=f"mlx-rl:{prog.removesuffix('.py')}")
    except Exception:
        return NullTap()
