"""Live-dashboard tap discovery (housekeeping dashboard on :8097).

Rollouts mirror their token streams to the machine's live dashboard so Urs
can watch training generations from his phone, exactly like :8084 traffic.
Same discovery pattern as the memory lease (machine.py): zero hard deps —
``~/code/housekeeping/dash/tap.py`` is probed at import time and any failure
degrades to a no-op tap.

Env:
  MLX_RL_DASHTAP=0        disable the tap entirely
  MLX_RL_DASHTAP=/path    override the probed tap.py location
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
    override = os.environ.get("MLX_RL_DASHTAP")
    if override == "0":
        return NullTap()
    path = override or os.path.expanduser("~/code/housekeeping/dash/tap.py")
    try:
        spec = importlib.util.spec_from_file_location("_dashtap_lib", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        prog = os.path.basename(sys.argv[0] or "mlx-rl")
        return mod.DashTap(src=f"mlx-rl:{prog.removesuffix('.py')}")
    except Exception:
        return NullTap()
