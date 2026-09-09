# experimental/ — per-experiment code, quarantined from the core

The rule (2026-08-22 audit): **`src/mlx_rl` and `scripts/` are the tested,
stable surface; everything that exists for one experiment lives here.**
Probe scripts, per-arc drivers, dataset spot-checks.

What that buys:

- `scripts/` stays small enough that every file in it is load-bearing and
  expected to work (`rl_dash.py` is served by launchd; `matrix_eval.py`,
  `promote_adapter.py`, `launch_detached.py`, the `*_calibrate.py` and
  `fetch_*.py` instruments are how runs and data get rebuilt).
- Nothing in here is a dependency of the core: moving or archiving a file
  in `experimental/` can never break training, serving, or the test suite.
- One-off drivers carry `# lifecycle: one-off (archive when <event>)`
  headers per the machine convention; archive them on their event.

This directory is tracked (history is the point — these files are the lab
record of how each result was produced), but it is internal-mirror material:
a public release branch strips it. Docs under `docs/` reference the paths
these files had when their experiments ran; those references are historical
and are not rewritten.

New experiment? Start the driver here, not in `scripts/`. If a tool proves
itself across several experiments, promote it to `scripts/` together with
tests.
