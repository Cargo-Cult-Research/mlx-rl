#!/usr/bin/env python3
"""Live run dashboard for mlx-rl — stdlib only.

Serves an interactive page (scripts/dashboard_page.html) plus a small JSON
API over the run dirs in runs/: metric curves, per-rollout trace browser,
and a server-side anomaly scan tuned to the failure classes real runs have
actually been bitten by:

  * think_len > max_new_tokens        -> SAGE budget breach        (error)
  * reward > 0 with unclosed think    -> grader leak / reward hack (error)
  * swap growth / update_s blowup     -> the swap cliff            (warn)

The high-frequency warn classes (whole group at the cap, active_groups == 0,
>50% len-capped, zero-variance groups) are ROLLED UP into rates with a
per-task split (anomaly_rollup) + a per-step event timeline, instead of one
row per occurrence — a 120-step run used to bury the panel in ~100 warns.

Run:      .venv/bin/python scripts/dashboard.py [--port 8377] [--runs runs]
Expose:   binds loopback only. If you need remote access, front it with an
          authenticated proxy or VPN — NEVER expose it publicly; traces
          contain raw model text.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "dashboard_page.html"
RUNS = ROOT.parent / "runs"

# rfcs.py loaded by file path — keeps this server stdlib-only (no mlx import).
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "rfcs", ROOT.parent / "src" / "mlx_rl" / "rfcs.py")
_rfcs_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rfcs_mod)
rfcs = _rfcs_mod.rfcs


def inject_rfcs(metrics: list[dict], samples: list[dict]) -> None:
    """Post-hoc RFCS for runs whose trainer predates the rfcs field: annotate
    sampled completions and merge a per-step mean into the metrics rows as
    rfcs_sample_mean (kept distinct from the trainer's all-rollouts rfcs_mean:
    samples only cover the first prompt's group)."""
    per_step: dict[int, list[float]] = {}
    for s in samples:
        ans = (s.get("meta") or {}).get("answer")
        if ans is None:
            continue
        for c in s.get("completions", []):
            parts = c.get("parts") or {}
            v = parts.get("rfcs")
            if v is None:
                if parts.get("correct") != 1.0:
                    continue
                text = c.get("text") or ""
                if "</think>" not in text:
                    continue
                v = rfcs(text.split("</think>", 1)[0], str(ans))
                if v is None:
                    continue
                c["rfcs"] = round(v, 4)
            per_step.setdefault(s["step"], []).append(v)
    for m in metrics:
        vals = per_step.get(m.get("step"))
        if vals and "rfcs_sample_mean" not in m:
            m["rfcs_sample_mean"] = round(sum(vals) / len(vals), 4)

# Runs are "live" if their metrics/samples changed this recently (seconds).
LIVE_S = 900


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # mid-write torn line: the poller catches it next tick
    return out


def _mtime(*paths: Path) -> float:
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def list_runs(runs_dir: Path) -> list[dict]:
    out = []
    for d in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if not d.is_dir():
            continue
        metrics_p, samples_p = d / "metrics.jsonl", d / "samples.jsonl"
        config_p = d / "config.json"
        if not (metrics_p.exists() or samples_p.exists() or config_p.exists()):
            continue
        metrics = _read_jsonl(metrics_p)
        steps = [m["step"] for m in metrics if "reward_mean" in m]
        cfg = {}
        if config_p.exists():
            try:
                cfg = json.loads(config_p.read_text())
            except json.JSONDecodeError:
                pass
        mtime = _mtime(metrics_p, samples_p, config_p)
        live = (time.time() - mtime) < LIVE_S
        died = (d / "ABORTED").exists() or (
            not live and bool(steps) and cfg.get("steps")
            and max(steps) < cfg["steps"]
            and not any(m.get("final") for m in metrics)
        )
        out.append({
            "name": d.name,
            "mtime": mtime,
            "age_s": round(time.time() - mtime, 1),
            "live": live,
            "died": died,
            "last_step": max(steps) if steps else 0,
            "steps_cfg": cfg.get("steps"),
            "model": Path(cfg.get("model", "")).name or None,
            "task": cfg.get("task"),
            "sage_r": cfg.get("sage_r"),
        })
    out.sort(key=lambda r: -r["mtime"])
    return out


def _zero_var(comps: list[dict]) -> bool:
    rs = [c.get("reward") or 0.0 for c in comps]
    return max(rs) - min(rs) < 1e-6


def anomaly_rollup(cfg: dict, metrics: list[dict], samples: list[dict]) -> dict:
    """The high-frequency warn classes, grouped into rates instead of one row
    per occurrence (they dominated the panel — v8 emitted ~100 of them), and
    task-attributed where the samples allow: the signal bottleneck is likely
    systematic by task (arith all-pass vs math all-fail look identical in
    active_groups but need opposite fixes).

    Returns {summary: [{kind,label,n,total,rate,per_task?}],
             events: [{step,kind,task}]} — events feed the timeline strip.
    """
    cap = cfg.get("max_new_tokens") or 0
    events: list[dict] = []
    summary: list[dict] = []
    train = [m for m in metrics if "reward_mean" in m]

    def step_rate(kind: str, label: str, pred) -> None:
        hits = [m["step"] for m in train if pred(m)]
        for st in hits:
            events.append({"step": st, "kind": kind, "task": None})
        if train:
            summary.append({"kind": kind, "label": label, "n": len(hits),
                            "total": len(train),
                            "rate": round(len(hits) / len(train), 4)})

    step_rate("no_signal", "active_groups = 0 — update skipped",
              lambda m: m.get("active_groups") == 0)
    step_rate("len_capped", ">50% of rollouts at the length cap",
              lambda m: (m.get("frac_len_capped") or 0) > 0.5)

    # Group-level, task-attributable (samples = the first prompt's group per
    # step, so these rates are over that sampled subset). Only meaningful for
    # training runs — oracle/probe dirs mix decode conditions in one row.
    if cfg.get("group_size"):

        def group_rate(kind: str, label: str, pred) -> None:
            hits: list[tuple] = []
            per_task: dict[str, int] = {}
            totals: dict[str, int] = {}
            for s in samples:
                comps = s.get("completions", [])
                if len(comps) < 2:
                    continue
                task = (s.get("meta") or {}).get("_task") or "?"
                totals[task] = totals.get(task, 0) + 1
                if pred(comps):
                    hits.append((s.get("step"), task))
                    per_task[task] = per_task.get(task, 0) + 1
            for st, task in hits:
                events.append({"step": st, "kind": kind, "task": task})
            if totals:
                summary.append({
                    "kind": kind, "label": label, "n": len(hits),
                    "total": sum(totals.values()),
                    "rate": round(len(hits) / sum(totals.values()), 4),
                    "per_task": {t: round(per_task.get(t, 0) / n, 4)
                                 for t, n in sorted(totals.items())}})

        group_rate("group_truncated", "whole sampled group at the cap",
                   lambda comps: bool(cap) and all(
                       (c.get("len") or 0) >= cap for c in comps))
        group_rate("zero_var_fail",
                   "zero-variance group, all FAIL — no gradient",
                   lambda comps: _zero_var(comps)
                   and (comps[0].get("reward") or 0) <= 0)
        group_rate("zero_var_pass",
                   "zero-variance group, all pass — no gradient",
                   lambda comps: _zero_var(comps)
                   and (comps[0].get("reward") or 0) > 0)

    return {"summary": summary, "events": events}


def scan_anomalies(
    cfg: dict,
    metrics: list[dict],
    samples: list[dict],
    run_dir: Path | None = None,
    live: bool = True,
) -> list[dict]:
    """Each anomaly: {level: error|warn|info, step, msg}. Tuned to observed
    failure classes, not hypothetical ones."""
    a: list[dict] = []
    cap = cfg.get("max_new_tokens") or 0

    # Death detection (a swap-guard kill is otherwise invisible — the run
    # just stops progressing with nothing in-band to say why).
    aborted = run_dir / "ABORTED" if run_dir else None
    if aborted and aborted.exists():
        reason = aborted.read_text().strip().splitlines()
        a.append({"level": "error", "step": None,
                  "msg": f"run ABORTED — {reason[0] if reason else 'no reason recorded'}"})
    elif not live and cfg.get("steps"):
        done = [m["step"] for m in metrics if "reward_mean" in m]
        finished = any(m.get("final") for m in metrics)
        if done and not finished and max(done) < cfg["steps"]:
            a.append({"level": "error", "step": max(done),
                      "msg": f"run DIED at step {max(done)}/{cfg['steps']} — "
                             "no final eval, no writes since; check the log "
                             "tail and `ARM ... exit` lines"})

    for s in samples:
        step, comps = s.get("step"), s.get("completions", [])
        if not comps:
            continue
        for i, c in enumerate(comps):
            tl = c.get("think_len")
            if cap and tl is not None and tl > cap:
                a.append({"level": "error", "step": step,
                          "msg": f"SAGE think_len {tl} > max_new_tokens {cap} "
                                 f"(completion {i}) — reasoning budget breached"})
            closed = (c.get("parts") or {}).get("think_closed")
            if closed == 0.0 and (c.get("reward") or 0) > 0:
                a.append({"level": "error", "step": step,
                          "msg": f"reward {c['reward']:.2f} granted on an UNCLOSED "
                                 f"think block (completion {i}) — grader leak"})
        # (whole-group truncation moved to anomaly_rollup — was one warn
        # row per step, dominating the panel)

    train = [m for m in metrics if "reward_mean" in m]
    upd = [m.get("update_s", 0.0) for m in train]
    med_upd = statistics.median(upd) if upd else 0.0
    prev_eval = None
    for m in metrics:
        step = m.get("step")
        # (active_groups == 0 moved to anomaly_rollup)
        if (m.get("swap_gb") or 0) > 1.0:
            a.append({"level": "warn", "step": step,
                      "msg": f"swap grew {m['swap_gb']:.1f} GB above baseline — "
                             "approaching the paging cliff"})
        if med_upd > 60 and m.get("update_s", 0) > 3 * med_upd:
            a.append({"level": "warn", "step": step,
                      "msg": f"update_s {m['update_s']:.0f}s is >3x the median "
                             f"({med_upd:.0f}s) — swap/thrash suspect"})
        # (frac_len_capped > 0.5 moved to anomaly_rollup)
        if "eval_reward" in m:
            if prev_eval is not None and m["eval_reward"] < prev_eval - 0.15:
                a.append({"level": "info", "step": step,
                          "msg": f"eval_reward dropped {prev_eval:.2f} → "
                                 f"{m['eval_reward']:.2f}"})
            prev_eval = m["eval_reward"]
    order = {"error": 0, "warn": 1, "info": 2}
    a.sort(key=lambda x: (order[x["level"]], -(x["step"] or 0)))
    return a


class Handler(BaseHTTPRequestHandler):
    runs_dir: Path = RUNS

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        try:
            if not parts:
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif parts == ["api", "runs"]:
                self._json({"now": time.time(), "runs": list_runs(self.runs_dir)})
            elif parts[:2] == ["api", "run"] and len(parts) == 3:
                self._run_detail(parts[2])
            elif parts == ["api", "log"]:
                self._log_tail(parse_qs(url.query))
            else:
                self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:  # keep the dashboard up no matter what
            self._json({"error": repr(e)}, 500)

    def _run_dir(self, name: str) -> Path | None:
        d = (self.runs_dir / name).resolve()
        if d.parent != self.runs_dir.resolve() or not d.is_dir():
            return None
        return d

    def _run_detail(self, name: str) -> None:
        d = self._run_dir(name)
        if d is None:
            return self._json({"error": "no such run"}, 404)
        cfg = {}
        if (d / "config.json").exists():
            try:
                cfg = json.loads((d / "config.json").read_text())
            except json.JSONDecodeError:
                pass
        metrics = _read_jsonl(d / "metrics.jsonl")
        samples = _read_jsonl(d / "samples.jsonl")
        inject_rfcs(metrics, samples)
        mtime = _mtime(d / "metrics.jsonl", d / "samples.jsonl", d / "config.json")
        live = (time.time() - mtime) < LIVE_S
        anomalies = scan_anomalies(cfg, metrics, samples, run_dir=d, live=live)
        self._json({
            "name": name,
            "now": time.time(),
            "mtime": mtime,
            "age_s": round(time.time() - mtime, 1),
            "live": live,
            "died": any(x["level"] == "error" and
                        ("ABORTED" in x["msg"] or "DIED" in x["msg"])
                        for x in anomalies),
            "config": cfg,
            "metrics": metrics,
            "samples": samples,
            "anomalies": anomalies,
            "rollup": anomaly_rollup(cfg, metrics, samples),
        })

    def _log_tail(self, q: dict) -> None:
        name = (q.get("name") or ["train.log"])[0]
        n = int((q.get("n") or ["120"])[0])
        p = (self.runs_dir / name).resolve()
        if (p.parent != self.runs_dir.resolve() or p.suffix != ".log"
                or not p.exists()):
            return self._json({"error": "no such log"}, 404)
        lines = p.read_text(errors="replace").splitlines()[-n:]
        self._json({"name": name, "lines": lines})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8377)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (keep loopback; front with an "
                         "authenticated proxy for remote access)")
    ap.add_argument("--runs", default=str(RUNS))
    args = ap.parse_args()
    Handler.runs_dir = Path(args.runs).resolve()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mlx-rl dashboard on http://{args.host}:{args.port} "
          f"(runs: {Handler.runs_dir})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
