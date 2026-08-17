#!/usr/bin/env python3
# lifecycle: core
"""rl-dash — watch an mlx-rl training run's raw data live.

Read-only, stdlib-only page on :8105 (tailnet name rl-dash.strawrunway.com
via the strawrunway private Caddy). Tails the newest run under runs/ (or
--run) — metrics.jsonl for the per-step scoreboard and samples.jsonl for the
first group of every step: each member's reward, regime, tool calls (query ->
hits / found), how it ended, and the visible reply, with the full transcript
(injected tool responses included) one click away. Reading raw samples is how
every reward hack in this project was caught; this puts them on the phone.

The token stream itself (each episode as it decodes) is on the housekeeping
dashboard (dash.strawrunway.com) via mlx_rl.dashtap; this page is the
outcome view next to it. Nothing here writes anything.

Run:  python3 scripts/rl_dash.py [--port 8105] [--runs runs] [--run runs/<dir>]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_STEPS = 60          # metrics rows shown
N_SAMPLE_STEPS = 6    # sample groups shown (newest first)
MAX_TEXT = 6000       # chars of full transcript per member

_KEYS = ["step", "reward_mean", "reward_std", "active_groups", "mean_len",
         "frac_called", "frac_correct", "frac_grounded", "frac_abstain",
         "frac_denial", "frac_no_reply", "frac_tool_cap", "frac_len_capped",
         "kl", "gen_s", "update_s", "gen_tok_s", "peak_gb"]


def _newest_run(runs: Path) -> Path | None:
    cands = [p for p in runs.iterdir() if p.is_dir() and (p / "metrics.jsonl").exists()]
    if not cands:
        return None
    return max(cands, key=lambda p: (p / "metrics.jsonl").stat().st_mtime)


def _tail_jsonl(path: Path, n: int, max_bytes: int = 4_000_000) -> list[dict]:
    """Last n JSON lines without reading a multi-GB samples file whole."""
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - max_bytes))
        chunk = f.read()
    lines = chunk.split(b"\n")
    if size > max_bytes:
        lines = lines[1:]  # first line may be partial
    out = []
    for l in lines[-n:] if n else lines:
        l = l.strip()
        if not l:
            continue
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def _state(run: Path) -> dict:
    cfg = {}
    try:
        cfg = json.loads((run / "config.json").read_text())
    except Exception:
        pass
    metrics = _tail_jsonl(run / "metrics.jsonl", 0)
    steps = [m for m in metrics if "reward_mean" in m]
    evals = [m for m in metrics if "eval_reward" in m]
    samples = _tail_jsonl(run / "samples.jsonl", N_SAMPLE_STEPS)
    for s in samples:
        for c in s.get("completions", []):
            if len(c.get("text", "")) > MAX_TEXT:
                c["text"] = c["text"][:MAX_TEXT] + " …[truncated]"
    return {
        "run": run.name,
        "aborted": (run / "ABORTED").read_text() if (run / "ABORTED").exists() else None,
        "config": {k: cfg.get(k) for k in ("task", "steps", "batch_prompts", "group_size",
                                           "max_new_tokens", "max_tool_rounds", "lr",
                                           "kl_coef", "inject_r", "micro_batch")},
        "task_kwargs": cfg.get("task_kwargs"),
        "metrics_mtime": (run / "metrics.jsonl").stat().st_mtime if (run / "metrics.jsonl").exists() else None,
        "steps": [{k: m.get(k) for k in _KEYS if k in m} for m in steps[-N_STEPS:]],
        "evals": [{k: v for k, v in m.items() if k == "step" or k.startswith("eval_")}
                  for m in evals[-8:]],
        "samples": list(reversed(samples)),
        "now": time.time(),
    }


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rl-dash</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#c9d1d9;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
header{position:sticky;top:0;background:#161b22;border-bottom:1px solid #30363d;padding:8px 12px;display:flex;gap:16px;flex-wrap:wrap;align-items:baseline}
h1{font-size:14px;margin:0;color:#58a6ff}
.dim{color:#8b949e}
.warn{color:#f85149;font-weight:bold}
main{padding:8px 12px;max-width:1400px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:2px 6px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}
th{color:#8b949e;position:sticky;top:38px;background:#0d1117}
td:first-child,th:first-child{text-align:left}
.wrap{overflow-x:auto}
.grp{border:1px solid #30363d;border-radius:6px;margin:10px 0;background:#161b22}
.grp .hd{padding:6px 10px;border-bottom:1px solid #30363d;color:#c9d1d9}
.grp .hd b{color:#d2a8ff}
.mem{padding:6px 10px;border-bottom:1px solid #21262d}
.mem:last-child{border-bottom:0}
.r{display:inline-block;min-width:52px;font-weight:bold}
.pos{color:#3fb950}.neg{color:#f85149}.zero{color:#8b949e}
.tag{display:inline-block;padding:0 6px;border-radius:8px;background:#21262d;color:#8b949e;margin-right:4px}
.inj{background:#3b2f00;color:#d29922}
.vis{white-space:pre-wrap;color:#c9d1d9;margin:4px 0 0 0}
.calls{color:#79c0ff;margin:2px 0}
details{margin-top:4px}
summary{cursor:pointer;color:#8b949e}
pre.full{white-space:pre-wrap;background:#0d1117;padding:6px;border-radius:4px;max-height:60vh;overflow:auto;color:#a5d6ff}
h2{font-size:13px;color:#8b949e;margin:14px 0 4px;text-transform:uppercase;letter-spacing:.06em}
</style></head><body>
<header><h1>rl-dash</h1><span id="run" class="dim">…</span><span id="age" class="dim"></span><span id="abort" class="warn"></span></header>
<main>
<h2>steps</h2><div class="wrap"><table id="steps"></table></div>
<h2>eval</h2><div class="wrap"><table id="evals"></table></div>
<h2>samples — first prompt's whole group, newest step first</h2>
<div id="samples"></div>
</main>
<script>
const KEYS=["step","reward_mean","reward_std","active_groups","mean_len","frac_called","frac_correct","frac_grounded","frac_abstain","frac_denial","frac_no_reply","frac_tool_cap","frac_len_capped","kl","gen_s","update_s","gen_tok_s","peak_gb"];
const SHORT={reward_mean:"reward",reward_std:"±",active_groups:"active",mean_len:"len",frac_called:"called",frac_correct:"correct",frac_grounded:"grounded",frac_abstain:"abstain",frac_denial:"denial",frac_no_reply:"noreply",frac_tool_cap:"toolcap",frac_len_capped:"lencap",gen_s:"gen s",update_s:"upd s",gen_tok_s:"tok/s",peak_gb:"peak GB"};
function esc(s){return (s??"").toString().replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function fmt(k,v){if(v==null)return"";if(typeof v!=="number")return esc(v);if(k==="step"||k==="active_groups")return v;if(k==="mean_len"||k==="gen_tok_s"||k==="gen_s"||k==="update_s")return v.toFixed(0);return v.toFixed(2)}
function table(el,rows,keys){if(!rows.length){el.innerHTML="<tr><td class=dim>nothing yet</td></tr>";return}
 const ks=keys||Object.keys(rows[0]);let h="<tr>"+ks.map(k=>"<th>"+esc((SHORT[k]||k).replace(/^eval_/,""))+"</th>").join("")+"</tr>";
 for(const r of rows.slice().reverse())h+="<tr>"+ks.map(k=>"<td>"+fmt(k,r[k])+"</td>").join("")+"</tr>";el.innerHTML=h}
function rcls(x){return x>0?"pos":(x<0?"neg":"zero")}
function member(c){const p=c.parts||{};const calls=(c.tool_calls||[]).map(t=>`↳ ${esc((t.args||{}).query||t.name||"?")} → hits ${t.hits??"?"}${t.found_target?" ✓found":""}`).join("<br>");
 const tags=[c.injected?'<span class="tag inj">injected</span>':"",c.finish?`<span class=tag>${esc(c.finish)}</span>`:"",`<span class=tag>${c.len} tok</span>`,p.regime_known?'<span class=tag>known</span>':"",p.regime_post?'<span class=tag>post</span>':"",p.regime_future?'<span class=tag>future</span>':"",p.regime_fictional?'<span class=tag>fictional</span>':"",
  p.answered?`<span class=tag>answer${p.correct?" ✓":" ✗"}</span>`:"",p.abstain?'<span class=tag>abstain</span>':"",p.denial?'<span class=tag>denial</span>':"",p.no_reply?'<span class=tag>no reply</span>':""].join("");
 const txt=c.text||"";const marker="<|im_start|>assistant\n";let vis=txt.includes(marker)?txt.slice(txt.lastIndexOf(marker)+marker.length):txt;vis=vis.replace(/^<think>\n\n<\/think>\n\n/,"").replace(/<\|im_end\|>|<\|endoftext\|>/g,"");
 return `<div class=mem><span class="r ${rcls(c.reward)}">${(c.reward>=0?"+":"")+Number(c.reward).toFixed(2)}</span> ${tags}${calls?`<div class=calls>${calls}</div>`:""}<div class=vis>${esc(vis.slice(0,700))}${vis.length>700?" …":""}</div><details><summary>full transcript</summary><pre class=full>${esc(txt)}</pre></details></div>`}
function samples(list){const el=document.getElementById("samples");if(!list.length){el.innerHTML="<div class=dim>nothing yet</div>";return}
 el.innerHTML=list.map(s=>{const m=s.meta||{};return `<div class=grp><div class=hd><b>step ${s.step}</b> · ${esc(m.regime||"")} · today ${esc(m.today||"")} · pub ${esc(m.published||"—")} · ${esc(m.qtype||"")}<br><span class=dim>${esc(m.question||JSON.stringify(m).slice(0,200))}</span></div>${(s.completions||[]).map(member).join("")}</div>`}).join("")}
async function tick(){try{const r=await fetch("/api/state",{cache:"no-store"});const s=await r.json();
 document.getElementById("run").textContent=s.run+"  "+JSON.stringify(s.config);
 const age=s.metrics_mtime?Math.round(s.now-s.metrics_mtime):null;document.getElementById("age").textContent=age==null?"":`last write ${age}s ago`;
 document.getElementById("abort").textContent=s.aborted?("ABORTED: "+s.aborted):"";
 table(document.getElementById("steps"),s.steps,KEYS.filter(k=>s.steps.some(r=>k in r)));
 table(document.getElementById("evals"),s.evals);samples(s.samples)}catch(e){document.getElementById("age").textContent="fetch failed"}}
tick();setInterval(tick,15000);
</script></body></html>"""


def make_handler(runs_dir: Path, pinned: Path | None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?", 1)[0]
            if p == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if p == "/api/state":
                run = pinned or _newest_run(runs_dir)
                if run is None:
                    return self._send(200, json.dumps({"run": None, "steps": [], "evals": [],
                                                       "samples": [], "now": time.time()}).encode(),
                                      "application/json")
                return self._send(200, json.dumps(_state(run)).encode(), "application/json")
            if p == "/health":
                return self._send(200, b"ok", "text/plain")
            self._send(404, b"not found", "text/plain")
    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8105)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--run", default=None, help="pin one run dir (default: newest)")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port),
                              make_handler(Path(a.runs), Path(a.run) if a.run else None))
    print(f"rl-dash on http://{a.host}:{a.port} runs={a.runs}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
