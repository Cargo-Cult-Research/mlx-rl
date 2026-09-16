#!/usr/bin/env python3
# lifecycle: core
"""rl-dash — watch an mlx-rl training run's raw data live.

Read-only, stdlib-only page on :8105 (tailnet name rl-dash.strawrunway.com
via the strawrunway private Caddy), plus a timer-gated PUBLIC mirror on
:8106 (tunnel path strawrunway.com/rl-dash — same two-faced pattern as
dash./cyber/social): off by default, "share public N h" from the private
page or `curl -X POST 'http://127.0.0.1:8105/mirror?on=8'` (`?off=1` to
close), expiry enforced server-side per request, so it always shuts itself
off. The mirror is read-only and follows whatever run is newest, which is
why it is a deliberate act, not the default. Tails the newest run under runs/ (or
--run) — metrics.jsonl for the per-step scoreboard and samples.jsonl for the
first group of every step: each member's reward, regime, tool calls (query ->
hits / found), how it ended, and the visible reply, with the full transcript
(injected tool responses included) one click away. Reading raw samples is how
every reward hack in this project was caught; this puts them on the phone.

The token stream itself (each episode as it decodes) is on the housekeeping
dashboard (dash.strawrunway.com) via mlx_rl.dashtap; this page is the
outcome view next to it. Three more views hang off it: /labbook (training
curves per experiment manifest in runs/experiments/, plus everything found
on disk), /matrix (the transfer matrix), and /review (human labels on
sampled replies -- the ONE thing here that writes, to
runs/human-review/labels.jsonl; the mirror timer file is the other).

Run:  python3 scripts/rl_dash.py [--port 8105] [--public-port 8106] [--runs runs] [--run runs/<dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
MIRROR_FILE = Path(__file__).resolve().parent / ".rl-dash-public-until"  # gitignored
MAX_HOURS = 24.0
PUB_PREFIX = "/rl-dash"


def mirror_until() -> float:
    try:
        return float(MIRROR_FILE.read_text().strip())
    except Exception:
        return 0.0


def mirror_on() -> bool:
    return time.time() < mirror_until()


def set_mirror(hours: float | None) -> float:
    if hours is None:
        MIRROR_FILE.unlink(missing_ok=True)
        return 0.0
    until = time.time() + hours * 3600
    MIRROR_FILE.write_text(f"{until:.0f}\n")
    return until
N_STEPS = 60          # metrics rows shown
N_SAMPLE_STEPS = 6    # sample groups shown (newest first)
MAX_TEXT = 6000       # chars of full transcript per member

_KEYS = ["step", "reward_mean", "reward_std", "active_groups", "mean_len",
         "frac_called", "frac_correct", "frac_grounded", "frac_abstain",
         "frac_denial", "frac_no_reply", "frac_tool_cap", "frac_len_capped",
         "kl", "gen_s", "update_s", "gen_tok_s", "peak_gb",
         "web_search_live", "web_search_hit", "web_fetch_live", "web_fetch_hit", "web_errors",
         "web_search_real", "web_search_fallback_error", "web_search_fallback_empty", "frac_fallback"]


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


def scan_anomalies(cfg: dict, metrics: list[dict], samples: list[dict], run: Path) -> list[dict]:
    """Each anomaly: {level: error|warn|info, step, msg}. Tuned to observed
    failure classes, not hypothetical ones -- the grader-leak and SAGE-breach
    checks are how those bugs were caught the first time."""
    a: list[dict] = []
    cap = cfg.get("max_new_tokens") or 0
    train = [m for m in metrics if "reward_mean" in m]
    if not (run / "ABORTED").exists() and cfg.get("steps") and train:
        # A swap-guard kill or a crash is otherwise invisible: the run just
        # stops progressing with nothing in-band to say why.
        last = max(m["step"] for m in train)
        finished = any(m.get("final") for m in metrics)
        stale = time.time() - (run / "metrics.jsonl").stat().st_mtime > 86400
        if not finished and last < cfg["steps"] and stale:
            a.append({"level": "error", "step": last,
                      "msg": f"run DIED at step {last}/{cfg['steps']} -- no final eval, "
                             "no writes for a day; check the log tail"})
    for smp in samples:
        step = smp.get("step")
        for i, c in enumerate(smp.get("completions", [])):
            tl = c.get("think_len")
            if cap and tl is not None and tl > cap:
                a.append({"level": "error", "step": step,
                          "msg": f"SAGE think_len {tl} > max_new_tokens {cap} "
                                 f"(completion {i}) -- reasoning budget breached"})
            if (c.get("parts") or {}).get("think_closed") == 0.0 and (c.get("reward") or 0) > 0:
                a.append({"level": "error", "step": step,
                          "msg": f"reward {c['reward']:.2f} granted on an UNCLOSED think "
                                 f"block (completion {i}) -- grader leak"})
    upd = sorted(m.get("update_s", 0.0) for m in train)
    med_upd = upd[len(upd) // 2] if upd else 0.0
    prev_eval = None
    for m in metrics:
        step = m.get("step")
        if (m.get("swap_gb") or 0) > 1.0:
            a.append({"level": "warn", "step": step,
                      "msg": f"swap grew {m['swap_gb']:.1f} GB above baseline -- "
                             "approaching the paging cliff"})
        if med_upd > 60 and m.get("update_s", 0) > 3 * med_upd:
            a.append({"level": "warn", "step": step,
                      "msg": f"update_s {m['update_s']:.0f}s is >3x the median "
                             f"({med_upd:.0f}s) -- swap/thrash suspect"})
        if m.get("no_update") and m.get("active_groups") == 0:
            a.append({"level": "warn", "step": step, "msg": "no active groups -- update skipped"})
        if "eval_reward" in m:
            if prev_eval is not None and m["eval_reward"] < prev_eval - 0.15:
                a.append({"level": "info", "step": step,
                          "msg": f"eval_reward dropped {prev_eval:.2f} -> {m['eval_reward']:.2f}"})
            prev_eval = m["eval_reward"]
    order = {"error": 0, "warn": 1, "info": 2}
    a.sort(key=lambda x: (order[x["level"]], -(x["step"] or 0)))
    return a


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
        "metrics_mtime": (run / "metrics.jsonl").stat().st_mtime if (run / "metrics.jsonl").exists() else None,
        "steps": [{k: m[k] for k in _KEYS if k in m} for m in steps[-N_STEPS:]],
        "evals": [{k: v for k, v in m.items() if k == "step" or k.startswith("eval_")}
                  for m in evals[-8:]],
        "samples": list(reversed(samples)),
        "anomalies": scan_anomalies(cfg, metrics, samples, run)[:40],
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
#share button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:1px 8px;font:inherit;cursor:pointer;margin-left:6px}
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
<header><h1>rl-dash</h1><span id="run" class="dim">…</span><span id="age" class="dim"></span><span id="abort" class="warn"></span><span id="share" class="dim"></span></header>
<main>
<div id="anomalies"></div>
<h2>steps</h2><div class="wrap"><table id="steps"></table></div>
<h2>eval</h2><div class="wrap"><table id="evals"></table></div>
<h2>samples — first prompt's whole group, newest step first</h2>
<div id="samples"></div>
</main>
<script>
const KEYS=__KEYS__;
const SHORT={reward_mean:"reward",reward_std:"±",active_groups:"active",mean_len:"len",frac_called:"called",frac_correct:"correct",frac_grounded:"grounded",frac_abstain:"abstain",frac_denial:"denial",frac_no_reply:"noreply",frac_tool_cap:"toolcap",frac_len_capped:"lencap",gen_s:"gen s",update_s:"upd s",gen_tok_s:"tok/s",peak_gb:"peak GB",web_search_live:"srch live",web_search_hit:"srch hit",web_fetch_live:"fetch live",web_fetch_hit:"fetch hit",web_errors:"web err",web_search_real:"real",web_search_fallback_error:"fb err",web_search_fallback_empty:"fb empty",frac_fallback:"ep fallback"};
function esc(s){return (s??"").toString().replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function fmt(k,v){if(v==null)return"";if(typeof v!=="number")return esc(v);if(k==="step"||k==="active_groups"||k.startsWith("web_"))return v;if(k==="mean_len"||k==="gen_tok_s"||k==="gen_s"||k==="update_s")return v.toFixed(0);return v.toFixed(2)}
function table(el,rows,keys){if(!rows.length){el.innerHTML="<tr><td class=dim>nothing yet</td></tr>";return}
 const ks=keys||Object.keys(rows[0]);let h="<tr>"+ks.map(k=>"<th>"+esc((SHORT[k]||k).replace(/^eval_/,""))+"</th>").join("")+"</tr>";
 for(const r of rows.slice().reverse())h+="<tr>"+ks.map(k=>"<td>"+fmt(k,r[k])+"</td>").join("")+"</tr>";el.innerHTML=h}
function rcls(x){return x>0?"pos":(x<0?"neg":"zero")}
function member(c,step,i){const p=c.parts||{};const calls=(c.tool_calls||[]).map(t=>`↳ ${esc((t.args||{}).query||t.name||"?")} → hits ${t.hits??"?"}${t.found_target?" ✓found":""}`).join("<br>");
 const tags=[c.injected?'<span class="tag inj">injected</span>':"",c.finish?`<span class=tag>${esc(c.finish)}</span>`:"",`<span class=tag>${c.len} tok</span>`,p.regime_known?'<span class=tag>known</span>':"",p.regime_post?'<span class=tag>post</span>':"",p.regime_future?'<span class=tag>future</span>':"",p.regime_fictional?'<span class=tag>fictional</span>':"",
  p.answered?`<span class=tag>answer${p.correct?" ✓":" ✗"}</span>`:"",p.abstain?'<span class=tag>abstain</span>':"",p.denial?'<span class=tag>denial</span>':"",p.no_reply?'<span class=tag>no reply</span>':""].join("");
 const txt=c.text||"";const marker="<|im_start|>assistant\n";let vis=txt.includes(marker)?txt.slice(txt.lastIndexOf(marker)+marker.length):txt;vis=vis.replace(/^<think>\n\n<\/think>\n\n/,"").replace(/<\|im_end\|>|<\|endoftext\|>/g,"");
 return `<div class=mem><span class="r ${rcls(c.reward)}">${(c.reward>=0?"+":"")+Number(c.reward).toFixed(2)}</span> ${tags}${calls?`<div class=calls>${calls}</div>`:""}<div class=vis>${esc(vis.slice(0,700))}${vis.length>700?" …":""}</div><details id="d-${step}-${i}"><summary>full transcript</summary><pre class=full>${esc(txt)}</pre></details></div>`}
function samples(list){const el=document.getElementById("samples");if(!list.length){el.innerHTML="<div class=dim>nothing yet</div>";return}
 // Re-render only when the sample set changed, and keep whatever the reader
 // had expanded: the page refreshes itself every 15 s and a collapsing
 // transcript mid-read is the wart this guards against.
 const sig=list.map(s=>s.step+":"+(s.completions||[]).length).join(",");if(el.dataset.sig===sig)return;
 const open=new Set([...el.querySelectorAll("details[open]")].map(d=>d.id));
 el.innerHTML=list.map(s=>{const m=s.meta||{};return `<div class=grp><div class=hd><b>step ${s.step}</b> · ${esc(m.regime||"")} · today ${esc(m.today||"")} · pub ${esc(m.published||"—")} · ${esc(m.qtype||"")}<br><span class=dim>${esc(m.question||JSON.stringify(m).slice(0,200))}</span></div>${(s.completions||[]).map((c,i)=>member(c,s.step,i)).join("")}</div>`}).join("");
 el.dataset.sig=sig;for(const id of open){const d=document.getElementById(id);if(d)d.open=true}}
const B=location.pathname.replace(/\/$/,"");
async function share(h){await fetch(B+"/mirror?"+(h?("on="+h):"off=1"),{method:"POST"});tick()}
function shareBar(s){const el=document.getElementById("share");if(!("private" in s)){el.innerHTML=s.mirror_until?`public mirror · expires ${new Date(s.mirror_until*1000).toLocaleTimeString()}`:"";return}
 const on=s.mirror_until&&s.mirror_until*1000>Date.now();el.innerHTML=(on?`public mirror ON until ${new Date(s.mirror_until*1000).toLocaleTimeString()} <button onclick="share(0)">off</button>`:`public mirror off <button onclick="share(8)">share public 8h</button>`)}
async function tick(){try{const r=await fetch(B+"/api/state",{cache:"no-store"});const s=await r.json();shareBar(s);
 document.getElementById("run").textContent=s.run+"  "+JSON.stringify(s.config);
 const age=s.metrics_mtime?Math.round(s.now-s.metrics_mtime):null;document.getElementById("age").textContent=age==null?"":`last write ${age}s ago`;
 document.getElementById("abort").textContent=s.aborted?("ABORTED: "+s.aborted):"";
 document.getElementById("anomalies").innerHTML=(s.anomalies||[]).map(a=>`<div class="${a.level==="error"?"warn":"dim"}">${esc(a.level)}${a.step!=null?" @"+a.step:""}: ${esc(a.msg)}</div>`).join("");
 table(document.getElementById("steps"),s.steps,KEYS.filter(k=>s.steps.some(r=>k in r)));
 table(document.getElementById("evals"),s.evals);samples(s.samples)}catch(e){document.getElementById("age").textContent="fetch failed"}}
tick();setInterval(tick,15000);
</script></body></html>"""



LABBOOK_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>lab book — rl</title>
<style>
:root{--bg:#0d1117;--fg:#c9d1d9;--dim:#8b949e;--line:#21262d;--card:#161b22;
      --good:#3fb950;--bad:#f85149;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.wrap{max-width:1180px;margin:0 auto;padding:22px 18px 60px}
h1{font-size:19px;margin:0 0 4px} h2{font-size:15px;margin:30px 0 10px;color:var(--fg)}
.q{color:var(--dim);margin:0 0 14px;max-width:74ch}
.notes{background:var(--card);border:1px solid var(--line);border-radius:7px;padding:11px 14px;margin:0 0 8px}
.notes li{color:var(--dim);margin:3px 0} .notes ul{margin:0;padding-left:18px}
.legs{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:13px}
.leg{background:var(--card);border:1px solid var(--line);border-radius:7px;padding:11px 13px}
.leg h3{margin:0 0 2px;font-size:14px;font-weight:600}
.meta{color:var(--dim);font-size:12px;margin-bottom:7px}
.tag{display:inline-block;padding:0 6px;border-radius:9px;font-size:11px;border:1px solid var(--line)}
.running{color:var(--accent);border-color:var(--accent)}.done{color:var(--good);border-color:var(--good)}
.stalled{color:var(--bad);border-color:var(--bad)}
.ended{color:var(--dim);border-color:var(--line)}
table{border-collapse:collapse;width:100%;margin-top:6px}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600;font-size:12px}
td.diag{outline:1px dashed #6e7681;outline-offset:-3px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.scroll{overflow-x:auto}
.foot{color:var(--dim);font-size:12px;margin-top:26px;border-top:1px solid var(--line);padding-top:9px}
svg.panel{display:block;width:100%;height:auto;margin-bottom:2px}
.ptitle{fill:#c9d1d9;font-size:10.5px;font-weight:600}
.ptick{fill:#8b949e;font-size:9px}.pleg{fill:#8b949e;font-size:9px}
</style></head><body><div class="wrap">
<h1 id="title">lab book</h1><p class="q" id="q"></p>
<div class="notes"><ul id="notes"></ul></div>
<div id="idxwrap"><h2>runs on disk <span class="meta" id="idxinfo"></span></h2>
<div class="scroll"><table id="index"></table></div></div>
<h2>training curves <span class="meta" id="curveinfo"></span></h2>
<div class="legs" id="legs"></div>
<h2>judge tokens <span class="meta" id="judgeinfo"></span></h2>
<div class="legs" id="judge"></div>
<h2>results — adapter (row) evaluated on subject (column)</h2>
<div class="scroll"><table id="tbl"></table></div>
<p class="meta">Dashed outline = the subject that adapter was trained on. Every other cell is leave-one-out.</p>
<div class="foot" id="foot"></div>
</div><script>
const F=(v,d=2)=>v==null?"—":(v>=0?"+":"")+v.toFixed(d);
function fmt(v){const a=Math.abs(v);return a>=10?v.toFixed(0):a>=1?v.toFixed(1):v.toFixed(2);}
// One panel with real axes: y ticks at min/mid/max, x ticks at first/last step.
// series = [{pts:[[x,y]...], kind:"dots"|"line", color, label}]
function panel(title,series,opts){
  const W=520,H=138,L=44,R=10,T=38,B=20;       // T leaves a row for title + legend
  const iw=W-L-R, ih=H-T-B;
  const ys=series.flatMap(s=>s.pts.map(p=>p[1]));
  if(!ys.length) return `<div class="meta">${title}: no data yet</div>`;
  let lo=opts&&opts.ymin!=null?opts.ymin:Math.min(...ys);
  let hi=opts&&opts.ymax!=null?opts.ymax:Math.max(...ys);
  if(hi===lo){hi=lo+1;}
  const xs=series.flatMap(s=>s.pts.map(p=>p[0]));
  const x0=Math.min(...xs), x1=Math.max(...xs)||1;
  const X=v=>L+((v-x0)/((x1-x0)||1))*iw, Y=v=>T+ih-((v-lo)/(hi-lo))*ih;
  let g=`<svg viewBox="0 0 ${W} ${H}" class="panel">`;
  g+=`<text x="${L}" y="13" class="ptitle">${title}</text>`;
  // legend
  let lx=L;                                    // legend sits under the title
  for(const s of series){ if(!s.label) continue;
    g+=`<rect x="${lx}" y="19" width="8" height="8" fill="${s.color}"/><text x="${lx+11}" y="26" class="pleg">${s.label}</text>`;
    lx+=s.label.length*6.4+24; }
  // y grid + labels
  for(const t of [lo,(lo+hi)/2,hi]){
    g+=`<line x1="${L}" y1="${Y(t).toFixed(1)}" x2="${W-R}" y2="${Y(t).toFixed(1)}" stroke="#21262d"/>`;
    g+=`<text x="${L-5}" y="${(Y(t)+3.5).toFixed(1)}" class="ptick" text-anchor="end">${fmt(t)}</text>`;}
  if(lo<0&&hi>0) g+=`<line x1="${L}" y1="${Y(0).toFixed(1)}" x2="${W-R}" y2="${Y(0).toFixed(1)}" stroke="#484f58" stroke-dasharray="3 3"/>`;
  // x labels
  const xl=(opts&&opts.xlabel)||"step";
  g+=`<text x="${L}" y="${H-6}" class="ptick">${(opts&&opts.x0label)||x0}</text>`;
  g+=`<text x="${W-R}" y="${H-6}" class="ptick" text-anchor="end">${(opts&&opts.x1label)||x1}</text>`;
  g+=`<text x="${(L+W-R)/2}" y="${H-6}" class="ptick" text-anchor="middle">${xl}</text>`;
  for(const s of series){
    if(!s.pts.length) continue;
    if(s.kind==="dots") g+=s.pts.map(p=>`<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="1.8" fill="${s.color}" fill-opacity="0.75"/>`).join("");
    else{ g+=`<path d="${s.pts.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join("")}" fill="none" stroke="${s.color}" stroke-width="1.6"/>`;
          g+=s.pts.map(p=>`<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2.1" fill="${s.color}"/>`).join(""); }
  }
  return g+"</svg>";
}
function charts(steps,evals){
  if(!steps.length&&!evals.length) return '<div class="meta">no steps yet</div>';
  const S=(k)=>steps.filter(s=>s[k]!=null).map(s=>[s.step,s[k]]);
  const E=(k)=>evals.filter(e=>e[k]!=null).map(e=>[e.step,e[k]]);
  let out="";
  out+=panel("reward",[
    {pts:S("reward"),kind:"dots",color:"#58a6ff",label:"train"},
    {pts:E("eval_reward"),kind:"line",color:"#3fb950",label:"held-out"}]);
  out+=panel("behaviour (held-out, 0–1)",[
    {pts:E("eval_called"),kind:"line",color:"#d29922",label:"calls tool"},
    {pts:E("eval_abstain"),kind:"line",color:"#a371f7",label:"abstains"},
    {pts:E("eval_correct"),kind:"line",color:"#79c0ff",label:"correct"}],{ymin:0,ymax:1});
  // Any eval_<subject>_reward key is a subject this run did NOT train on --
  // scored live via --eval-cells, or retrofitted from checkpoints afterwards.
  const subs=[...new Set(evals.flatMap(e=>Object.keys(e))
    .map(k=>/^eval_([a-z0-9]+)_reward$/.exec(k))
    .filter(Boolean).map(m=>m[1]))].sort();
  if(subs.length){
    const C=["#3fb950","#d29922","#a371f7","#58a6ff"];
    out+=panel("held-out subjects — reward",[
      {pts:E("eval_reward"),kind:"line",color:"#8b949e",label:"trained subject"},
      ...subs.map((s,i)=>({pts:E(`eval_${s}_reward`),kind:"line",color:C[i%C.length],
                           label:s}))]);
    out+=panel("held-out subjects — calls tool",
      subs.map((s,i)=>({pts:E(`eval_${s}_called`),kind:"line",color:C[i%C.length],
                        label:s})),{ymin:0,ymax:1});
  }
  out+=panel("KL from base",[{pts:S("kl"),kind:"line",color:"#f85149",label:"per step"}],{ymin:0});
  out+=panel("gradient norm (clip at 1.0)",[
    {pts:S("grad_norm"),kind:"dots",color:"#d29922",label:"before clipping"},
    {pts:S("grad_norm").map(p=>[p[0],1.0]),kind:"line",color:"#484f58",label:"clip"}],{ymin:0});
  out+=panel("gradient batch",[
    {pts:S("n_seqs"),kind:"dots",color:"#8b949e",label:"sequences in update"}],{ymin:0});
  return out;
}
const $=id=>document.getElementById(id);
const EXP=new URLSearchParams(location.search).get("exp");
const esc=t=>String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
function ago(t){const s=Math.round(Date.now()/1000-t);
  if(s<90)return s+"s ago"; if(s<5400)return Math.round(s/60)+" min ago"; return (s/3600).toFixed(1)+" h ago";}
function legCard(name,leg,spec){
  const st=leg.status||"?";
  const cls={running:"running",done:"done",ended:"ended"}[st]||"stalled";
  const n=(leg.steps||[]).length, last=n?leg.steps[n-1].step:0;
  const subj=(spec.diagonal||{})[name]||"";
  const pace=leg.median_step_s?(leg.median_step_s/60).toFixed(1)+" min/step":"";
  const bits=[subj,`${n} steps (to ${last})`,pace,leg.updated?"updated "+ago(leg.updated):""].filter(Boolean);
  return `<div class="leg"><h3>${esc(name)} <span class="tag ${cls}">${st}</span></h3>
    <div class="meta">${esc(bits.join(" · "))}</div>
    ${charts(leg.steps||[],leg.evals||[])}</div>`;
}
// One line per run, newest first. No grouping, no cap: this is the answer to
// "what has run", and a run missing from it is a bug.
function indexTable(index){
  const when=t=>{const d=new Date(t*1000);
    return d.toLocaleDateString(undefined,{month:"short",day:"numeric"})+" "
      +d.toLocaleTimeString(undefined,{hour:"2-digit",minute:"2-digit"});};
  let h='<tr><th>updated</th><th>run</th><th>subject</th><th>steps</th>'
       +'<th>state</th><th>written up</th></tr>';
  for(const r of index){
    const cls={running:"running",done:"done",ended:"ended"}[r.status]||"stalled";
    const steps=(r.last_step==null?"—":r.last_step)+(r.steps?" / "+r.steps:"");
    h+=`<tr><td>${esc(when(r.ts))}</td><td>${esc(r.name)}</td>`
      +`<td>${esc(r.cell||"—")}</td><td>${esc(steps)}</td>`
      +`<td><span class="tag ${cls}">${esc(r.status)}</span></td>`
      +`<td>${r.filed?"yes":'<span class="meta">no</span>'}</td></tr>`;
  }
  return h;
}
function resultsTable(table,spec){
  const arms=Object.keys(table||{});
  if(!arms.length) return '<tr><td class="meta">no evaluations recorded yet</td></tr>';
  const cells=spec.cells&&spec.cells.length?spec.cells
    :[...new Set(arms.flatMap(a=>Object.keys(table[a])))];
  const diag=spec.diagonal||{};
  let h='<tr><th>adapter</th>'+cells.map(c=>`<th>${esc(c)}</th>`).join("")+'</tr>';
  for(const a of arms){
    h+=`<tr><td>${esc(a)}</td>`+cells.map(c=>{
      const v=table[a][c];
      if(!v||v.reward==null) return '<td>—</td>';
      const on=diag[a]===c?' diag':'';
      const k=v.reward>=0?'pos':'neg';
      return `<td class="${k}${on}">${F(v.reward)}${v.n?` <span class="ptick">n=${v.n}</span>`:""}</td>`;
    }).join("")+'</tr>';
  }
  return h;
}
async function load(){
  try{
    const r=await fetch("/api/labbook"+(EXP?"?exp="+encodeURIComponent(EXP):""),{cache:"no-store"});
    const s=await r.json();
    if(!s.experiment){$("q").textContent="no experiment manifests in runs/experiments/";return;}
    const spec=s.spec||{};
    document.title=(spec.title||"lab book")+" — rl";
    $("title").textContent=spec.title||"lab book";
    $("q").textContent=spec.question||"";
    $("notes").innerHTML=(spec.notes||[]).map(n=>`<li>${esc(n)}</li>`).join("")
      ||'<li>no notes</li>';
    const idx=spec.index||[];
    $("idxwrap").style.display=idx.length?"":"none";
    if(idx.length){
      $("index").innerHTML=indexTable(idx);
      const nlive=idx.filter(r=>r.status==="running").length;
      $("idxinfo").textContent=`${idx.length} on disk`+(nlive?` · ${nlive} running`:"");
    }
    const names=Object.keys(s.legs||{});
    const live=names.filter(n=>s.legs[n].status==="running").length;
    $("curveinfo").textContent=`${names.length} run${names.length===1?"":"s"}`
      +(live?` · ${live} running`:"");
    $("legs").innerHTML=names.map(n=>legCard(n,s.legs[n],spec)).join("")
      ||'<div class="meta">no runs in this manifest</div>';
    $("tbl").innerHTML=resultsTable(s.table,spec);
    judgePanels();
    const links=(s.experiments||[]).map(e=>e===s.experiment?`<b>${esc(e)}</b>`
      :`<a href="?exp=${encodeURIComponent(e)}" style="color:var(--accent)">${esc(e)}</a>`).join(" · ");
    $("foot").innerHTML=`experiment: ${links} · refreshed ${new Date().toLocaleTimeString()}`;
  }catch(e){ $("q").textContent="load failed: "+e; }
}
async function judgePanels(){
  try{
    const r=await fetch("/api/judge_usage",{cache:"no-store"});
    const u=await r.json(), d=u.days||[];
    if(!d.length){$("judge").innerHTML='<div class="meta">no judge calls logged</div>';return;}
    const M=v=>v/1e6, ix=d.map((_,i)=>i);
    const S=k=>d.map((r,i)=>[i,M(r[k])]);
    const first=d[0].day.slice(5), last=d[d.length-1].day.slice(5);
    const ax={xlabel:"day",x0label:first,x1label:last,ymin:0};
    // Billed and local on separate panels: same units, but one is spend and
    // the other is only heat, and a shared axis would hide the handover.
    let h='<div class="leg"><h3>billed (Opus / Sonnet)</h3>'
      +'<div class="meta">cumulative, millions of tokens · cache-weighted cost line is what actually bills</div>'
      +panel("cumulative billed tokens",[
        {pts:S("cum_billed_input"),kind:"line",color:"#58a6ff",label:"input (raw)"},
        {pts:S("cum_billed_equiv"),kind:"line",color:"#d29922",label:"input (cache-weighted)"},
        {pts:S("cum_billed_output"),kind:"line",color:"#f85149",label:"output"}],ax)
      +panel("billed tokens per day",[
        {pts:S("billed_input"),kind:"dots",color:"#58a6ff",label:"input"},
        {pts:S("billed_output"),kind:"dots",color:"#f85149",label:"output"}],ax)
      +'</div>';
    const anyLocal=d.some(r=>r.cum_local_input>0);
    const localCalls=Object.entries(u.per_model||{}).filter(([k])=>k.startsWith("local:"))
      .reduce((a,[,v])=>a+(v.calls||0),0);
    h+='<div class="leg"><h3>local judge (free)</h3>'
      +'<div class="meta">same units, no spend — the work that moved off the API'
      +(anyLocal?'':` · ${localCalls} calls made before token counting was added, so the`
        +' curve starts flat rather than at zero work')+'</div>'
      +panel("cumulative local tokens",[
        {pts:S("cum_local_input"),kind:"line",color:"#3fb950",label:"input"},
        {pts:S("cum_local_output"),kind:"line",color:"#a371f7",label:"output"}],ax)
      +panel("local tokens per day",[
        {pts:S("local_input"),kind:"dots",color:"#3fb950",label:"input"},
        {pts:S("local_output"),kind:"dots",color:"#a371f7",label:"output"}],ax)
      +'</div>';
    $("judge").innerHTML=h;
    const tot=u.totals||{};
    $("judgeinfo").textContent=`billed ${M(tot.billed_equiv||0).toFixed(1)}M in `
      +`(cache-weighted) · ${M(tot.billed_output||0).toFixed(2)}M out · `
      +`${(tot.billed_calls||0)} calls`;
  }catch(e){ $("judge").innerHTML='<div class="meta">usage unavailable: '+e+'</div>'; }
}
load();setInterval(load,20000);
</script></body></html>"""



REVIEW_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rl-dash · review</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#c9d1d9;font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:760px;margin:0 auto;padding:12px}
h1{font-size:15px;color:#58a6ff;margin:0 0 8px}.dim{color:#8b949e;font-size:13px}
.card{border:1px solid #30363d;border-radius:8px;background:#161b22;padding:12px;margin:10px 0}
.q{font-size:16px;color:#e6edf3}.lbl{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin-top:10px}
.reply{white-space:pre-wrap;background:#0d1117;padding:8px;border-radius:6px;font-size:14px}
.calls{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#79c0ff}
.judge{display:inline-block;padding:2px 8px;border-radius:10px;background:#21262d;margin-right:6px}
button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:10px 12px;font:inherit;margin:4px 4px 0 0;cursor:pointer}
button.ok{border-color:#238636}button.bad{border-color:#da3633}
textarea{width:100%;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px;font:inherit;margin-top:6px}
.prog{color:#8b949e;font-size:13px;float:right}
</style></head><body>
<h1>rl-dash · human review <span id="prog" class="prog"></span></h1>
<div class="dim">Blind: which model produced the reply is hidden. You are checking the <b>judge</b> (what did the reply commit to?) and the <b>grade</b> (was it right, given the gold?). Tap one button per item; notes optional.</div>
<div id="card" class="card">loading…</div>
<script>
const B=location.pathname.replace(/\/review\/?$/,"");
let cur=null;
function esc(s){return (s??"").toString().replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
async function next(){const r=await fetch(B+"/api/review/next",{cache:"no-store"});const s=await r.json();document.getElementById("prog").textContent=`${s.done}/${s.total} labelled`;
 if(!s.item){document.getElementById("card").innerHTML="<b>All done — thank you.</b>";return}
 cur=s.item;const it=cur;const calls=(it.tool_calls||[]).map(c=>`↳ ${esc(c.query||"")} → ${c.capped?"cap":("hits "+(c.hits??"?"))}${c.found?" ✓target":""}${c.fallback?" (index)":""}`).join("<br>");
 document.getElementById("card").innerHTML=`
  <div class=q>${esc(it.question)}</div>
  <div class=dim>stated today ${esc(it.today)} · ${esc(it.regime)}${it.turn?" · turn "+(it.turn+1):""} · gold: <b>${esc((it.gold||[]).join(" / ")||"— (no such paper)")}</b>${it.published?" · published "+esc(it.published):""}</div>
  ${calls?`<div class=lbl>tool trace</div><div class=calls>${calls}</div>`:""}
  <div class=lbl>reply the user saw</div><div class=reply>${esc(it.visible||"(no reply)")}</div>
  <div class=lbl>judge said</div><span class=judge>${esc(it.judge_kind)}</span>${it.judge_kind==="answer"?`<span class=judge>${it.correct?"graded correct":"graded wrong"}</span>`:""}<span class=judge>reward ${it.reward}</span>
  <div class=lbl>your call</div>
  <button class=ok onclick="lab('agree')">✓ judge &amp; grade right</button>
  <button class=bad onclick="lab('should_be_answer')">it commits to an answer</button>
  <button class=bad onclick="lab('should_be_abstain')">it declines / hedges</button>
  <button class=bad onclick="lab('should_be_denial')">it asserts non-existence</button>
  <button class=bad onclick="lab('grade_wrong')">kind right, correctness wrong</button>
  <button onclick="lab('skip')">skip</button>
  <textarea id=note rows=2 placeholder="note (optional): e.g. answers about a different paper; hedge is fake; ..."></textarea>`}
async function lab(label){const note=document.getElementById("note").value;await fetch(B+"/api/review/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:cur.id,label,note})});next()}
next();
</script></body></html>"""

REVIEW_Q = ROOT / "runs" / "human-review" / "queue.jsonl"
REVIEW_L = ROOT / "runs" / "human-review" / "labels.jsonl"


def _review_state():
    items = [json.loads(l) for l in REVIEW_Q.read_text().splitlines() if l.strip()] if REVIEW_Q.exists() else []
    done = set()
    if REVIEW_L.exists():
        for l in REVIEW_L.read_text().splitlines():
            if l.strip():
                try:
                    done.add(json.loads(l)["id"])
                except Exception:
                    pass
    nxt = next((it for it in items if it["id"] not in done), None)
    if nxt is not None:
        nxt = {k: v for k, v in nxt.items() if k not in ("arm", "src")}  # blind
    return {"total": len(items), "done": len(done), "item": nxt}


MATRIX_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rl-dash · transfer matrix</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#c9d1d9;font:14px/1.45 -apple-system,system-ui,sans-serif;padding:12px;max-width:1200px;margin:0 auto}
h1{font-size:15px;color:#58a6ff;margin:0 0 6px}.dim{color:#8b949e;font-size:13px}
table{border-collapse:separate;border-spacing:3px;margin-top:10px}
th{color:#8b949e;font-weight:normal;font-size:12px;padding:4px 6px;text-align:left;vertical-align:bottom}
th.rot{writing-mode:vertical-rl;transform:rotate(180deg);height:150px;white-space:nowrap}
td{padding:0;min-width:86px}
.cell{border-radius:6px;padding:6px 8px;text-align:center;font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#e6edf3;position:relative}
.cell small{display:block;font-size:10px;color:rgba(230,237,243,.75)}
.trained{outline:2px dashed #e3b341;outline-offset:-2px}
.arm{font-weight:600;color:#d2a8ff;white-space:nowrap;padding:0 8px}
.legend{margin:10px 0;font-size:12px;color:#8b949e}
.sw{display:inline-block;width:14px;height:12px;border-radius:3px;vertical-align:middle;margin:0 3px 0 10px}
select{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:2px 6px;font:inherit}
</style></head><body>
<h1>transfer matrix <span id="run" class="dim"></span></h1>
<div class="dim">rows = adapters (and stacks), columns = cells (one per domain). Colour = <b>change over the prompt-only base</b> in that cell (green better, red worse); number = reward; small = base. Dashed outline = the adapter was <b>trained</b> on that cell — the diagonal; everything else is transfer. <select id="pick"></select></div>
<div id="grid"></div>
<div class="legend"><span class="sw" style="background:#b62324"></span>−1.5 <span class="sw" style="background:#5a1e1e"></span>−0.5 <span class="sw" style="background:#21262d"></span>0 <span class="sw" style="background:#1f4d2b"></span>+0.5 <span class="sw" style="background:#2ea043"></span>+1.5 &nbsp;·&nbsp; base row shown in grey with the absolute reward</div>
<script>
const B=location.pathname.replace(/\/matrix\/?$/,"");
function esc(s){return (s??"").toString().replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function col(d){if(d==null)return"#21262d";const x=Math.max(-1.5,Math.min(1.5,d))/1.5;const t=Math.abs(x);
 const g=[46,160,67],r=[182,35,36],n=[33,38,45];const c=x>=0?g:r;return `rgb(${c.map((v,i)=>Math.round(n[i]+(v-n[i])*t)).join(",")})`}
function render(res){const cells=res.cells;const arms=Object.keys(res.arms);const trained=res.trained||{};
 let h="<table><tr><th></th>"+cells.map(c=>`<th class=rot>${esc(c)}</th>`).join("")+"</tr>";
 for(const a of arms){h+=`<tr><td class=arm>${esc(a)}</td>`;const base=res.arms.base?res.arms.base.cells:{};
  for(const c of cells){const v=res.arms[a].cells[c];if(!v){h+="<td><div class=cell style='background:#161b22'>—</div></td>";continue}
   const d=(a==="base"||!base[c])?null:v.reward-base[c].reward;const tr=(trained[a]||[]).includes(c);
   const tip=`called ${v.called?.toFixed(2)} correct ${v.correct?.toFixed(2)} abstain ${v.abstain?.toFixed(2)} denial ${v.denial?.toFixed(2)} n=${v.n}`;
   h+=`<td><div class="cell ${tr?"trained":""}" style="background:${a==="base"?"#30363d":col(d)}" title="${esc(tip)}">${v.reward>=0?"+":""}${v.reward.toFixed(2)}${a!=="base"&&base[c]?`<small>Δ ${d>=0?"+":""}${d.toFixed(2)}</small>`:`<small>${a==="base"?"base":""}</small>`}</div></td>`}
  h+="</tr>"}
 document.getElementById("grid").innerHTML=h+"</table>"}
async function load(name){const r=await fetch(B+"/api/matrix"+(name?("?run="+encodeURIComponent(name)):""),{cache:"no-store"});const s=await r.json();
 document.getElementById("run").textContent=s.run?`· ${s.run} (n=${s.results.n}×${s.results.k})`:"(no results yet)";
 const p=document.getElementById("pick");if(p.options.length===0){for(const n of s.runs){const o=document.createElement("option");o.value=n;o.textContent=n;p.appendChild(o)}p.value=s.run;p.onchange=()=>load(p.value)}
 if(s.results)render(s.results)}
load();
</script></body></html>"""

MATRIX_DIR = ROOT / "runs" / "matrix"


def _matrix_state(name: str | None):
    runs = sorted([p.name for p in MATRIX_DIR.iterdir() if (p / "results.json").exists()],
                  reverse=True) if MATRIX_DIR.exists() else []
    if not runs:
        return {"run": None, "runs": [], "results": None}
    pick = name if name in runs else runs[0]
    res = json.loads((MATRIX_DIR / pick / "results.json").read_text())
    legs = MATRIX_DIR / "legs.json"   # {arm: [trained cells]}
    if legs.exists():
        try:
            res["trained"] = json.loads(legs.read_text())
        except Exception:
            pass
    return {"run": pick, "runs": runs, "results": res}

OFF_PAGE = (b"<!doctype html><html><head><meta charset='utf-8'>"
            b"<meta name='viewport' content='width=device-width'><title>rl-dash</title></head>"
            b"<body style='background:#0d1117;color:#8b949e;font:14px ui-monospace,Menlo,monospace;"
            b"text-align:center;padding-top:20vh'>the public mirror is currently off</body></html>")



EXPERIMENTS_DIR = ROOT / "runs" / "experiments"
# metrics.jsonl names, mapped to the short names the page draws. Checked
# against a real file -- guessing these is how the first version drew nothing.
# Only what the labbook page draws (S("...") in its script); "reward" also
# marks a row as a training step.
CURVE_KEYS = {"reward_mean": "reward", "kl": "kl", "n_seqs": "n_seqs",
              "grad_norm": "grad_norm"}


def _leg_curve(run_dir: Path) -> dict:
    """Training curve straight from the run's metrics.jsonl — read on every
    request, so a page open during a run keeps growing with it."""
    m = run_dir / "metrics.jsonl"
    if not m.exists():
        return {"steps": [], "evals": [], "status": "not started"}
    steps, evals, raw_ts = [], [], []
    for line in m.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:      # a half-written final line while training
            continue
        raw_ts.append(d)
        row = {"step": d.get("step")}
        for src, dst in CURVE_KEYS.items():
            if src in d:
                row[dst] = d[src]
        if "reward" in row:          # baseline row carries only eval_* keys
            steps.append(row)
        if any(k.startswith("eval_") for k in d):
            evals.append({"step": d.get("step"),
                          **{k: v for k, v in d.items() if k.startswith("eval_")}})
    last = m.stat().st_mtime
    status, median_gap = _run_status(run_dir, last, [r["ts"] for r in raw_ts if r.get("ts")])
    return {"steps": steps, "evals": evals, "status": status, "updated": last,
            "median_step_s": round(median_gap)}


def _run_status(run_dir: Path, last: float, ts: list[float]) -> tuple[str, float]:
    """done | running | stalled | ended, plus the median step interval.

    "Stalled" has to be judged against THIS run's pace, not a constant: a
    fixed 15-minute window called a healthy run stalled as soon as steps got
    slower (fixing the sign-biased pruning took steps from 9 to 17 minutes,
    because 3.7x more sequences reach the update). Allow three median step
    intervals, floored so a fast run is not marked stalled on one slow step.
    And "stalled" is a call to action, so it has to expire: a run that
    stopped a week ago is history, not a problem; calling seven of those
    stalled buries the one that actually died an hour ago. One rule for the
    index table and the leg cards -- they used to disagree on the same page."""
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
    median_gap = gaps[len(gaps) // 2] if gaps else 0.0
    age = time.time() - last
    if (run_dir / "promoted" / "adapters.safetensors").exists():
        return "done", median_gap
    if age < max(900.0, 3.0 * median_gap):
        return "running", median_gap
    return ("stalled" if age < 86400 else "ended"), median_gap


def _merge_heldout(leg: dict, extra_dir: Path) -> int:
    """Fold a retrofit's held-out scores into a leg's eval rows, by step.

    A run that only scored its own subject can have the unseen subjects added
    afterwards from its checkpoints. Those land in their own directory, so
    they are merged here rather than plotted as a separate run -- they are the
    same policy at the same steps, and belong on the same axes.
    """
    m = extra_dir / "metrics.jsonl"
    if not m.exists():
        return 0
    by_step = {e.get("step"): e for e in leg.get("evals", [])}
    n = 0
    for line in m.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        extra = {k: v for k, v in d.items() if k.startswith("eval_")}
        if not extra:
            continue
        step = d.get("step")
        if step in by_step:
            by_step[step].update(extra)
        else:
            row = {"step": step, **extra}
            by_step[step] = row
            leg.setdefault("evals", []).append(row)
        n += 1
    leg["evals"] = sorted(leg.get("evals", []), key=lambda e: e.get("step") or 0)
    return n


def _judge_usage() -> dict:
    """Daily judge token usage. Reads scripts/judge_usage.py rather than
    re-summing the call logs here: two aggregators over the same files drift,
    and a spend number that disagrees with itself is worse than none."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from judge_usage import collect, series
    except Exception as e:                              # noqa: BLE001
        return {"days": [], "error": f"{type(e).__name__}: {e}"}
    days = series()
    per = collect()["per_model"]
    tot = {"billed_equiv": 0.0, "billed_output": 0, "billed_calls": 0}
    for name, v in per.items():
        if name.startswith("local:"):
            continue
        tot["billed_equiv"] += v.get("billable_equiv", 0)
        tot["billed_output"] += v.get("output", 0)
        tot["billed_calls"] += v.get("calls", 0)
    return {"days": days, "per_model": per, "totals": tot, "now": time.time()}


AUTO_NAME = "all-runs"
AUTO_WINDOW_DAYS = 21     # how far back the auto page looks
AUTO_MAX_LEGS = 20        # and how many it will draw; the drop is reported


def _discover_runs() -> list[dict]:
    """Every training run on disk, newest first — no manifest required.

    The lab book used to draw only runs named in runs/experiments/*.json, so
    a run was invisible until someone remembered to file it, and "invisible"
    looks exactly like "not running". A run dir is anything under runs/ with
    both a config.json and a metrics.jsonl; resume/ subdirs are the trainer's
    own bookkeeping, not runs.
    """
    seen, out = set(), []
    for m in ROOT.joinpath("runs").glob("**/metrics.jsonl"):
        d = m.parent
        if "resume" in d.parts or not (d / "config.json").exists():
            continue
        rel = d.relative_to(ROOT).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            cfg = json.loads((d / "config.json").read_text())
        except (json.JSONDecodeError, OSError):
            cfg = {}
        kw = cfg.get("task_kwargs") or {}
        cell = None
        if kw.get("domain"):
            cell = kw["domain"]
        last_step, stamps = None, []
        for row in _tail_jsonl(m, 12):   # enough to pace the stall window
            if row.get("ts"):
                stamps.append(row["ts"])
            last_step = row.get("step", last_step)
        ts = m.stat().st_mtime
        status, _ = _run_status(d, ts, stamps)
        out.append({"name": d.name, "rel": rel, "ts": ts, "cell": cell,
                    "steps": cfg.get("steps"), "last_step": last_step,
                    "status": status})
    out.sort(key=lambda r: r["ts"], reverse=True)
    return out


def _auto_spec(manifests: list[Path]) -> dict:
    """The manifest nobody has to write.

    Curated manifests still own the narrative -- title, question, the notes
    that say what a number means. This one owns completeness: anything not
    filed anywhere still shows up here, so a run cannot be lost.
    """
    claimed = set()
    for f in manifests:
        try:
            spec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        claimed.update((spec.get("legs") or {}).values())
    runs = _discover_runs()
    cutoff = time.time() - AUTO_WINDOW_DAYS * 86400
    recent = [r for r in runs if r["ts"] >= cutoff]
    shown, dropped_cap = recent[:AUTO_MAX_LEGS], max(0, len(recent) - AUTO_MAX_LEGS)
    notes = [
        "Found on disk, not filed by hand: any directory under runs/ with a "
        "config.json and a metrics.jsonl appears here within seconds of its "
        "first step. Nothing has to be registered for a run to be visible.",
        "Smoke tests, retrofits and abandoned attempts are included on "
        "purpose -- this page answers 'what has run', not 'what counts'. "
        "A run worth reasoning about gets a manifest in runs/experiments/, "
        "which is where the title, the question and the caveats live.",
        f"Showing runs touched in the last {AUTO_WINDOW_DAYS} days: "
        f"{len(recent)} of {len(runs)} on disk.",
    ]
    if dropped_cap:
        notes.append(f"{dropped_cap} more are inside the window but past the "
                     f"{AUTO_MAX_LEGS}-run drawing limit -- widen "
                     f"AUTO_MAX_LEGS or file them in a manifest.")
    unfiled = [r["name"] for r in shown if r["rel"] not in claimed]
    notes.append("Not in any manifest: "
                 + (", ".join(unfiled) if unfiled else "none — all filed."))
    # The index is every run, flat and chronological. The chart section below
    # it is capped because 70 sets of axes is not a page anyone reads -- but
    # the cap must never again decide whether a run is *visible*, only whether
    # it is *drawn*.
    index = [{**r, "filed": r["rel"] in claimed} for r in runs]
    return {"title": "All runs",
            "question": "Every run on disk, newest first, whether or not "
                        "anyone has written it up. The switcher below leads "
                        "to the curated experiments.",
            "index": index,
            "notes": notes,
            "legs": {r["name"]: r["rel"] for r in shown},
            "diagonal": {r["name"]: r["cell"] for r in shown if r["cell"]},
            "cells": sorted({r["cell"] for r in shown if r["cell"]}),
            "eval_dirs": []}


def _labbook_state(name: str | None):
    files = sorted(EXPERIMENTS_DIR.glob("*.json")) if EXPERIMENTS_DIR.exists() else []
    # The auto page is always offered, so an empty runs/experiments/ is no
    # longer an empty lab book.
    names = [f.stem for f in files] + [AUTO_NAME]
    # No ?exp= means "show me what has run", so the flat chronological list
    # wins by default. Guessing which curated manifest was most interesting
    # meant the page you landed on depended on file mtimes.
    pick = name if name in names else AUTO_NAME
    spec = (_auto_spec(files) if pick == AUTO_NAME
            else json.loads((EXPERIMENTS_DIR / f"{pick}.json").read_text()))
    legs = {k: _leg_curve(ROOT / v) for k, v in spec.get("legs", {}).items()}
    for leg_name, d in (spec.get("heldout") or {}).items():
        if leg_name in legs:
            _merge_heldout(legs[leg_name], ROOT / d)
    table = {}
    for d in spec.get("eval_dirs", []):
        for res in sorted((ROOT / d).glob("*/results.json")):
            try:
                r = json.loads(res.read_text())
            except json.JSONDecodeError:
                continue
            for arm, v in r.get("arms", {}).items():
                for cell, agg in v.get("cells", {}).items():
                    table.setdefault(arm, {})[cell] = {
                        "reward": agg.get("reward"), "n": agg.get("n"),
                        "correct": agg.get("correct"), "called": agg.get("called"),
                        "missing": agg.get("missing"), "abstain": agg.get("abstain")}
    return {"experiment": pick, "experiments": names, "spec": spec,
            "legs": legs, "table": table, "now": time.time()}


def _finite(o):
    """Strip NaN/Infinity before serialising.

    JSON has no NaN. Python writes a bare `NaN` token anyway, and the
    browser's JSON.parse rejects the whole document -- so one run with an
    empty evaluation (mean of no samples) blanked every page with
    "load failed: SyntaxError: The string did not match the expected
    pattern." Non-finite becomes null, which the charts already skip.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def _body(obj, **kw) -> bytes:
    return json.dumps(_finite(obj), **kw).encode()


def make_handler(runs_dir: Path, pinned: Path | None, public: bool):
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

        def _path(self) -> str:
            p = self.path.split("?", 1)[0]
            if public:  # cloudflared does not strip the matched prefix
                if p == PUB_PREFIX:
                    p = "/"
                elif p.startswith(PUB_PREFIX + "/"):
                    p = p[len(PUB_PREFIX):]
            return p

        def do_POST(self):
            p = self._path()
            if not public and p == "/api/review/label":
                n = int(self.headers.get("Content-Length", "0") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    rec = {"id": str(body.get("id", ""))[:40], "label": str(body.get("label", ""))[:32],
                           "note": str(body.get("note", ""))[:2000], "t": time.time()}
                except Exception:
                    return self._send(400, b"bad json", "text/plain")
                REVIEW_L.parent.mkdir(parents=True, exist_ok=True)
                with REVIEW_L.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return self._send(200, b"ok", "text/plain")
            if public or p != "/mirror":
                return self._send(404, b"not found", "text/plain")
            q = parse_qs(urlsplit(self.path).query)
            if "off" in q:
                until = set_mirror(None)
            else:
                try:
                    hours = float(q.get("on", ["8"])[0])
                except ValueError:
                    hours = 8.0
                until = set_mirror(min(max(hours, 0.1), MAX_HOURS))
            self._send(200, _body({"until": until}), "application/json")

        def do_GET(self):
            p = self._path()
            if public and p in ("/", "/api/state", "/matrix", "/matrix/", "/api/matrix") and not mirror_on():
                if p == "/":
                    return self._send(200, OFF_PAGE, "text/html; charset=utf-8")
                return self._send(403, b'{"error":"mirror is off"}', "application/json")
            if p == "/":
                return self._send(200, PAGE.replace("__KEYS__", json.dumps(_KEYS)).encode(),
                                  "text/html; charset=utf-8")
            if p in ("/matrix", "/matrix/"):
                return self._send(200, MATRIX_PAGE.encode(), "text/html; charset=utf-8")
            if p == "/api/matrix":
                q = parse_qs(urlsplit(self.path).query)
                return self._send(200, _body(_matrix_state(q.get("run", [None])[0])),
                                  "application/json")
            if not public and p in ("/labbook", "/labbook/"):
                return self._send(200, LABBOOK_PAGE.encode(), "text/html; charset=utf-8")
            if not public and p == "/api/judge_usage":
                return self._send(200, _body(_judge_usage()),
                                  "application/json")
            if not public and p == "/api/labbook":
                q = parse_qs(urlsplit(self.path).query)
                return self._send(200, _body(_labbook_state(q.get("exp", [None])[0])),
                                  "application/json")
            if not public and p in ("/review", "/review/"):
                return self._send(200, REVIEW_PAGE.encode(), "text/html; charset=utf-8")
            if not public and p == "/api/review/next":
                return self._send(200, _body(_review_state(), ensure_ascii=False),
                                  "application/json")
            if p == "/api/state":
                run = pinned or _newest_run(runs_dir)
                st = (_state(run) if run is not None else
                      {"run": None, "steps": [], "evals": [], "samples": [], "now": time.time()})
                st["mirror_until"] = mirror_until()
                if not public:
                    st["private"] = True
                return self._send(200, _body(st), "application/json")
            if p == "/health":
                return self._send(200, b"ok", "text/plain")
            self._send(404, b"not found", "text/plain")
    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8105)
    ap.add_argument("--public-port", type=int, default=8106,
                    help="timer-gated public mirror (0 = none)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--run", default=None, help="pin one run dir (default: newest)")
    a = ap.parse_args()
    runs, pinned = Path(a.runs), (Path(a.run) if a.run else None)
    if a.public_port:
        pub = ThreadingHTTPServer((a.host, a.public_port), make_handler(runs, pinned, True))
        threading.Thread(target=pub.serve_forever, daemon=True).start()
        print(f"rl-dash public mirror on http://{a.host}:{a.public_port} (gated)", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(runs, pinned, False))
    print(f"rl-dash on http://{a.host}:{a.port} runs={a.runs}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
