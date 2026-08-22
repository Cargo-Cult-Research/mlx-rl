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
outcome view next to it. Nothing here writes anything.

Run:  python3 scripts/rl_dash.py [--port 8105] [--public-port 8106] [--runs runs] [--run runs/<dir>]
"""
from __future__ import annotations

import argparse
import json
import threading
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
<h2>steps</h2><div class="wrap"><table id="steps"></table></div>
<h2>eval</h2><div class="wrap"><table id="evals"></table></div>
<h2>samples — first prompt's whole group, newest step first</h2>
<div id="samples"></div>
</main>
<script>
const KEYS=["step","reward_mean","reward_std","active_groups","mean_len","frac_called","frac_correct","frac_grounded","frac_abstain","frac_denial","frac_no_reply","frac_tool_cap","frac_len_capped","kl","gen_s","update_s","gen_tok_s","peak_gb","web_search_live","web_search_hit","web_fetch_live","web_fetch_hit","web_errors","web_search_real","web_search_fallback_error","web_search_fallback_empty","frac_fallback"];
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
 table(document.getElementById("steps"),s.steps,KEYS.filter(k=>s.steps.some(r=>k in r)));
 table(document.getElementById("evals"),s.evals);samples(s.samples)}catch(e){document.getElementById("age").textContent="fetch failed"}}
tick();setInterval(tick,15000);
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
<div class="dim">rows = adapters (and stacks), columns = cells (domain:situation). Colour = <b>change over the prompt-only base</b> in that cell (green better, red worse); number = reward; small = base. Dashed outline = the adapter was <b>trained</b> on that cell — the diagonal; everything else is transfer. <select id="pick"></select></div>
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
   const tip=`called ${v.called?.toFixed(2)} correct ${v.correct?.toFixed(2)} abstain ${v.abstain?.toFixed(2)} denial ${v.denial?.toFixed(2)} fab-prov ${v.fabricated_provenance?.toFixed(2)} n=${v.n}`;
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
CURVE_KEYS = {"reward_mean": "reward", "reward_std": "reward_std",
              "frac_correct": "correct", "frac_called": "called",
              "frac_abstain": "abstain", "mean_len": "mean_len",
              "active_groups": "active_groups",
              "groups_skipped_stage1": "skipped", "kl": "kl",
              "gen_tok_s": "gen_tok_s", "peak_gb": "peak_gb"}


def _leg_curve(run_dir: Path) -> dict:
    """Training curve straight from the run's metrics.jsonl — read on every
    request, so a page open during a run keeps growing with it."""
    m = run_dir / "metrics.jsonl"
    if not m.exists():
        return {"steps": [], "evals": [], "status": "not started"}
    steps, evals = [], []
    for line in m.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:      # a half-written final line while training
            continue
        row = {"step": d.get("step")}
        for src, dst in CURVE_KEYS.items():
            if src in d:
                row[dst] = d[src]
        if "reward" in row:          # baseline row carries only eval_* keys
            steps.append(row)
        if any(k.startswith("eval_") for k in d):
            evals.append({"step": d.get("step"),
                          **{k: v for k, v in d.items() if k.startswith("eval_")}})
    done = (run_dir / "promoted" / "adapters.safetensors").exists()
    last = m.stat().st_mtime
    status = "done" if done else ("running" if time.time() - last < 900 else "stalled")
    return {"steps": steps, "evals": evals, "status": status, "updated": last}


def _labbook_state(name: str | None):
    files = sorted(EXPERIMENTS_DIR.glob("*.json")) if EXPERIMENTS_DIR.exists() else []
    names = [f.stem for f in files]
    if not names:
        return {"experiment": None, "experiments": []}
    pick = name if name in names else names[0]
    spec = json.loads((EXPERIMENTS_DIR / f"{pick}.json").read_text())
    legs = {k: _leg_curve(ROOT / v) for k, v in spec.get("legs", {}).items()}
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
.legs{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:13px}
.leg{background:var(--card);border:1px solid var(--line);border-radius:7px;padding:11px 13px}
.leg h3{margin:0 0 2px;font-size:14px;font-weight:600}
.meta{color:var(--dim);font-size:12px;margin-bottom:7px}
.tag{display:inline-block;padding:0 6px;border-radius:9px;font-size:11px;border:1px solid var(--line)}
.running{color:var(--accent);border-color:var(--accent)}.done{color:var(--good);border-color:var(--good)}
.stalled{color:var(--bad);border-color:var(--bad)}
table{border-collapse:collapse;width:100%;margin-top:6px}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600;font-size:12px}
td.diag{outline:1px dashed #6e7681;outline-offset:-3px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.scroll{overflow-x:auto}
.foot{color:var(--dim);font-size:12px;margin-top:26px;border-top:1px solid var(--line);padding-top:9px}
svg{display:block;width:100%;height:82px}
</style></head><body><div class="wrap">
<h1 id="title">lab book</h1><p class="q" id="q"></p>
<div class="notes"><ul id="notes"></ul></div>
<h2>training curves <span class="meta" id="curveinfo"></span></h2>
<div class="legs" id="legs"></div>
<h2>results — adapter (row) evaluated on subject (column)</h2>
<div class="scroll"><table id="tbl"></table></div>
<p class="meta">Dashed outline = the subject that adapter was trained on. Every other cell is leave-one-out.</p>
<div class="foot" id="foot"></div>
</div><script>
const F=(v,d=2)=>v==null?"—":(v>=0?"+":"")+v.toFixed(d);
function path(pts,w,h,lo,hi){
  if(!pts.length) return "";
  const sx=pts.length>1?w/(pts.length-1):0, r=(hi-lo)||1;
  return pts.map((v,i)=>(i?"L":"M")+(i*sx).toFixed(1)+","+(h-((v-lo)/r)*h).toFixed(1)).join("");
}
function curve(steps,evals){
  // Training reward is per-step and noisy: draw it as points, because a
  // connecting line invents a trend between samples that are independent
  // draws. The held-out eval is smooth and sparse, so it keeps its line.
  const w=320,h=82, r=steps.map(s=>s.reward).filter(v=>v!=null);
  if(!r.length) return '<div class="meta">no steps yet</div>';
  const ev=evals.map(e=>e.eval_reward).filter(v=>v!=null);
  const all=r.concat(ev), lo=Math.min(...all), hi=Math.max(...all), rng=(hi-lo)||1;
  const zero=h-((0-lo)/rng)*h;
  const sx=r.length>1?w/(r.length-1):0;
  let g=`<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">`;
  if(zero>=0&&zero<=h) g+=`<line x1="0" y1="${zero.toFixed(1)}" x2="${w}" y2="${zero.toFixed(1)}" stroke="#30363d" stroke-dasharray="3 3"/>`;
  g+=r.map((v,i)=>`<circle cx="${(i*sx).toFixed(1)}" cy="${(h-((v-lo)/rng)*h).toFixed(1)}" r="1.9" fill="#58a6ff" fill-opacity="0.8"/>`).join("");
  if(ev.length>1) g+=`<path d="${path(ev,w,h,lo,hi)}" fill="none" stroke="#3fb950" stroke-width="1.5"/>`;
  else if(ev.length===1) g+=`<circle cx="0" cy="${(h-((ev[0]-lo)/rng)*h).toFixed(1)}" r="2.4" fill="#3fb950"/>`;
  return g+"</svg>";
}
async function tick(){
  const s=await (await fetch("/api/labbook")).json();
  if(!s.experiment){document.getElementById("foot").textContent="no experiments defined";return;}
  document.getElementById("title").textContent=s.spec.title||s.experiment;
  document.getElementById("q").textContent=s.spec.question||"";
  document.getElementById("notes").innerHTML=(s.spec.notes||[]).map(n=>`<li>${n}</li>`).join("");
  let tot=0;
  document.getElementById("legs").innerHTML=Object.entries(s.legs).map(([n,l])=>{
    const last=l.steps.length?l.steps[l.steps.length-1]:null; tot+=l.steps.length;
    const ev=l.evals.length?l.evals[l.evals.length-1]:null;
    return `<div class="leg"><h3>${n} <span class="tag ${l.status}">${l.status}</span></h3>
      <div class="meta">step ${last?last.step:0}${last&&last.reward!=null?" · reward "+F(last.reward):""}
      ${ev&&ev.eval_reward!=null?" · eval "+F(ev.eval_reward):""}</div>${curve(l.steps,l.evals)}
      <div class="meta">blue dots = train reward per step · green = held-out eval</div></div>`;}).join("");
  document.getElementById("curveinfo").textContent=`${tot} steps logged across ${Object.keys(s.legs).length} runs`;
  const cells=s.spec.cells||[], diag=s.spec.diagonal||{};
  const arms=Object.keys(s.table).sort((a,b)=>a==="base"?-1:b==="base"?1:a.localeCompare(b));
  let t=`<tr><th>adapter</th>${cells.map(c=>`<th>${c.split(":")[0]}</th>`).join("")}</tr>`;
  for(const a of arms){
    t+=`<tr><td>${a}</td>`+cells.map(c=>{
      const v=s.table[a][c];
      const cls=(diag[a]===c?"diag ":"")+(v?(v.reward>=0?"pos":"neg"):"");
      return `<td class="${cls}">${v?F(v.reward):"—"}</td>`;}).join("")+"</tr>";
  }
  document.getElementById("tbl").innerHTML=t;
  document.getElementById("foot").textContent="pulled live from metrics.jsonl and results.json · "+new Date().toLocaleTimeString();
}
tick(); setInterval(tick,10000);
</script></body></html>"""

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
            self._send(200, json.dumps({"until": until}).encode(), "application/json")

        def do_GET(self):
            p = self._path()
            if public and p in ("/", "/api/state", "/matrix", "/matrix/", "/api/matrix") and not mirror_on():
                if p == "/":
                    return self._send(200, OFF_PAGE, "text/html; charset=utf-8")
                return self._send(403, b'{"error":"mirror is off"}', "application/json")
            if p == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if p in ("/matrix", "/matrix/"):
                return self._send(200, MATRIX_PAGE.encode(), "text/html; charset=utf-8")
            if p == "/api/matrix":
                q = parse_qs(urlsplit(self.path).query)
                return self._send(200, json.dumps(_matrix_state(q.get("run", [None])[0])).encode(),
                                  "application/json")
            if not public and p in ("/labbook", "/labbook/"):
                return self._send(200, LABBOOK_PAGE.encode(), "text/html; charset=utf-8")
            if not public and p == "/api/labbook":
                q = parse_qs(urlsplit(self.path).query)
                return self._send(200, json.dumps(_labbook_state(q.get("exp", [None])[0])).encode(),
                                  "application/json")
            if not public and p in ("/review", "/review/"):
                return self._send(200, REVIEW_PAGE.encode(), "text/html; charset=utf-8")
            if not public and p == "/api/review/next":
                return self._send(200, json.dumps(_review_state(), ensure_ascii=False).encode(),
                                  "application/json")
            if p == "/api/state":
                run = pinned or _newest_run(runs_dir)
                st = (_state(run) if run is not None else
                      {"run": None, "steps": [], "evals": [], "samples": [], "now": time.time()})
                st["mirror_until"] = mirror_until()
                if not public:
                    st["private"] = True
                return self._send(200, json.dumps(st).encode(), "application/json")
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
