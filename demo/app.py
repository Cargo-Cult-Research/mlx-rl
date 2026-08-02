#!/usr/bin/env python3
# lifecycle: core
"""Public side-by-side demo: base Qwen3.6-35B vs the qa-gloveC-200 RL pair.

Serves one page (demo/page.html) plus a "duel" endpoint that runs the SAME
prompt twice against the :8084 lens host — arm A the pristine base weights,
arm B the calibrated-honesty LoRA with its trained system prompt (the glove)
— and streams both completions back over one SSE response. The lens host
swaps LoRA per request, so both arms share one resident model.

Public boundary (exposed at rl.strawrunway.com via the strawrunway tunnel),
so the guardrails live here:
  * prompt length cap, fixed max_tokens/temperature (no client knobs)
  * per-IP token bucket + one duel in flight globally
  * adapter list is hardcoded — no pass-through
  * visitor feedback appends to demo/flags.jsonl (gitignored), size-capped

If the resting backend ever injects the glove at the serving proxy
(lens+c200 default), point --upstream at the INNER server port instead of
:8084 so arm A stays glove-free — the proxy would otherwise glove both arms.

Run:  python3 demo/app.py [--port 8092] [--upstream http://127.0.0.1:8084]
"""
from __future__ import annotations

import argparse
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "page.html"
FLAGS = ROOT / "flags.jsonl"
GLOVE_PATH = Path.home() / "models/adapters/qa-gloveC-200-20260731/GLOVE.txt"
ADAPTER = "c200"

MAX_PROMPT_CHARS = 400
MAX_TOKENS = 192          # the training horizon
TEMPERATURE = 0.7
RATE_PER_MIN = 4
RATE_BURST = 2
FLAGS_MAX_BYTES = 5 * 2**20

_duel_lock = threading.Semaphore(1)
_buckets: dict[str, list[float]] = {}
_buckets_lock = threading.Lock()


def _rate_ok(ip: str) -> bool:
    now = time.monotonic()
    with _buckets_lock:
        tokens, last = _buckets.get(ip, [float(RATE_BURST), now])
        tokens = min(RATE_BURST, tokens + (now - last) * RATE_PER_MIN / 60.0)
        if tokens < 1.0:
            _buckets[ip] = [tokens, now]
            return False
        _buckets[ip] = [tokens - 1.0, now]
        return True


def make_handler(upstream: str):
    up = urlparse(upstream)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _client_ip(self) -> str:
            xff = self.headers.get("X-Forwarded-For")
            return xff.split(",")[0].strip() if xff \
                else self.client_address[0]

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ------------------------------------------------------------ GET
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/status":
                self._json(200, self._status())
            else:
                self._json(404, {"error": "not found"})

        def _status(self) -> dict:
            """ready iff the lens host answers AND serves the c200 adapter
            AND the glove file is readable. Anything else -> the page shows
            its offline banner (the model slot is busy with research)."""
            if not GLOVE_PATH.exists():
                return {"ready": False, "reason": "glove file missing"}
            try:
                conn = http.client.HTTPConnection(up.hostname, up.port,
                                                  timeout=3)
                conn.request("GET", "/health")
                r = conn.getresponse()
                h = json.loads(r.read()) if r.status == 200 else {}
                conn.close()
            except Exception:
                return {"ready": False, "reason": "model slot offline"}
            if ADAPTER not in (h.get("available_adapters") or []):
                return {"ready": False,
                        "reason": "a different model holds the slot"}
            return {"ready": True, "model": h.get("model")}

        # ------------------------------------------------------------ POST
        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/flag":
                return self._do_flag()
            if path != "/api/duel":
                return self._json(404, {"error": "not found"})
            ip = self._client_ip()
            if not _rate_ok(ip):
                return self._json(429, {"error":
                    "rate limited — wait a few seconds"})
            try:
                req = json.loads(self.rfile.read(
                    min(int(self.headers.get("Content-Length", 0)), 8192)))
            except Exception:
                return self._json(400, {"error": "bad json"})
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                return self._json(400, {"error": "empty prompt"})
            if len(prompt) > MAX_PROMPT_CHARS:
                return self._json(400, {"error":
                    f"prompt too long (max {MAX_PROMPT_CHARS} chars)"})
            if not self._status().get("ready"):
                return self._json(503, {"error": "demo offline"})
            if not _duel_lock.acquire(blocking=False):
                return self._json(503, {"error":
                    "a duel is already running — try again in ~30s"})
            try:
                self._duel(prompt)
            finally:
                _duel_lock.release()

        def _duel(self, prompt: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            glove = GLOVE_PATH.read_text().strip()
            arms = [("base", None, None), ("rl", glove, ADAPTER)]
            try:
                for arm, system, adapter in arms:
                    self._emit({"arm": arm, "start": True})
                    ok = self._stream_arm(arm, prompt, system, adapter)
                    self._emit({"arm": arm, "done": True, "ok": ok})
                self._emit({"all_done": True})
            except BrokenPipeError:
                pass  # visitor left; upstream sees the close

        def _emit(self, obj: dict) -> None:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        def _stream_arm(self, arm: str, prompt: str,
                        system: str | None, adapter: str | None) -> bool:
            msgs = ([{"role": "system", "content": system}] if system else []) \
                + [{"role": "user", "content": prompt}]
            payload = {
                "model": "qwen36-lens", "messages": msgs,
                "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                "stream": True, "enable_thinking": False, "adapter": adapter,
            }
            try:
                conn = http.client.HTTPConnection(up.hostname, up.port,
                                                  timeout=600)
                conn.request("POST", "/v1/chat/completions",
                             body=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
                r = conn.getresponse()
                if r.status != 200:
                    r.read()
                    conn.close()
                    self._emit({"arm": arm, "error": f"backend {r.status}"})
                    return False
                while True:
                    line = r.readline()
                    if not line:
                        break
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"]
                    except Exception:
                        continue
                    text = delta.get("content")
                    if text:
                        self._emit({"arm": arm, "delta": text})
                conn.close()
                return True
            except BrokenPipeError:
                raise
            except Exception:
                self._emit({"arm": arm, "error": "backend unreachable"})
                return False

        def _do_flag(self) -> None:
            """Visitor feedback — the edge-case harvest. Appends one line;
            contents are untrusted visitor data, size-capped, never executed."""
            ip = self._client_ip()
            if not _rate_ok(ip):
                return self._json(429, {"error": "rate limited"})
            try:
                req = json.loads(self.rfile.read(
                    min(int(self.headers.get("Content-Length", 0)), 16384)))
            except Exception:
                return self._json(400, {"error": "bad json"})
            verdict = str(req.get("verdict", ""))[:24]
            if verdict not in ("rl_wrong", "base_better", "both_wrong",
                               "interesting"):
                return self._json(400, {"error": "bad verdict"})
            if FLAGS.exists() and FLAGS.stat().st_size > FLAGS_MAX_BYTES:
                return self._json(503, {"error": "flag store full"})
            with FLAGS.open("a") as fh:
                fh.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "verdict": verdict,
                    "prompt": str(req.get("prompt", ""))[:MAX_PROMPT_CHARS],
                    "base": str(req.get("base", ""))[:2000],
                    "rl": str(req.get("rl", ""))[:2000],
                }) + "\n")
            return self._json(200, {"ok": True})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--upstream", default="http://127.0.0.1:8084")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port),
                                make_handler(args.upstream))
    print(f"mlx-rl demo: 127.0.0.1:{args.port} -> {args.upstream}",
          flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
