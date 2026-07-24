"""SAGE-decode serving host — the oracle's decoder as an inference backend.

Serves qwen36 (MLX 4-bit weights) with SAGE confidence-guided decoding
(arXiv 2602.08354, see docs/sage-paper-notes.md) on every request: step-wise
beam over the think phase, Φ ranking, top-h </think> acceptance, then ordinary
sampling for the visible answer. The oracle measured 0.806 vs 0.681 greedy on
the held-out mix at ~7x decode cost — this makes that trade available to any
client of the endpoint.

OpenAI /v1/chat/completions only. SAGE is a beam — there is no incremental
token stream; `stream: true` is honored as a single terminal SSE chunk after
the beam finishes (clients with idle timeouts beware: a request takes
~1-3 min).

Request extensions:
  "sage":    {"m": 2, "tr": 0.5, "max_steps": 64, "max_step_tokens": 256,
              "answer_reserve": 256}          — per-request overrides
  "adapter": "<run-name>"                     — mlx-rl LoRA from --adapters-dir
              (e.g. "my-sage-run"); null/absent = base model

Single-flight: one generation at a time (beam owns the GPU); requests queue
on the lock. All mx work runs on ONE worker thread (MLX streams are
per-thread state — the 0.31.x lesson).

Run:  .venv/bin/python scripts/sage_server.py [--port 8080]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULTS = {"m": 2, "tr": 0.5, "max_steps": 64, "max_step_tokens": 256,
            "answer_reserve": 256}


class SageServer:
    def __init__(self, profile: str, served_name: str,
                 adapters_dir: str | None):
        from mlx_lm import load as mlx_load

        from mlx_rl.profiles import get_profile
        from mlx_rl.train import _step_delim_ids

        self.prof = get_profile(profile)
        self.served_name = served_name
        self.model, self.tokenizer = mlx_load(self.prof.model)
        self.chat_kwargs = {**self.prof.chat_kwargs,
                            **self.prof.think_chat_kwargs}
        self.eos = set(getattr(self.tokenizer, "eos_token_ids", None)
                       or [self.tokenizer.eos_token_id])
        self.eos |= set(self.prof.extra_eos)
        self.step_delim = _step_delim_ids(self.tokenizer)
        self.adapters_dir = Path(adapters_dir) if adapters_dir else None
        self.adapter: str | None = None
        self.lock = threading.Lock()
        self.worker = ThreadPoolExecutor(max_workers=1,
                                         thread_name_prefix="mx-worker")

    # ---------------------------------------------------------- adapters
    def _set_adapter(self, name: str | None) -> None:
        """Swap the mlx-rl LoRA adapter (worker thread only). mlx-rl format:
        geometry in config.json + stepped adapter-*.safetensors."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.tuner.lora import LoRALinear

        if name == self.adapter:
            return

        def walk(container, items):
            for key, child in items:
                if isinstance(child, LoRALinear):
                    yield container, key, child
                elif isinstance(child, nn.Module):
                    yield from walk(child, child.children().items())
                elif isinstance(child, dict):
                    yield from walk(child, child.items())
                elif isinstance(child, list):
                    yield from walk(child, enumerate(child))

        for parent, key, lora in list(
                walk(self.model, self.model.children().items())):
            parent[key] = lora.linear
        self.adapter = None
        if name is None:
            return
        if self.adapters_dir is None:
            raise ValueError("no --adapters-dir configured")
        # runs/<name>/adapters/{adapter_config.json, adapter-XXXXX.safetensors}
        path = self.adapters_dir / name
        if (path / "adapters").is_dir():
            path = path / "adapters"
        cfg_path = path / "adapter_config.json"
        if not cfg_path.exists():
            raise ValueError(f"unknown adapter {name!r}")
        cfg = json.loads(cfg_path.read_text())
        snaps = sorted(path.glob("adapter-*.safetensors"))
        if not snaps:
            raise ValueError(f"adapter {name!r} has no safetensors")
        from mlx_lm.tuner.utils import linear_to_lora_layers
        linear_to_lora_layers(self.model, int(cfg["num_layers"]), {
            "rank": cfg["rank"], "scale": cfg["scale"], "dropout": 0.0,
            "keys": cfg["keys"],
        })
        self.model.load_weights(str(snaps[-1]), strict=False)
        mx.eval(self.model.parameters())
        self.adapter = name

    # ---------------------------------------------------------- generate
    def complete(self, req: dict) -> dict:
        """One SAGE completion (runs under self.lock, executes on worker)."""
        def job():
            from mlx_rl.engine import sage_completion
            from mlx_rl.rollout import encode_prompt
            from mlx_rl.train import _completion_text

            self._set_adapter(req.get("adapter"))
            sage = {**DEFAULTS, **(req.get("sage") or {})}
            budget = int(req.get("max_tokens") or 4096)
            prompt = encode_prompt(self.tokenizer, req.get("messages", []),
                                   **self.chat_kwargs)
            comp = sage_completion(
                self.model, list(prompt), self.prof.think_end,
                eos=self.eos, step_delim=self.step_delim,
                m=int(sage["m"]), tr=float(sage["tr"]),
                max_new_tokens=budget,
                max_reasoning_steps=int(sage["max_steps"]),
                max_step_tokens=int(sage["max_step_tokens"]),
                think_temperature=1.0,
                answer_temperature=float(req.get("temperature") or 1.0),
                answer_reserve=int(sage["answer_reserve"]),
            )
            text = _completion_text(self.tokenizer, comp)
            import mlx.core as mx
            mx.clear_cache()  # beam KV is dead weight between requests
            return {
                "text": text,
                "finish": "length" if comp.finish_reason == "length" else "stop",
                "prompt_tokens": len(prompt),
                "completion_tokens": len(comp.tokens),
                "think_len": comp.think_len,
            }

        return self.worker.submit(job).result()


def make_handler(srv: SageServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"model": srv.served_name,
                                 "adapter": srv.adapter, "decode": "sage",
                                 **DEFAULTS})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": srv.served_name, "object": "model",
                     "owned_by": "mlx-rl"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(
                    int(self.headers["Content-Length"])))
            except Exception:
                return self._json(400, {"error": "bad json"})
            with srv.lock:
                try:
                    out = srv.complete(req)
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                except Exception as e:  # noqa: BLE001 — surface, don't wedge
                    return self._json(500, {"error": repr(e)})
            rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            msg = {"role": "assistant", "content": out["text"]}
            choice = {"index": 0, "message": msg,
                      "finish_reason": out["finish"],
                      "sage": {"think_len": out["think_len"],
                               "adapter": srv.adapter}}
            resp = {"id": rid, "object": "chat.completion",
                    "created": int(time.time()), "model": srv.served_name,
                    "choices": [choice],
                    "usage": {"prompt_tokens": out["prompt_tokens"],
                              "completion_tokens": out["completion_tokens"],
                              "total_tokens": out["prompt_tokens"]
                                              + out["completion_tokens"]}}
            if req.get("stream"):
                # no incremental stream from a beam: one terminal chunk
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunk = {"id": rid, "object": "chat.completion.chunk",
                         "created": resp["created"],
                         "model": srv.served_name,
                         "choices": [{"index": 0,
                                      "delta": {"role": "assistant",
                                                "content": out["text"]},
                                      "finish_reason": out["finish"]}],
                         "usage": resp["usage"]}
                try:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                except BrokenPipeError:
                    pass
                return
            self._json(200, resp)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--served-name", default="qwen36-sagedecode")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--adapters-dir",
                    default=str(Path(__file__).resolve().parent.parent / "runs"),
                    help="mlx-rl runs/ dir: request ext \"adapter\": "
                         "\"<run-name>\" serves that run's latest LoRA")
    args = ap.parse_args()
    srv = SageServer(args.profile, args.served_name, args.adapters_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(srv))
    print(f"sage_server: {args.served_name} on :{args.port} "
          f"(profile {args.profile}, adapters {args.adapters_dir})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
