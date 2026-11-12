"""goethe-engine serve: the OpenAI-compatible upstream goethed proxies to (stdlib http.server, one engine thread).

Routes: POST /v1/chat/completions, POST /v1/completions (JSON or SSE), GET /v1/models, GET /health.
Requests from many handler threads join one Engine; the engine thread steps it (continuous batching) and fans the
token deltas out to per-request queues. Localhost by default."""
import argparse
import hmac
import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .tokenizer import ByteTokenizer, HFTokenizer, TextStream

MAX_BODY = 8 << 20
UNSUPPORTED = ("tools", "tool_choice", "functions", "function_call", "logprobs", "top_logprobs", "logit_bias",
               "suffix", "best_of", "echo")


class ApiError(Exception):
    def __init__(self, status, msg, kind="invalid_request_error"):
        super().__init__(msg)
        self.status, self.kind = status, kind


class Choice:
    """One output sequence: detokenizer and text-level stop strings (with the holdback a split stop string needs)."""

    def __init__(self, tok, stops):
        self.ts, self.stops, self.held, self.text, self.stopped, self.finish = TextStream(tok), stops, "", "", False, None
        self.ntok = 0

    def feed(self, ids, final=False):
        """Text that is safe to send now; sets stopped when a stop string completes (text before it is kept)."""
        if self.stopped:
            return ""
        self.ntok += len(ids)
        self.held += self.ts.feed(ids) + (self.ts.flush() if final else "")
        cut = min((i for i in (self.held.find(s) for s in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            out, self.held, self.stopped, self.finish = self.held[:cut], "", True, "stop"
        else:
            keep = 0 if final else max((len(s) - 1 for s in self.stops), default=0)
            out, self.held = self.held[:len(self.held) - keep], self.held[len(self.held) - keep:]
        self.text += out
        return out


class Server:
    def __init__(self, engine, tok, name, ctx, api_key=None):
        self.engine, self.tok, self.name, self.ctx, self.key = engine, tok, name, ctx, api_key
        self.pending, self.wake, self.stop, self.dead = {}, threading.Event(), False, None
        self.created = int(time.time())
        self.thread = threading.Thread(target=self._loop, daemon=True, name="engine")

    def _loop(self):
        try:
            while not self.stop:
                if not self.engine.has_work():
                    self.wake.wait(0.05)
                    self.wake.clear()
                    continue
                for o in self.engine.step():
                    q = self.pending.get(o.rid)
                    if q is not None:
                        q.put(o)
        except BaseException as e:   # the engine is gone: every waiting request fails, /health turns 503
            self.dead = f"{type(e).__name__}: {e}"
            for q in list(self.pending.values()):
                q.put(None)

    def submit(self, prompt, params):
        rid = uuid.uuid4().hex
        q = self.pending[rid] = queue.Queue()
        try:
            self.engine.add_request(rid, prompt, params)
        except ValueError as e:
            del self.pending[rid]
            raise ApiError(400, str(e))
        self.wake.set()
        return rid, q

    def cancel(self, rid):
        self.pending.pop(rid, None)
        self.engine.abort(rid)
        self.wake.set()


def parse_stop(v):
    if v is None:
        return ()
    v = [v] if isinstance(v, str) else v
    if not isinstance(v, list) or len(v) > 4 or not all(isinstance(s, str) and s for s in v):
        raise ApiError(400, "stop must be a non-empty string or up to 4 of them")
    return tuple(v)


def sampling(body, srv, plen):
    from .sampling import SamplingParams
    for k in UNSUPPORTED:
        if body.get(k) not in (None, False):
            raise ApiError(400, f"{k} is not supported")
    for k in ("frequency_penalty", "presence_penalty"):
        if body.get(k) not in (None, 0, 0.0):
            raise ApiError(400, f"{k} is not supported")
    if (body.get("response_format") or {}).get("type", "text") != "text":
        raise ApiError(400, "response_format is not supported")
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    num = lambda k, d, t: d if body.get(k) is None else t(body[k])
    try:
        return SamplingParams(temperature=num("temperature", 1.0, float), top_p=num("top_p", 1.0, float),
                              top_k=num("top_k", 0, int), seed=num("seed", None, int),
                              max_tokens=num("max_tokens", srv.ctx, int) if mt is None else int(mt),
                              n=num("n", 1, int), stop_token_ids=())
    except (TypeError, ValueError) as e:
        raise ApiError(400, f"bad sampling parameter: {e}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    srv = None

    def log_message(self, *a):
        pass

    # --- plumbing
    def _json(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, e):
        self._json(e.status, {"error": {"message": str(e), "type": e.kind, "param": None, "code": None}})

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse(self, obj):
        raw = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(raw), raw))
        self.wfile.flush()

    def _authed(self):
        k = self.srv.key
        if not k:
            return True
        got = self.headers.get("Authorization", "")
        return hmac.compare_digest(got.encode(), f"Bearer {k}".encode())

    def do_GET(self):
        s, path = self.srv, self.path.split("?")[0]
        if path == "/health":
            return self._json(503, {"status": "dead", "error": s.dead}) if s.dead else self._json(200, {"status": "ok"})
        if not self._authed():
            return self._error(ApiError(401, "invalid api key", "authentication_error"))
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [dict(id=s.name, object="model", created=s.created,
                                                                    owned_by="goethe-engine", max_model_len=s.ctx)]})
        self._error(ApiError(404, f"no route {path}", "not_found_error"))

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if not self._authed():
                raise ApiError(401, "invalid api key", "authentication_error")
            if path not in ("/v1/chat/completions", "/v1/completions"):
                raise ApiError(404, f"no route {path}", "not_found_error")
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ApiError(413, "body too large")
            try:
                body = json.loads(self.rfile.read(n))
            except ValueError:
                raise ApiError(400, "body is not JSON")
            if not isinstance(body, dict):
                raise ApiError(400, "body must be an object")
            if self.srv.dead:
                raise ApiError(503, f"engine stopped: {self.srv.dead}", "server_error")
            self._complete(body, path.endswith("chat/completions"))
        except ApiError as e:
            self._error(e)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # --- completions
    def _prompt(self, body, chat):
        tok = self.srv.tok
        try:
            if chat:
                ms = body.get("messages")
                if not isinstance(ms, list) or not ms:
                    raise ApiError(400, "messages must be a non-empty list")
                msgs = []
                for m in ms:
                    c = m.get("content")
                    if isinstance(c, list):
                        c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
                    if not isinstance(c, str) or not isinstance(m.get("role"), str):
                        raise ApiError(400, "each message needs a role and text content")
                    msgs.append(dict(role=m["role"], content=c))
                return tok.chat(msgs)
            p = body.get("prompt")
            if isinstance(p, list) and len(p) == 1 and isinstance(p[0], str):
                p = p[0]
            if isinstance(p, str):
                return tok.encode(p)
            if isinstance(p, list) and p and all(isinstance(t, int) for t in p):
                return p
        except (KeyError, AttributeError, TypeError) as e:
            raise ApiError(400, f"bad request: {e}")
        except ValueError as e:
            raise ApiError(400, str(e))
        raise ApiError(400, "prompt must be a string, a one-element list of one, or a list of token ids")

    def _complete(self, body, chat):
        s = self.srv
        prompt = self._prompt(body, chat)
        params, stops = sampling(body, s, len(prompt)), parse_stop(body.get("stop"))
        stream = bool(body.get("stream"))
        usage_on = bool((body.get("stream_options") or {}).get("include_usage"))
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        rid, q = s.submit(prompt, params)
        obj = "text_completion" if not chat else "chat.completion.chunk" if stream else "chat.completion"
        mk = lambda ch, **kw: dict(id=cid, object=obj, created=int(time.time()), model=s.name, choices=ch, **kw)

        def choice(i, text, fin=None):
            if chat:
                return dict(index=i, finish_reason=fin, **({"delta": {"content": text}} if stream else
                                                           {"message": {"role": "assistant", "content": text}}))
            return dict(index=i, text=text, finish_reason=fin)

        chs, closed, engine_done = {}, set(), set()
        try:
            if stream:
                self._sse_start()
                if chat:
                    for i in range(params.n):
                        self._sse(mk([dict(index=i, delta={"role": "assistant", "content": ""}, finish_reason=None)]))
            while len(closed) < params.n:
                try:
                    o = q.get(timeout=1)
                except queue.Empty:
                    o = None if s.dead else False
                if o is None:
                    raise ApiError(503, f"engine stopped: {s.dead}", "server_error")
                if o is False:
                    continue
                if o.finished and o.finish_reason == "error":
                    raise ApiError(500, o.error or "engine error", "server_error")
                if o.finished:
                    engine_done.add(o.index)
                if o.index in closed:
                    continue
                c = chs.setdefault(o.index, Choice(s.tok, stops))
                text = c.feed(o.token_ids, final=o.finished)
                if o.finished and not c.finish:
                    c.finish = "length" if o.finish_reason == "length" else "stop"
                if c.finish:
                    closed.add(o.index)
                if stream:
                    if text:
                        self._sse(mk([choice(o.index, text)]))
                    if c.finish:
                        self._sse(mk([choice(o.index, "", c.finish)]))
            if len(engine_done) < params.n:
                s.cancel(rid)   # a stop string ended the choices early: free the engine's work
            u = dict(prompt_tokens=len(prompt), completion_tokens=sum(c.ntok for c in chs.values()))
            u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]
            if stream:
                if usage_on:
                    self._sse(mk([], usage=u))
                self._sse(b"[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                self._json(200, mk([choice(i, chs[i].text, chs[i].finish) for i in sorted(chs)], usage=u))
        except (BrokenPipeError, ConnectionResetError):
            s.cancel(rid)
            self.close_connection = True
        except ApiError as e:
            s.cancel(rid)
            if stream:   # headers are out: an error event, then end the stream
                self._sse({"error": {"message": str(e), "type": e.kind}})
                self._sse(b"[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.close_connection = True
            else:
                raise
        finally:
            s.pending.pop(rid, None)


def build(args):
    """(engine, tokenizer, ctx) from the CLI args: a pack backend, or the CPU reference model with --ref."""
    from .engine import Engine
    from .scheduler import Config
    if args.ref:
        from .ref import RefModel, TorchBackend
        tok = ByteTokenizer()
        ctx = args.context_length or 512
        bs, seqs = 16, args.max_running_requests or 8
        be = TorchBackend(RefModel(vocab=tok.vocab_size, seed=0), num_blocks=seqs * (-(-ctx // bs)) + 8, block_size=bs,
                          max_seqs=seqs)
    else:
        if not (args.pack and args.weights):
            raise SystemExit("serve needs --pack DIR and --weights PATH... (or --ref for the CPU reference model)")
        from .gc_c import GcBackend
        if len(args.weights) > 2:
            raise SystemExit("--weights takes the target file and optionally a drafter file (speculative pair)")
        be = GcBackend(args.pack, args.weights[0], B=args.max_running_requests or 8, drafter_weights=args.weights[1] if len(args.weights) > 1 else None)
        tdir = args.tokenizer or os.path.dirname(os.path.abspath(args.weights[0]))
        if not os.path.exists(os.path.join(tdir, "tokenizer.json")) and not args.tokenizer:
            raise SystemExit("no tokenizer: pass --tokenizer DIR (a Hugging Face tokenizer directory)")
        tok = HFTokenizer(tdir)
        ctx = args.context_length or be.max_len
    seqs = min(args.max_running_requests or be.max_seqs, be.max_seqs)
    eng = Engine(be, Config(max_num_seqs=seqs, max_model_len=ctx, prefix_cache=getattr(be, "paged", True)),
                 eos_ids=tok.eos_ids)
    return eng, tok, ctx


def parser():
    ap = argparse.ArgumentParser(prog="goethe-engine serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--context-length", type=int)
    ap.add_argument("--max-running-requests", type=int)
    ap.add_argument("--served-model-name")
    ap.add_argument("--api-key")
    ap.add_argument("--pack", metavar="DIR", help="signed kernel pack directory")
    ap.add_argument("--weights", nargs="+", metavar="PATH", help="target weights, then a drafter's for a speculative pair")
    ap.add_argument("--tokenizer", metavar="DIR", help="Hugging Face tokenizer dir (default: the weights' dir)")
    ap.add_argument("--ref", action="store_true", help="CPU reference model (dev, tests)")
    return ap


def make_server(args):
    eng, tok, ctx = build(args)
    s = Server(eng, tok, args.served_model_name or "goethe-engine", ctx, args.api_key)
    s.thread.start()
    h = type("H", (Handler,), {"srv": s})
    httpd = ThreadingHTTPServer((args.host, args.port), h)
    httpd.daemon_threads = True
    return httpd, s


def main(argv):
    from .pack import PackError
    args = parser().parse_args(argv)
    try:
        httpd, s = make_server(args)
    except PackError as e:
        sys.exit(f"goethe-engine: pack REFUSED: {e}")
    print(f"goethe-engine serving {s.name} on http://{args.host}:{httpd.server_address[1]}", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        s.stop = True
