"""goethe-engine serve: the OpenAI-compatible upstream goethed proxies to (stdlib http.server, one engine thread).

Routes: POST /v1/chat/completions, POST /v1/completions (JSON or SSE) for a text model; POST /v1/embeddings for an embedding model;
POST /v1/audio/transcriptions for an encoder-decoder (JSON {frames: [[f32] * F] * T, prompt: decoder ids, max_tokens}: the features the
model's encoder takes); GET /v1/models, GET /health.
Requests from many handler threads join one Engine; the engine thread steps it (continuous batching) and fans the
token deltas out to per-request queues. Localhost by default."""
import argparse
import base64
import hmac
import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import numerics_kw as NK, pack as P_
from .tokenizer import ByteTokenizer, HFTokenizer, Turn

MAX_BODY = 8 << 20
UNSUPPORTED = ("functions", "function_call", "logprobs", "top_logprobs", "logit_bias", "suffix", "best_of", "echo")
CHAT_ONLY = ("tools", "tool_choice", "parallel_tool_calls", "reasoning_effort")
# The body fields each route reads; any other non-null field is refused (unsupported_parameter), never silently ignored.
FIELDS = {"generate": ("model", "messages", "prompt", "stream", "stream_options", "user", "max_tokens", "max_completion_tokens", "temperature",
                       "top_p", "top_k", "seed", "n", "stop", "frequency_penalty", "presence_penalty", "response_format") + UNSUPPORTED + CHAT_ONLY,
          "embed": ("model", "input", "encoding_format", "user", "normalize", "dimensions", "truncation")}


def known_fields(body, task):
    k = next((k for k, v in body.items() if v is not None and k not in FIELDS[task]), None)
    if k is not None:
        raise ApiError(400, f"{k} is not a parameter of this endpoint", code="unsupported_parameter", param=k)


class ApiError(Exception):
    """An HTTP error in the one shape every engine front door uses: {"error": {type, code (spec/errors.json), param, message}}."""

    def __init__(self, status, msg, kind="invalid_request_error", code=None, param=None):
        super().__init__(msg)
        self.status, self.kind, self.code, self.param = status, kind, code, param

    def body(self):
        return {"error": {"message": str(self), "type": self.kind, "param": self.param, "code": self.code}}


class Choice:
    """One output sequence: its turn's parts, stop strings on its content, the first tool call only when one_call."""

    def __init__(self, tok, stops, fmt=None, one_call=False):
        self.turn, self.stops, self.one_call, self.held, self.text, self.reasoning = Turn(tok, fmt), stops, one_call, "", "", ""
        self.calls, self.stopped, self.finish, self.ntok = [], False, None, 0

    def feed(self, ids, final=False):
        """The parts safe to send now; a completed stop string ends the content before it."""
        if self.stopped:
            return []
        self.ntok += len(ids)
        out = []
        for k, v in self.turn.feed(ids, final):
            if k == "content":
                self.held += v
            elif k == "reasoning":
                self.reasoning += v
                out.append((k, v))
            elif not (self.one_call and self.calls):
                self.calls.append(dict(id="call_" + uuid.uuid4().hex[:24], type="function", function=v))
                out.append((k, dict(index=len(self.calls) - 1, **self.calls[-1])))
        cut = min((i for i in (self.held.find(s) for s in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            text, self.held, self.stopped, self.finish = self.held[:cut], "", True, "stop"
        else:
            keep = 0 if final else max((len(s) - 1 for s in self.stops), default=0)
            text, self.held = self.held[:len(self.held) - keep], self.held[len(self.held) - keep:]
        if text:
            self.text += text
            out.append(("content", text))
        return out


def message(m):
    """An OpenAI chat message as a template takes it: text content, tool call arguments parsed from their JSON string."""
    bad = lambda why: ApiError(400, why, code="invalid_parameter", param="messages")
    if not isinstance(m, dict) or not isinstance(m.get("role"), str):
        raise bad("each message needs a role")
    c = m.get("content")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    if not isinstance(c, str) and c is not None:
        raise bad("message content must be text")
    out = dict(role=m["role"], content=c or "", **{k: m[k] for k in ("name", "tool_call_id", "reasoning_content") if isinstance(m.get(k), str)})
    calls = m.get("tool_calls") or []
    if not isinstance(calls, list):
        raise bad("tool_calls must be a list")
    for tc in calls:
        f = tc.get("function") if isinstance(tc, dict) else None
        a = f.get("arguments") if isinstance(f, dict) else None
        try:
            a = json.loads(a) if isinstance(a, str) and a.strip() else a or {}
        except ValueError:
            raise bad("a tool call's arguments are not JSON")
        if not (isinstance(f, dict) and isinstance(f.get("name"), str) and isinstance(a, dict)):
            raise bad("a tool call needs function.name and object arguments")
        out.setdefault("tool_calls", []).append(dict(id=tc.get("id"), type="function", function=dict(name=f["name"], arguments=a)))
    return out


def chat_options(body, tok):
    """Template arguments of a chat request; tools are refused when the engine cannot read the model's calls back."""
    tools, choice = body.get("tools"), body.get("tool_choice")
    if choice not in (None, "auto", "none"):
        raise ApiError(400, "tool_choice must be auto or none", code="unsupported_parameter", param="tool_choice")
    if tools is not None and not (isinstance(tools, list) and all(isinstance(t, dict) for t in tools)):
        raise ApiError(400, "tools must be a list of tool objects", code="invalid_parameter", param="tools")
    if tools and tok.turn is None:
        raise ApiError(400, "tools are not supported: the engine cannot read this model's tool calls", code="unsupported_parameter", param="tools")
    kw = dict(tools=tools if tools and choice != "none" else None)
    if body.get("reasoning_effort") is not None:
        kw["reasoning_effort"] = body["reasoning_effort"]
    return kw


KIND_WORD = {"decoder": "text", "embedding": "embedding", "encoder-decoder": "speech", "media": "speech"}   # the words customers see


class Server:
    def __init__(self, engine, tok, name, ctx, api_key=None):
        self.engine, self.tok, self.name, self.ctx, self.key = engine, tok, name, ctx, api_key
        self.embedding = hasattr(engine, "submit") and hasattr(engine, "embed")
        self.seq2seq = hasattr(engine, "frames")
        self.media = hasattr(engine, "text") and hasattr(engine, "transcribe")      # MediaRunner: a media model (audio into a decoder)
        self.lock, self.instruction = threading.Lock(), None
        self.pending, self.wake, self.stop, self.dead = {}, threading.Event(), False, None
        self.created = int(time.time())
        self.thread = threading.Thread(target=self._loop, daemon=True, name="engine")

    def _loop(self):
        try:
            while not self.stop:
                if self.embedding or self.media or not self.engine.has_work():   # embedding / media runners: no engine loop
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
            raise ApiError(400, str(e), code=getattr(e, "code", "invalid_request"), param=getattr(e, "param", None))
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
        raise ApiError(400, "stop must be a non-empty string or up to 4 of them", code="invalid_parameter", param="stop")
    return tuple(v)


def sampling(body, srv, plen):
    known_fields(body, "generate")
    for k in UNSUPPORTED:
        if body.get(k) not in (None, False):
            raise ApiError(400, f"{k} is not supported", code="unsupported_parameter", param=k)
    for k in ("frequency_penalty", "presence_penalty"):
        if body.get(k) not in (None, 0, 0.0):
            raise ApiError(400, f"{k} is not supported", code="unsupported_parameter", param=k)
    if (body.get("response_format") or {}).get("type", "text") != "text":
        raise ApiError(400, "response_format is not supported", code="unsupported_parameter", param="response_format")
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    req = dict(temperature=body.get("temperature", 1.0), top_p=body.get("top_p"), top_k=body.get("top_k"), seed=body.get("seed"),
               max_tokens=srv.ctx if mt is None else mt, n=body.get("n"))
    from .sampling import SamplingParams
    return SamplingParams(**checked("generate", req, srv))


def checked(task, req, srv):
    """params.check against the served pack's bucket (the one mechanism of api.Model), as an HTTP 400 with the error code."""
    from . import params
    from .errors import EngineError
    try:
        return params.check(task, req, getattr(getattr(srv.engine, "be", None) or getattr(srv.engine, "backend", None), "bucket", None))
    except EngineError as e:
        raise ApiError(400, str(e), code=e.code, param=e.param)


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
        self._json(e.status, e.body())

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
                                                                    owned_by="goethe-engine", max_model_len=s.ctx,
                                                                    kind=KIND_WORD["embedding" if s.embedding else "media" if s.media else "encoder-decoder" if s.seq2seq else "decoder"])]})
        self._error(ApiError(404, f"no route {path}", "not_found_error"))

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self.close_connection = True     # the body stays unread: the connection cannot carry another request
                raise ApiError(413, "body too large", code="invalid_request")
            raw = self.rfile.read(n)             # read before any refusal, else a kept-alive connection parses it as the next request
            if not self._authed():
                raise ApiError(401, "invalid api key", "authentication_error")
            if path not in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/audio/transcriptions"):
                raise ApiError(404, f"no route {path}", "not_found_error")
            kind = "embedding" if self.srv.embedding else "encoder-decoder" if self.srv.seq2seq else "media" if self.srv.media else "decoder"
            want = {"/v1/embeddings": ("embedding",), "/v1/audio/transcriptions": ("encoder-decoder", "media")}.get(path, ("decoder",))
            if kind not in want:
                word = KIND_WORD[kind]
                raise ApiError(400, f"{path}: this model is {'an' if word[0] in 'ae' else 'a'} {word} model, not "
                                    f"{' or '.join(('an ' if KIND_WORD[w][0] in 'ae' else 'a ') + KIND_WORD[w] for w in dict.fromkeys(want))} model",
                               code="unsupported_task")
            if self.srv.media:
                return self._transcribe_media(raw)
            try:
                body = json.loads(raw)
            except ValueError:
                raise ApiError(400, "body is not JSON", code="invalid_request")
            if not isinstance(body, dict):
                raise ApiError(400, "body must be an object", code="invalid_request")
            if self.srv.dead:
                raise ApiError(503, f"engine stopped: {self.srv.dead}", "server_error")
            if self.srv.embedding:
                return self._embeddings(body)
            if self.srv.seq2seq:
                return self._transcribe(body)
            self._complete(body, path.endswith("chat/completions"))
        except ApiError as e:
            self._error(e)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # --- embeddings
    def _embeddings(self, body):
        s, inp = self.srv, body.get("input")
        if body.get("encoding_format", "float") not in ("float", "base64"):
            raise ApiError(400, "encoding_format must be float or base64", code="invalid_parameter", param="encoding_format")
        if isinstance(inp, str) or (isinstance(inp, list) and inp and all(isinstance(t, int) for t in inp)):
            inp = [inp]
        if not isinstance(inp, list) or not inp or not all(isinstance(x, str) or (isinstance(x, list) and x and all(
                isinstance(t, int) for t in x)) for x in inp):
            raise ApiError(400, "input must be a string, a list of strings, a list of token ids or a list of such lists", code="invalid_parameter",
                           param="input")
        known_fields(body, "embed")
        items = [s.tok.embed_input(x) if isinstance(x, str) else x for x in inp]
        r = checked("embed", dict(normalize=body.get("normalize", True) is not False, dimensions=body.get("dimensions"),
                                  truncation=body.get("truncation")), s)
        if r.get("truncation") == "right":       # as api.Model.embed: cut each input to the longest the pack embeds
            be = getattr(s.engine, "be", None) or getattr(s.engine, "backend", None)
            items = [list(t)[:min(be.max_len, be.max_cols)] for t in items]
        try:
            res = s.engine.submit(items, norm=r["normalize"] and "dimensions" not in r).get()
        except P_.PackError as e:
            raise ApiError(400, str(e), code=e.code)
        if isinstance(res, BaseException):
            raise ApiError(500, f"{type(res).__name__}: {res}", "server_error")
        if "dimensions" in r:
            from .embed import normalize
            res = res[:, :r["dimensions"]]
            res = normalize(res) if r["normalize"] else res
        rows = []
        for i, r in enumerate(res):
            e = base64.b64encode(r.astype("<f4").tobytes()).decode() if body.get("encoding_format") == "base64" else r.tolist()
            rows.append(dict(object="embedding", index=i, embedding=e))
        n = sum(len(t) for t in items)
        self._json(200, dict(object="list", data=rows, model=s.name, usage=dict(prompt_tokens=n, total_tokens=n)))

    # --- transcriptions (media model): OpenAI multipart
    def _transcribe_media(self, raw):
        """POST /v1/audio/transcriptions, multipart/form-data: file (WAV in the core path; other containers through ffmpeg when it is on
        PATH), model, prompt (the instruction: only when the pack declares it), response_format json | text, temperature 0 only,
        language (only when declared). The prompt is the text model's chat template over the media marker and the instruction."""
        import email.parser
        s = self.srv
        ct = self.headers.get("Content-Type", "")
        if not ct.startswith("multipart/form-data"):
            raise ApiError(400, "body must be multipart/form-data (file, model, ...)", code="invalid_request")
        msg = email.parser.BytesParser().parsebytes(b"Content-Type: " + ct.encode() + b"\r\n\r\n" + raw)
        if not msg.is_multipart():
            raise ApiError(400, "multipart body without parts", code="invalid_request")
        fields, upload = {}, None
        for part in msg.get_payload():
            name = part.get_param("name", header="content-disposition")
            if name == "file":
                upload = part.get_payload(decode=True)
            elif name:
                fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "replace")
        if not upload:
            raise ApiError(400, "file is required", code="invalid_parameter", param="file")
        for k in fields:
            if k not in ("model", "prompt", "response_format", "temperature", "language"):
                raise ApiError(400, f"{k} is not supported", code="unsupported_parameter", param=k)
        if fields.get("temperature") not in (None, "", "0", "0.0"):
            raise ApiError(400, "temperature: transcription is greedy (temperature 0 only)", code="unsupported_parameter", param="temperature")
        fmt = fields.get("response_format", "json")
        if fmt not in ("json", "text"):
            raise ApiError(400, "response_format must be json or text", code="invalid_parameter", param="response_format")
        bucket = s.engine.be.bucket
        req = {k: fields[k] for k in ("language",) if fields.get(k)}
        if fields.get("prompt"):
            req["instruction"] = fields["prompt"]
        r = checked("transcribe", req, s)
        instr = r.get("instruction", (bucket.get("io") or {}).get("instruction", s.instruction or ""))
        pcm = decode_audio(upload, s.engine.be.rate)
        max_new = (bucket.get("io") or {}).get("max_new_tokens") or max(1, bucket["maxctx"] // 4)
        try:
            with s.lock:
                prompt = s.engine.text.render(s.engine.be.marker + instr)
                ids = s.engine.transcribe([pcm], prompt, max_new)
                text = s.engine.text.detokenize(ids)
                usage = dict(input_tokens=s.engine.last_input, output_tokens=len(ids), total_tokens=s.engine.last_input + len(ids))
        except P_.PackError as e:
            raise ApiError(400, str(e), code=e.code)
        if fmt == "text":
            b = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("X-Usage", json.dumps(usage))          # a text reply carries its usage in a header
            self.end_headers()
            self.wfile.write(b)
            return
        self._json(200, dict(text=text, usage=usage))

    # --- transcriptions (encoder-decoder)
    def _transcribe(self, body):
        import numpy as np
        s = self.srv
        try:
            frames = np.asarray(body.get("frames"), dtype=np.float32)
            prompt = body.get("prompt")
            if not (isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt)):
                raise ValueError("prompt must be the decoder's start token ids")
            params = sampling(dict({k: v for k, v in body.items() if k != "frames"}, temperature=0), s, len(prompt))
            rid = uuid.uuid4().hex
            q = s.pending[rid] = queue.Queue()
            s.engine.add_request(rid, prompt, params, frames=frames)
        except (ValueError, TypeError, RuntimeError) as e:
            s.pending.pop(rid if "rid" in locals() else None, None)
            raise ApiError(400, str(e), code=getattr(e, "code", "invalid_request"), param=getattr(e, "param", None))
        s.wake.set()
        ids = []
        while True:
            o = q.get()
            if o is None:
                raise ApiError(503, f"engine stopped: {s.dead}", "server_error")
            ids += o.token_ids
            if o.finished:
                break
        s.pending.pop(rid, None)
        self._json(200, dict(text=s.tok.decode(ids), tokens=ids))

    # --- completions
    def _prompt(self, body, chat):
        tok = self.srv.tok
        try:
            if chat:
                ms = body.get("messages")
                if not isinstance(ms, list) or not ms:
                    raise ApiError(400, "messages must be a non-empty list", code="invalid_parameter", param="messages")
                return tok.chat([message(m) for m in ms], **chat_options(body, tok))
            k = next((k for k in CHAT_ONLY if body.get(k) is not None), None)
            if k:
                raise ApiError(400, f"{k} is a chat parameter: use /v1/chat/completions", code="unsupported_parameter", param=k)
            p = body.get("prompt")
            if isinstance(p, list) and len(p) == 1 and isinstance(p[0], str):
                p = p[0]
            if isinstance(p, str):
                return tok.encode(p)
            if isinstance(p, list) and p and all(isinstance(t, int) for t in p):
                return p
        except (KeyError, AttributeError, TypeError) as e:
            raise ApiError(400, f"bad request: {e}", code="invalid_request")
        except ValueError as e:
            raise ApiError(400, str(e), code="invalid_parameter", param="prompt")
        raise ApiError(400, "prompt must be a string, a one-element list of one, or a list of token ids", code="invalid_parameter", param="prompt")

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

        def delta(i, k, v):
            if not chat:
                return dict(index=i, text=v, finish_reason=None)
            return dict(index=i, finish_reason=None, delta={"tool_calls": [v]} if k == "tool" else {"reasoning_content" if k == "reasoning" else "content": v})

        def final(i, c):
            if not chat:
                return dict(index=i, text=c.text, finish_reason=c.finish)
            msg = dict(role="assistant", content=c.text if c.text or not c.calls else None)
            msg.update({k: v for k, v in (("reasoning_content", c.reasoning), ("tool_calls", c.calls)) if v})
            return dict(index=i, finish_reason=c.finish, message=msg)

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
                c = chs.setdefault(o.index, Choice(s.tok, stops, s.tok.turn if chat else None, body.get("parallel_tool_calls") is False))
                parts = c.feed(o.token_ids, final=o.finished)
                if o.finished and not c.finish:
                    c.finish = "length" if o.finish_reason == "length" else "tool_calls" if c.calls else "stop"
                if c.finish:
                    closed.add(o.index)
                if stream:
                    for k, v in parts:
                        self._sse(mk([delta(o.index, k, v)]))
                    if c.finish:
                        self._sse(mk([dict(delta(o.index, "content", ""), finish_reason=c.finish)]))
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
                self._json(200, mk([final(i, chs[i]) for i in sorted(chs)], usage=u))
        except (BrokenPipeError, ConnectionResetError):
            s.cancel(rid)
            self.close_connection = True
        except ApiError as e:
            s.cancel(rid)
            if stream:   # headers are out: an error event, then end the stream
                self._sse(e.body())
                self._sse(b"[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.close_connection = True
            else:
                raise
        finally:
            s.pending.pop(rid, None)


def decode_audio(data, rate):
    """float32 mono PCM at rate: a WAV at that rate in the core path (stdlib); anything else (another rate, mp3, flac, ogg) through
    ffmpeg when it is on PATH, else refused (code unsupported_media, HTTP 415)."""
    import shutil
    import subprocess
    import numpy as np
    from .media import read_wav
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            return read_wav(data, rate)
        except P_.PackError as e:
            if "Hz" not in str(e) or not shutil.which("ffmpeg"):
                raise ApiError(415 if "Hz" in str(e) else 400, str(e), code="unsupported_media", param="file")
    if not shutil.which("ffmpeg"):
        raise ApiError(415, f"the upload is not a {rate} Hz WAV and ffmpeg is not on PATH to convert it", code="unsupported_media", param="file")
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(rate), "pipe:1"],
                       input=data, capture_output=True, timeout=600)
    if r.returncode or not r.stdout:
        raise ApiError(415, f"ffmpeg could not decode the upload: {r.stderr.decode(errors='replace')[-200:]}", code="unsupported_media", param="file")
    return np.frombuffer(r.stdout, np.float32).copy()


def build(args):
    """(engine, tokenizer, ctx) from the CLI args: a pack backend, or the CPU reference model with --ref."""
    from . import registry
    from .scheduler import Config
    if args.ref and args.ref_type == "encoder-decoder":
        from .seq2seq import RefSeq2Seq, RefSeq2SeqBackend
        tok, ctx = ByteTokenizer(), args.context_length or 64
        seqs = args.max_running_requests or 4
        be = RefSeq2SeqBackend(RefSeq2Seq(vocab=257, seed=0), num_blocks=seqs * (-(-ctx // 4)) + 8, block_size=4, max_seqs=seqs)
        return registry.runner(be, Config(max_num_seqs=seqs, max_model_len=ctx), eos_ids=tok.eos_ids), tok, ctx
    if args.ref and args.ref_type == "embedding":
        from .embed import RefEmbed
        from .ref import RefModel
        tok, ctx = ByteTokenizer(), args.context_length or 64
        be = RefEmbed(RefModel(vocab=257, seed=0), max_cols=4 * ctx, max_seqs=args.max_running_requests or 8, max_len=ctx)
        return registry.runner(be, None), tok, ctx
    if args.ref:
        from .ref import RefModel, TorchBackend
        tok = ByteTokenizer()
        ctx = args.context_length or 512
        bs, seqs = 16, args.max_running_requests or 8
        be = TorchBackend(RefModel(vocab=tok.vocab_size, seed=0), num_blocks=seqs * (-(-ctx // bs)) + 8, block_size=bs,
                          max_seqs=seqs)
    else:
        if not (args.pack and args.weights):
            raise SystemExit("serve needs --pack FILE and --weights PATH... (or --ref for the CPU reference model)")
        if len(args.weights) > 2 and registry.pack_type(args.pack) != "media":
            raise SystemExit("--weights takes the target model (a GGUF file, or a checkpoint directory) and optionally a drafter file (speculative pair)")
        from . import api
        media = registry.pack_type(args.pack) == "media"         # a media model's weights are its projector and its text model
        mo = api.Model([args.pack] + list(args.overlay or []), args.weights if media else args.weights[0],
                       drafter=args.weights[1] if len(args.weights) > 1 and not media else None,
                       device=args.device, store=args.store, B=args.max_running_requests or 8, numerics=args.numerics, act=args.act,
                       kv=args.kv, config=dict(max_num_seqs=args.max_running_requests, max_model_len=args.context_length,
                                               kv_dtype=None if args.kv == "auto" else args.kv))
        if mo.type == "media":                   # the library tokenizes; the text model GGUF carries the template and the pieces
            return mo.runner, None, args.context_length or mo.backend.bucket["maxctx"]
        tdir = args.tokenizer or os.path.dirname(os.path.abspath(args.weights[0]))
        if not os.path.exists(os.path.join(tdir, "tokenizer.json")) and not args.tokenizer:
            raise SystemExit("no tokenizer: pass --tokenizer DIR (a Hugging Face tokenizer directory)")
        tok = HFTokenizer(tdir)
        if mo.type == "decoder":
            mo.runner.eos = set(tok.eos_ids)        # the pack opens (and is verified) before the tokenizer is read
        ctx = mo.cfg.max_model_len if mo.cfg else args.context_length or mo.backend.max_len   # clamped to the handle's limit
        return mo.runner, tok, ctx
    seqs = min(args.max_running_requests or be.max_seqs, be.max_seqs)
    eng = registry.runner(be, Config(max_num_seqs=seqs, max_model_len=ctx, kv_dtype="f16" if args.kv == "auto" else args.kv, prefix_cache=getattr(be, "prefix_cache", True)),
                 eos_ids=tok.eos_ids)
    return eng, tok, ctx


def parser():
    ap = argparse.ArgumentParser(prog="goethe-engine serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--context-length", type=int)
    ap.add_argument("--max-running-requests", type=int)
    ap.add_argument("--served-model-name")
    ap.add_argument("--api-key", default=os.environ.get("GOETHE_ENGINE_API_KEY"))
    ap.add_argument("--pack", metavar="FILE", help="signed kernel pack (.gcpack, signature in FILE.sig)")
    ap.add_argument("--weights", nargs="+", metavar="PATH", help="target weights, then a drafter's for a speculative pair (a media model: "
                                                                 "its projector, then its text model)")
    ap.add_argument("--instruction", help="media models: the transcription instruction when the pack's bucket declares none (io.instruction)")
    ap.add_argument("--tokenizer", metavar="DIR", help="Hugging Face tokenizer dir (default: the weights' dir)")
    ap.add_argument("--numerics", choices=NK.CHOICES, default="invariant",
                    help="reference: bitwise llama.cpp, one stream; invariant: independent of the batch (invariant-hp: 16-bit "
                         "activations); tolerance: deterministic until a tolerance bucket exists; old names exact | deterministic warn")
    ap.add_argument("--act", choices=("auto", "a8", "a16"), default="auto", help="activation precision of deterministic buckets")
    ap.add_argument("--kv", choices=("auto", "f16", "fp8"), default="auto", help="KV cache dtype (auto: f16 when the pack has it, else fp8)")
    ap.add_argument("--ref", action="store_true", help="CPU reference model (dev, tests)")
    ap.add_argument("--ref-type", choices=("decoder", "embedding", "encoder-decoder"), default="decoder", help="model type of the --ref model")
    ap.add_argument("--overlay", action="append", metavar="FILE", help="signed overlay of the pack (more arches, formats): repeatable")
    ap.add_argument("--store", metavar="DIR", help="pack 4 blob store directory (default $GOETHE_BLOB_STORE)")
    ap.add_argument("--device", type=int, default=-1, help="device ordinal (default: the current device)")
    return ap


def make_server(args):
    eng, tok, ctx = build(args)
    s = Server(eng, tok, args.served_model_name or "goethe-engine", ctx, args.api_key)
    s.instruction = getattr(args, "instruction", None)
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
