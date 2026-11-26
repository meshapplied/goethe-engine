"""Worker protocol v1 server (spec/proto/worker.proto): one worker per device over a UNIX stream socket; models are api.Model handles.

Frames: 4-byte little-endian length + one serialized Request / Reply. The message classes come from the compiled descriptor set
spec/proto/worker.desc (protoc --descriptor_set_out), so any protobuf runtime >= 4 reads them. Each request runs on its own thread
(concurrent generates batch in the model's engine loop, cancel reaches a running one); embed and transcribe calls on one model are
serialized (a handle is not thread safe). Errors are Reply.error {code (spec/errors.json), message}.

  python -m goethe_engine worker --socket PATH"""
import json
import os
import socket
import struct
import threading

from . import __version__, api, pack as P
from .errors import EngineError

PROTOCOL = (1, 1)          # the protocol versions this worker speaks (min, max)
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec", "proto")


def messages():
    """(Request, Reply) classes from the compiled descriptor set."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    fds = descriptor_pb2.FileDescriptorSet()
    with open(os.path.join(SPEC, "worker.desc"), "rb") as f:
        fds.ParseFromString(f.read())
    pool = descriptor_pool.DescriptorPool()
    for fd in fds.file:
        pool.Add(fd)
    get = getattr(message_factory, "GetMessageClass", None) or message_factory.MessageFactory(pool).GetPrototype
    return tuple(get(pool.FindMessageTypeByName(f"goethe.worker.v1.{n}")) for n in ("Request", "Reply"))


def send(sock, msg, lock=None):
    b = msg.SerializeToString()
    with lock or threading.Lock():
        sock.sendall(struct.pack("<I", len(b)) + b)


def recv(sock, cls):
    """The next message, or None at end of stream."""
    def exact(n):
        buf = b""
        while len(buf) < n:
            c = sock.recv(n - len(buf))
            if not c:
                return None
            buf += c
        return buf
    h = exact(4)
    if h is None:
        return None
    b = exact(struct.unpack("<I", h)[0])
    m = cls()
    m.ParseFromString(b or b"")
    return m


class Worker:
    def __init__(self, device=0):
        self.Request, self.Reply = messages()
        self.device, self.models, self.lock = device, {}, threading.Lock()

    def device_record(self):
        try:
            d = P.device(self.device)
            return dict(platform=d["platform"], name=d["name"], cc=d["cc"], sms=d["sms"] or 0, free_bytes=P.device_free(self.device) or 0)
        except Exception:                      # no device visible (CPU host): the handshake still answers
            return dict(platform="none")

    # --- handlers: each returns a Reply body (field name, message dict) or yields several for a stream
    def hello(self, r):
        lo, hi = max(r.min_protocol, PROTOCOL[0]), min(r.max_protocol, PROTOCOL[1])
        if lo > hi:
            raise P.PackError(f"protocol {r.min_protocol}..{r.max_protocol} requested, this worker speaks {PROTOCOL[0]}..{PROTOCOL[1]}")
        return "welcome", dict(protocol=hi, engine_version=__version__, device=self.device_record(),
                               models=[self.model_record(k) for k in self.models])

    def model_record(self, k):
        m = self.models[k]["model"]
        return dict(model_id=k, task=m.kind, capabilities_json=json.dumps(self.models[k]["caps"]))

    def load(self, r):
        opts = json.loads(r.options_json or "{}")
        if r.model_id in self.models:
            raise P.PackError(f"model {r.model_id!r} is loaded already")
        pack = list(r.pack)
        m = api.Model(pack if len(pack) > 1 else pack[0], list(r.weights), drafter=list(r.drafter) or None, device=self.device,
                      trusted=tuple(opts.pop("trusted")) if "trusted" in opts else None, **opts)
        ent = dict(model=m, caps=api.capabilities(pack[0]), lock=threading.Lock(), server=None)
        if m.kind == "generate":
            from .serve import Server
            ent["server"] = Server(m.runner, None, r.model_id, m.backend.max_len)
            ent["server"].thread.start()
        self.models[r.model_id] = ent
        return "loaded", self.model_record(r.model_id)

    def unload(self, r):
        ent = self.get(r.model_id)
        if ent["server"]:
            ent["server"].stop = True
        with ent["lock"]:
            ent["model"].close()
        del self.models[r.model_id]
        return "ok", {}

    def warm(self, r):
        ent = self.get(r.model_id)
        m = ent["model"]
        if m.kind == "generate":                  # warm graph buckets + one token through the engine loop
            if getattr(m.backend, "graph_buckets", ()):
                ev = threading.Event()
                m.runner.inbox.append(("warm", ev))
                ent["server"].wake.set()
                ev.wait()
            _, q = ent["server"].submit([0], m.sampling(max_tokens=1))
            while not (q.get() or type("E", (), dict(finish_reason="error"))).finish_reason:
                pass
        elif m.kind == "embed":
            with ent["lock"]:
                m.embed([[0]])
        return "ok", {}

    def health(self, r):
        return "health", dict(ok=True, model_ids=sorted(self.models))

    def metrics(self, r):
        out = {k: (v["model"].runner.metrics() if hasattr(v["model"].runner, "metrics") else {}) for k, v in self.models.items()}
        return "metrics", dict(json=json.dumps(out))

    def get(self, k):
        if k not in self.models:
            raise P.PackError(f"model {k!r} is not loaded (loaded: {sorted(self.models)})")
        return self.models[k]

    def generate(self, r, rid, cancels=None):
        ent = self.get(r.model_id)
        m, srv = ent["model"], ent["server"]
        sp = m.sampling(**json.loads(r.params_json or "{}"))
        key, q = srv.submit(list(r.prompt.ids), sp)
        if cancels is not None:
            cancels[rid] = lambda: srv.cancel(key)
        ids = []
        while True:
            o = q.get()
            if o is None:
                raise RuntimeError(f"engine stopped: {srv.dead}")
            if o.finish_reason == "error":
                raise RuntimeError(o.error or "error")
            ids += o.token_ids
            if r.stream or o.finish_reason:
                yield "chunk", dict(ids=o.token_ids if r.stream else ids, done=bool(o.finish_reason), finish_reason=o.finish_reason or "")
            if o.finish_reason:
                return

    def embed(self, r):
        ent = self.get(r.model_id)
        with ent["lock"]:
            x = ent["model"].embed([list(t.ids) for t in r.items], **json.loads(r.params_json or "{}"))
        return "embed", dict(dim=int(x.shape[1]) if len(x) else 0, rows=x.reshape(-1).tolist())

    def transcribe(self, r):
        import numpy as np
        ent = self.get(r.model_id)
        with ent["lock"]:
            ids = ent["model"].transcribe([np.asarray(r.pcm, np.float32)], r.prompt, **json.loads(r.params_json or "{}"))
        return "transcribe", dict(ids=list(ids))

    # --- connection
    def handle(self, conn):
        wl, cancels, greeted = threading.Lock(), {}, False

        def reply(rid, field, body):
            rep = self.Reply(id=rid)
            getattr(rep, field).CopyFrom(getattr(self.Reply(), field).__class__(**body)) if body else getattr(rep, field).SetInParent()
            send(conn, rep, wl)

        def error(rid, e):
            code = e.code if isinstance(e, EngineError) else "error"
            send(conn, self.Reply(id=rid, error=dict(code=code, message=f"{type(e).__name__}: {e}", param=getattr(e, "param", None) or "")), wl)

        def run(req):
            kind = req.WhichOneof("body")
            try:
                if kind == "generate":
                    for field, body in self.generate(req.generate, req.id, cancels):
                        reply(req.id, field, body)
                else:
                    reply(req.id, *getattr(self, kind)(getattr(req, kind)))
            except Exception as e:
                error(req.id, e)
            finally:
                cancels.pop(req.id, None)

        while True:
            req = recv(conn, self.Request)
            if req is None:
                break
            kind = req.WhichOneof("body")
            if not greeted and kind != "hello":
                error(req.id, P.PackError("the first request must be hello"))
                continue
            if kind == "hello":
                try:
                    reply(req.id, *self.hello(req.hello))
                    greeted = True
                except Exception as e:
                    error(req.id, e)
            elif kind == "cancel":
                f = cancels.get(req.cancel.target)
                f and f()
                reply(req.id, "ok", {})
            else:
                threading.Thread(target=run, args=(req,), daemon=True).start()
        conn.close()


def serve(path, device=0, ready=None):
    """Accept connections on the UNIX socket path (mode 0600) until the process ends; ready (an Event) is set once listening."""
    w = Worker(device)
    if os.path.exists(path):
        os.unlink(path)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(path)
    os.chmod(path, 0o600)
    s.listen(16)
    if ready:
        ready.set()
    while True:
        c, _ = s.accept()
        threading.Thread(target=w.handle, args=(c,), daemon=True).start()


class Client:
    """A minimal client (tests, tools): call(field, **body) -> list of Reply messages of that request (a stream ends with done)."""

    def __init__(self, path):
        self.Request, self.Reply = messages()
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(path)
        self.n = 0

    def call(self, field, **body):
        self.n += 1
        req = self.Request(id=self.n)
        getattr(req, field).CopyFrom(getattr(self.Request(), field).__class__(**body)) if body else getattr(req, field).SetInParent()
        send(self.s, req)
        out = []
        while True:
            r = recv(self.s, self.Reply)
            out.append(r)
            k = r.WhichOneof("body")
            if k != "chunk" or r.chunk.done:
                return out
