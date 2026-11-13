"""Backend over a gc-c pack: the pack's libgck.so through the C ABI of abi/gc.h (ctypes, no compiled code here).

The pack is staged into a private directory and verified there (pack.stage); the library is loaded from that copy and
gc_open gets the same directory, so what was verified is what runs. The weights were hashed here, hence
GC_OPEN_NO_CHECK. One handle, not thread safe: the engine calls it from one thread. Capabilities come from the
manifest (pack_backend.caps_of_bucket), never from symbols."""
import ctypes as C
import os

import torch

from . import pack as P
from .backend import KVGeometry
from .pack_backend import caps_of_bucket, numerics_of

ABI_MAJOR, ABI_MINOR = 1, 0   # 1.1 adds state_pages (a 1.0 library ignores the open field and reports none)
OK, E_CAPACITY = 0, -7
OPEN_SPEC, OPEN_NO_CHECK = 1, 2
L_PAGED, L_DEVICE_ARGMAX, L_SPEC = 1, 2, 4
W_GREEDY, W_SPEC, W_FEATS = 1, 2, 4
S_HOST_LOGITS = 1
KV_DTYPE = {0: "f16", 1: "fp8"}
P_ = C.POINTER


class OpenParams(C.Structure):
    _fields_ = [("size", C.c_uint32), ("flags", C.c_uint32), ("pack_dir", C.c_char_p), ("weights", P_(C.c_char_p)),
                ("n_weights", C.c_uint32), ("device", C.c_int32), ("buckets", P_(C.c_char_p)),
                ("n_buckets", C.c_uint32), ("pool_blocks", C.c_uint32), ("spec_policy", C.c_char_p),
                ("scratch_dir", C.c_char_p), ("state_pages", C.c_uint32)]


class Layout(C.Structure):
    _fields_ = [("size", C.c_uint32)] + [(k, C.c_uint32) for k in (
        "flags", "vocab", "max_cols", "max_seqs", "max_ctx", "block_size", "num_blocks", "layers", "kv_heads",
        "head_dim", "kv_dtype", "spec_width", "feat_width")] + [(k, C.c_uint64) for k in (
        "kv_bytes", "state_bytes", "device_bytes")] + [("state_pages", C.c_uint32), ("pad1", C.c_uint32)]


class Work(C.Structure):
    _fields_ = [("size", C.c_uint32), ("flags", C.c_uint32), ("sid", C.c_uint64), ("pos", C.c_uint32),
                ("n_tokens", C.c_uint32), ("tokens", P_(C.c_int32)), ("parents", P_(C.c_int32)),
                ("blocks", P_(C.c_uint32)), ("n_blocks", C.c_uint32), ("want", C.c_uint32)]


class StepIn(C.Structure):
    _fields_ = [("size", C.c_uint32), ("flags", C.c_uint32), ("work", P_(P_(Work))), ("n_work", C.c_uint32),
                ("pad", C.c_uint32)]


class StepOut(C.Structure):
    _fields_ = [("size", C.c_uint32), ("n_rows", C.c_uint32), ("tokens", P_(C.c_int32)), ("tokens_cap", C.c_uint32),
                ("pad0", C.c_uint32), ("counts", P_(C.c_uint32)), ("logits", C.c_void_p), ("logits_cap", C.c_uint64),
                ("logits_dev", C.c_uint64), ("feats", C.c_void_p), ("feats_cap", C.c_uint64)]


def bind(path):
    """The library at path with its gc.h signatures; refuses a different ABI major or an older minor."""
    L = C.CDLL(path)
    sig = dict(gc_abi_version=(C.c_uint32, []), gc_open=(C.c_int, [P_(OpenParams), P_(C.c_void_p)]),
               gc_layout_get=(C.c_int, [C.c_void_p, P_(Layout)]), gc_step=(C.c_int, [C.c_void_p, P_(StepIn), P_(StepOut)]),
               gc_commit=(C.c_int, [C.c_void_p, C.c_uint64, C.c_uint32, P_(C.c_uint32), C.c_uint32]),
               gc_copy_blocks=(C.c_int, [C.c_void_p, P_(C.c_uint32), P_(C.c_uint32), C.c_uint32]),
               gc_release=(C.c_int, [C.c_void_p, C.c_uint64]), gc_last_error=(C.c_char_p, [C.c_void_p]),
               gc_close=(None, [C.c_void_p]))
    for n, (res, args) in sig.items():
        f = getattr(L, n)
        f.restype, f.argtypes = res, args
    v = L.gc_abi_version()
    if v >> 16 != ABI_MAJOR or (v & 0xFFFF) < ABI_MINOR:
        raise P.PackError(f"{path}: C ABI {v >> 16}.{v & 0xFFFF}, this engine needs {ABI_MAJOR}.{ABI_MINOR}+")
    return L


def arr(ctype, xs):
    return (ctype * len(xs))(*xs)


class GcBackend:
    def __init__(self, pack, weights, *, B=8, ctx=None, contract=None, sku=None, pool_blocks=None,
                 drafter_weights=None, policy=None, trusted=None, device=-1, state_pages=4,
                 numerics="deterministic", act="auto", kv="auto"):
        """state_pages: snapshot pages a page pack (no paged KV) gets for its prefix cache; a paged pack takes none."""
        self.h = self.stage = None
        self.stage = P.stage(pack, trusted=trusted)
        try:
            self._open(weights, B, ctx, contract, sku, pool_blocks, drafter_weights, policy, device, state_pages, (numerics, act, kv))
        except BaseException:
            self.close()
            raise

    def _open(self, weights, B, ctx, contract, sku, pool_blocks, drafter_weights, policy, device, state_pages, req):
        m = self.m = self.stage.manifest
        dev = None if sku else P.device()
        self.spec = drafter_weights is not None
        ws = [weights] if isinstance(weights, str) else list(weights)
        if self.spec:
            b, db, pol = P.spec_pair(m, sku=sku, dev=dev, contract=contract, ctx=ctx)
            ids, used, pol_s = [b["id"], db["id"]], [b.get("weights", 0), db.get("weights", 1)], policy or self._policy(pol)
            allw = [None] * len(m["model"]["weights"])
            allw[used[0]], allw[used[1]] = ws[0], drafter_weights
            self.bucket, lad = b, [b]
        else:
            # one sequence: the pack's single stream bucket when it has one (a token a launch, its own numerics contract);
            # else, and for B > 1, the smallest batch bucket
            want = P.wanted(*req, single=B == 1)
            P.need_bucket(m, sku, want, f"numerics={req[0]} act={req[1]} kv={req[2]} for {B} sequence(s)")
            b = (B == 1 and P.single_stream(m, sku=sku, dev=dev, contract=contract, ctx=ctx, want=want)) or \
                P.select(m, sku=sku, dev=dev, contract=contract, B=max(B, 2), ctx=ctx, want=want)
            lad = P.ladder(m, b)
            pf = P.prefill(m, b)   # a prompt chunk of W tokens a launch (the runtime ignores it on a paged pack)
            ids, pol_s, pol = [x["id"] for x in lad] + ([pf["id"]] if pf else []), None, None
            used, allw = [x.get("weights", 0) for x in lad + ([pf] if pf else [])] + [b.get("weights", 0)], ws
            self.bucket = b
        P.check_weights(m, allw, P.memo_dir(), used)
        top = lad[-1]
        libpath = os.path.join(P.materialize(self.stage, top), top["lib"])
        self.L = bind(libpath)
        paths = [w for w in allw if w is not None]
        keep = [arr(C.c_char_p, [os.fsencode(w) for w in paths]), arr(C.c_char_p, [i.encode() for i in ids])]
        p = OpenParams(C.sizeof(OpenParams), OPEN_NO_CHECK | (OPEN_SPEC if self.spec else 0), os.fsencode(self.stage.dir),
                       keep[0], len(paths), device, keep[1], len(ids), pool_blocks or 0,
                       pol_s.encode() if pol_s else None, None, state_pages)
        h = C.c_void_p()
        rc = self.L.gc_open(C.byref(p), C.byref(h))
        if rc:
            raise P.PackError(f"gc_open {rc}: {(self.L.gc_last_error(None) or b'').decode()}")
        self.h = h.value
        lo = Layout(size=C.sizeof(Layout))
        self._ck(self.L.gc_layout_get(h, C.byref(lo)))
        self.layout = lo
        self.paged = bool(lo.flags & L_PAGED)
        self.state_pages = lo.state_pages   # snapshot pages: ids num_blocks ..., only copy_blocks addresses them
        self.prefix_cache = not self.spec and (self.paged or self.state_pages > 0)
        self.vocab_size, self.max_seqs, self.max_len, self.feat_width = lo.vocab, lo.max_seqs, lo.max_ctx, lo.feat_width
        self.geometry = KVGeometry(lo.block_size, lo.num_blocks, lo.layers, lo.kv_heads, lo.head_dim,
                                   KV_DTYPE.get(lo.kv_dtype, "f16"))
        self.device_argmax = bool(lo.flags & L_DEVICE_ARGMAX)
        if lo.flags & L_SPEC:
            self.spec_width = lo.spec_width
            self.max_len = lo.max_ctx - 2 * lo.spec_width - 2
        self.caps = caps_of_bucket(b, dict(kind=pol["kind"]) if self.spec else None)
        self.numerics = numerics_of(m, b)
        self.stats = dict(steps=0)

    @staticmethod
    def _policy(pol):
        kw = ",".join(f"{k}={v}" for k, v in pol.items() if k != "kind")
        return f"ar:{pol['K']}" if pol["kind"] == "ar" else f"{pol['kind']}:{kw}"

    def _ck(self, rc):
        if rc < 0:
            raise RuntimeError(f"gc call failed ({rc}): {(self.L.gc_last_error(self.h) or b'').decode()}")

    def step(self, work):
        lo, n = self.layout, len(work)
        greedy = self.device_argmax and all(w.greedy for w in work)
        keep, ws = [], []
        for w in work:
            fl = (W_GREEDY if greedy else 0) | (W_SPEC if w.spec else 0) | (W_FEATS if w.feats else 0)
            toks, blk = arr(C.c_int32, w.tokens), arr(C.c_uint32, w.blocks)
            par = arr(C.c_int32, w.parents) if w.parents is not None else None
            keep += [toks, blk, par]
            ws.append(Work(C.sizeof(Work), fl, w.sid, w.pos, len(w.tokens), toks, par, blk,
                           len(w.blocks), w.want))
        pw = arr(P_(Work), [C.pointer(x) for x in ws])
        rows = sum(max(w.want, lo.spec_width if w.spec else 0) for w in work)
        if self.spec:
            rows = max(rows, lo.spec_width)   # a pair handle wants room for a whole round on every step, a prompt chunk included
        out = StepOut(size=C.sizeof(StepOut))
        counts = (C.c_uint32 * n)()
        if greedy:
            tok = (C.c_int32 * max(rows, 1))()
            out.tokens, out.tokens_cap, out.counts = tok, rows, counts
        else:
            rows = sum(w.want for w in work)
            lg = torch.empty(rows, self.vocab_size, dtype=torch.float32)
            out.logits, out.logits_cap = lg.data_ptr(), lg.numel()
        nf = sum(len(w.tokens) for w in work if w.feats)
        if nf:
            ft = torch.empty(nf, self.feat_width, dtype=torch.float32)
            out.feats, out.feats_cap = ft.data_ptr(), ft.numel()
        self._ck(self.L.gc_step(self.h, C.byref(StepIn(C.sizeof(StepIn), 0 if greedy else S_HOST_LOGITS, pw, n, 0)),
                                C.byref(out)))
        self.stats["steps"] += 1
        if greedy:
            res, o = [], 0
            for c in counts:
                res.append(list(tok[o:o + c]))
                o += c
        else:
            res = lg[:out.n_rows]
        return (res, ft) if nf else res

    def commit(self, sid, base, path):
        self._ck(self.L.gc_commit(self.h, sid, base, arr(C.c_uint32, path), len(path)))

    def copy_blocks(self, pairs):
        self._ck(self.L.gc_copy_blocks(self.h, arr(C.c_uint32, [s for s, _ in pairs]),
                                       arr(C.c_uint32, [d for _, d in pairs]), len(pairs)))

    def release(self, sid):
        self._ck(self.L.gc_release(self.h, sid))

    def close(self):
        if self.h:
            self.L.gc_close(self.h)
            self.h = None
        if self.stage:
            self.stage.close()

    __del__ = close
