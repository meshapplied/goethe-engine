"""Embedding models (model.type embedding): one encoder pass per sequence, pooled in the graph, no KV, no decode loop.

EmbedRunner   the serving loop: queued requests are packed into launches of at most the bucket's rows (layout max_cols tokens) and
              max_seqs sequences, run through backend.embed and returned as float32 rows. A row never depends on its batch (the
              backend's contract), so packing changes throughput only.
GcEmbed       backend over a pack's encode bucket (kind "encode"): its library through gc.h 1.2 gc_embed (GC_L_EMBED).
RefEmbed      CPU reference: ref.RefModel's layers without a causal mask, pooled (mean | cls | last) and optionally L2 normalized."""
import ctypes as C
import math
import os
import queue
import threading

import torch

from . import pack as P
from .gc_c import OK, E_CAPACITY, OPEN_NO_CHECK, OpenParams, StepIn, Work, arr
from .ref import RefModel, rms

L_EMBED = 1 << 4
S_HOST_LOGITS, S_NORMALIZE = 1, 2
POOLS = {0: "none", 1: "mean", 2: "cls", 3: "last"}
P_ = C.POINTER


class Layout12(C.Structure):
    """gc.h 1.2 gc_layout: the 1.1 fields, then embd_dim, embd_pool, enc_frames, enc_feat."""
    _fields_ = [("size", C.c_uint32)] + [(k, C.c_uint32) for k in (
        "flags", "vocab", "max_cols", "max_seqs", "max_ctx", "block_size", "num_blocks", "layers", "kv_heads",
        "head_dim", "kv_dtype", "spec_width", "feat_width")] + [(k, C.c_uint64) for k in (
        "kv_bytes", "state_bytes", "device_bytes")] + [(k, C.c_uint32) for k in (
        "state_pages", "pad1", "embd_dim", "embd_pool", "enc_frames", "enc_feat")]


class EmbedOut(C.Structure):
    _fields_ = [("size", C.c_uint32), ("n_rows", C.c_uint32), ("rows", P_(C.c_float)), ("rows_cap", C.c_uint64)]


def normalize(x):
    """L2 rows as llama.cpp common_embd_normalize(embd_norm 2): float products summed in double, scaled by float(1 / norm)."""
    out = torch.empty_like(x)
    for i, r in enumerate(x.float().numpy()):
        s = 0.0
        for v in (r * r).tolist():          # float32 products, summed in order in double
            s += v
        s = math.sqrt(s)
        out[i] = x[i] * torch.tensor(1.0 / s if s > 0 else 0.0, dtype=torch.float32)
    return out


def encode_buckets(m):
    """The encode buckets of a manifest (kind "encode")."""
    return [b for b in m["buckets"] if P.axes(b)[1] == "encode"]


class GcEmbed:
    """An encode bucket of a gc-c pack: gc_open on that one bucket, gc_embed per launch."""

    def __init__(self, pack, weights, *, device=-1, contract=None, sku=None, trusted=None, store=None, numerics="invariant", **_):
        self.h = self.stage = None
        self.stage = P.stage(pack, trusted=trusted, defer=True, store=store)
        try:
            self._open(weights, device, contract, sku, numerics)
        except BaseException:
            self.close()
            raise

    def _open(self, weights, device, contract, sku, numerics):
        ws = [weights] if isinstance(weights, str) else list(weights)
        m0 = self.stage.manifest
        m = P.restrict(P.for_variant(m0, P.pick_variant(m0, ws)))
        m = self.m = dict(m, buckets=encode_buckets(m))
        P.need(m["buckets"], "no encode bucket in this pack (model.type embedding needs one)")
        dev = None if sku else P.device(None if device < 0 else device)
        want = P.wanted(numerics, single=True)
        b = self.bucket = P.select(m, sku=sku, dev=dev, contract=contract, verify=True, want=want, kind="encode")
        P.check_weights(m, ws)
        self.stage.ensure([b])
        lib = os.path.join(P.materialize(self.stage, b), b["lib"])
        L = self.L = C.CDLL(lib)
        for n, res, args in (("gc_abi_version", C.c_uint32, []), ("gc_open", C.c_int, [P_(OpenParams), P_(C.c_void_p)]),
                             ("gc_layout_get", C.c_int, [C.c_void_p, P_(Layout12)]),
                             ("gc_embed", C.c_int, [C.c_void_p, P_(StepIn), P_(EmbedOut)]),
                             ("gc_release", C.c_int, [C.c_void_p, C.c_uint64]), ("gc_last_error", C.c_char_p, [C.c_void_p]),
                             ("gc_close", None, [C.c_void_p])):
            f = getattr(L, n)
            f.restype, f.argtypes = res, args
        v = L.gc_abi_version()
        P.need(v >> 16 == 1 and (v & 0xFFFF) >= 2, f"{lib}: C ABI {v >> 16}.{v & 0xFFFF}, gc_embed needs 1.2+")
        keep = [arr(C.c_char_p, [os.fsencode(w) for w in ws]), arr(C.c_char_p, [b["id"].encode()])]
        p = OpenParams(C.sizeof(OpenParams), OPEN_NO_CHECK, os.fsencode(self.stage.dir), keep[0], len(ws), device, keep[1], 1, 0,
                       None, None, 0)
        h = C.c_void_p()
        rc = L.gc_open(C.byref(p), C.byref(h))
        if rc:
            raise P.PackError(f"gc_open {rc}: {(L.gc_last_error(None) or b'').decode()}")
        self.h = h.value
        lo = Layout12(size=C.sizeof(Layout12))
        self._ck(L.gc_layout_get(self.h, C.byref(lo)))
        P.need(lo.flags & L_EMBED, f"bucket {b['id']}: the library does not report GC_L_EMBED")
        self.layout, self.dim, self.pool = lo, lo.embd_dim, POOLS.get(lo.embd_pool, "none")
        self.max_cols, self.max_seqs, self.max_len = lo.max_cols, lo.max_seqs, lo.max_ctx
        self.numerics = dict(contract=b["contract"], determinism=b.get("determinism"))

    def _ck(self, rc):
        if rc < 0:
            raise RuntimeError(f"gc call failed ({rc}): {(self.L.gc_last_error(self.h) or b'').decode()}")

    def embed(self, items, norm=True):
        """float32 [len(items), dim]: the pooled row of each token list (L2 normalized when norm)."""
        if not items:
            return torch.empty(0, self.dim)
        keep = [(C.c_int32 * len(t))(*t) for t in items]
        ws = [Work(C.sizeof(Work), 0, i, 0, len(t), k, None, None, 0, 1) for i, (t, k) in enumerate(zip(items, keep))]
        wp = (P_(Work) * len(ws))(*[C.pointer(w) for w in ws])
        si = StepIn(C.sizeof(StepIn), S_HOST_LOGITS | (S_NORMALIZE if norm else 0), wp, len(ws), 0)
        buf = (C.c_float * (len(items) * self.dim))()
        o = EmbedOut(C.sizeof(EmbedOut), 0, buf, len(buf))
        self._ck(self.L.gc_embed(self.h, C.byref(si), C.byref(o)))
        return torch.frombuffer(bytearray(buf), dtype=torch.float32).view(len(items), self.dim).clone()

    def close(self):
        if self.h:
            self.L.gc_close(self.h)
            self.h = None
        if self.stage:
            self.stage.close()
            self.stage = None

    __del__ = close


class RefEmbed:
    """CPU reference: RefModel's attention and MLP layers over the whole sequence with no causal mask (an encoder), the final
    norm, pooled; one sequence at a time, so a row never depends on the batch."""

    def __init__(self, model=None, pool="mean", max_cols=64, max_seqs=8, max_len=64):
        self.M, self.pool = model or RefModel(), pool
        self.dim, self.max_cols, self.max_seqs, self.max_len = self.M.d, max_cols, max_seqs, max_len
        self.numerics = dict(contract="ref", determinism="exact")

    def rows(self, t):
        M, T = self.M, len(t)
        x = M.emb[torch.tensor(t)]
        for w in M.layers:
            h = torch.stack([rms(r) for r in x])
            q = torch.stack([M._rope((w["q"] @ h[i]).view(M.H, M.dh), i) for i in range(T)])
            k = torch.stack([M._rope((w["k"] @ h[i]).view(M.KVH, M.dh), i) for i in range(T)])
            v = torch.stack([(w["v"] @ h[i]).view(M.KVH, M.dh) for i in range(T)])
            g = M.H // M.KVH
            sc = torch.einsum("ikgd,jkd->ikgj", q.view(T, M.KVH, g, M.dh), k) / math.sqrt(M.dh)
            a = torch.einsum("ikgj,jkd->ikgd", torch.softmax(sc, -1), v).reshape(T, -1)
            x = x + a @ w["o"].T
            h = torch.stack([rms(r) for r in x])
            x = x + (torch.nn.functional.silu(h @ w["g"].T) * (h @ w["u"].T)) @ w["dn"].T
        return torch.stack([rms(r) for r in x])

    def embed(self, items, norm=True):
        out = []
        for t in items:
            r = self.rows(t)
            out.append(r.mean(0) if self.pool == "mean" else r[0] if self.pool == "cls" else r[-1])
        x = torch.stack(out).float() if out else torch.empty(0, self.dim)
        return normalize(x) if norm else x

    def close(self):
        pass


class EmbedRunner:
    """Packs requests into launches of at most backend.max_cols tokens and backend.max_seqs sequences, in arrival order."""

    def __init__(self, backend, cfg=None, **_):
        self.be = backend
        self.q = queue.Queue()
        self.stats = dict(launches=0, rows=0)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def check(self, tokens):
        P.need(1 <= len(tokens) <= self.be.max_len and len(tokens) <= self.be.max_cols,
               f"input of {len(tokens)} tokens: this model embeds 1 .. {min(self.be.max_len, self.be.max_cols)}")

    def batches(self, items):
        """Launches (lists of indices into items): a new one when the next item would pass the bucket's rows or sequences."""
        out, cur, n = [], [], 0
        for i, t in enumerate(items):
            if cur and (n + len(t) > self.be.max_cols or len(cur) == self.be.max_seqs):
                out.append(cur)
                cur, n = [], 0
            cur.append(i)
            n += len(t)
        return out + ([cur] if cur else [])

    def embed(self, items, norm=True):
        """float32 [len(items), dim], computed now on the calling thread (tests, offline use)."""
        for t in items:
            self.check(t)
        rows = [None] * len(items)
        for b in self.batches(items):
            for i, r in zip(b, self.be.embed([items[i] for i in b], norm)):
                rows[i] = r
            self.stats["launches"] += 1
            self.stats["rows"] += len(b)
        return torch.stack(rows) if rows else torch.empty(0, self.be.dim)

    def submit(self, items, norm=True):
        """Queued for the runner thread, packed with other callers' requests; returns a queue that receives the rows (or an error)."""
        for t in items:
            self.check(t)
        done = queue.Queue(1)
        self.q.put((items, norm, done))
        return done

    def _loop(self):
        while True:
            reqs = [self.q.get()]
            while not self.q.empty():
                reqs.append(self.q.get_nowait())
            for norm in (True, False):
                grp = [r for r in reqs if r[1] == norm]
                if not grp:
                    continue
                flat = [t for r in grp for t in r[0]]
                try:
                    rows = self.embed(flat, norm)
                except Exception as e:      # the error goes to every waiting caller
                    for r in grp:
                        r[2].put(e)
                    continue
                k = 0
                for items, _, done in grp:
                    done.put(rows[k:k + len(items)])
                    k += len(items)

    def numerics(self):
        return self.be.numerics
