"""Backend over a gc-c pack: the pack's libgck.so through the C ABI of abi/gc.h (ctypes, no compiled code here).

The pack is staged into a private directory and verified there (pack.stage); the library is loaded from that copy and
gc_open gets the same directory, so what was verified is what runs. Weights are identified by structure (tensor table and
metadata keys, pack.check_weights), never hashed: GC_OPEN_NO_CHECK tells the runtime the loader made that check. One handle, not thread safe: the engine calls it from one thread. Capabilities come from the
manifest (pack_backend.caps_of_bucket), never from symbols."""
import ctypes as C
import os
import sys

import numpy as np

from . import pack as P
from .backend import KVGeometry, Work as W
from .pack_backend import caps_of_bucket, numerics_of

ABI_MAJOR, ABI_MINOR = 1, 0   # 1.1 adds state_pages (a 1.0 library ignores the open field and reports none)
OK, E_DEVICE, E_CAPACITY = 0, -4, -7
OPEN_SPEC, OPEN_NO_CHECK = 1, 2
L_PAGED, L_DEVICE_ARGMAX, L_SPEC, L_BOUNDED = 1, 2, 4, 8
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
               gc_state_save=(C.c_int, [C.c_void_p, C.c_uint64, C.c_uint32]),
               gc_state_restore=(C.c_int, [C.c_void_p, C.c_uint32, C.c_uint64]),
               gc_release=(C.c_int, [C.c_void_p, C.c_uint64]), gc_last_error=(C.c_char_p, [C.c_void_p]),
               gc_close=(None, [C.c_void_p]))
    for n, (res, args) in sig.items():
        if n.startswith("gc_state_") and not hasattr(L, n):   # 1.1 and only with GC_L_BOUNDED: a 1.0 library has neither
            continue
        f = getattr(L, n)
        f.restype, f.argtypes = res, args
    v = L.gc_abi_version()
    if v >> 16 != ABI_MAJOR or (v & 0xFFFF) < ABI_MINOR:
        raise P.PackError(f"{path}: C ABI {v >> 16}.{v & 0xFFFF}, this engine needs {ABI_MAJOR}.{ABI_MINOR}+")
    return L


def arr(ctype, xs):
    return (ctype * len(xs))(*xs)


def forced(m, dev, bid):
    """(ladder, prefill bucket) that runs manifest bucket bid: a batch bucket with its ladder, a one-column stream bucket alone, a prefill
    bucket beside the narrowest batch ladder of its sku, contract, role, maxctx and revision; refused when bid is not in this branch
    or does not run on the device."""
    by = {b["id"]: b for b in m["buckets"]}
    P.need(bid in by, f"bucket {bid} is not in the branch these weights select ({sorted(by)})")
    b = by[bid]
    if dev is not None:
        why = P.fits(dev, b["requires"], m)
        P.need(not why, f"bucket {bid} does not run on {dev.get('name')}: {why}")
    if b.get("prefill"):
        base = [x for x in m["buckets"] if x["batch"] and all(x[k] == b[k] for k in ("sku", "contract", "role", "maxctx")) and P.rev(x) == P.rev(b)]
        P.need(base, f"prefill bucket {bid} has no batch ladder of its contract beside it")
        return P.ladder(m, min(base, key=lambda x: x["ncol"])), b
    return (P.ladder(m, b) if b["batch"] else [b]), None


class GcBackend:
    def __init__(self, pack, weights, *, B=8, ctx=None, contract=None, sku=None, pool_blocks=None,
                 drafter_weights=None, policy=None, trusted=None, device=-1, state_pages=4,
                 numerics="invariant", act="auto", kv="auto", store=None, plen=None, price=None, allow_broken=False, bucket=None):
        """bucket: open exactly this manifest bucket id (verification: a batch bucket with its ladder, a stream bucket alone, a prefill
        bucket beside the narrowest batch ladder of its contract) instead of selecting. store: the pack 4 blob store (P.open_store). plen: the prompt length the handle serves (picks among priced prefill buckets). state_pages: snapshot pages a page pack (no paged KV) or a bounded paged pack gets for its prefix cache."""
        self.h = self.stage = None
        self.note, self.launch_ws, failed = None, 0, []
        self.stage = P.stage(pack, trusted=trusted, defer=True, store=store)
        try:
            while True:
                self._open(weights, B, ctx, contract, sku, pool_blocks, drafter_weights, policy, device, state_pages, (numerics, act, kv), plen,
                           price, allow_broken, bucket)
                if self.spec:
                    break
                top = self.ladder[-1]["ncol"]
                try:
                    self._warm()
                except P.OutOfMemory as e:
                    # the ladder's first launches need device memory the manifest's resident does not count (local memory, workspace):
                    # the next narrower ladder (never a forced bucket: a gate opens exactly one)
                    nar = [x["ncol"] for x in self.m["buckets"] if x["batch"] and x["role"] == "target" and x["sku"] == self.bucket["sku"] and x["ncol"] < top]
                    if bucket is not None or not nar:
                        raise
                    self._say(f"the {top}-column ladder cannot launch ({e})")
                    self.L.gc_close(self.h)
                    self.h, B = None, max(nar)
                    failed.append(top)
                    continue
                if failed and not self.launch_ws and self.warm_bytes:
                    # the launch memory measured on this narrower ladder: the wider one once more, its KV pool leaving room for it
                    self.launch_ws = self.warm_bytes
                    self._say(f"launch memory measured {self.launch_ws / 1e9:.2f} GB on the {top}-column ladder; retrying the {max(failed)}-column one with room for it")
                    self.L.gc_close(self.h)
                    self.h, B = None, max(failed)
                    continue
                break
        except BaseException:
            self.close()
            raise

    def _open(self, weights, B, ctx, contract, sku, pool_blocks, drafter_weights, policy, device, state_pages, req, plen=None, price=None, allow_broken=False, bucket=None):
        lim = P.ctx_limit(self.stage.manifest, ctx)       # a model limit refuses a longer context before anything loads
        self.device = device
        self.spec = drafter_weights is not None
        ws = [weights] if isinstance(weights, str) else list(weights)
        dws = None if drafter_weights is None else [drafter_weights] if isinstance(drafter_weights, str) else list(drafter_weights)
        free, got = P.device_free(device), {}

        def pick(m, dev):
            if self.spec:
                b, db, pol = P.spec_pair(m, sku=sku, dev=dev, contract=contract, ctx=ctx)
                got.update(ids=[b["id"], db["id"]], pol=pol, lad=[b], pf=None)
                return b, [b, db], b
            if bucket is not None:
                lad, pf = forced(m, dev, bucket)
                P.fit(lad + ([pf] if pf else []), state_pages, free, pool_blocks, self.launch_ws + P.overhead(m, lad[0]["sku"]), lad[0]["id"])
                load = lad + ([pf] if pf else [])
                got.update(ids=[x["id"] for x in load], pol=None, lad=lad, pf=pf)
                return (pf or lad[-1]), load, lad[-1]
            # one sequence: the pack's single stream bucket when it has one (a token a launch, its own numerics contract);
            # else, and for B > 1, the smallest batch bucket
            want = P.wanted(*req, single=B == 1)
            P.need_bucket(m, sku, want, f"numerics={req[0]} act={req[1]} kv={req[2]} for {B} sequence(s)")

            def at(cx, w):
                b = (w == 1 and P.single_stream(m, sku=sku, dev=dev, contract=contract, ctx=cx, want=want)) or \
                    P.select(m, sku=sku, dev=dev, contract=contract, B=max(w, 2), ctx=cx, want=want)
                if w == 1:
                    b = P.with_prefill(m, b, sku=sku, dev=dev, contract=contract, ctx=cx, want=want)
                return b, P.ladder(m, b), P.prefill(m, b, plen)   # prefill: a prompt chunk of W tokens a launch (ignored on a paged pack)
            # the requested batch, then the narrower batch buckets: the widest whose weights, slots, pool and scratch fit is opened
            widths = [B] + sorted({x["ncol"] for x in m["buckets"] if x["batch"] and x["role"] == "target" and 1 < x["ncol"] < B}, reverse=True)
            b, lad, pf = P.open_fit(at, m, ctx, lambda bl: P.fit(bl, state_pages, free, pool_blocks, self.launch_ws + P.overhead(m, bl[0]["sku"]), bl[0]["id"]), widths)
            load = lad + ([pf] if pf else [])
            got.update(ids=[x["id"] for x in load], pol=None, lad=lad, pf=pf)
            return b, load, lad[-1]

        br = P.open_branch(self.stage, ws, pick, drafter_files=dws, sku=sku, ordinal=device, price=price, allow_broken=allow_broken)
        m, b, lad, pf, ids, pol = br.m, br.bucket, got["lad"], got["pf"], got["ids"], got["pol"]
        self.m, self.bucket, self.ladder, self.pf = m, b, lad, pf
        pol_s = (policy or self._policy(pol)) if self.spec else None
        if not self.spec:   # weights, slots and the smallest pool must fit the device; no pool_blocks: the largest that fits
            pool_blocks, self.resident = P.fit(lad + ([pf] if pf else []), state_pages, free, pool_blocks, self.launch_ws + P.overhead(m, b["sku"]), b["id"])
        libpath = br.lib
        self.L = bind(libpath)
        paths = ws + ([None] + dws if dws else [])     # groups: the target's files, a NULL, a drafter's (the runtime finds tensors by name)
        keep = [arr(C.c_char_p, [None if w is None else os.fsencode(w) for w in paths]), arr(C.c_char_p, [i.encode() for i in ids])]
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
        self.bounded = bool(lo.flags & L_BOUNDED)   # window ring or recurrent state in the sequence's slot: a prefix is not shareable by block
        self.prefix_cache = not self.spec and ((self.paged and not self.bounded) or self.state_pages > 0)
        self.vocab_size, self.max_seqs, self.max_len, self.feat_width = lo.vocab, lo.max_seqs, lo.max_ctx, lo.feat_width
        self.geometry = KVGeometry(lo.block_size, lo.num_blocks, lo.layers, lo.kv_heads, lo.head_dim,
                                   KV_DTYPE.get(lo.kv_dtype, "f16"))
        self.device_argmax = bool(lo.flags & L_DEVICE_ARGMAX)
        if lo.flags & L_SPEC:
            self.spec_width = lo.spec_width
            self.max_len = lo.max_ctx - 2 * lo.spec_width - 2
        if lim:
            self.max_len, self.limit = min(self.max_len, lim[0]), lim   # the engine refuses longer sequences (max_model_len)
        self.caps = caps_of_bucket(b, dict(kind=pol["kind"]) if self.spec else None)
        self.numerics = numerics_of(m, b)
        self.stats = dict(steps=0)

    @staticmethod
    def _policy(pol):
        kw = ",".join(f"{k}={v}" for k, v in pol.items() if k != "kind")
        return f"ar:{pol['K']}" if pol["kind"] == "ar" else f"{pol['kind']}:{kw}"

    def _ck(self, rc):
        if rc < 0:
            why = (self.L.gc_last_error(self.h) or b'').decode()
            if rc == E_DEVICE and "out of memory" in why:
                raise P.OutOfMemory(f"gc call failed ({rc}): {why}")
            raise RuntimeError(f"gc call failed ({rc}): {why}")

    def _say(self, note):
        self.note = note
        print(f"goethe-engine: {note}", file=sys.stderr, flush=True)

    def _warm(self):
        """One step of every loaded bucket at the first and at the last positions of the context (k one-token sequences per ladder width
        k; a 2-token prompt through a loaded prefill chain), so a launch that cannot get its device memory (local memory, workspace: not
        in the manifest's resident; a long-position step variant) fails at open, not on a request. warm_bytes: the memory they took.
        Only an out-of-memory refusal counts at the last positions (a runtime may refuse a fresh sequence there for other reasons)."""
        f0, n, L = P.device_free(self.device), min(self.max_seqs, self.geometry.num_blocks), self.layout.max_ctx
        bs, nb = self.geometry.block_size, self.geometry.num_blocks
        full = list(range(min(nb, -(-L // bs))))              # one context's block table, shared: the rows are scratch
        blocks = (lambda i: full) if self.paged else (lambda i: [i])
        for pos in (0, L - 3):
            steps = [[W(i + 1, [0], pos, blocks(i), 1) for i in range(min(k, n))] for k in sorted({x["ncol"] for x in self.ladder})]
            if self.pf:
                steps.append([W(1, [0, 0], pos, blocks(0), 1)])
            for ws in steps:
                try:
                    self.step(ws)
                except RuntimeError:
                    if not pos:
                        raise
                finally:
                    for w in ws:
                        try:
                            self.release(w.sid)
                        except (RuntimeError, P.OutOfMemory):
                            pass
        f1 = P.device_free(self.device)
        self.warm_bytes = f0 - f1 if f0 is not None and f1 is not None else None
        self.stats["steps"] = 0

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
            lg = np.empty((rows, self.vocab_size), np.float32)
            out.logits, out.logits_cap = lg.ctypes.data, lg.size
        nf = sum(len(w.tokens) for w in work if w.feats)
        if nf:
            ft = np.empty((nf, self.feat_width), np.float32)
            out.feats, out.feats_cap = ft.ctypes.data, ft.size
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

    def save_state(self, sid, snap):
        self._ck(self.L.gc_state_save(self.h, sid, snap))

    def restore_state(self, snap, sid):
        self._ck(self.L.gc_state_restore(self.h, snap, sid))

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
