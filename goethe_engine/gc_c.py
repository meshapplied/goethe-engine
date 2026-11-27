"""Backend over a gc-c pack: the pack's libgck.so through the C ABI of abi/gc.h (ctypes, no compiled code here).

The pack is staged into a private directory and verified there (pack.stage); the library is loaded from that copy and
gc_open gets the same directory, so what was verified is what runs. Weights are identified by structure (tensor table and
metadata keys, pack.check_weights), never hashed: GC_OPEN_NO_CHECK tells the runtime the loader made that check. One handle, not thread safe: the engine calls it from one thread. Capabilities come from the
manifest (pack_backend.caps_of_bucket), never from symbols."""
import array
import ctypes as C
import os
import sys

import numpy as np

from . import pack as P
from .backend import KVGeometry, Work as W
from .pack_backend import caps_of_bucket, numerics_of

ABI_MAJOR, ABI_MINOR = 1, 0   # 1.1 adds state_pages, 1.2 embed/seq2seq, 1.3 device sample/bitmask/async/flat
OK, E_DEVICE, E_CAPACITY = 0, -4, -7
OPEN_SPEC, OPEN_NO_CHECK = 1, 2
L_PAGED, L_DEVICE_ARGMAX, L_SPEC, L_BOUNDED = 1, 2, 4, 8
L_DEVICE_SAMPLE, L_DEVICE_BITMASK, L_ASYNC, L_FLAT = 1 << 6, 1 << 7, 1 << 8, 1 << 9
W_GREEDY, W_SPEC, W_FEATS, W_SAMPLE, W_BITMASK, W_PREV_TOKEN = 1, 2, 4, 8, 16, 32
S_HOST_LOGITS, S_FLAT = 1, 4
KV_DTYPE = {0: "f16", 1: "fp8"}
P_ = C.POINTER
_P_I32, _P_U32, _P_U64, _P_F32 = P_(C.c_int32), P_(C.c_uint32), P_(C.c_uint64), P_(C.c_float)
_I32 = "i" if array.array("i").itemsize == 4 else "l"
_U32 = "I" if array.array("I").itemsize == 4 else "L"


class OpenParams(C.Structure):
    _fields_ = [("size", C.c_uint32), ("flags", C.c_uint32), ("pack_dir", C.c_char_p), ("weights", P_(C.c_char_p)),
                ("n_weights", C.c_uint32), ("device", C.c_int32), ("buckets", P_(C.c_char_p)),
                ("n_buckets", C.c_uint32), ("pool_blocks", C.c_uint32), ("spec_policy", C.c_char_p),
                ("scratch_dir", C.c_char_p), ("state_pages", C.c_uint32)]


class Layout(C.Structure):
    _fields_ = [("size", C.c_uint32)] + [(k, C.c_uint32) for k in (
        "flags", "vocab", "max_cols", "max_seqs", "max_ctx", "block_size", "num_blocks", "layers", "kv_heads",
        "head_dim", "kv_dtype", "spec_width", "feat_width")] + [(k, C.c_uint64) for k in (
        "kv_bytes", "state_bytes", "device_bytes")] + [
        ("state_pages", C.c_uint32), ("pad1", C.c_uint32),
        ("embd_dim", C.c_uint32), ("embd_pool", C.c_uint32), ("enc_frames", C.c_uint32), ("enc_feat", C.c_uint32),
        ("n_graph_buckets", C.c_uint32), ("graph_buckets", C.c_uint32 * 16), ("pad2", C.c_uint32)]


class Work(C.Structure):
    _fields_ = [("size", C.c_uint32), ("flags", C.c_uint32), ("sid", C.c_uint64), ("pos", C.c_uint32),
                ("n_tokens", C.c_uint32), ("tokens", _P_I32), ("parents", _P_I32),
                ("blocks", _P_U32), ("n_blocks", C.c_uint32), ("want", C.c_uint32),
                ("temperature", C.c_float), ("top_p", C.c_float), ("top_k", C.c_uint32),
                ("bitmask_words", C.c_uint32), ("seed", C.c_uint64), ("step_idx", C.c_uint64),
                ("bitmask", _P_U32)]


class BatchFlat(C.Structure):
    _fields_ = [("size", C.c_uint32), ("n_seqs", C.c_uint32), ("total_tokens", C.c_uint32),
                ("total_blocks", C.c_uint32), ("bitmask_words", C.c_uint32), ("pad0", C.c_uint32),
                ("sids", _P_U64), ("flags", _P_U32), ("pos", _P_U32), ("want", _P_U32),
                ("tok_offsets", _P_U32), ("tokens", _P_I32), ("parents", _P_I32),
                ("blk_offsets", _P_U32), ("blocks", _P_U32),
                ("temperature", _P_F32), ("top_p", _P_F32), ("top_k", _P_U32),
                ("seed", _P_U64), ("step_idx", _P_U64), ("bitmasks", _P_U32)]


class StepIn(C.Structure):
    _fields_ = [("size", C.c_uint32), ("flags", C.c_uint32), ("work", P_(P_(Work))), ("n_work", C.c_uint32),
                ("pad", C.c_uint32), ("pad_cols", C.c_uint32), ("pad1", C.c_uint32), ("flat", P_(BatchFlat))]


class StepOut(C.Structure):
    _fields_ = [("size", C.c_uint32), ("n_rows", C.c_uint32), ("tokens", _P_I32), ("tokens_cap", C.c_uint32),
                ("pad0", C.c_uint32), ("counts", _P_U32), ("logits", C.c_void_p), ("logits_cap", C.c_uint64),
                ("logits_dev", C.c_uint64), ("feats", C.c_void_p), ("feats_cap", C.c_uint64)]


_P_WORK = P_(Work)
_WORK_SZ, _FLAT_SZ, _STEP_IN_SZ, _STEP_OUT_SZ = C.sizeof(Work), C.sizeof(BatchFlat), C.sizeof(StepIn), C.sizeof(StepOut)


class _StepBuf:
    """Reusable ctypes/CSR input + output staging slot (2 slots enable ABI 1.3 async overlap)."""

    def __init__(self):
        self.ws_cap = self.cnt_cap = self.tok_cap = self.f_seq_cap = self.f_tok_cap = self.f_blk_cap = 0
        self.ws = self.pw = self.counts = self.tok_out = None
        self.wi, self.tok_in, self.par_in, self.has_par, self.bm_keep = [], [], [], [], []
        self.flat = BatchFlat(size=_FLAT_SZ)
        self.sin = StepIn(size=_STEP_IN_SZ)
        self.out = StepOut(size=_STEP_OUT_SZ)
        self.sin_ref, self.out_ref, self.flat_ref = C.byref(self.sin), C.byref(self.out), C.pointer(self.flat)
        self.f_sids = self.f_flags = self.f_pos = self.f_want = None
        self.f_tok_off = self.f_blk_off = self.f_tokens = self.f_parents = self.f_blocks = None
        self.f_temp = self.f_top_p = self.f_top_k = self.f_seed = self.f_step_idx = None
        self.bm_buf = self.lg = self.ft = None
        self.blk_key = None
        self.one_sid = self.one_bv = -1
        self.one_ar = self.one_blks = None
        self.in_flight = False
        self.meta = None

    def grow_counts(self, n):
        cap = max(n, self.cnt_cap * 2, 8)
        self.counts = (C.c_uint32 * cap)()
        self.out.counts = self.counts
        self.cnt_cap = cap

    def grow_ws(self, n):
        cap = max(n, self.ws_cap * 2, 8)
        ws = (Work * cap)()
        pw = (_P_WORK * cap)()
        wi, tok_in, par_in = [], [], []
        for i in range(cap):
            w = ws[i]
            w.size = _WORK_SZ
            tb = (C.c_int32 * 4)()
            w.tokens = tb
            wi.append(w)
            tok_in.append(tb)
            par_in.append(None)
            pw[i] = C.pointer(w)
        self.ws, self.pw, self.ws_cap = ws, pw, cap
        self.wi, self.tok_in, self.par_in, self.has_par = wi, tok_in, par_in, [False] * cap
        self.sin.work = pw
        if cap > self.cnt_cap:
            self.grow_counts(cap)

    def grow_flat_seqs(self, n):
        cap = max(n, self.f_seq_cap * 2, 8)
        self.f_seq_cap = cap
        self.f_sids = (C.c_uint64 * cap)()
        self.f_flags = (C.c_uint32 * cap)()
        self.f_pos = (C.c_uint32 * cap)()
        self.f_want = (C.c_uint32 * cap)()
        self.f_tok_off = (C.c_uint32 * (cap + 1))()
        self.f_blk_off = (C.c_uint32 * (cap + 1))()
        self.f_temp = (C.c_float * cap)()
        self.f_top_p = (C.c_float * cap)()
        self.f_top_k = (C.c_uint32 * cap)()
        self.f_seed = (C.c_uint64 * cap)()
        self.f_step_idx = (C.c_uint64 * cap)()
        fl = self.flat
        fl.sids, fl.flags, fl.pos, fl.want = self.f_sids, self.f_flags, self.f_pos, self.f_want
        fl.tok_offsets, fl.blk_offsets = self.f_tok_off, self.f_blk_off
        if cap > self.cnt_cap:
            self.grow_counts(cap)
        self.blk_key = None

    def grow_flat_toks(self, nt):
        cap = max(nt, self.f_tok_cap * 2, 16)
        self.f_tok_cap = cap
        self.f_tokens = (C.c_int32 * cap)()
        self.f_parents = (C.c_int32 * cap)()
        self.flat.tokens = self.f_tokens

    def grow_flat_blks(self, nb):
        cap = max(nb, self.f_blk_cap * 2, 64)
        self.f_blk_cap = cap
        self.f_blocks = (C.c_uint32 * cap)()
        self.flat.blocks = self.f_blocks
        self.blk_key = None


def bind(path):
    """The library at path with its gc.h signatures; refuses a different ABI major or an older minor."""
    L = C.CDLL(path)
    sig = dict(gc_abi_version=(C.c_uint32, []), gc_open=(C.c_int, [P_(OpenParams), P_(C.c_void_p)]),
               gc_layout_get=(C.c_int, [C.c_void_p, P_(Layout)]), gc_step=(C.c_int, [C.c_void_p, P_(StepIn), P_(StepOut)]),
               gc_step_begin=(C.c_int, [C.c_void_p, P_(StepIn), P_(StepOut)]),
               gc_step_wait=(C.c_int, [C.c_void_p, P_(StepOut)]),
               gc_commit=(C.c_int, [C.c_void_p, C.c_uint64, C.c_uint32, _P_U32, C.c_uint32]),
               gc_copy_blocks=(C.c_int, [C.c_void_p, _P_U32, _P_U32, C.c_uint32]),
               gc_state_save=(C.c_int, [C.c_void_p, C.c_uint64, C.c_uint32]),
               gc_state_restore=(C.c_int, [C.c_void_p, C.c_uint32, C.c_uint64]),
               gc_release=(C.c_int, [C.c_void_p, C.c_uint64]), gc_last_error=(C.c_char_p, [C.c_void_p]),
               gc_close=(None, [C.c_void_p]))
    for n, (res, args) in sig.items():
        if (n.startswith("gc_state_") or n.startswith("gc_step_")) and not hasattr(L, n):
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
    def __init__(self, pack, weights, *, B=None, ctx=None, contract=None, sku=None, pool_blocks=None,
                 drafter_weights=None, policy=None, trusted=None, device=-1, state_pages=4,
                 numerics="invariant", act="auto", kv="auto", store=None, plen=None, price=None, allow_broken=False, bucket=None):
        """bucket: open exactly this manifest bucket id (verification: a batch bucket with its ladder, a stream bucket alone, a prefill
        bucket beside the narrowest batch ladder of its contract) instead of selecting. store: the pack 4 blob store (P.open_store). plen: the prompt length the handle serves (picks among priced prefill buckets). state_pages: snapshot pages a page pack (no paged KV) or a bounded paged pack gets for its prefix cache."""
        if B is None:
            B = 1 if drafter_weights is not None else 8
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
        self.spec_name, self.spec_note = None, None
        ws = [weights] if isinstance(weights, str) else list(weights)
        dws = None if drafter_weights is None else [drafter_weights] if isinstance(drafter_weights, str) else list(drafter_weights)
        free, got = P.device_free(device), {}

        def pick(m, dev):
            want = P.wanted(*req, single=B == 1)
            if self.spec:
                # the pack's strategies in its preference order: the first that serves this many lanes and fits the device
                why = []
                for t, d, pol, name in P.spec_pairs(m, sku=sku, dev=dev, contract=contract, ctx=ctx, want=want):
                    try:
                        P.need(B <= P.lanes(t), f"the pair serves {P.lanes(t)} stream(s), {B} lanes asked")
                        P.fit([t, d], 0, free, None, self.launch_ws + P.overhead(m, t["sku"]), t["id"])
                    except P.PackError as e:
                        why.append(f"{name}: {e}")
                        continue
                    self.spec_name = name
                    got.update(ids=[t["id"], d["id"]], pol=pol, lad=[t], pf=None)
                    return t, [t, d], t
                self.spec, self.spec_note = False, "speculative decoding off: " + ("; ".join(why) or "the pack has no pair for this device")
            if bucket is not None:
                lad, pf = forced(m, dev, bucket)
                P.fit(lad + ([pf] if pf else []), state_pages, free, pool_blocks, self.launch_ws + P.overhead(m, lad[0]["sku"]), lad[0]["id"])
                load = lad + ([pf] if pf else [])
                got.update(ids=[x["id"] for x in load], pol=None, lad=lad, pf=pf)
                return (pf or lad[-1]), load, lad[-1]
            # one sequence: the pack's single stream bucket when it has one (a token a launch, its own numerics contract);
            # else, and for B > 1, the smallest batch bucket
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
        if self.spec_note:
            self._say(self.spec_note)
        pol_s = (policy or self._policy(pol)) if self.spec else None
        if not self.spec:   # weights, slots and the smallest pool must fit the device; no pool_blocks: the largest that fits
            pool_blocks, self.resident = P.fit(lad + ([pf] if pf else []), state_pages, free, pool_blocks, self.launch_ws + P.overhead(m, b["sku"]), b["id"])
        libpath = br.lib
        self.L = bind(libpath)
        paths = ws + ([None] + dws if self.spec and dws else [])     # groups: the target's files, a NULL, a drafter's (the runtime finds tensors by name)
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
        self.device_sample = bool(lo.flags & L_DEVICE_SAMPLE)
        self.device_bitmask = bool(lo.flags & L_DEVICE_BITMASK)
        self.async_step = bool(lo.flags & L_ASYNC) and hasattr(self.L, "gc_step_begin") and hasattr(self.L, "gc_step_wait")
        self.spec_prev_token = bool(self.async_step and self.spec and (b.get("spec_caps") or {}).get("prev_token"))
        self.flat_batch = bool(lo.flags & L_FLAT)
        self.graph_buckets = tuple(int(lo.graph_buckets[i]) for i in range(min(int(lo.n_graph_buckets), 16)) if lo.graph_buckets[i])
        if lo.flags & L_SPEC:
            self.spec_width = lo.spec_width
            self.max_len = lo.max_ctx - 2 * lo.spec_width - 2
        if lim:
            self.max_len, self.limit = min(self.max_len, lim[0]), lim   # the engine refuses longer sequences (max_model_len)
        self.caps = caps_of_bucket(b, dict(kind=pol["kind"]) if self.spec else None)
        self.numerics = numerics_of(m, b)
        self.stats = dict(steps=0)
        self._init_bufs()

    def _init_bufs(self):
        self._slots = (_StepBuf(), _StepBuf())
        self._slot_i = 0
        self._blk_cache = {}
        self._bm_words = (self.vocab_size + 31) // 32

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

    def _can_dev_tok(self, work):
        da = getattr(self, "device_argmax", False)
        ds = getattr(self, "device_sample", False)
        db = getattr(self, "device_bitmask", False)
        any_want = False
        for w in work:
            if not w.want:
                continue
            any_want = True
            if w.bitmask is not None and not db:
                return False
            if w.greedy and w.bitmask is None:
                if not da:
                    return False
            elif w.sample:
                if w.temperature > 0:
                    if not ds:
                        return False
                elif not da:
                    return False
            else:
                return False
        return True if any_want else da

    def _item_flags(self, w, dev_tok):
        fl = (W_SPEC if w.spec else 0) | (W_FEATS if w.feats else 0) | (W_PREV_TOKEN if w.prev_token else 0)
        if not dev_tok:
            return fl
        if not w.want or (w.greedy and w.bitmask is None):
            return fl | W_GREEDY
        fl |= W_SAMPLE if w.temperature > 0 else W_GREEDY
        if w.bitmask is not None:
            fl |= W_BITMASK
        return fl

    def _pack_flat(self, slot, work, dev_tok, pad_cols):
        n, spec_w = len(work), self.layout.spec_width
        if n > slot.f_seq_cap:
            slot.grow_flat_seqs(n)
        sids, flags, pos, want = slot.f_sids, slot.f_flags, slot.f_pos, slot.f_want
        tok_off = slot.f_tok_off
        total_toks = rows = nf = 0
        has_par = has_samp = has_bm = False
        for i, w in enumerate(work):
            sids[i] = w.sid
            fl = self._item_flags(w, dev_tok)
            flags[i] = fl
            pos[i] = w.pos
            wi = w.want
            want[i] = wi
            tok_off[i] = total_toks
            nt = len(w.tokens)
            total_toks += nt
            if w.parents is not None:
                has_par = True
            if fl & W_SAMPLE:
                has_samp = True
            if fl & W_BITMASK:
                has_bm = True
            rows += (spec_w if dev_tok and w.spec and spec_w > wi else wi)
            if w.feats:
                nf += nt
        tok_off[n] = total_toks
        if total_toks > slot.f_tok_cap:
            slot.grow_flat_toks(total_toks)
        f_toks = slot.f_tokens
        if total_toks == n and not has_par:
            for i, w in enumerate(work):
                f_toks[i] = w.tokens[0]
        else:
            f_par = slot.f_parents
            off = 0
            for w in work:
                toks = w.tokens
                nt = len(toks)
                if nt == 1:
                    f_toks[off] = toks[0]
                elif nt <= 16:
                    f_toks[off:off + nt] = toks
                else:
                    ar = array.array(_I32, toks)
                    C.memmove(C.addressof(f_toks) + off * 4, ar.buffer_info()[0], nt * 4)
                if has_par:
                    par = w.parents
                    if par is None:
                        for j in range(nt):
                            f_par[off + j] = j - 1
                    else:
                        f_par[off:off + nt] = par
                off += nt
        fl = slot.flat
        fl.n_seqs, fl.total_tokens = n, total_toks
        fl.parents = slot.f_parents if has_par else None
        if self.paged:
            key = tuple((w.sid, w.blocks_ver) for w in work) if all(w.blocks_ver >= 0 for w in work) else None
            if key is None or key != slot.blk_key:
                total_blks = sum(len(w.blocks) for w in work)
                if total_blks > slot.f_blk_cap:
                    slot.grow_flat_blks(total_blks)
                blk_off, f_blks = slot.f_blk_off, slot.f_blocks
                boff = 0
                for i, w in enumerate(work):
                    blk_off[i] = boff
                    blks = w.blocks
                    nb = len(blks)
                    if nb:
                        if nb <= 16:
                            f_blks[boff:boff + nb] = blks
                        else:
                            ar = array.array(_U32, blks)
                            C.memmove(C.addressof(f_blks) + boff * 4, ar.buffer_info()[0], nb * 4)
                        boff += nb
                blk_off[n] = boff
                fl.total_blocks = boff
                fl.blk_offsets, fl.blocks = blk_off, f_blks if boff else None
                slot.blk_key = key
        else:
            fl.total_blocks, fl.blk_offsets, fl.blocks = 0, None, None
        if has_samp:
            ft, fp, fk, fsd, fsi = slot.f_temp, slot.f_top_p, slot.f_top_k, slot.f_seed, slot.f_step_idx
            for i, w in enumerate(work):
                ft[i] = w.temperature
                fp[i] = w.top_p
                fk[i] = w.top_k
                fsd[i] = w.seed & 0xFFFFFFFFFFFFFFFF
                fsi[i] = w.step_idx & 0xFFFFFFFFFFFFFFFF
            fl.temperature, fl.top_p, fl.top_k, fl.seed, fl.step_idx = ft, fp, fk, fsd, fsi
        else:
            fl.temperature = fl.top_p = fl.top_k = fl.seed = fl.step_idx = None
        if has_bm:
            bw = self._bm_words
            if slot.bm_buf is None or slot.bm_buf.shape != (rows, bw):
                slot.bm_buf = np.empty((rows, bw), np.uint32)
            bm_buf = slot.bm_buf
            r0 = 0
            for w in work:
                wi = w.want
                if wi and w.bitmask is not None:
                    bm_buf[r0:r0 + wi] = np.ascontiguousarray(w.bitmask, dtype=np.uint32)
                r0 += wi
            fl.bitmask_words = bw
            fl.bitmasks = bm_buf.ctypes.data_as(_P_U32)
        else:
            fl.bitmask_words, fl.bitmasks = 0, None
        if dev_tok and self.spec and spec_w > rows:
            rows = spec_w
        sin = slot.sin
        sin.flags = (0 if dev_tok else S_HOST_LOGITS) | S_FLAT
        sin.n_work = n
        sin.pad_cols = pad_cols
        sin.flat = slot.flat_ref
        return rows, nf

    def _pack_work(self, slot, work, dev_tok, pad_cols):
        n, spec_w = len(work), self.layout.spec_width
        if n > slot.ws_cap:
            slot.grow_ws(n)
        ws, tok_in, par_in, has_par = slot.wi, slot.tok_in, slot.par_in, slot.has_par
        blk_cache = self._blk_cache
        if len(blk_cache) > max(self.max_seqs * 2, 128):
            blk_cache.clear()
        slot.bm_keep.clear()
        bw = self._bm_words
        rows = nf = 0
        for i, w in enumerate(work):
            wi = ws[i]
            fl = self._item_flags(w, dev_tok)
            wi.flags = fl
            wi.sid = w.sid
            wi.pos = w.pos
            toks = w.tokens
            nt = len(toks)
            wi.n_tokens = nt
            tb = tok_in[i]
            if nt == 1:
                tb[0] = toks[0]
            else:
                if nt > len(tb):
                    tb = (C.c_int32 * nt)()
                    tok_in[i] = tb
                    wi.tokens = tb
                if nt <= 16:
                    tb[:nt] = toks
                else:
                    ar = array.array(_I32, toks)
                    C.memmove(tb, ar.buffer_info()[0], nt * 4)
            par = w.parents
            if par is None:
                if has_par[i]:
                    wi.parents = None
                    has_par[i] = False
            else:
                pb = par_in[i]
                np_ = len(par)
                if pb is None or np_ > len(pb):
                    pb = (C.c_int32 * max(np_, 16))()
                    par_in[i] = pb
                pb[:np_] = par
                wi.parents = pb
                has_par[i] = True
            blks = w.blocks
            nb = len(blks)
            wi.n_blocks = nb
            if nb:
                bv = w.blocks_ver
                ent = blk_cache.get(w.sid)
                if ent is not None and ((bv >= 0 and ent[0] == bv) or (bv < 0 and type(blks) is list and ent[1] == blks)):
                    wi.blocks = ent[3]
                else:
                    ar = array.array(_U32, blks)
                    ptr = C.cast(ar.buffer_info()[0], _P_U32)
                    blk_cache[w.sid] = (bv, None if bv >= 0 else list(blks), ar, ptr)
                    wi.blocks = ptr
            else:
                wi.blocks = None
            want = w.want
            wi.want = want
            if fl & W_SAMPLE:
                wi.temperature, wi.top_p, wi.top_k = w.temperature, w.top_p, w.top_k
                wi.seed, wi.step_idx = w.seed & 0xFFFFFFFFFFFFFFFF, w.step_idx & 0xFFFFFFFFFFFFFFFF
            if fl & W_BITMASK:
                bm = np.ascontiguousarray(w.bitmask, dtype=np.uint32)
                slot.bm_keep.append(bm)
                wi.bitmask_words, wi.bitmask = bw, bm.ctypes.data_as(_P_U32)
            else:
                wi.bitmask_words, wi.bitmask = 0, None
            rows += (spec_w if dev_tok and w.spec and spec_w > want else want)
            if w.feats:
                nf += nt
        if dev_tok and self.spec and spec_w > rows:
            rows = spec_w   # a pair handle wants room for a whole round on every step, a prompt chunk included
        sin = slot.sin
        sin.flags = 0 if dev_tok else S_HOST_LOGITS
        sin.n_work = n
        sin.pad_cols = pad_cols
        sin.flat = None
        return rows, nf

    def _pack_out(self, slot, n, dev_tok, rows, nf):
        out = slot.out
        out.n_rows = 0
        lg = ft = None
        if dev_tok:
            need_tok = rows if rows > 0 else 1
            if need_tok > slot.tok_cap:
                cap = max(need_tok, slot.tok_cap * 2, 16)
                slot.tok_out = (C.c_int32 * cap)()
                slot.tok_cap = cap
                out.tokens = slot.tok_out
            out.tokens_cap = rows
            out.logits, out.logits_cap = None, 0
        else:
            out.tokens_cap = 0
            if slot.lg is not None and slot.lg.shape[0] >= rows and sys.getrefcount(slot.lg) == 2:
                lg = slot.lg
            else:
                lg = np.empty((rows, self.vocab_size), np.float32)
                slot.lg = lg
            out.logits, out.logits_cap = lg.ctypes.data, rows * self.vocab_size
        if nf:
            ft = np.empty((nf, self.feat_width), np.float32)
            out.feats, out.feats_cap = ft.ctypes.data, ft.size
        else:
            out.feats, out.feats_cap = None, 0
        slot.meta = (n, dev_tok, nf, lg, ft)

    def _pack_one(self, slot, w, pad_cols):
        # Fast O(1) path for single-sequence greedy/spec steps (avoids dict churn & loops, <5us/round).
        if slot.ws_cap < 1:
            slot.grow_ws(1)
        wi = slot.wi[0]
        wi.sid = w.sid
        wi.flags = (W_SPEC if w.spec else 0) | (W_PREV_TOKEN if w.prev_token else 0) | W_GREEDY
        wi.pos = w.pos
        wi.n_tokens = 1
        slot.tok_in[0][0] = w.tokens[0]
        if slot.has_par[0]:
            wi.parents = None
            slot.has_par[0] = False
        blks = w.blocks
        nb = len(blks) if self.paged else 0
        wi.n_blocks = nb
        if nb:
            bv = w.blocks_ver
            if bv >= 0 and slot.one_sid == w.sid and slot.one_bv == bv:
                wi.blocks = slot.one_blks
            else:
                ar = array.array(_U32, blks)
                ptr = C.cast(ar.buffer_info()[0], _P_U32)
                slot.one_sid, slot.one_bv, slot.one_ar, slot.one_blks = w.sid, bv, ar, ptr
                wi.blocks = ptr
        else:
            wi.blocks = None
        wi.want = w.want
        wi.bitmask_words, wi.bitmask = 0, None
        spec_w = self.layout.spec_width
        rows = spec_w if (self.spec and spec_w > w.want) else w.want
        sin = slot.sin
        sin.flags = 0
        sin.n_work = 1
        sin.pad_cols = pad_cols
        sin.flat = None
        return rows, 0

    def _pack(self, slot, work, pad_cols=0):
        dev_tok = self._can_dev_tok(work)
        if (dev_tok and len(work) == 1 and work[0].parents is None and not work[0].feats
                and work[0].bitmask is None and not work[0].sample and len(work[0].tokens) == 1):
            rows, nf = self._pack_one(slot, work[0], pad_cols)
        else:
            use_flat = getattr(self, "flat_batch", False) and not getattr(self, "spec", False)
            rows, nf = (self._pack_flat if use_flat else self._pack_work)(slot, work, dev_tok, pad_cols)
        self._pack_out(slot, len(work), dev_tok, rows, nf)
        return dev_tok

    def _unpack(self, slot):
        n, dev_tok, nf, lg, ft = slot.meta
        slot.meta = None
        if dev_tok:
            counts, tok = slot.counts, slot.tok_out
            if n == 1:
                c0 = counts[0]
                res = [[tok[0]] if c0 == 1 else tok[:c0]]
            else:
                res, o = [], 0
                for i in range(n):
                    c = counts[i]
                    res.append([tok[o]] if c == 1 else tok[o:o + c])
                    o += c
        else:
            nr = slot.out.n_rows
            res = lg if nr == lg.shape[0] else lg[:nr]
        return (res, ft) if nf else res

    def step(self, work, pad_cols=0):
        if not hasattr(self, "_slots"):
            self._init_bufs()
        slot = self._slots[0]
        self._pack(slot, work, pad_cols)
        self._ck(self.L.gc_step(self.h, slot.sin_ref, slot.out_ref))
        self.stats["steps"] += 1
        return self._unpack(slot)

    def step_begin(self, work, pad_cols=0):
        """ABI 1.3 GC_L_ASYNC: enqueue step on the handle's FIFO queue (depth <= 2) and return slot token."""
        if not hasattr(self, "_slots"):
            self._init_bufs()
        self._slot_i ^= 1
        slot = self._slots[self._slot_i]
        if slot.in_flight:
            raise RuntimeError("async step queue full (>2 in flight)")
        dev_tok = self._pack(slot, work, pad_cols)
        self._ck(self.L.gc_step_begin(self.h, slot.sin_ref, slot.out_ref))
        slot.in_flight = True
        return (slot, dev_tok)

    def step_wait(self, handle):
        """ABI 1.3 GC_L_ASYNC: wait for the oldest in-flight step and unpack its output."""
        slot, _ = handle
        try:
            self._ck(self.L.gc_step_wait(self.h, slot.out_ref))
        finally:
            slot.in_flight = False
        self.stats["steps"] += 1
        return self._unpack(slot)

    def warm_buckets(self, max_cols=None, dummy_block=0):
        """Warm each advertised graph_bucket rung once so graphs are captured before live traffic."""
        buckets = getattr(self, "graph_buckets", ())
        if not buckets:
            return
        from .backend import Work as _Work
        lim = min(max_cols, self.layout.max_cols) if max_cols else self.layout.max_cols
        sid = (1 << 63) - 1
        blks = [dummy_block] if self.paged else []
        try:
            for rung in buckets:
                if rung <= lim:
                    self.step([_Work(sid, [0], 0, blks, 1, greedy=True)], pad_cols=rung)
        finally:
            self.release(sid)

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
        if hasattr(self, "_blk_cache"):
            self._blk_cache.pop(sid, None)
        if hasattr(self, "_slots"):
            for sl in self._slots:
                sl.blk_key = None
                if sl.one_sid == sid:
                    sl.one_sid = sl.one_bv = -1
                    sl.one_ar = sl.one_blks = None
        self._ck(self.L.gc_release(self.h, sid))

    def close(self):
        if hasattr(self, "_slots"):
            if self.h and getattr(self, "async_step", False):
                for sl in self._slots:
                    if sl.in_flight:
                        try:
                            self.L.gc_step_wait(self.h, sl.out_ref)
                        except Exception:
                            pass
                        sl.in_flight = False
            for sl in self._slots:
                sl.lg = sl.ft = sl.bm_buf = None
        if hasattr(self, "_blk_cache"):
            self._blk_cache.clear()
        if self.h:
            self.L.gc_close(self.h)
            self.h = None
        if self.stage:
            self.stage.close()

    __del__ = close
