"""Runtime kind gc-plan v1: a data-only pack run by the engine itself through the CUDA driver API. A bucket's
files are plan.json (device buffers, the model tensors bound into them, the step's actions: kernel launches with their argument
bytes and pointer slots, memsets, device copies, inputs, outputs) and the cubins its launches name; no library, no cudart, no
cuBLAS. One CUDA graph per bucket, captured on first use; the step's inputs are generated here from the engine's own state.

A bucket of B sequences runs on cache blocks [0, B) (the graph fixes the first KV stream and recurrent slot): move_slot and
zero_slot move a sequence's blocks. Inputs (kinds of llama.cpp's graph inputs, the input layer on the device): token ids,
positions, the encoder KQ mask (non-causal; padded keys masked, pad queries kept finite), the symmetric sliding-window mask, mean
and cls pooling, per-row constants the compiler measured (io.inputs.fills), output ids.
"""
import dataclasses
import ctypes as C
import functools
import json
import os
import re
import struct

import numpy as np

from . import pack as P
from .structure import gguf_offsets

NEG_INF_F16 = np.uint16(0xFC00)
PDL, CLUSTER = 6, 4                   # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, _CLUSTER_DIMENSION
_HEXPAD = ["0" * (32 - k) if k else "" for k in range(32)]     # hex zeros padding an argument to 16 bytes


class _Attr(C.Structure):
    _fields_ = [("id", C.c_int), ("pad", C.c_int), ("val", C.c_ubyte * 64)]


class _Cfg(C.Structure):
    _fields_ = [("gx", C.c_uint), ("gy", C.c_uint), ("gz", C.c_uint), ("bx", C.c_uint), ("by", C.c_uint), ("bz", C.c_uint),
                ("smem", C.c_uint), ("stream", C.c_void_p), ("attrs", C.POINTER(_Attr)), ("nattrs", C.c_uint)]


class _Copy2D(C.Structure):         # CUDA_MEMCPY2D
    _fields_ = [("srcX", C.c_size_t), ("srcY", C.c_size_t), ("srcType", C.c_int), ("srcHost", C.c_void_p), ("srcDevice", C.c_uint64),
                ("srcArray", C.c_void_p), ("srcPitch", C.c_size_t), ("dstX", C.c_size_t), ("dstY", C.c_size_t), ("dstType", C.c_int),
                ("dstHost", C.c_void_p), ("dstDevice", C.c_uint64), ("dstArray", C.c_void_p), ("dstPitch", C.c_size_t),
                ("width", C.c_size_t), ("height", C.c_size_t)]


def base(name):
    """'CUDA0#self_kq_mask#0' -> 'self_kq_mask'; a view of an input is that input's kind ('rs_s_copy (view)')"""
    return (name.split("#")[1] if "#" in name else name).removesuffix(" (view)")


class Step:
    """one ubatch: tokens with their sequence, stream, cell and position; n_kv cells visible per stream; n_real: tokens past it are
    padding (an encoder chunk), masked out of attention and pooling"""
    def __init__(self, tokens, seqs, pos, cells, streams, kv_size, n_kv, outputs, n_real=None):
        self.tokens, self.seqs, self.pos, self.cells, self.streams = map(np.asarray, (tokens, seqs, pos, cells, streams))
        self.kv_size, self.n_kv, self.outputs = kv_size, n_kv, np.asarray(outputs)
        self.n_real = len(self.tokens) if n_real is None else n_real


@functools.lru_cache(maxsize=32)
def _enc_mask(K, n_real, n_swa, esize, pos):
    """an encoder's [T queries][K key cells] mask bytes (cached: items of one length and padding share it)"""
    pq = np.frombuffer(pos, np.int64)
    keep = np.zeros((len(pq), K), bool)
    keep[:, :n_real] = True
    if n_swa:                                # symmetric window (non-causal): |p_q - p_k| <= n_swa / 2
        keep &= np.abs(pq[:, None] - np.arange(K)[None, :]) <= n_swa // 2
        keep[n_real:, :n_real] = True        # pad queries past the window keep every real key (a row of only -inf is NaN, and
    if esize == 4:                           # NaN V poisons real rows even at weight 0; pooling drops pad rows)
        return np.where(keep, 0, -np.inf).astype(np.float32).tobytes()
    return np.where(keep, np.uint16(0), NEG_INF_F16).astype(np.uint16).tobytes()


def _page_tables(step, P, n):
    """int32 [q_indptr B + 1][kv_indptr B + 1][last_page_len B][indices], zero-padded to n bytes, for the step's requests in the
    order their rows come (B of them): step.pages [(rows, [page ids], kv_len)] when the backend allocated them, else each
    request's cells as the stream layout holds them (stream s: cells s * kv_size .. contiguous, page = cell / P)"""
    reqs = getattr(step, "pages", None)
    if reqs is None:
        reqs, order = [], {}
        for i, s in enumerate(step.seqs.tolist()):
            order.setdefault(s, []).append(i)
        for s, rows in order.items():
            L = int(step.pos[rows].max()) + 1
            first = int(step.streams[rows[0]]) * step.kv_size // P
            reqs.append((len(rows), list(range(first, first + -(-L // P))), L))
    B = len(reqs)
    q = np.concatenate([[0], np.cumsum([r for r, _, _ in reqs])])
    kvp = np.concatenate([[0], np.cumsum([len(p) for _, p, _ in reqs])])
    last = [L - (len(p) - 1) * P for _, p, L in reqs]
    out = np.concatenate([q, kvp, last, [x for _, p, _ in reqs for x in p]]).astype(np.int32).tobytes()
    if len(out) > n:
        raise ValueError(f"gc_pages: {len(out)} B of page tables, the plan holds {n}")
    return out + bytes(n - len(out))


def generate(name, n, step, hp):
    """bytes of the plan input `name` (n bytes in the plan) for this step; hp: the bucket's io.inputs (n_pos, causal, n_swa, fills)"""
    if "(view)" in name:                     # a slice of its base input at the element offset the compiler measured
        o = hp.get("views", {}).get(f"{name}@{n}")
        P.need(o is not None, f"plan input {name}: a view the pack does not place")
        full = generate(name.replace(" (view)", ""), n + 4 * o, step, hp)
        return full[4 * o:4 * o + n]
    k, T = base(name), len(step.tokens)
    if k in hp.get("fills", {}):
        return np.full(T, hp["fills"][k], np.float32).tobytes()
    if k == "inp_tokens":
        return step.tokens.astype(np.int32).tobytes()
    if k == "mean":
        m = np.zeros(T, np.float32)
        m[:step.n_real] = np.float32(1.0) / np.float32(step.n_real)
        return m.tobytes()
    if k == "cls":
        return np.zeros(1, np.int32).tobytes()
    mk = hp.get("masks", {}).get(k)          # an encoder mask: [T queries][K key cells], K = T padded to llama.cpp's 256 cells
    if mk and not hp.get("causal", True):   # every query sees the chunk's real keys (pad queries too: finite rows)
        return _enc_mask(n // (mk["esize"] * T), step.n_real, hp.get("n_swa", 0) if mk["swa"] else 0, mk["esize"],
                         step.pos.astype(np.int64).tobytes())
    if k == "gc_pages":                     # paged KV: the step's page tables (fakecuda's gc_pages, kern/fa_ggml.cu gc_fa_paged_*)
        return _page_tables(step, hp["page"], n)
    if k == "inp_pos":
        p = step.pos.astype(np.int32)
        return (np.concatenate([p, p, p, np.zeros_like(p)]) if hp.get("n_pos", 1) == 4 else p).tobytes()
    if k in ("attn_inp_k_idxs", "attn_inp_v_idxs"):
        return (step.streams.astype(np.int64) * step.kv_size + step.cells).tobytes()
    if k == "attn_inp_kq_mask":
        n_stream = len(np.unique(step.streams))
        tps = T // n_stream
        m = np.full((n_stream, tps, step.n_kv), NEG_INF_F16, np.uint16)
        for i, p in enumerate(step.pos):
            m[i // tps if n_stream > 1 else 0, i % tps if n_stream > 1 else i, :p + 1] = 0
        return m.tobytes()
    if k in ("attn_inp_k_idxs_swa", "attn_inp_v_idxs_swa"):   # a sliding-window cache: a ring of swa_cells cells per stream
        return (step.streams.astype(np.int64) * hp["swa_cells"] + step.pos % hp["swa_cells"]).tobytes()
    if k == "attn_inp_kq_mask_swa":   # ring cell j holds the newest written position = j (mod S); visible when 0 <= p - p_j < n_swa
        S, W, n_stream = hp["swa_cells"], hp["n_swa"], len(np.unique(step.streams))
        tps, K = T // n_stream, n // (2 * T)
        P.need(K * 2 * T == n, f"plan input {name}: {n} bytes are no [{T}][K] f16 mask")
        m, j = np.full((n_stream, tps, K), NEG_INF_F16, np.uint16), np.arange(K)
        for s in range(n_stream):
            last = int(step.pos[s * tps:(s + 1) * tps].max())    # the step writes its tokens before attending
            pj = last - (last - j) % S
            for r in range(tps):
                p = int(step.pos[s * tps + r])
                m[s, r, (pj >= 0) & (pj <= p) & (p - pj < W)] = 0
        return m.tobytes()
    if k in ("attn_inp_rel_pos", "attn_inp_rel_pos_swa"):   # relative-position attention: [T][K] int32 distances p - p_j of the
        swa = k.endswith("_swa")                             # cells the query sees (0 <= d < extent), extent where none (llama.cpp)
        E = hp["n_swa"] if swa else hp["rel_extent"]
        n_stream = len(np.unique(step.streams))
        tps, K = T // n_stream, n // (4 * T)
        P.need(K * 4 * T == n, f"plan input {name}: {n} bytes are no [{T}][K] int32 distances")
        d, j = np.full((n_stream, tps, K), E, np.int32), np.arange(K)
        for s in range(n_stream):
            rows = step.pos[s * tps:(s + 1) * tps].astype(np.int64)
            if swa:                                          # ring cell j holds the newest written position = j (mod S)
                S = hp["swa_cells"]
                last = int(rows.max())
                pj = last - (last - j) % S
            else:                                            # stream cell j holds position j (written up to the step's last)
                pj = np.where(j <= int(rows.max()), j, -1)
            for r, p in enumerate(rows):
                dist = p - pj
                d[s, r] = np.where((pj >= 0) & (dist >= 0) & (dist < E), dist, E)
        return d.tobytes()
    if k == "out_ids":
        return step.outputs.astype(np.int32).tobytes()
    if k == "rs_s_copy":
        return np.unique(step.seqs).astype(np.int32).tobytes()
    raise P.PackError(f"plan input {k}: not a kind this engine generates")


FAKE0, FAKE_SPAN = 0xFA00000000000000, 1 << 56


def _host(v):                    # a host address recorded inside a plan (process ASLR): never read on the device
    return isinstance(v, int) and not isinstance(v, bool) and (1 << 40) < v < FAKE0


def _lerp_int(v0, v1, t):
    num = (v1 - v0) * t[0]
    P.need(num % t[1] == 0, f"affine bucket: {v0} -> {v1} not affine at integer steps")
    return v0 + num // t[1]


def _fastdiv(d):                 # ggml's init_fastdiv_values: (mp, L, d)
    L = 0
    while L < 32 and (1 << L) < d:
        L += 1
    return ((1 << 32) * ((1 << L) - d) // d + 1) & 0xFFFFFFFF, L, d


def _lerp_arg(a0, a1, t, loose):
    """a launch argument at an interpolated width: device pointers as 64-bit words, fastdiv triples from their divisor, host
    addresses kept, the rest 32-bit affine; loose (a CUTLASS block): only its device pointers move"""
    b0, b1 = bytes.fromhex(a0), bytes.fromhex(a1)
    P.need(len(b0) == len(b1), "affine bucket: argument sizes differ")
    if b0 == b1:
        return a0
    if len(b0) == 12:
        u0, u1 = struct.unpack("<3I", b0), struct.unpack("<3I", b1)
        if _fastdiv(u0[2]) == u0 and _fastdiv(u1[2]) == u1:
            return struct.pack("<3I", *_fastdiv(_lerp_int(u0[2], u1[2], t))).hex()
    out, o = bytearray(b0), 0
    while o < len(b0):
        if o + 8 <= len(b0) and o % 8 == 0:
            p0, p1 = struct.unpack_from("<Q", b0, o)[0], struct.unpack_from("<Q", b1, o)[0]
            if FAKE0 <= p0 < FAKE0 + FAKE_SPAN and FAKE0 <= p1 < FAKE0 + FAKE_SPAN:
                struct.pack_into("<Q", out, o, _lerp_int(p0, p1, t))
                o += 8
                continue
            if _host(p0) and _host(p1):
                o += 8
                continue
        if o + 4 <= len(b0):
            v0, v1 = struct.unpack_from("<I", b0, o)[0], struct.unpack_from("<I", b1, o)[0]
            if v0 != v1 and not loose:
                struct.pack_into("<I", out, o, _lerp_int(v0, v1, t) & 0xFFFFFFFF)
            o += 4
        else:
            P.need(b0[o] == b1[o], "affine bucket: a differing byte outside a word")
            o += 1
    return out.hex()


def lerp_plan(x0, x1, t):
    """a plan at width w from the plans at w0 and w1 (t = (w - w0, w1 - w0)): every differing quantity affine in the width"""
    if isinstance(x0, dict) and "launch" in x0 and "args" in x0:
        loose = x0["launch"].startswith("_ZN7cutlass")
        return {**{k: lerp_plan(x0[k], x1[k], t) for k in x0 if k != "args"},
                "args": [_lerp_arg(a, b, t, loose) for a, b in zip(x0["args"], x1["args"])]}
    if isinstance(x0, dict):
        P.need(x0.keys() == x1.keys(), "affine bucket: plans differ in fields")
        return {k: lerp_plan(x0[k], x1[k], t) for k in x0}
    if isinstance(x0, list):
        P.need(len(x0) == len(x1), "affine bucket: plans differ in length")
        return [lerp_plan(a, b, t) for a, b in zip(x0, x1)]
    if x0 == x1 or (_host(x0) and _host(x1)):
        return x0
    P.need(isinstance(x0, int) and isinstance(x1, int) and not isinstance(x0, bool), f"affine bucket: {x0!r} vs {x1!r}")
    return _lerp_int(x0, x1, t)


_PDL = os.environ.get("GOETHE_PLAN_PDL", "1") != "0"     # 0: every launch without PDL (profiling: true kernel durations)
_PAD = os.environ.get("GOETHE_PAD", "1") != "0"          # 0: a short prefill tail runs token by token (the exact split)
_PACKED = os.environ.get("GOETHE_PACKED", "1") != "0"    # 0: concurrent prompts prefill one at a time
_SPLIT = os.environ.get("GOETHE_SPLIT", "0") != "0"    # 1: the scheduler splits a step over waiting prompts (prefill_share): off until
#                                                        packed buckets build their masks on the device (host masks: ~16 MB a launch at 4k)


_REFS = ("dst", "src", "copy", "copy2d", "memset", "A", "B", "C")


_IA = {}         # id(an init list) -> (that list, its _init_actions): one hashing per plan object (cleared per Runtime)


def _init_actions(p):
    """[(key, {compute buffer: rank in the action})] of p's init actions in order (e.g. NVFP4 weights repacked at load): a key is the
    action's content with every pointer slot zeroed, a weight or cache buffer by its rank among them and a compute buffer by its
    rank in the action and its size; equal keys write equal bytes (their inputs: the weights)"""
    import hashlib
    if id(p.get("init")) in _IA:
        return _IA[id(p.get("init"))][1]
    comp = {i for i, b in enumerate(p["buffers"]) if b["role"] == "compute"}
    srank = {b: k for k, b in enumerate(i for i, x in enumerate(p["buffers"]) if x["role"] != "compute")}
    out = []
    for a in p.get("init", ()):
        names = [v[0] for k in _REFS if isinstance(v := a.get(k), (list, tuple)) and len(v) == 2]
        names += [b for ps in a.get("ptrs", ()) for _, b, _ in ps] + [t["gaddr"][0] for _, _, t in a.get("tmaps", ())]
        loc = {}
        for b in names:
            if b in comp and b not in loc:
                loc[b] = len(loc)
        ref = lambda b: ["d", loc[b], p["buffers"][b]["size"]] if b in loc else ["s", srank[b]]
        x = {k: v for k, v in a.items() if k not in ("args", "ptrs", "tmaps", "after") + _REFS}
        x.update({k: [ref(v[0]), v[1]] for k in _REFS if isinstance(v := a.get(k), (list, tuple)) and len(v) == 2})
        if "args" in a:
            args = [bytearray.fromhex(h) for h in a["args"]]
            for i, ps in enumerate(a.get("ptrs", ())):
                for o, _, _ in ps:
                    args[i][o:o + 8] = bytes(8)
            for i, o, _ in a.get("tmaps", ()):
                args[i][o:o + 128] = bytes(128)
            x.update(args=[g.hex() for g in args], ptrs=[[[o, ref(b), off] for o, b, off in ps] for ps in a.get("ptrs", ())],
                     tmaps=[[i, o, dict(t, gaddr=[ref(t["gaddr"][0]), t["gaddr"][1]])] for i, o, t in a.get("tmaps", ())])
        out.append((hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest(), loc))
    if p.get("init"):
        _IA[id(p["init"])] = (p["init"], out)
    return out


def _derived(p):
    """{compute buffer: (init action key, rank)}: the buffers p's init actions write, shared by every bucket and alternative of
    the model whose init has the same action; None when two init actions name one compute buffer (then the bucket's own)"""
    out = {}
    for key, loc in _init_actions(p):
        for b, r in loc.items():
            if b in out:
                return None
            out[b] = (key, r)
    return out


_PAGE = 2 << 20     # the driver's allocation granularity: a cuMemAlloc takes whole 2 MiB pages (Qwen3.8 P512's 378 init outputs:
#                     413 of the 502 MiB measured past the bytes asked for; the rest, modules and graphs, is the slack's)


def _pg(n):
    return -(-max(n, 1) // _PAGE) * _PAGE


def _device_parts(p):
    """({(init action key, rank): bytes}, arena bytes, own bytes): plan p's compute buffers as a Runtime holds them (each allocation
    in whole pages): the outputs of its init actions (one copy per distinct action, shared by every bucket), the rest carved from
    the arena, or all its own when its init is not shareable"""
    d = _derived(p) if p.get("init") else {}
    if d is None:
        return {}, 0, sum(_pg(b["size"]) for b in p["buffers"] if b["role"] == "compute")
    return ({kr: _pg(p["buffers"][b]["size"]) for b, kr in d.items()},
            sum(-(-max(b["size"], 1) // 4096) * 4096 for i, b in enumerate(p["buffers"]) if b["role"] == "compute" and i not in d), 0)


def _device_bytes(plans, share=None):
    """(derived, arena, own): the device bytes a Runtime allocates for the plans' compute buffers: derived (init outputs, one copy
    per distinct init action), the arena (the largest bucket's other scratch), own (compute of plans whose init is not shareable).
    The one account _streams and the Runtime both use"""
    share, pool, arena, own = share or {}, {}, 0, 0
    for pid, p in plans.items():
        if pid not in share:
            q, a, o = _device_parts(p)
            pool.update(q)
            arena, own = max(arena, a), own + o
    return sum(pool.values()), arena and _pg(arena), own


def _account(plans, free, share=None, dev=None):
    """the device bytes a Runtime of the plans takes in free bytes: weights, compute (dev, else _device_bytes: derived, arena,
    own), a slack of 1 GiB + free/50 (cuBLAS, modules, graphs) and a page per cache buffer (its rounding), stream (one stream's
    caches); nseq: the streams recorded"""
    first = next(iter(plans.values()))
    nseq = first["nseq"]
    der, arena, own = dev or _device_bytes(plans, share)
    caches = first.get("caches", {}).values()
    return dict(weights=sum(_pg(b["size"]) for b in first["buffers"] if b["role"] == "weights"), derived=der, arena=arena, own=own,
                slack=(1 << 30) + free // 50 + _PAGE * len({b for b, _, _ in caches}), stream=sum(n // nseq for _, _, n in caches),
                nseq=nseq)


def _streams(plans, free, share=None, dev=None):
    """the streams free device bytes hold beside everything else of _account, at most the recorded nseq: the one fit rule of the
    Runtime and of a compiler choosing what a pack carries"""
    a = _account(plans, free, share, dev)
    fixed = a["weights"] + a["derived"] + a["arena"] + a["own"] + a["slack"]
    return min(a["nseq"], (free - fixed) // a["stream"]) if a["stream"] else a["nseq"]


def does_not_fit(plans, free, needed=2, share=None, dev=None, where="this GPU"):
    """None where free device bytes hold the plans at `needed` streams (_streams), else the DoesNotFit naming every term: the
    one test a Runtime and a compiler deciding whether a pack kind ships make"""
    have = _streams(plans, free, share, dev)
    if have >= needed:
        return None
    a = _account(plans, free, share, dev)
    M = lambda v: f"{v >> 20}"
    need = a["weights"] + a["derived"] + a["arena"] + a["own"] + a["slack"] + needed * a["stream"]
    why = "" if a["nseq"] >= needed else f" (the pack records {a['nseq']})"
    return P.DoesNotFit(f"does not fit {where}: {needed} streams{why} need {M(need)} MiB (weights {M(a['weights'])}, init outputs "
                        f"{M(a['derived'])}, arena {M(a['arena'])}, unshared scratch {M(a['own'])}, {needed} x {M(a['stream'])} a "
                        f"stream, slack {M(a['slack'])}), {M(free)} MiB free holds {max(have, 0)}", need, free, max(have, 0), needed)


def _fit_streams(plans, free, min_streams=0, share=None, where="this GPU"):
    """the plans with their caches held for as many streams as this device's memory takes beside the weights and the largest
    bucket's scratch (at most the recorded nseq; GOETHE_NSEQ forces one): every cache keeps its per-stream bytes and order, the
    caches of a buffer packed after one another, and every reference into a cache buffer (launch pointers, copies, GEMM operands,
    TMA descriptors, inputs, outputs) moved to the cache's new place. Streams are cache-major (a stream's rows at s * row), so the
    first nseq streams of each cache are where they were relative to the cache; buckets need B <= nseq streams (the backend's
    ladder reads nseq). A pack recorded on a large GPU thus runs on a smaller one, with fewer sequences at once."""
    first = next(iter(plans.values()))
    nseq = first["nseq"]
    if os.environ.get("GOETHE_NSEQ"):
        want = int(os.environ["GOETHE_NSEQ"])
        P.need(min(want, nseq) >= max(2, min_streams), f"GOETHE_NSEQ={want}: {min(want, nseq)} streams, the plans need {max(2, min_streams)}")
    else:
        want = _streams(plans, free, share)
        if min(want, nseq) < max(2, min_streams):
            raise does_not_fit(plans, free, max(2, min_streams), share, where=where)
    if want >= nseq:
        return plans
    out = {}
    for pid, p in plans.items():
        q = dict(p, caches=dict(p["caches"]), buffers=list(p["buffers"]))   # only what moves is copied (plans may share objects:
        moves = {}                                           # an affine width is its end plan) buffer -> [(old off, old end, new off)]
        for b in {b for b, _, _ in p["caches"].values()}:
            end = 0
            for name, (bb, o, n) in sorted(((k, v) for k, v in p["caches"].items() if v[0] == b), key=lambda kv: kv[1][1]):
                moves.setdefault(b, []).append((o, o + n, end))
                q["caches"][name] = [b, end, n // nseq * want]
                end = -(-(end + n // nseq * want) // 256) * 256
            q["buffers"][b] = dict(q["buffers"][b], size=max(end, 256))
        memo = {}

        def mv(b, off):                                      # (once per buffer and offset: a plan names the same ones again)
            if (b, off) not in memo:
                memo[b, off] = mv1(b, off)
            return memo[b, off]

        def mv1(b, off):                                     # the cache holding off (caches abut: a cache's end is the next
            for o, e, no in moves.get(b, ()):                # one's base, which belongs to the next); else an end pointer
                if o <= off < e:
                    return no + off - o
            for o, e, no in moves.get(b, ()):
                if off == e:
                    return no + off - o
            P.need(b not in moves, f"gc-plan: a reference into cache buffer {b} at {off} is in no cache")
            return off
        def act(a):                                          # the action with its cache references moved (a copy only if one does)
            c = {k: [v[0], mv(*v)] for k in ("dst", "src", "copy", "copy2d", "memset", "A", "B", "C")
                 if isinstance(v := a.get(k), list) and len(v) == 2 and v[0] in moves}
            if any(b in moves for ps in a.get("ptrs", ()) for _, b, _ in ps):
                c["ptrs"] = [[[o, b, mv(b, off) if b in moves else off] for o, b, off in ps] for ps in a["ptrs"]]
            if any(t["gaddr"][0] in moves for _, _, t in a.get("tmaps", ())):
                c["tmaps"] = [[i, o, dict(t, gaddr=[t["gaddr"][0], mv(*t["gaddr"])]) if t["gaddr"][0] in moves else t] for i, o, t in a["tmaps"]]
            return {**a, **c} if c else a
        q["actions"] = [act(a) for a in p["actions"]]
        q["nseq"] = want
        out[pid] = q
    return out


class DeviceOutOfMemory(P.OutOfMemory, RuntimeError):
    """CUDA_ERROR_OUT_OF_MEMORY (2) from the driver: an allocation, a module load or a launch past the device's memory"""


_HELD = {}       # one model's uploaded weights kept across Runtimes in this process: key, device addresses, device, bytes


def _weights_key(plans, files, device):
    """what a Runtime's uploaded weights are: the weight files (path, size, mtime), the device, the weight buffers' sizes and the
    tensors bound into them; None when a plan has init actions (they may write the shared buffers: never kept)"""
    import hashlib
    if any(p.get("init") for p in plans.values()):
        return None
    first = next(iter(plans.values()))
    st = [(os.path.abspath(f), os.stat(f).st_size, os.stat(f).st_mtime_ns) for f in files]
    lay = [b["size"] for b in first["buffers"] if b["role"] == "weights"]
    return hashlib.sha256(json.dumps([st, device, lay, first["weights"]], sort_keys=True).encode()).hexdigest()


def _release_held(cu):
    """the held weights freed and their context hold released"""
    if _HELD.get("key"):
        for a in _HELD["addrs"]:
            cu.cuMemFree_v2(C.c_uint64(a))
        cu.cuDevicePrimaryCtxRelease(_HELD["dev"])
    _HELD.clear()


class Runtime:
    """the device side of a gc-plan pack: one context, the plans' buffers, their cubins, the weights, a CUDA graph per bucket"""

    def __init__(self, plans, blob, files, device=0, share=None, min_streams=0, keep_weights=False):
        """plans: {bucket id: plan dict}; blob(rel, bucket id) -> bytes of that bucket file; files: the model's weight files; share:
        {bucket id: bucket id} whose scratch it reuses (the widths of an affine bucket: the widest one's buffers hold every width's);
        min_streams: the streams the caller's buckets need (a PackError before anything is allocated or read when fewer fit);
        keep_weights: leave the uploaded weights on the device after close() for the next Runtime of the same weights in this
        process (one Runtime per bucket, as gc test's S3 builds them); a Runtime of other weights frees them first"""
        cu = self.cu = C.CDLL("libcuda.so.1")
        self.ck(cu.cuInit(0), "cuInit")
        dev, self.ctx = C.c_int(), C.c_void_p()
        self.ck(cu.cuDeviceGet(C.byref(dev), max(device, 0)), "device")
        self.ck(cu.cuDevicePrimaryCtxRetain(C.byref(self.ctx), dev), "context")
        self.dev = dev
        self.ck(cu.cuCtxSetCurrent(self.ctx), "context")
        self.st = C.c_void_p()
        self.ck(cu.cuStreamCreate(C.byref(self.st), 1), "stream")
        wkey = _weights_key(plans, files, device)
        if _HELD.get("key") != wkey:         # another model's (or no) weights held: free them before measuring the device
            _release_held(cu)
        held = _HELD.get("key") == wkey and wkey is not None
        free = self._free_bytes() + (_HELD["cost"] if held else 0)     # (as if this Runtime uploaded them itself)
        share = share or {}
        _IA.clear()
        name = (C.c_char * 256)()
        self.ck(cu.cuDeviceGetName(name, 256, dev), "device name")
        try:                                 # the cache capacity this device holds (a pack is GPU-agnostic)
            plans = _fit_streams(plans, free, min_streams, share, where=name.value.decode())
        except P.PackError:                  # (refused before any allocation: the stream and context hold released)
            cu.cuStreamDestroy_v2(self.st)
            cu.cuDevicePrimaryCtxRelease(dev)
            self.ctx = None
            raise
        # weights and caches (kv) are shared by every bucket, by role and order (one model, one cache layout); scratch (compute)
        # buffers are each bucket's own (llama.cpp's allocator sizes and splits them per graph)
        first = next(iter(plans.values()))
        shared_layout = [(b["role"], b["size"]) for b in first["buffers"] if b["role"] != "compute"]
        self.shared, self.allocs, self.bases, self.hosts = [], [], {}, []
        wi, cost = [i for i, (r, _) in enumerate(shared_layout) if r == "weights"], 0
        for i, (r, n) in enumerate(shared_layout):   # zeroed, weights too: ggml zeroes a quantized tensor's row padding at load
            if held and r == "weights":
                self.shared.append(_HELD["addrs"][wi.index(i)])
                continue
            f0 = self._free_bytes() if r == "weights" else 0
            self.shared.append(self._alloc(n, zero=True))
            cost += f0 - self._free_bytes() if r == "weights" else 0     # the device memory the weights take
        arena_need, self.derived, run_init, ran = 0, {}, {}, set()   # (init action key, rank) -> address; pid -> init actions it runs
        for pid, p in sorted(plans.items(), key=lambda kv: kv[0] in share):     # the owners first
            P.need([(b["role"], b["size"]) for b in p["buffers"] if b["role"] != "compute"] == shared_layout,
                   f"bucket {pid}: weights or caches laid out differently from the other buckets")
            it = iter(self.shared)
            if pid in share:
                own = plans[share[pid]]["buffers"]
                P.need(len(own) == len(p["buffers"]) and all(b["size"] <= c["size"] for b, c in zip(p["buffers"], own)),
                       f"bucket {pid}: scratch larger than its affine bucket's widest")
                self.bases[pid] = list(self.bases[share[pid]])
                continue
            d = _derived(p) if p.get("init") else {}
            if d is None:                    # an init whose outputs are not shareable: every compute buffer the bucket's own
                self.bases[pid] = [next(it) if b["role"] != "compute" else self._alloc(b["size"], zero=True) for b in p["buffers"]]
                run_init[pid] = p["init"]
                continue
            acts = _init_actions(p)
            run_init[pid] = [a for a, (k, _) in zip(p.get("init", ()), acts) if k not in ran]   # each distinct init action once
            ran |= {k for k, _ in acts}
            for b in d:                      # init outputs: one copy per distinct init action, shared by every bucket
                if d[b] not in self.derived:
                    self.derived[d[b]] = self._alloc(p["buffers"][b]["size"], zero=True)
            off, self.bases[pid] = 0, []     # one bucket runs at a time: every bucket's scratch is carved from one arena
            for i, b in enumerate(p["buffers"]):
                if b["role"] != "compute":
                    self.bases[pid].append(next(it))
                elif i in d:
                    self.bases[pid].append(self.derived[d[i]])
                else:
                    self.bases[pid].append(off)
                    off += -(-max(b["size"], 1) // 4096) * 4096
            arena_need = max(arena_need, off)
        if arena_need:                       # the widest bucket's scratch, zeroed once; offsets become addresses
            arena = self._alloc(arena_need, zero=True)
            for pid, p in plans.items():
                if pid not in share and (d := _derived(p) if p.get("init") else {}) is not None:
                    self.bases[pid] = [x + arena if b["role"] == "compute" and i not in d else x
                                       for i, (x, b) in enumerate(zip(self.bases[pid], p["buffers"]))]
            for pid in share:
                self.bases[pid] = list(self.bases[share[pid]])
        if any('"gemm"' in json.dumps(p.get("actions")) for p in plans.values()):
            self._blas()                     # cuBLAS made before any capture (it allocates; a capture forbids that)
        self.plans, self.fn, self.graphs, mods = plans, {}, {}, {}
        for pid, p in plans.items():
            for k, rel in p["kernels"].items():
                if k in self.fn:
                    continue
                if rel not in mods:
                    m, img = C.c_void_p(), blob(rel, pid)
                    self.ck(cu.cuModuleLoadData(C.byref(m), img), f"cubin {rel}")
                    mods[rel] = m
                f = C.c_void_p()
                self.ck(cu.cuModuleGetFunction(C.byref(f), mods[rel], k.encode()), k[:60])
                self.fn[k] = f
        self.mods = mods
        if not held:
            self._weights(first["weights"], files, self.bases[next(iter(plans))])
            if wkey is not None and keep_weights:     # kept for the next Runtime of these weights in this process
                addrs = [self.shared[i] for i in wi]
                self.allocs = [a for a in self.allocs if a not in addrs]
                self.ck(cu.cuDevicePrimaryCtxRetain(C.byref(C.c_void_p()), self.dev), "context")   # the held weights' own hold
                _HELD.update(key=wkey, addrs=addrs, dev=self.dev, cost=cost)
        for pid in run_init:                   # init actions (e.g. a weight relaid for another kernel): each distinct one once, after
            keep = []                          # the weights
            for a in run_init[pid]:
                self._issue(a, keep, self.bases[pid])
            self.ck(cu.cuStreamSynchronize(self.st), "init")

    def _alloc(self, n, zero):
        a = C.c_uint64()
        self.ck(self.cu.cuMemAlloc_v2(C.byref(a), C.c_size_t(max(n, 1))), f"alloc {n} B")
        if zero:
            self.ck(self.cu.cuMemsetD8_v2(a, C.c_ubyte(0), C.c_size_t(max(n, 1))), "zero")
        self.allocs.append(a.value)
        return a.value

    def ck(self, rc, what):
        if rc:
            s = C.c_char_p()
            self.cu.cuGetErrorString(rc, C.byref(s))
            raise (DeviceOutOfMemory if rc == 2 else RuntimeError)(f"{what}: CUDA {rc} {s.value.decode() if s.value else ''}")

    def _weights(self, bind, files, base, chunk=1 << 28):
        """every tensor the plan binds, from the weight files by name (raw bytes: the plan reads them as llama.cpp does), through
        two pinned staging buffers: a thread fills chunk k + 1 from the file's map while the copy engine moves chunk k (pageable
        tensor-by-tensor copies ran at ~2 GB/s). O(bytes)"""
        import concurrent.futures as cf
        import mmap
        left, work, maps = dict(bind), [], []
        for path in files:
            off = gguf_offsets(path)
            f = open(path, "rb")
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)      # (copy on write: never written, but addressable)
            maps.append((f, m))
            for name in [n for n in left if n in off]:
                buf, o, n = left.pop(name)
                P.need(off[name] + n <= len(m), f"{path}: tensor {name} is short")
                for k in range(0, n, chunk):
                    work.append((base[buf] + o + k, m, off[name] + k, min(chunk, n - k), name))
        P.need(not left, f"weights: {len(left)} plan tensors in no weight file (first {next(iter(left), '')})")
        self.ck(self.cu.cuCtxSynchronize(), "zeroed buffers")       # (the buffers' zeroing ran on the legacy stream)
        bufs, evs = [], []
        for _ in range(2):
            h, e = C.c_void_p(), C.c_void_p()
            self.ck(self.cu.cuMemAllocHost_v2(C.byref(h), C.c_size_t(chunk)), "pinned staging")
            self.ck(self.cu.cuEventCreate(C.byref(e), 0), "event")
            bufs.append(h.value)
            evs.append(e)

        def fill(k):
            _, m, o, n, _ = work[k]
            C.memmove(bufs[k % 2], C.addressof(C.c_char.from_buffer(m, o)), n)
        with cf.ThreadPoolExecutor(1) as ex:
            nxt = ex.submit(fill, 0) if work else None
            for k, (dst, _, _, n, name) in enumerate(work):
                nxt.result()
                self.ck(self.cu.cuMemcpyHtoDAsync_v2(C.c_uint64(dst), C.c_void_p(bufs[k % 2]), C.c_size_t(n), self.st), name)
                self.ck(self.cu.cuEventRecord(evs[k % 2], self.st), "event")
                if k + 1 < len(work):
                    if k >= 1:
                        self.ck(self.cu.cuEventSynchronize(evs[(k + 1) % 2]), "sync")   # the buffer chunk k + 1 refills is free
                    nxt = ex.submit(fill, k + 1)
        self.ck(self.cu.cuStreamSynchronize(self.st), "weights")
        for h, e in zip(bufs, evs):
            self.cu.cuMemFreeHost(C.c_void_p(h))
            self.cu.cuEventDestroy_v2(e)
        for f, m in maps:
            m.close()
            f.close()

    def _issue(self, a, keep, base, st=None):
        cu, st = self.cu, st or self.st
        addr = lambda w: base[w[0]] + w[1]
        if "launch" in a:
            tm = {}
            for i, o, t in a.get("tmaps", ()):
                tm.setdefault(i, []).append((o, t))
            at, n, hs = [], 0, []            # the arguments in one buffer, each at a 16-byte boundary (as malloc placed them)
            for h in a["args"]:
                at.append(n)
                n += -(-len(h) // 32) * 16
                hs += [h, _HEXPAD[len(h) % 32]]
            raw = bytearray.fromhex("".join(hs) or "00")
            for i, (s, ptrs) in enumerate(zip(at, a["ptrs"])):
                for o, buf, off in ptrs:
                    struct.pack_into("<Q", raw, s + o, base[buf] + off)
                for o, t in tm.get(i, ()):     # a TMA descriptor in the argument: encoded by this driver at the real address
                    raw[s + o:s + o + 128] = self._tmap(t, base[t["gaddr"][0]] + t["gaddr"][1])
            cb = (C.c_char * len(raw)).from_buffer(raw)
            a0 = C.addressof(cb)              # (addresses, not ctypes.cast: 6x cheaper a launch)
            pp = (C.c_void_p * len(at))(*[a0 + s for s in at])
            keep += [raw, cb, pp]
            f = self.fn[a["launch"]]
            if a["smem"] > 48 * 1024:
                self.ck(cu.cuFuncSetAttribute(f, 8, a["smem"]), "max dynamic smem")
            attrs = []
            if any(a.get("cluster") or ()):
                x = _Attr(CLUSTER)
                C.memmove(C.addressof(x.val), (C.c_uint * 3)(*a["cluster"]), 12)
                attrs.append(x)
            if a.get("pdl") and _PDL:
                x = _Attr(PDL)
                x.val[0] = 1
                attrs.append(x)
            if attrs:
                cfg = _Cfg(*a["g"], *a["b"], a["smem"], st, (_Attr * len(attrs))(*attrs), len(attrs))
                self.ck(cu.cuLaunchKernelEx(C.byref(cfg), f, pp, None), a["launch"][:60])
            else:
                self.ck(cu.cuLaunchKernel(f, *a["g"], *a["b"], a["smem"], st, pp, None), a["launch"][:60])
        elif "memset" in a:
            self.ck(cu.cuMemsetD8Async(C.c_uint64(addr(a["memset"])), C.c_ubyte(a["v"]), C.c_size_t(a["n"]), st), "memset")
        elif "copy2d" in a:
            m = _Copy2D(srcType=2, srcDevice=addr(a["copy2d"]), srcPitch=a["sp"], dstType=2, dstDevice=addr(a["dst"]),
                        dstPitch=a["dp"], width=a["w"], height=a["h"])
            self.ck(cu.cuMemcpy2DAsync_v2(C.byref(m), st), "2d copy")
        elif "copy" in a:
            self.ck(cu.cuMemcpyDtoDAsync_v2(C.c_uint64(addr(a["dst"])), C.c_uint64(addr(a["copy"])), C.c_size_t(a["n"]), st),
                    "copy")
        elif "gemm" in a:                    # a library GEMM S3 picked (cuBLAS GemmEx, cuBLASLt underneath), captured like a launch
            g = a["gemm"]
            h, al, be = self._blas(), bytes.fromhex(g["alpha"]), bytes.fromhex(g["beta"])
            if g.get("ptrarray"):            # A, B, C: device arrays of operand pointers (filled by an earlier kernel of the step)
                self.ck_blas(self.blas.cublasGemmBatchedEx(h, g["ta"], g["tb"], g["m"], g["n"], g["k"], al, C.c_uint64(addr(a["A"])), g["at"],
                                                           g["lda"], C.c_uint64(addr(a["B"])), g["bt"], g["ldb"], be, C.c_uint64(addr(a["C"])),
                                                           g["ct"], g["ldc"], g["gemm"], g["comp"], g["algo"]), "cublasGemmBatchedEx")
                return
            P.need(g["gemm"] == 1, "a strided batched cuBLAS GEMM: not carried yet")
            self.ck_blas(self.blas.cublasGemmEx(h, g["ta"], g["tb"], g["m"], g["n"], g["k"], al, C.c_uint64(addr(a["A"])), g["at"], g["lda"],
                                                C.c_uint64(addr(a["B"])), g["bt"], g["ldb"], be, C.c_uint64(addr(a["C"])), g["ct"], g["ldc"],
                                                g["comp"], g["algo"]), "cublasGemmEx")

    def _issue_dag(self, acts, keep, base):
        """a step's actions captured over a few streams along the plan's dependencies (epack's "after": RAW, WAR, WAW by
        tensor range): an action runs on the stream of its latest dependency (a chain stays on one stream), waits on events
        only for dependencies issued on another one, and every stream joins the capture's own before the outputs. A library
        GEMM stays on the capture's stream (its handle's). The kernels and their arguments are the linear plan's: the step's
        result is the same; only independent launches may overlap. O(actions + edges)."""
        cu, n = self.cu, int(os.environ.get("GOETHE_DAG_STREAMS", "4"))
        if len(getattr(self, "xst", [])) < n - 1:
            self.xst = getattr(self, "xst", [])
            while len(self.xst) < n - 1:
                s = C.c_void_p()
                self.ck(cu.cuStreamCreate(C.byref(s), 1), "dag stream")
                self.xst.append(s)
        need = len(acts) + n + 1
        self.evs = getattr(self, "evs", [])
        while len(self.evs) < need:
            e = C.c_void_p()
            self.ck(cu.cuEventCreate(C.byref(e), 2), "dag event")     # CU_EVENT_DISABLE_TIMING
            self.evs.append(e)
        sts = [self.st] + self.xst[:n - 1]
        fork = self.evs[len(acts)]
        self.ck(cu.cuEventRecord(fork, self.st), "dag fork")
        for s in sts[1:]:
            self.ck(cu.cuStreamWaitEvent(s, fork, 0), "dag fork wait")
        on, last, rr = {}, [-1] * n, 0
        for i, a in enumerate(acts):
            if "input" in a or "output" in a:
                continue
            deps = [d for d in a["after"] if d in on]
            if "gemm" in a:
                k = 0
            elif deps:
                k = on[max(deps)]
            else:
                k, rr = rr % n, rr + 1
            for d in deps:
                if on[d] != k:
                    self.ck(cu.cuStreamWaitEvent(sts[k], self.evs[d], 0), "dag wait")
            self._issue(a, keep, base, sts[k])
            self.ck(cu.cuEventRecord(self.evs[i], sts[k]), "dag record")
            on[i], last[k] = k, i
        for j, s in enumerate(sts[1:], 1):
            e = self.evs[len(acts) + j]
            self.ck(cu.cuEventRecord(e, s), "dag join")
            self.ck(cu.cuStreamWaitEvent(self.st, e, 0), "dag join wait")

    def _blas(self):
        """cuBLAS on this context and stream, with its own fixed workspace (graph capture needs one), made at first use"""
        if getattr(self, "blas_h", None) is None:  # (Runtime init, before any capture)
            import ctypes.util
            self.blas = C.CDLL(os.environ.get("GOETHE_CUBLAS") or ctypes.util.find_library("cublas") or "libcublas.so.12")
            h = C.c_void_p()
            self.ck_blas(self.blas.cublasCreate_v2(C.byref(h)), "cublasCreate")
            ws = 32 << 20
            self.ck_blas(self.blas.cublasSetStream_v2(h, self.st), "cublasSetStream")
            self.ck_blas(self.blas.cublasSetWorkspace_v2(h, C.c_uint64(self._alloc(ws, zero=False)), C.c_size_t(ws)), "cublasSetWorkspace")
            self.blas_h = h
        return self.blas_h

    def ck_blas(self, rc, what):
        if rc:
            raise RuntimeError(f"{what}: cuBLAS status {rc}")

    def _free_bytes(self):
        free, total = C.c_size_t(), C.c_size_t()
        self.ck(self.cu.cuMemGetInfo_v2(C.byref(free), C.byref(total)), "memory")
        return free.value

    def _tmap(self, t, gaddr):
        m, r = (C.c_ubyte * 128)(), t["rank"]
        u64, u32 = C.c_uint64 * r, C.c_uint32 * r
        self.ck(self.cu.cuTensorMapEncodeTiled(m, t["dt"], r, C.c_void_p(gaddr), u64(*t["dims"]), (C.c_uint64 * max(r - 1, 1))(*t["strides"]),
                                               u32(*t["box"]), u32(*t["es"]), t["il"], t["sw"], t["l2"], t["oob"]), "tensor map")
        return bytes(m)

    def _pinned(self, n):
        h = C.c_void_p()
        self.ck(self.cu.cuMemAllocHost_v2(C.byref(h), C.c_size_t(max(n, 1))), f"pinned {n} B")
        C.memset(h, 0, max(n, 1))             # zeroed: a graph run from it before any step (warm) reads valid ids
        self.hosts.append(h.value)
        return h.value

    def _graph(self, pid, argmax=0):
        """bucket pid's step as one CUDA graph: its inputs uploaded from a pinned staging buffer, its actions, its outputs (argmax=n:
        the last one reduced on the device to n row argmaxes) read back into pinned buffers. A step is: fill, launch, sync"""
        key = (pid, argmax)
        if key not in self.graphs:
            p, base, keep, g, ge = self.plans[pid], self.bases[pid], [], C.c_void_p(), C.c_void_p()
            ins = [a for a in p["actions"] if "input" in a]
            stage = self._pinned(sum(a["n"] for a in ins))
            oa = [a for a in p["actions"] if "output" in a]
            outs = [(self._pinned(4 * argmax if argmax and k == len(oa) - 1 else a["n"]),
                     4 * argmax if argmax and k == len(oa) - 1 else a["n"]) for k, a in enumerate(oa)]
            if argmax and getattr(self, "_amax", None) is None:    # no allocation while capturing
                self._amax = self._alloc(4 * 4096, zero=False)
            self.ck(self.cu.cuStreamBeginCapture_v2(self.st, 0), "begin capture")
            o = 0
            for a in ins:
                self.ck(self.cu.cuMemcpyHtoDAsync_v2(C.c_uint64(base[a["dst"][0]] + a["dst"][1]), C.c_void_p(stage + o), C.c_size_t(a["n"]),
                                                     self.st), a["input"])
                o += a["n"]
            if os.environ.get("GOETHE_DAG") == "1" and p["actions"] and all("after" in a for a in p["actions"]):
                self._issue_dag(p["actions"], keep, base)
            else:
                for a in p["actions"]:
                    self._issue(a, keep, base)
            for k, (a, (h, n)) in enumerate(zip(oa, outs)):
                src = base[a["src"][0]] + a["src"][1]
                if argmax and k == len(oa) - 1:
                    prm = [C.c_uint64(src), C.c_int(a["n"] // 4 // argmax), C.c_uint64(self._amax)]
                    arr = (C.c_void_p * 3)(*[C.cast(C.pointer(x), C.c_void_p) for x in prm])
                    keep += [prm, arr]
                    self.ck(self.cu.cuLaunchKernel(self.fn["gc_argmax_rows"], argmax, 1, 1, 1024, 1, 1, 0, self.st, arr, None), "argmax")
                    src = self._amax
                self.ck(self.cu.cuMemcpyDtoHAsync_v2(C.c_void_p(h), C.c_uint64(src), C.c_size_t(n), self.st), "output")
            self.ck(self.cu.cuStreamEndCapture(self.st, C.byref(g)), "end capture")
            self.ck(self.cu.cuGraphInstantiateWithFlags(C.byref(ge), g, C.c_ulonglong(0)), "instantiate")
            self.graphs[key] = (ge, keep, ins, stage, outs)
        return self.graphs[key]

    def warm(self, greedy=None):
        """capture every bucket's CUDA graph now (otherwise on first use) and run each once (zero inputs: token 0 at position 0), so
        the first real step finds its modules loaded; greedy {pid: n}: also its n-row argmax form. Every cache zeroed afterwards."""
        for pid in self.plans:
            self._graph(pid)
        for pid, n in (greedy or {}).items():
            self._graph(pid, n)
        for g in self.graphs.values():
            self.ck(self.cu.cuGraphLaunch(g[0], self.st), "warm launch")
        for s in range(self.plans[next(iter(self.plans))]["nseq"]):     # every cell finite again (zero_slot leaves KV cells)
            self.zero_slot(s, state=False)

    def step(self, pid, step, argmax=0):
        """run bucket pid on this step: its inputs generated into the staging buffer, its graph launched; the step's outputs (all of
        them, in order). argmax=n: the last output is n logits rows reduced on the device to their argmax (int32 each)"""
        pid = getattr(self, "alias", {}).get(pid, pid)     # the JIT's pick for this bucket on this device
        ge, _, ins, stage, outs = self._graph(pid, argmax)
        hp, o = self.plans[pid]["io"]["inputs"], 0
        for a in ins:
            b = generate(a["input"], a["n"], step, hp)
            P.need(len(b) == a["n"], f"input {a['input']}: {len(b)} B, the plan wants {a['n']}")
            C.memmove(stage + o, b, a["n"])
            o += a["n"]
        self.ck(self.cu.cuGraphLaunch(ge, self.st), "graph launch")
        self.ck(self.cu.cuStreamSynchronize(self.st), "sync")
        return [C.string_at(h, n) for h, n in outs]

    def _rows(self, state=False, kv=False):   # (device address of sequence block 0, block bytes) of every cache (state: recurrent
        p = next(iter(self.plans))                # ones only); kv: and whether it is a KV cache (rows of cells), not a state
        base = self.bases[p]
        for k, (b, off, n) in self.plans[p].get("caches", {}).items():
            rec = bool(re.match(r"cache_[rs]_l", k))
            if not state or rec:
                yield (base[b] + off, n // self.plans[p]["nseq"], not rec) if kv else (base[b] + off, n // self.plans[p]["nseq"])

    @functools.cached_property
    def cell_major(self):        # flash attention in the plans (llama.cpp's context-wide choice): V not transposed, every KV row
        fa = ("flash_attn", "BatchDecode", "Prefill", "gc_fa_")     # [cells][heads][dim], so a stream's first n cells are a prefix
        return any(any(x in a.get("launch", "") for x in fa) for pl in self.plans.values() for a in pl["actions"])

    def copy_slot(self, src, dst, cells=None, kv_size=None):
        """stream src's caches into stream dst: whole, or (cells, kv_size: cell-major KV caches) only their first cells cells (a
        sequence's computed positions; the rest is masked), recurrent state whole"""
        for a, row, kv in self._rows(kv=True):
            n = row
            if kv and cells is not None and kv_size and self.cell_major and row % kv_size == 0:
                n = min(row, row // kv_size * cells)
            if n:
                self.ck(self.cu.cuMemcpyDtoDAsync_v2(C.c_uint64(a + dst * row), C.c_uint64(a + src * row), C.c_size_t(n), self.st),
                        "copy slot")
        self.ck(self.cu.cuStreamSynchronize(self.st), "sync")

    def zero_slot(self, s, state=True):
        """stream s zeroed: its recurrent state (a new sequence starts from zero); state=False: its KV cells too (a new sequence
        needs no KV zeroing: cells past its positions are masked to weight 0 and hold finite values, warm() zeroes them once)"""
        rows = list(self._rows(state))
        if not rows:
            return
        for a, row in rows:
            self.ck(self.cu.cuMemsetD8Async(C.c_uint64(a + s * row), C.c_ubyte(0), C.c_size_t(row), self.st), "zero slot")
        self.ck(self.cu.cuStreamSynchronize(self.st), "sync")

    def close(self):
        if getattr(self, "ctx", None):
            for g in self.graphs.values():
                self.cu.cuGraphExecDestroy(g[0])
            for h in self.hosts:
                self.cu.cuMemFreeHost(C.c_void_p(h))
            for m in self.mods.values():
                self.cu.cuModuleUnload(m)
            for a in self.allocs:
                self.cu.cuMemFree_v2(C.c_uint64(a))
            self.cu.cuStreamDestroy_v2(self.st)
            self.cu.cuDevicePrimaryCtxRelease(self.dev)
            self.ctx = None


def _open(pack, weights, kinds, *, trusted, store, sku, device, price, allow_broken):
    """(stage, manifest, [buckets of kind], Runtime): a gc-plan pack staged and verified, the buckets of that kind of the variant the
    weights match, their plans on the device"""
    stage = P.stage(pack, trusted=trusted, defer=True, store=store)
    ws = [weights] if isinstance(weights, str) else list(weights)
    sel = {}

    def pick(m, dev):
        bs = [b for b in m["buckets"] if P.axes(b)[1] in kinds]
        P.need(bs, f"the pack has no {'/'.join(kinds)} bucket")
        sel["all"] = bs
        return bs[0], bs, bs[0]
    br = P.open_branch(stage, ws, pick, sku=sku, ordinal=device, price=price, allow_broken=allow_broken)
    bs = [b for b in sel["all"] if b["variant"] == br.bucket["variant"]]
    path = lambda b, rel: os.path.join(stage.dir, "blobs", b["files"][rel])
    plans, byid, share, out, cands = {}, {}, {}, [], []
    for b in bs:
        p0 = json.load(open(path(b, b["lib"])))
        a = b.get("affine")
        alts = b.get("alts", ())                         # S3's near ties: the JIT times them on this device (jit)
        if not a:
            plans[b["id"]], byid[b["id"]] = p0, b
            for i, alt in enumerate(alts):
                plans[f"{b['id']}~{i}"], byid[f"{b['id']}~{i}"] = json.load(open(path(b, alt["lib"]))), b
            out.append(b)
            if alts:
                cands.append((b, b["id"], [b["id"]], len(alts)))
            continue
        p1, (w0, w1) = json.load(open(path(b, a["lib1"]))), (a["n_kv0"], a["n_kv1"])     # every width between, step 256
        top = f"{b['id']}@{w1}"
        apair = [(json.load(open(path(b, x["lib"]))), json.load(open(path(b, x["lib1"])))) for x in alts if "lib1" in x]
        for w in range(w0, w1 + 1, 256):
            pid = f"{b['id']}@{w}"
            lerp = lambda q0, q1: q0 if w == w0 else q1 if w == w1 else lerp_plan(q0, q1, (w - w0, w1 - w0))
            plans[pid] = lerp(p0, p1)
            byid[pid] = b
            out.append({**b, "id": pid, "io": plans[pid]["io"]})
            for i, (q0, q1) in enumerate(apair):           # an alternative segment, at the same widths
                plans[f"{pid}~{i}"], byid[f"{pid}~{i}"] = lerp(q0, q1), b
                if w != w1:
                    share[f"{pid}~{i}"] = f"{top}~{i}"
            if w != w1:
                share[pid] = top
        if apair:
            cands.append(({**b, "id": top, "io": plans[top]["io"]}, top, [f"{b['id']}@{w}" for w in range(w0, w1 + 1, 256)], len(apair)))
    rt = Runtime(plans, lambda rel, pid: open(path(byid[pid], rel), "rb").read(), ws, device, share)
    base = {}                                      # each chunk length's narrowest and widest buckets (and the one-sequence decode's):
    for b in out:                                  # timed once on this device, the backend's price of a launch, affine in the width
        k = (b["kind"], b.get("prefill") if b["kind"] == "prefill" else b.get("ncol"), b.get("ncol", 1))   # between (_split, _packed)
        if k[0] == "prefill" or (k[0] == "decode" and k[1] == 1):
            for e, better in ((0, lambda x, y: x < y), (1, lambda x, y: x > y)):
                if (k, e) not in base or better(b["io"]["n_kv"], base[k, e]["io"]["n_kv"]):
                    base[k, e] = b
    reps = {c[1] for c in cands}
    jit(rt, cands + [(b, b["id"], [], 0) for b in {b["id"]: b for b in base.values()}.values() if b["id"] not in reps], pack)
    return stage, br.m, out, rt


def jit(rt, buckets, pack):
    """the pack's near-tie alternatives (S3's, per bucket) timed on this device, the fastest serving each bucket: in-graph, round
    robin over the candidates (drift cancels), median of the launches; cached per (GPU UUID, pack, driver) in
    ~/.goethe/tune/<uuid>-<pack sha>.json (a later start reads it; GOETHE_JIT=0 skips, =force re-times); each bucket's timings
    also by its own content (its files' hashes, the plan's io and streams, this code) in tune/<uuid>-<driver>/, so a new pack
    re-times only the buckets that changed"""
    import hashlib
    import time
    if not buckets or os.environ.get("GOETHE_JIT") == "0":             # buckets: [(bucket, timed plan id, member ids, n alts)]
        return
    uu, drv = (C.c_ubyte * 16)(), C.c_int()
    rt.ck(rt.cu.cuDeviceGetUuid_v2(uu, rt.dev), "uuid")
    rt.ck(rt.cu.cuDriverGetVersion(C.byref(drv)), "driver")
    h = hashlib.sha256()
    with open(pack, "rb") as f:
        h.update(f.read(1 << 20))
    key = f"{bytes(uu).hex()}-{h.hexdigest()[:16]}-{drv.value}"
    cache = os.path.join(os.environ.get("GOETHE_HOME", os.path.expanduser("~/.goethe")), "tune", key + ".json")
    if os.path.exists(cache) and os.environ.get("GOETHE_JIT") != "force":
        d = json.load(open(cache))
        rt.alias = {k: v for k, v in d["pick"].items() if v in rt.plans}
        rt.cost = {k: v[k] / 1e6 for k, v in d["us"].items()}
        return
    e0, e1, ms = C.c_void_p(), C.c_void_p(), C.c_float()
    rt.ck(rt.cu.cuEventCreate(C.byref(e0), 0), "event")
    rt.ck(rt.cu.cuEventCreate(C.byref(e1), 0), "event")
    pick, t0, log, rt.cost = {}, time.perf_counter(), {}, {}
    gdir = os.path.join(os.path.dirname(cache), f"{bytes(uu).hex()}-{drv.value}")
    own = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    for b, rep, members, na in buckets:           # (bucket, the plan timed: an affine bucket's widest, its widths, alternatives)
        if b.get("ncol", 1) > rt.plans[rep]["nseq"] - 1:    # more streams than this device's caches hold (_fit_streams)
            continue
        gk = hashlib.sha256(json.dumps([own, sorted(b["files"].items()), rt.plans[rep]["io"], rt.plans[rep]["nseq"], na,
                                        os.environ.get("GOETHE_DAG")], sort_keys=True).encode()).hexdigest()
        gf = os.path.join(gdir, gk + ".json")
        if os.path.exists(gf) and os.environ.get("GOETHE_JIT") != "force":     # these plans timed on this device before
            med = {rep + sfx: v for sfx, v in json.load(open(gf)).items()}
        else:
            med = _time_bucket(rt, b, rep, na, e0, e1, ms)
            os.makedirs(gdir, exist_ok=True)
            json.dump({pid[len(rep):]: v for pid, v in med.items()}, open(gf + ".tmp", "w"))
            os.replace(gf + ".tmp", gf)
        best = min(med, key=med.get)
        for m in members:                         # every width of the bucket takes the pick
            pick[m] = m + best[len(rep):]
        log[rep] = {pid: round(v * 1000, 1) for pid, v in med.items()}
        rt.cost[rep] = med[rep] / 1000
    for a, row in rt._rows():                     # the timing steps wrote the caches: zeroed for the requests to come
        rt.ck(rt.cu.cuMemsetD8Async(C.c_uint64(a), C.c_ubyte(0), C.c_size_t(row * next(iter(rt.plans.values()))["nseq"]), rt.st), "zero")
    rt.ck(rt.cu.cuStreamSynchronize(rt.st), "sync")
    rt.alias = {k: v for k, v in pick.items() if v != k}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump({"pick": pick, "us": log, "seconds": round(time.perf_counter() - t0, 2)}, open(cache + ".tmp", "w"), indent=1)
    os.replace(cache + ".tmp", cache)


def _time_bucket(rt, b, rep, na, e0, e1, ms):
    """{plan id: median ms} of bucket b's timed plan rep and its na alternatives: one real step each, then the graphs round robin"""
    io, n = rt.plans[rep]["io"], b.get("ncol") if b["kind"] == "decode" else b.get("prefill", 1)
    w, kv = io["n_kv"], io.get("kv_size", io["n_kv"])
    pos = max(w - (256 if b["kind"] == "decode" else n), 0)    # the widest positions the bucket serves
    S = b.get("ncol", 1) if b["kind"] == "prefill" else 1     # a packed prefill: S sequences of n tokens, sequence s in stream s
    p, sq = list(range(pos, pos + n)) * S, [s for s in range(S) for _ in range(n)]
    st = Step([1] * n, range(n), [pos] * n, [pos] * n, range(n), kv, w, range(n)) if b["kind"] == "decode" else \
        Step([1] * n * S, sq, p, p, sq, kv, w, [s * n + n - 1 for s in range(S)])
    pids = [rep] + [f"{rep}~{i}" for i in range(na)]
    ge = {}
    for pid in pids:                          # one real step each (fills the graph's inputs), then its graph alone
        rt.step(pid, st)
        ge[pid] = rt._graph(pid)[0]
    ts = {pid: [] for pid in pids}
    for _ in range(7):
        for pid in pids:
            rt.ck(rt.cu.cuEventRecord(e0, rt.st), "event")
            rt.ck(rt.cu.cuGraphLaunch(ge[pid], rt.st), "launch")
            rt.ck(rt.cu.cuEventRecord(e1, rt.st), "event")
            rt.ck(rt.cu.cuEventSynchronize(e1), "sync")
            rt.ck(rt.cu.cuEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
            ts[pid].append(ms.value)
    return {pid: float(np.median(v[2:])) for pid, v in ts.items()}


class PlanEmbed:
    """An embedding model from a gc-plan pack: encode buckets, one per chunk length T; an item runs in the smallest T that holds it,
    padded (pad keys masked, pooling over the real tokens)."""

    def __init__(self, pack, weights, *, device=-1, trusted=None, store=None, sku=None, price=None, allow_broken=False,
                 numerics="invariant", **_):
        self.rt = self.stage = None
        self.stage, self.m, bs, self.rt = _open(pack, weights, ("encode",), trusted=trusted, store=store, sku=sku, device=device,
                                                price=price, allow_broken=allow_broken)
        self.ladder = sorted((b["maxctx"], b["id"]) for b in bs)
        self.bucket = dict(bs[0], maxctx=self.ladder[-1][0])
        io = self.bucket.get("io") or {}
        self.dim, self.pool = io["out"]["n_embd"], io.get("pool", "mean")
        self.max_len = self.max_cols = self.ladder[-1][0]
        self.max_seqs = 1
        self.numerics = dict(contract=self.bucket["contract"], determinism=self.bucket.get("determinism"))

    def embed(self, items, norm=True):
        """float32 [len(items), dim]: the pooled row of each token list (L2 normalized when norm)"""
        out = np.empty((len(items), self.dim), np.float32)
        for i, t in enumerate(items):
            n = len(t)
            P.need(0 < n <= self.max_len, f"item {i}: {n} tokens, the pack takes 1..{self.max_len}")
            T, pid = next(x for x in self.ladder if x[0] >= n)
            step = Step(list(t) + [0] * (T - n), [0] * T, range(T), range(T), [0] * T, T, T, range(T), n_real=n)
            v = np.frombuffer(self.rt.step(pid, step)[0], np.float32)[:self.dim]
            out[i] = v / max(float(np.linalg.norm(v)), 1e-12) if norm else v
        return out

    def close(self):
        if self.rt:
            self.rt.close()
            self.rt = None
        if self.stage:
            self.stage.close()
            self.stage = None

    __del__ = close


class PlanBackend:
    """A decoder from a gc-plan pack, as a page backend (paged=False: an engine block is one sequence; the backend keeps its own map
    of sequences to llama.cpp streams and recurrent slots). Prefill and decode buckets share one KV width n_kv: a bucket serves any
    positions whose cells fit it (cells past the position are masked). A bucket's graph fixes its streams: prefill buckets run
    stream 0, a B-sequence decode bucket streams [0, B). So live sequences are kept packed in streams [0, n) (a finished one's
    hole takes the last), the top stream is spare: a prefill runs in stream 0 with its occupant parked in the spare; a step that
    decodes every live sequence runs the smallest decode bucket of at least that many (its rows past the live ones in free
    streams, their outputs dropped), any other decode runs one sequence at a time the same way. A token run splits exactly into
    the prefill chunk lengths, largest first; a shorter tail runs padded in the smallest prefill chunk that holds it (causal
    attention, row-wise ops), or, for a recurrent model (padding would advance its state), through the one-sequence decode bucket. Buckets come in KV widths n_kv (llama.cpp's cells per stream, padded
    to 256): a launch takes the width llama.cpp would, pad(last position + 1), so short contexts read few cells and the numerics
    are llama.cpp's own."""
    batches_all_live = True     # a decode bucket runs every live sequence (cache blocks [0, n)): the scheduler prefills first

    @property
    def prefill_align(self):    # the largest prefill chunk: a prompt piece of whole chunks runs in whole buckets
        return max(self.chunks) if self.chunks else 1
    paged, state_pages, prefix_cache, bounded = False, 0, False, False

    def __init__(self, pack, weights, *, device=-1, trusted=None, store=None, sku=None, price=None, allow_broken=False, **_):
        self.rt = self.stage = None
        self.stage, self.m, bs, self.rt = _open(pack, weights, ("prefill", "decode"), trusted=trusted, store=store, sku=sku,
                                                device=device, price=price, allow_broken=allow_broken)
        io = {b["id"]: b["io"] for b in bs}
        self.decs = {(b["ncol"], io[b["id"]]["n_kv"]): b["id"] for b in bs if b["kind"] == "decode"}   # (sequences, n_kv) -> bucket
        self.pres = {(b["prefill"], io[b["id"]]["n_kv"]): b["id"] for b in bs if b["kind"] == "prefill" and b.get("ncol", 1) == 1}
        # packed prefills: S prompts of T tokens in streams [0, S) at once, (S, T) -> bucket (their positions from 0)
        self.packs = {(b["ncol"], b["prefill"], io[b["id"]]["n_kv"]): b["id"] for b in bs if b["kind"] == "prefill" and b.get("ncol", 1) > 1}
        self.widths = sorted({w for _, w in self.decs} | {w for _, w in self.pres})
        P.need(any(n == 1 for n, _ in self.decs), "the pack has no one-sequence decode bucket")
        self.dec = self.decs[max(k for k in self.decs if k[0] == 1)]
        self.chunks = sorted({t for t, _ in self.pres}, reverse=True)
        cs = getattr(self.rt, "cost", {})               # each chunk length's launch time at its narrowest width (jit)
        tab = [(t, self.pres) for t in self.chunks] + [(1, self.decs)]     # 1: a token through the one-sequence decode bucket
        self.price = {t: sorted((w, cs[p]) for (n, w), p in tb.items() if n == t and p in cs) for t, tb in tab}
        if not all(self.price.values()):                # not every length timed: launched tokens for all
            self.price = {}
        self.pprice = {k[:2]: sorted((w, cs[p]) for (s, t, w), p in self.packs.items() if (s, t) == k[:2] and p in cs) for k in self.packs}
        self.bucket = next(b for b in bs if b["id"] == self.dec)
        self.n_kv, self.kv_size = self.widths[-1], io[self.dec]["kv_size"]
        P.need(all(x["kv_size"] == self.kv_size for x in io.values()), "buckets recorded at different capacities")
        self.nseq = self.rt.plans[self.dec]["nseq"]
        over = {v for k, v in list(self.decs.items()) + list(self.packs.items()) if k[0] > self.nseq - 1}
        self.decs = {k: v for k, v in self.decs.items() if v not in over}      # the streams this device's caches hold
        self.packs = {k: v for k, v in self.packs.items() if v not in over}
        for pid in [x for x in self.rt.plans if x.split("~")[0] in over]:      # (and their JIT alternatives): never warmed or run
            del self.rt.plans[pid]
        self.spare = self.nseq - 1
        self.vocab_size, self.max_len = io[self.dec]["out"]["vocab"], self.n_kv
        self.max_seqs = max(1, min(max(n for n, _ in self.decs), self.nseq - 1))
        self.live = []                       # sids, index = stream
        self.used = {}                       # sid -> cells computed
        from .backend import KVGeometry
        self.geometry = KVGeometry(block_size=self.max_len, num_blocks=self.max_seqs, layers=1, kv_heads=1, head_dim=1)
        rec = any(re.match(r"cache_[rs]_l", k) for k in self.rt.plans[self.dec].get("caches", {}))
        self.caps = {"recurrent": "chain"} if rec else {}
        # paged KV (a pack recorded with GC_PAGED): attention reads pages of `page` cells through the gc_pages tables, so a
        # sequence's cells are anywhere in the cache (page 0: the padding rows' scratch); no stream moves, no spare
        self.page = self.rt.plans[self.dec]["io"]["inputs"].get("page")
        if self.page:
            P.need(not rec, "gc-plan: paged KV with recurrent state")
            self.free = list(range(self.nseq * self.kv_size // self.page - 1, 0, -1))
            self.tab = {}                    # sid -> its pages
            self.max_seqs = max(1, min(max(n for n, _ in self.decs), self.nseq))
        self.numerics = dict(contract=self.bucket["contract"], determinism=self.bucket.get("determinism"))

    def warm(self):
        """every bucket's graph captured now, the decode buckets in their greedy (device argmax) form too"""
        self.rt.warm({**{pid: n for (n, _), pid in self.decs.items()}, **{pid: k[0] for k, pid in self.packs.items()}}
                     if "gc_argmax_rows" in self.rt.fn else None)

    def prefill_share(self, n, budget):
        """tokens per prompt when n prompts prefill in one step (the scheduler's split): the budget evenly, a whole number of the
        widest packed chunk for n streams, so packed prefill runs them at once (_packed's measured prices may still run them
        alone); None (first come first served): no packed bucket for n, less than one chunk each, or (measured) its launch costs
        more per real token than the cheapest single-prompt chunk"""
        fit = [k for k in self.packs if k[0] >= n]
        if not _SPLIT or not _PACKED or not _PAD or self.caps.get("recurrent") or not fit:
            return None
        S = min(k[0] for k in fit)
        T = max(k[1] for k in fit if k[0] == S)
        pts = self.pprice.get((S, T))
        if self.price and pts:                   # measured, both at one width (the middle of the packed bucket's, which a prompt's
            w = (pts[0][0] + pts[-1][0]) / 2     # chunks cross): a packed launch's seconds per real token vs the cheapest chunk's
            at = lambda q: q[0][1] if q[-1][0] == q[0][0] else \
                q[0][1] + (q[-1][1] - q[0][1]) * (min(max(w, q[0][0]), q[-1][0]) - q[0][0]) / (q[-1][0] - q[0][0])
            if at(pts) / (n * T) >= min(at(p) / t for t, p in self.price.items() if t > 1):     # not cheaper: no split
                return None
        per = budget // n
        return per - per % T if per >= T else None

    def _stream(self, sid):
        if sid not in self.live:
            P.need(len(self.live) < self.max_seqs, f"more than {self.max_seqs} live sequences")
            self.live.append(sid)
            self.rt.zero_slot(len(self.live) - 1)     # a new sequence starts from zero recurrent state
        return self.live.index(sid)

    def _pick(self, table, n, end):
        """the bucket of table {(n, n_kv): id} for n whose KV width is the one llama.cpp reads when the last cell used is end - 1
        (end padded to 256), else the narrowest wider one (cells past the position are masked); None: no width holds it.
        Paged KV: the narrowest (attention reads the page tables, not the width)"""
        ws = sorted(w for k, w in table if k == n and (getattr(self, "page", None) or w >= -(-end // 256) * 256))
        return (table[(n, ws[0])], ws[0]) if ws else (None, None)

    def _split(self, pos, r, priced=False):
        """[(chunk, real tokens)] for r tokens from pos, the least measured time (else launched tokens), then the fewest launches:
        exact chunks, or (not
        for a recurrent model: padding would advance its state; GOETHE_PAD=0) one padded chunk holding the rest (causal attention
        and row-wise ops: padding changes no real row; its KV cells sit past the last real position, masked, then overwritten by
        decode). No chunk fits: token by token through the one-sequence decode bucket (only then: it measured ~10x a padded chunk)"""
        price = getattr(self, "price", {})          # {chunk: [(width, seconds)]} timed at load (else launched tokens)
        paged = getattr(self, "page", None)
        picks = {}

        def pick(t, end):                            # (pid, width) of chunk t ending at end; a pick depends on end only through its
            k = (t, end if paged else -(-end // 256))   # 256-cell width class: cached, O(C r / 256) picks for r tokens, C chunks
            if k not in picks:
                picks[k] = self._pick(self.pres, t, end) if t > 1 else self._pick(self.decs, 1, end)
            return picks[k]

        def cost(t, q):                              # the launch at the width it takes: affine between the timed widths (paged KV:
            if t not in price:                       # at the chunk's end, its attention's length; each width was timed ending there)
                return t
            w, pts = pick(t, q + t)[1], price[t]
            if paged:
                w = min(max(q + t, pts[0][0]), pts[-1][0])
            (w0, c0), (w1, c1) = pts[0], pts[-1]
            return c0 if w1 == w0 else c0 + (c1 - c0) * (w - w0) / (w1 - w0)
        pad_ok = _PAD and not self.caps.get("recurrent")
        # the states reachable from 0 tokens done (a fitting chunk's step, else one token) in one forward pass, then best[j] =
        # (seconds, launches) to finish from j over them in reverse: O(|R| C) for |R| reachable states and C chunks (sparse for large
        # chunks, at most r + 1), no recursion, back-pointers how[j] = (chunk, real tokens) instead of per-state plan copies
        steps, reach, stack = {}, {0}, [0]
        while stack:
            j = stack.pop()
            q, left = pos + j, r - j
            if left == 0:
                continue
            opts = [(t, j + t if t <= left else None) for t in self.chunks         # in chunk order (ties: the first, as before);
                    if (t <= left or pad_ok) and pick(t, q + t)[0] is not None]  # k None: padded, the rest in one launch
            steps[j] = opts or [(1, j + 1)]
            for _, k in steps[j]:
                if k is not None and k not in reach:
                    reach.add(k)
                    stack.append(k)
        best, how = {r: (0, 0)}, {}
        for j in sorted(reach - {r}, reverse=True):
            q, c = pos + j, None
            for t, k in steps[j]:
                x = ((best[k][0] + cost(t, q), best[k][1] + 1), (t, t)) if k is not None else ((cost(t, q), 1), (t, r - j))
                if c is None or x[0] < c[0]:
                    c = x
            best[j], how[j] = c
        if priced:
            return best[0][0]                        # its measured seconds
        plan, j = [], 0
        while j < r:
            plan.append(how[j])
            j += how[j][0] if how[j][1] == how[j][0] else r - j
        return plan

    def _cells(self, sid, pos):
        """paged KV: the cells of sid's positions pos (its pages, taken from the free list as it grows)"""
        t = self.tab.setdefault(sid, [])
        while len(t) <= max(pos) // self.page:
            P.need(self.free, "gc-plan: paged KV: no free page")
            t.append(self.free.pop())
        return [t[x // self.page] * self.page + x % self.page for x in pos]

    def _run(self, toks, pos, sid=None):
        """logits row of the last token of toks run at positions pos .. pos + len - 1 in stream 0 (paged KV: in sid's pages)"""
        i, row = 0, None
        for T, n in self._split(pos, len(toks)):
            q = pos + i
            pid, W = self._pick(self.pres, T, q + T) if T > 1 else self._pick(self.decs, 1, q + 1)
            P.need(pid is not None, f"position {q + T} past the pack's KV widths")
            p = list(range(q, q + T))
            tk = list(toks[i:i + n]) + [toks[i + n - 1]] * (T - n)
            st = Step(tk, [0] * T, p, self._cells(sid, p) if self.page else p, [0] * T, self.kv_size, W, [n - 1] if T > 1 else [0])
            if self.page:
                st.pages = [(T, list(self.tab[sid]), q + T)]
            row = self.rt.step(pid, st)[-1]
            i += n
        return np.frombuffer(row, np.float32)[:self.vocab_size]

    def _packed(self, pre, greedy=False):
        """prompt pieces filling the live streams at once (every live sequence is one of them, all from one position q0) in packed
        prefills: the smallest S >= their count, the smallest T >= their longest (else the largest), chunk by chunk (q = 0, T,
        2T, ...) at the width that holds q + T; each stream's chunk padded to T (causal: padding changes no real row; a finished
        stream's padding writes cells past its prompt, masked, then overwritten by decode), free streams' rows dropped; a sequence
        takes the row of its last token from the chunk that holds it; past the widest packed width the rest runs alone.
        {sid: last row} of those it ran (none: not this step's shape); greedy: {sid: token}, each row's argmax on the device"""
        if not self.packs or self.page or len(pre) < 2 or self.caps.get("recurrent") or not _PAD or not _PACKED or len({w.pos for w in pre}) != 1:
            return {}                            # (prompt pieces at one common position: the scheduler feeds long prompts in chunks)
        for w in pre:
            self._stream(w.sid)
        if sorted(self.live) != sorted(w.sid for w in pre):
            return {}
        n = len(pre)
        fit = {(k[0], k[1]) for k in self.packs if k[0] >= n}
        if not fit:
            return {}
        S = min(k[0] for k in fit)
        Ts = sorted(t for s, t in fit if s == S)
        ws = [{w.sid: w for w in pre}[self.live[s]] for s in range(n)]
        longest = max(len(w.tokens) for w in ws)
        T = next((t for t in Ts if t >= longest), Ts[-1])
        widths = {k[2]: pid for k, pid in self.packs.items() if k[:2] == (S, T)}
        q0 = ws[0].pos
        pts = self.pprice.get((S, T))
        if self.price and pts:                   # measured: the packed launches vs every prompt alone (the cheaper runs)
            lin = lambda w: pts[0][1] if pts[-1][0] == pts[0][0] else \
                pts[0][1] + (pts[-1][1] - pts[0][1]) * (w - pts[0][0]) / (pts[-1][0] - pts[0][0])
            fit = lambda q: next((x for x in sorted(widths) if x >= -(-(q + T) // 256) * 256), None)
            qs = range(q0, q0 + longest, T)
            if any(fit(q) is None for q in qs) or \
                    sum(lin(fit(q)) for q in qs) >= sum(self._split(q0, len(w.tokens), priced=True) for w in ws):
                return {}
        rows, q = {}, q0
        while q < q0 + longest:
            W = next((x for x in sorted(widths) if x >= -(-(q + T) // 256) * 256), None)
            if W is None:
                break
            toks, sq, pos, outs = [], [], [], []
            for s in range(S):
                t = list(ws[s].tokens[q - q0:q - q0 + T]) if s < n else []
                t = t or [ws[s].tokens[-1] if s < n else 0]
                last = len(ws[s].tokens) - 1 - (q - q0) if s < n else -1
                outs.append(s * T + (last if 0 <= last < T else 0))
                toks += t + [t[-1]] * (T - len(t))
                sq += [s] * T
                pos += list(range(q, q + T))
            st = Step(toks, sq, pos, pos, sq, self.kv_size, W, outs)
            if greedy:
                P.need(self.rt_vocab(widths[W], S) == self.vocab_size, "gc-plan: logits rows wider than the vocabulary")
                out = np.frombuffer(self.rt.step(widths[W], st, argmax=S)[-1], np.int32)
            else:
                out = np.frombuffer(self.rt.step(widths[W], st)[-1], np.float32).reshape(S, -1)
            for s in range(n):
                if q - q0 <= len(ws[s].tokens) - 1 < q - q0 + T:
                    rows[ws[s].sid] = int(out[s]) if greedy else out[s, :self.vocab_size].copy()
            q += T
        for w in ws:                             # past the widest packed width: the rest alone
            if w.sid not in rows:
                rows[w.sid] = self._alone(dataclasses.replace(w, tokens=w.tokens[q - q0:], pos=q))
        return rows

    def _alone(self, w):
        """one sequence's run in stream 0, its stream's occupant parked in the spare (paged KV: in its own pages, nothing moves)"""
        k = self._stream(w.sid)
        if self.page:
            return self._run(list(w.tokens), w.pos, w.sid)
        occ, end = self.live[0], w.pos + len(w.tokens)
        if k:                                    # only computed cells move (a fresh prompt's: none)
            self.rt.copy_slot(0, self.spare, self.used.get(occ, 0), self.kv_size)
            self.rt.copy_slot(k, 0, w.pos, self.kv_size)
        row = self._run(list(w.tokens), w.pos)
        if k:
            self.rt.copy_slot(0, k, end, self.kv_size)
            self.rt.copy_slot(self.spare, 0, self.used.get(occ, 0), self.kv_size)
        return row

    def step(self, work):
        for w in work:
            P.need(w.want in (0, 1) and not w.spec and not w.feats and w.parents is None,
                   "gc-plan decoders return the last row (or none, a prompt chunk before its last) so far")
            P.need(w.pos + len(w.tokens) <= self.max_len, f"position {w.pos + len(w.tokens)} past the pack's {self.max_len}")
        rows = {}
        greedy = all(w.greedy or not w.want for w in work) and "gc_argmax_rows" in self.rt.fn     # tokens, not logits
        for w in work:                         # cells each sequence has computed once this step runs (what a stream move copies)
            self.used[w.sid] = max(self.used.get(w.sid, 0), w.pos + len(w.tokens))
        dec = [w for w in work if len(w.tokens) == 1 and w.sid in self.live]
        isdec = {id(w) for w in dec}          # by identity: a work item's == compares every field
        rows.update(self._packed([w for w in work if id(w) not in isdec], greedy))
        for w in work:
            if id(w) not in isdec and w.sid not in rows:
                rows[w.sid] = self._alone(w)
        n = len(dec) if self.page else len(self.live)
        end = max((w.pos for w in dec), default=0) + 1
        pid = B = W = None                  # the smallest ladder batch B >= n with a width that holds end
        for b in sorted({k for k, _ in self.decs if k >= n}):
            pid, W = self._pick(self.decs, b, end)
            if pid:
                B = b
                break
        if dec and self.page and pid:         # paged KV: the decoding sequences in their pages, rows past them on page 0
            cells = [self._cells(w.sid, [w.pos])[0] for w in dec]
            st = Step([w.tokens[0] for w in dec] + [0] * (B - len(dec)), list(range(B)), [w.pos for w in dec] + [0] * (B - len(dec)),
                      cells + [0] * (B - len(dec)), [0] * B, self.kv_size, W, list(range(B)))
            st.pages = [(1, list(self.tab[w.sid]), w.pos + 1) for w in dec] + [(1, [0], 1)] * (B - len(dec))
            if greedy:
                P.need(self.rt_vocab(pid, B) == self.vocab_size, "gc-plan: logits rows wider than the vocabulary")
                ids = np.frombuffer(self.rt.step(pid, st, argmax=B)[-1], np.int32)
                for i, w in enumerate(dec):
                    rows[w.sid] = int(ids[i])
            else:
                out = np.frombuffer(self.rt.step(pid, st)[-1], np.float32).reshape(B, -1)
                for i, w in enumerate(dec):
                    rows[w.sid] = out[i, :self.vocab_size]
        elif dec and {w.sid for w in dec} == set(self.live) and pid:     # every live sequence, one batch
            by = {w.sid: w for w in dec}
            ws = [by[s] for s in self.live]
            # rows n .. B-1 run in free streams (the live ones are packed in [0, n)), position 0, their outputs dropped: a free
            # stream's state is garbage until a sequence takes it (zeroed then)
            st = Step([w.tokens[0] for w in ws] + [0] * (B - n), list(range(B)), [w.pos for w in ws] + [0] * (B - n),
                      [w.pos for w in ws] + [0] * (B - n), list(range(B)), self.kv_size, W, list(range(B)))
            if greedy:                    # the argmax of each row's first vocab_size logits (the buffer's rows may be wider)
                P.need(self.rt_vocab(pid, B) == self.vocab_size, "gc-plan: logits rows wider than the vocabulary")
                ids = np.frombuffer(self.rt.step(pid, st, argmax=B)[-1], np.int32)
                for i, s in enumerate(self.live):
                    rows[s] = int(ids[i])
            else:
                out = np.frombuffer(self.rt.step(pid, st)[-1], np.float32).reshape(B, -1)
                for i, s in enumerate(self.live):
                    rows[s] = out[i, :self.vocab_size]
        else:
            for w in dec:
                rows[w.sid] = self._alone(w)
        if greedy:                        # tokens per work item (the engine's list form), a row's argmax where one came back
            return [[rows[w.sid] if isinstance(rows[w.sid], int) else int(np.argmax(np.nan_to_num(rows[w.sid])))] if w.want else []
                    for w in work]
        keep = [rows[w.sid] for w in work if w.want]
        return np.stack(keep) if keep else np.empty((0, self.vocab_size), np.float32)

    def rt_vocab(self, pid, n):           # logits per row of bucket pid's last output
        return next(a for a in reversed(self.rt.plans[pid]["actions"]) if "output" in a)["n"] // 4 // n

    def copy_blocks(self, pairs):
        for src, dst in pairs:
            self.rt.copy_slot(src, dst)

    def snapshot(self):
        """the live sequences as they stand: their streams' cache rows (paged KV: every page) copied to host memory and this
        backend's own bookkeeping; restore(snap) puts both back, so a step sequence runs again from here. O(cache bytes held).
        One snapshot at a time: the host side is one pinned buffer per cache, kept and grown across snapshots (closed with
        the backend)"""
        import copy
        n = self.nseq if self.page else len(self.live)
        self.rt.ck(self.rt.cu.cuCtxSynchronize(), "sync")
        bufs, host = getattr(self, "_snap_bufs", {}), []
        for i, (a, row) in enumerate(self.rt._rows()):
            k = row * n
            if k and bufs.get(i, (0, 0))[1] < k:
                if i in bufs:
                    self.rt.cu.cuMemFreeHost(C.c_void_p(bufs[i][0]))
                h = C.c_void_p()
                self.rt.ck(self.rt.cu.cuMemAllocHost_v2(C.byref(h), C.c_size_t(k)), f"snapshot ({k >> 20} MiB pinned)")
                bufs[i] = (h.value, k)
            if k:
                self.rt.ck(self.rt.cu.cuMemcpyDtoH_v2(C.c_void_p(bufs[i][0]), C.c_uint64(a), C.c_size_t(k)), "snapshot")
            host.append((bufs.get(i, (0, 0))[0], k))
        self._snap_bufs = bufs
        own = {k: copy.deepcopy(getattr(self, k)) for k in ("live", "used", "free", "tab") if hasattr(self, k)}
        return host, own

    def restore(self, snap):
        """the state snapshot() took, back on the device and in this backend"""
        import copy
        host, own = snap
        self.rt.ck(self.rt.cu.cuCtxSynchronize(), "sync")
        for (a, _), (h, k) in zip(self.rt._rows(), host):
            if k:
                self.rt.ck(self.rt.cu.cuMemcpyHtoD_v2(C.c_uint64(a), C.c_void_p(h), C.c_size_t(k)), "restore")
        for k, v in own.items():
            setattr(self, k, copy.deepcopy(v))

    def release(self, sid):
        """the sequence's stream is freed: the last live sequence moves into the hole (streams stay packed); paged KV: its pages
        go back to the free list"""
        if self.page:
            self.free.extend(self.tab.pop(sid, []))
            if sid in self.live:
                self.live.remove(sid)
            self.used.pop(sid, None)
            return
        if sid not in self.live:
            return
        k, last = self.live.index(sid), len(self.live) - 1
        if k != last:
            self.rt.copy_slot(last, k, self.used.get(self.live[last]), self.kv_size)
            self.live[k] = self.live[last]
        self.live.pop()
        self.used.pop(sid, None)

    def close(self):
        if self.rt:
            for h, _ in getattr(self, "_snap_bufs", {}).values():     # (snapshot()'s pinned buffers)
                self.rt.cu.cuMemFreeHost(C.c_void_p(h))
            self._snap_bufs = {}
            self.rt.close()
            self.rt = None
        if self.stage:
            self.stage.close()
            self.stage = None

    __del__ = close
