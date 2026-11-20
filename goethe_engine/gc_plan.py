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


class Runtime:
    """the device side of a gc-plan pack: one context, the plans' buffers, their cubins, the weights, a CUDA graph per bucket"""

    def __init__(self, plans, blob, files, device=0, share=None):
        """plans: {bucket id: plan dict}; blob(rel, bucket id) -> bytes of that bucket file; files: the model's weight files; share:
        {bucket id: bucket id} whose scratch it reuses (the widths of an affine bucket: the widest one's buffers hold every width's)"""
        cu = self.cu = C.CDLL("libcuda.so.1")
        self.ck(cu.cuInit(0), "cuInit")
        dev, self.ctx = C.c_int(), C.c_void_p()
        self.ck(cu.cuDeviceGet(C.byref(dev), max(device, 0)), "device")
        self.ck(cu.cuDevicePrimaryCtxRetain(C.byref(self.ctx), dev), "context")
        self.dev = dev
        self.ck(cu.cuCtxSetCurrent(self.ctx), "context")
        self.st = C.c_void_p()
        self.ck(cu.cuStreamCreate(C.byref(self.st), 1), "stream")
        # weights and caches (kv) are shared by every bucket, by role and order (one model, one cache layout); scratch (compute)
        # buffers are each bucket's own (llama.cpp's allocator sizes and splits them per graph)
        first = next(iter(plans.values()))
        shared_layout = [(b["role"], b["size"]) for b in first["buffers"] if b["role"] != "compute"]
        self.shared, self.allocs, self.bases, self.hosts = [], [], {}, []
        for r, n in shared_layout:          # zeroed, weights too: ggml zeroes a quantized tensor's row padding at load
            self.shared.append(self._alloc(n, zero=True))
        share, arena_need = share or {}, 0
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
            if p.get("init"):                # init actions write their buffers once: those stay the bucket's own
                self.bases[pid] = [next(it) if b["role"] != "compute" else self._alloc(b["size"], zero=True) for b in p["buffers"]]
                continue
            off, self.bases[pid] = 0, []     # one bucket runs at a time: every bucket's scratch is carved from one arena
            for b in p["buffers"]:
                if b["role"] != "compute":
                    self.bases[pid].append(next(it))
                else:
                    self.bases[pid].append(off)
                    off += -(-max(b["size"], 1) // 4096) * 4096
            arena_need = max(arena_need, off)
        if arena_need:                       # the widest bucket's scratch, zeroed once; offsets become addresses
            arena = self._alloc(arena_need, zero=True)
            for pid, p in plans.items():
                if pid not in share and not p.get("init"):
                    self.bases[pid] = [x + arena if b["role"] == "compute" else x for x, b in zip(self.bases[pid], p["buffers"])]
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
        self._weights(first["weights"], files, self.bases[next(iter(plans))])
        for pid, p in plans.items():           # init actions (e.g. a weight relaid for another kernel): once, after the weights
            keep = []
            for a in p.get("init", ()):
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
            raise RuntimeError(f"{what}: CUDA {rc} {s.value.decode() if s.value else ''}")

    def _weights(self, bind, files, base):
        """every tensor the plan binds, from the weight files by name (raw bytes: the plan reads them as llama.cpp does)"""
        left = dict(bind)
        for path in files:
            off = gguf_offsets(path)
            with open(path, "rb") as f:
                for name in [n for n in left if n in off]:
                    buf, o, n = left.pop(name)
                    f.seek(off[name])
                    b = f.read(n)
                    P.need(len(b) == n, f"{path}: tensor {name} is short")
                    self.ck(self.cu.cuMemcpyHtoD_v2(C.c_uint64(base[buf] + o), b, C.c_size_t(n)), name)
        P.need(not left, f"weights: {len(left)} plan tensors in no weight file (first {next(iter(left), '')})")

    def _issue(self, a, keep, base):
        cu, st = self.cu, self.st
        addr = lambda w: base[w[0]] + w[1]
        if "launch" in a:
            args = []
            tm = {}
            for i, o, t in a.get("tmaps", ()):
                tm.setdefault(i, []).append((o, t))
            for i, (raw, ptrs) in enumerate(zip(a["args"], a["ptrs"])):
                raw = bytearray.fromhex(raw)
                for o, buf, off in ptrs:
                    struct.pack_into("<Q", raw, o, base[buf] + off)
                for o, t in tm.get(i, ()):     # a TMA descriptor in the argument: encoded by this driver at the real address
                    raw[o:o + 128] = self._tmap(t, base[t["gaddr"][0]] + t["gaddr"][1])
                args.append(C.create_string_buffer(bytes(raw), len(raw)))
            pp = (C.c_void_p * len(args))(*[C.cast(x, C.c_void_p) for x in args])
            keep += [args, pp]
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
            P.need(g["gemm"] == 1, "a batched cuBLAS GEMM: not carried yet")
            h, al, be = self._blas(), bytes.fromhex(g["alpha"]), bytes.fromhex(g["beta"])
            self.ck_blas(self.blas.cublasGemmEx(h, g["ta"], g["tb"], g["m"], g["n"], g["k"], al, C.c_uint64(addr(a["A"])), g["at"], g["lda"],
                                                C.c_uint64(addr(a["B"])), g["bt"], g["ldb"], be, C.c_uint64(addr(a["C"])), g["ct"], g["ldc"],
                                                g["comp"], g["algo"]), "cublasGemmEx")

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
    ~/.goethe/tune/<uuid>-<pack sha>.json (a later start reads it; GOETHE_JIT=0 skips, =force re-times)"""
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
    for b, rep, members, na in buckets:           # (bucket, the plan timed: an affine bucket's widest, its widths, alternatives)
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
        med = {pid: float(np.median(v[2:])) for pid, v in ts.items()}
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
        fits = lambda t, q: self._pick(self.pres, t, q + t)[0] is not None
        price = getattr(self, "price", {})          # {chunk: [(width, seconds)]} timed at load (else launched tokens)

        def cost(t, q):                              # the launch at the width it takes: affine between the timed widths
            if t not in price:
                return t
            w, pts = self._pick(self.pres, t, q + t)[1] if t > 1 else self._pick(self.decs, 1, q + 1)[1], price[t]
            (w0, c0), (w1, c1) = pts[0], pts[-1]
            return c0 if w1 == w0 else c0 + (c1 - c0) * (w - w0) / (w1 - w0)
        pad_ok = _PAD and not self.caps.get("recurrent")
        memo = {}

        def best(q, r):                           # ((tokens, launches), plan)
            if r == 0:
                return (0, 0), []
            if (q, r) not in memo:
                c = []
                for t in self.chunks:
                    if t <= r and fits(t, q):
                        (k, n), pl = best(q + t, r - t)
                        c.append(((k + cost(t, q), n + 1), [(t, t)] + pl))
                    elif t > r and pad_ok and fits(t, q):
                        c.append(((cost(t, q), 1), [(t, r)]))
                if not c:
                    (k, n), pl = best(q + 1, r - 1)
                    c.append(((k + cost(1, q), n + 1), [(1, 1)] + pl))
                memo[q, r] = min(c, key=lambda x: x[0])
            return memo[q, r]
        return best(pos, r)[0][0] if priced else best(pos, r)[1]   # priced: its measured seconds

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
            self.rt.close()
            self.rt = None
        if self.stage:
            self.stage.close()
            self.stage = None

    __del__ = close
