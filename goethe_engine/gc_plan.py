"""Runtime kind gc-plan v1: a data-only pack run by the engine itself through the CUDA driver API. A bucket's
files are plan.json (device buffers, the model tensors bound into them, the step's actions: kernel launches with their argument
bytes and pointer slots, memsets, device copies, inputs, outputs) and the cubins its launches name; no library, no cudart, no
cuBLAS. One CUDA graph per bucket, captured on first use; the step's inputs are generated here from the engine's own state.

A bucket of B sequences runs on cache blocks [0, B) (the graph fixes the first KV stream and recurrent slot): move_slot and
zero_slot move a sequence's blocks. Inputs (kinds of llama.cpp's graph inputs, the input layer on the device): token ids,
positions, the encoder KQ mask (non-causal; padded keys masked, pad queries kept finite), the symmetric sliding-window mask, mean
and cls pooling, per-row constants the compiler measured (io.inputs.fills), output ids.
"""
import ctypes as C
import json
import os
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
    """'CUDA0#self_kq_mask#0' -> 'self_kq_mask'"""
    return name.split("#")[1] if "#" in name else name


class Step:
    """one ubatch: tokens with their sequence, stream, cell and position; n_kv cells visible per stream; n_real: tokens past it are
    padding (an encoder chunk), masked out of attention and pooling"""
    def __init__(self, tokens, seqs, pos, cells, streams, kv_size, n_kv, outputs, n_real=None):
        self.tokens, self.seqs, self.pos, self.cells, self.streams = map(np.asarray, (tokens, seqs, pos, cells, streams))
        self.kv_size, self.n_kv, self.outputs = kv_size, n_kv, np.asarray(outputs)
        self.n_real = len(self.tokens) if n_real is None else n_real


def generate(name, n, step, hp):
    """bytes of the plan input `name` (n bytes in the plan) for this step; hp: the bucket's io.inputs (n_pos, causal, n_swa, fills)"""
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
        K = n // (mk["esize"] * T)
        keep = np.zeros((T, K), bool)
        keep[:, :step.n_real] = True
        if mk["swa"] and hp.get("n_swa"):   # symmetric window (non-causal): |p_q - p_k| <= n_swa / 2
            pq, pk = step.pos.astype(np.int64), np.arange(K)
            keep &= np.abs(pq[:, None] - pk[None, :]) <= hp["n_swa"] // 2
        if mk["esize"] == 4:
            return np.where(keep, 0, -np.inf).astype(np.float32).tobytes()
        return np.where(keep, np.uint16(0), NEG_INF_F16).astype(np.uint16).tobytes()
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
    if k == "out_ids":
        return step.outputs.astype(np.int32).tobytes()
    if k == "rs_s_copy":
        return np.unique(step.seqs).astype(np.int32).tobytes()
    raise P.PackError(f"plan input {k}: not a kind this engine generates")


class Runtime:
    """the device side of a gc-plan pack: one context, the plans' buffers, their cubins, the weights, a CUDA graph per bucket"""

    def __init__(self, plans, blob, files, device=0):
        """plans: {bucket id: plan dict}; blob(rel, bucket id) -> bytes of that bucket file; files: the model's weight files"""
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
        self.shared, self.allocs, self.bases = [], [], {}
        for r, n in shared_layout:          # zeroed, weights too: ggml zeroes a quantized tensor's row padding at load
            self.shared.append(self._alloc(n, zero=True))
        for pid, p in plans.items():
            P.need([(b["role"], b["size"]) for b in p["buffers"] if b["role"] != "compute"] == shared_layout,
                   f"bucket {pid}: weights or caches laid out differently from the other buckets")
            it = iter(self.shared)
            self.bases[pid] = [next(it) if b["role"] != "compute" else self._alloc(b["size"], zero=True) for b in p["buffers"]]
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
            for raw, ptrs in zip(a["args"], a["ptrs"]):
                raw = bytearray.fromhex(raw)
                for o, buf, off in ptrs:
                    struct.pack_into("<Q", raw, o, base[buf] + off)
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
            if a.get("pdl"):
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
        elif "gemm" in a:
            raise P.PackError("a plan calls cuBLAS: not a gc-plan pack")

    def _graph(self, pid):
        if pid not in self.graphs:
            keep, g, ge = [], C.c_void_p(), C.c_void_p()
            self.ck(self.cu.cuStreamBeginCapture_v2(self.st, 0), "begin capture")
            for a in self.plans[pid]["actions"]:
                self._issue(a, keep, self.bases[pid])
            self.ck(self.cu.cuStreamEndCapture(self.st, C.byref(g)), "end capture")
            self.ck(self.cu.cuGraphInstantiateWithFlags(C.byref(ge), g, C.c_ulonglong(0)), "instantiate")
            self.graphs[pid] = (ge, keep)
        return self.graphs[pid][0]

    def step(self, pid, step):
        """run bucket pid on this step: its inputs uploaded, its graph launched; the step's outputs (all of them, in order)"""
        p, keep, base = self.plans[pid], [], self.bases[pid]
        ge = self._graph(pid)
        for a in p["actions"]:
            if "input" in a:
                b = generate(a["input"], a["n"], step, p["io"]["inputs"])
                P.need(len(b) == a["n"], f"input {a['input']}: {len(b)} B, the plan wants {a['n']}")
                h = C.create_string_buffer(b, len(b))
                keep.append(h)
                self.ck(self.cu.cuMemcpyHtoDAsync_v2(C.c_uint64(base[a["dst"][0]] + a["dst"][1]), h, C.c_size_t(len(b)), self.st), a["input"])
        self.ck(self.cu.cuGraphLaunch(ge, self.st), "graph launch")
        outs = []
        for a in (a for a in p["actions"] if "output" in a):
            h = C.create_string_buffer(a["n"])
            self.ck(self.cu.cuMemcpyDtoHAsync_v2(h, C.c_uint64(base[a["src"][0]] + a["src"][1]), C.c_size_t(a["n"]), self.st), "output")
            outs.append(h)
        self.ck(self.cu.cuStreamSynchronize(self.st), "sync")
        return [h.raw for h in outs]

    def close(self):
        if getattr(self, "ctx", None):
            for ge, _ in self.graphs.values():
                self.cu.cuGraphExecDestroy(ge)
            for m in self.mods.values():
                self.cu.cuModuleUnload(m)
            for a in self.allocs:
                self.cu.cuMemFree_v2(C.c_uint64(a))
            self.cu.cuStreamDestroy_v2(self.st)
            self.cu.cuDevicePrimaryCtxRelease(self.dev)
            self.ctx = None


def _open(pack, weights, kind, *, trusted, store, sku, device, price, allow_broken):
    """(stage, manifest, [buckets of kind], Runtime): a gc-plan pack staged and verified, the buckets of that kind of the variant the
    weights match, their plans on the device"""
    stage = P.stage(pack, trusted=trusted, defer=True, store=store)
    ws = [weights] if isinstance(weights, str) else list(weights)
    sel = {}

    def pick(m, dev):
        bs = [b for b in m["buckets"] if P.axes(b)[1] == kind]
        P.need(bs, f"the pack has no {kind} bucket")
        sel["all"] = bs
        return bs[0], bs, bs[0]
    br = P.open_branch(stage, ws, pick, sku=sku, ordinal=device, price=price, allow_broken=allow_broken)
    bs = [b for b in sel["all"] if b["variant"] == br.bucket["variant"]]
    path = lambda b, rel: os.path.join(stage.dir, "blobs", b["files"][rel])
    plans = {b["id"]: json.load(open(path(b, b["lib"]))) for b in bs}
    byid = {b["id"]: b for b in bs}
    rt = Runtime(plans, lambda rel, pid: open(path(byid[pid], rel), "rb").read(), ws, device)
    return stage, br.m, bs, rt


class PlanEmbed:
    """An embedding model from a gc-plan pack: encode buckets, one per chunk length T; an item runs in the smallest T that holds it,
    padded (pad keys masked, pooling over the real tokens)."""

    def __init__(self, pack, weights, *, device=-1, trusted=None, store=None, sku=None, price=None, allow_broken=False,
                 numerics="invariant", **_):
        self.rt = self.stage = None
        self.stage, self.m, bs, self.rt = _open(pack, weights, "encode", trusted=trusted, store=store, sku=sku, device=device,
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
