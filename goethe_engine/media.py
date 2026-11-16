"""Media models (model.type media): a decoder whose prompt carries media (io.media: audio) encoded into embedding rows by the pack's
media library (bucket kind encode, abi/gcm.h). The engine knows the bucket fields only (lib, io.media, io.in, io.out, io.decode,
provenance), never a model.

GcMedia      backend over the encode bucket's library: tokenize a templated prompt into text / media chunks, encode a media item
             into rows, and (io.decode) run the backbone in the library (the pinned-body tier); the bound cuBLAS / cuBLASLt / launch
             mode must equal the bucket's provenance (a pinned body reproduces its reference only on those)
MediaRunner  transcribe(pcm, prompt): chunks -> rows per media chunk -> decode; read_wav: PCM16 / float WAV -> float32 mono"""
import ctypes as C
import json
import os
import struct

import numpy as np

from . import pack as P

CHUNK_TEXT, CHUNK_AUDIO = 0, 1
E_CAPACITY = -7
PINNED_REF = ("cublas", "cublasLt", "pdl")   # provenance fields a pinned body is reproducible under
P_ = C.POINTER


class Chunk(C.Structure):
    _fields_ = [("kind", C.c_uint32), ("n", C.c_uint32), ("ids", P_(C.c_int32)), ("rows", P_(C.c_float))]


def media_buckets(m):
    """The encode buckets of a manifest that declare a media kind (io.media)."""
    return [b for b in m["buckets"] if P.axes(b)[1] == "encode" and (b.get("io") or {}).get("media")]


def check_provenance(b, bound):
    """Refuse a pinned body whose bound library versions (gcm_provenance) differ from the bucket's provenance, or that launches with
    programmatic dependent launch on (the pinned reference is not repeatable with it)."""
    pv = b.get("provenance") or {}
    if pv.get("body") != "ggml-pinned":
        return
    ver = lambda x: x.get("version") if isinstance(x, dict) else x      # {"version": "M.m.p", ...} or "M.m.p"
    pv, bound = {k: ver(pv.get(k)) for k in PINNED_REF}, {k: ver(bound.get(k)) for k in PINNED_REF}
    miss = [k for k in PINNED_REF if not pv.get(k)]
    P.need(not miss, f"bucket {b['id']}: a pinned body's provenance lacks {miss}; refused")
    P.need(pv["pdl"] == "off", f"bucket {b['id']}: provenance pdl {pv['pdl']!r}, a pinned body needs pdl off; refused")
    bad = {k: (pv[k], bound[k]) for k in PINNED_REF if pv[k] != bound.get(k)}
    P.need(not bad, f"bucket {b['id']}: body pinned to {', '.join(f'{k} {a}' for k, (a, _) in bad.items())}, this process bound "
                    f"{', '.join(f'{k} {c}' for k, (_, c) in bad.items())}; refused (bind the provenance versions first)")


def bind(path):
    L = C.CDLL(path)
    H = C.c_void_p
    for n, res, args in (("gcm_abi_version", C.c_uint32, []), ("gcm_open", C.c_int, [C.c_char_p, C.c_char_p, C.c_int, P_(H)]),
                         ("gcm_n_embd", C.c_int, [H]), ("gcm_marker", C.c_char_p, [H]),
                         ("gcm_tokenize", C.c_int, [H, C.c_char_p, P_(Chunk), C.c_uint32, P_(C.c_uint32)]),
                         ("gcm_encode_audio", C.c_int, [H, P_(C.c_float), C.c_uint32, P_(C.c_float), C.c_uint32, P_(C.c_uint32)]),
                         ("gcm_provenance", C.c_int, [H, C.c_char_p, C.c_uint32]),
                         ("gcm_last_error", C.c_char_p, [H]), ("gcm_close", None, [H])):
        f = getattr(L, n)
        f.restype, f.argtypes = res, args
    if hasattr(L, "gcm_decode"):
        L.gcm_decode.restype = C.c_int
        L.gcm_decode.argtypes = [H, P_(Chunk), C.c_uint32, C.c_int, C.c_uint32, P_(C.c_int32), P_(C.c_uint32)]
    return L


class GcMedia:
    """The media encode bucket of a pack: its library opened on the variant's weight files in model.weights order (gcm_open: the
    media projector, then the text model)."""

    def __init__(self, pack, weights, *, device=-1, sku=None, trusted=None, store=None, numerics="invariant", **_):
        self.h = self.stage = None
        self.stage = P.stage(pack, trusted=trusted, defer=True, store=store)
        try:
            self._open([weights] if isinstance(weights, str) else list(weights), device, sku, numerics)
        except BaseException:
            self.close()
            raise

    def _open(self, ws, device, sku, numerics):
        want = P.wanted(numerics, single=True)

        def pick(m, dev):
            b = P.select(m, sku=sku, dev=dev, want=lambda x: want(x) and bool((x.get("io") or {}).get("media")), kind="encode")
            return b, [b], b
        br = P.open_branch(self.stage, ws, pick, sku=sku, ordinal=device)
        self.m, b = dict(br.m, buckets=media_buckets(br.m)), br.bucket
        self.bucket, io = b, b["io"]
        self.io = io
        need = [br.m["model"]["weights"][i]["file"] for i in br.m["variants"][b["variant"]]["weights"]]
        P.need(len(ws) == len(need) == 2, f"bucket {b['id']}: gcm_open takes the variant's files in model.weights order {need} "
                                          f"(projector, text model); {len(ws)} given")
        L = self.L = bind(br.lib)
        v = L.gcm_abi_version()
        P.need(v >> 16 == 1, f"{b['lib']}: media ABI {v >> 16}.{v & 0xFFFF}, this engine speaks 1.x")
        h = C.c_void_p()
        rc = L.gcm_open(os.fsencode(ws[0]), os.fsencode(ws[1]), max(device, 0), C.byref(h))   # -1 (current device): ordinal 0
        P.need(rc == 0, f"gcm_open {rc}: {(L.gcm_last_error(None) or b'').decode()}")
        self.h = h.value
        buf = C.create_string_buffer(4096)
        self._ck(L.gcm_provenance(self.h, buf, len(buf)))
        self.bound = json.loads(buf.value.decode())
        check_provenance(b, self.bound)
        self.n_embd = L.gcm_n_embd(self.h)
        want = (io.get("out") or {}).get("n_embd")
        P.need(want in (None, self.n_embd), f"bucket {b['id']}: io.out.n_embd {want}, the library reports {self.n_embd}")
        self.media, self.rate = io["media"], (io.get("in") or {}).get("rate")
        self.marker = L.gcm_marker(self.h).decode()
        self.can_decode = bool(io.get("decode")) and hasattr(L, "gcm_decode")
        self.numerics = dict(contract=b["contract"], determinism=b.get("determinism"), provenance=b.get("provenance"))

    def _ck(self, rc):
        if rc < 0:
            raise RuntimeError(f"gcm call failed ({rc}): {(self.L.gcm_last_error(self.h) or b'').decode()}")

    def tokenize(self, prompt):
        """[(kind, ids or None)]: the text chunks' ids and a placeholder per media item, in prompt order."""
        cap, n = 16, C.c_uint32()
        while True:
            cs = (Chunk * cap)()
            rc = self.L.gcm_tokenize(self.h, prompt.encode(), cs, cap, C.byref(n))
            if rc != E_CAPACITY:
                break
            cap = n.value
        self._ck(rc)
        return [(c.kind, [c.ids[i] for i in range(c.n)] if c.kind == CHUNK_TEXT else None) for c in cs[:n.value]]

    def encode_audio(self, pcm):
        """float32 rows [n_tok, n_embd] of one mono PCM clip at the bucket's rate."""
        x = np.ascontiguousarray(pcm, dtype=np.float32)
        n, cap = C.c_uint32(), 1
        while True:
            out = np.empty((cap, self.n_embd), np.float32)
            rc = self.L.gcm_encode_audio(self.h, x.ctypes.data_as(P_(C.c_float)), len(x), out.ctypes.data_as(P_(C.c_float)), cap, C.byref(n))
            if rc != E_CAPACITY:
                break
            cap = n.value
        self._ck(rc)
        return out[:n.value].copy()

    def decode(self, chunks, max_new, greedy=True):
        """Generated ids after the chunks [(kind, ids | rows)], in the library (io.decode)."""
        P.need(self.can_decode, f"bucket {self.bucket['id']}: no in-library decode (io.decode); the compiled backbone's supplied-rows "
                                "prefill is not in this pack")
        keep, cs = [], (Chunk * len(chunks))()
        for c, (k, x) in zip(cs, chunks):
            a = np.ascontiguousarray(x, dtype=np.int32 if k == CHUNK_TEXT else np.float32)
            keep.append(a)
            c.kind, c.n = k, len(a)
            if k == CHUNK_TEXT:
                c.ids = a.ctypes.data_as(P_(C.c_int32))
            else:
                c.rows = a.ctypes.data_as(P_(C.c_float))
        out, n = (C.c_int32 * max_new)(), C.c_uint32()
        self._ck(self.L.gcm_decode(self.h, cs, len(chunks), int(greedy), max_new, out, C.byref(n)))
        return list(out[:n.value])

    def close(self):
        if self.h:
            self.L.gcm_close(self.h)
            self.h = None
        if self.stage:
            self.stage.close()
            self.stage = None

    __del__ = close


def read_wav(path, rate=None):
    """float32 mono samples of a PCM16 / PCM32 / float32 WAV (channels averaged); refuses a rate other than `rate`."""
    with open(path, "rb") as f:
        raw = f.read()
    P.need(raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", f"{path}: not a RIFF WAVE file")
    i, fmt, data = 12, None, None
    while i + 8 <= len(raw):
        tag, n = raw[i:i + 4], struct.unpack("<I", raw[i + 4:i + 8])[0]
        if tag == b"fmt ":
            fmt = struct.unpack("<HHIIHH", raw[i + 8:i + 24])
        elif tag == b"data":
            data = raw[i + 8:i + 8 + n]
        i += 8 + n + (n & 1)
    P.need(fmt and data is not None, f"{path}: no fmt or data chunk")
    code, ch, sr, _, _, bits = fmt
    P.need(rate is None or sr == rate, f"{path}: {sr} Hz, the bucket takes {rate} Hz (resample first)")
    dt = {(1, 16): ("<i2", 32768.0), (1, 32): ("<i4", 2147483648.0), (3, 32): ("<f4", 1.0)}.get((code, bits))
    P.need(dt, f"{path}: WAV format {code} with {bits} bits is not read here")
    x = np.frombuffer(data[:len(data) // (bits // 8) * (bits // 8)], dt[0]).astype(np.float32) / dt[1]
    return x.reshape(-1, ch).mean(1) if ch > 1 else x


class MediaRunner:
    """transcribe: the prompt's chunks, each media marker's item encoded into rows, then the decode."""

    def __init__(self, backend, cfg=None, **_):
        self.be = backend

    def transcribe(self, media, prompt, max_tokens=2048):
        """Generated ids of prompt (chat-templated, one backend.marker per item of media, a list of PCM arrays) at temperature 0."""
        chunks, k = [], 0
        for kind, ids in self.be.tokenize(prompt):
            if kind == CHUNK_TEXT:
                chunks.append((kind, ids))
                continue
            P.need(k < len(media), f"the prompt has more media markers than the {len(media)} item(s) given")
            chunks.append((kind, self.be.encode_audio(media[k])))
            k += 1
        P.need(k == len(media), f"{len(media)} media item(s), the prompt has {k} marker(s) ({self.be.marker!r})")
        return self.be.decode(chunks, max_tokens)

    def numerics(self):
        return self.be.numerics
