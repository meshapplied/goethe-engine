"""Encoder-decoder models (model.type encoder-decoder: whisper-style ASR, nemotron speech): the encoder runs once per sequence on float
frames [T, F] and writes that sequence's cross-attention KV into a pool of its own (one cross slot per sequence slot, geometry
cross_geometry), read-only; the decoder is the ordinary step plus cross attention, so the Engine loop runs unchanged.

Seq2SeqRunner  Engine whose requests carry frames; a sequence is encoded the first time it reaches a step (also after a preemption:
               release frees both pools, the encoder is deterministic), and the prefix cache is off (a decoder prefix depends on its audio)
GcSeq2Seq      backend over a gc-c pack whose library reports GC_L_SEQ2SEQ: gc.h 1.2 gc_encode, then gc_step as for a decoder
RefSeq2Seq     CPU reference model and its backend: ref_seq2seq.py (torch), importable from here"""
import ctypes as C

import numpy as np

from .backend import KVGeometry
from .engine import Engine
from .embed import Layout12
from .gc_c import GcBackend, Layout
from .scheduler import Config

L_SEQ2SEQ = 1 << 5


def __getattr__(name):   # the torch reference model loads only when asked for
    if name in ("RefSeq2Seq", "RefSeq2SeqBackend"):
        from . import ref_seq2seq
        return getattr(ref_seq2seq, name)
    raise AttributeError(name)


class GcSeq2Seq(GcBackend):
    """A gc-c encoder-decoder pack: the decoder ladder as GcBackend opens it, plus gc_encode (gc.h 1.2) into the library's cross pool."""

    def __init__(self, pack, weights, **kw):
        super().__init__(pack, weights, **kw)
        lo = self.layout
        if not lo.flags & L_SEQ2SEQ:
            self.close()
            raise RuntimeError(f"bucket {self.bucket['id']}: the library does not report GC_L_SEQ2SEQ")
        self.L.gc_encode.restype = C.c_int
        self.L.gc_encode.argtypes = [C.c_void_p, C.c_uint64, C.POINTER(C.c_float), C.c_uint32, C.c_uint32]
        lo2 = Layout12(size=C.sizeof(Layout12))             # the 1.2 fields (enc_frames, enc_feat) past the 1.1 layout
        self._ck(self.L.gc_layout_get(self.h, C.cast(C.pointer(lo2), C.POINTER(Layout))))
        self.enc_frames, self.enc_feat = lo2.enc_frames, lo2.enc_feat
        io = (self.bucket.get("io") or {}).get("cross") or {}
        self.cross_geometry = KVGeometry(io.get("max_frames", self.enc_frames), self.max_seqs, io.get("layers", 0), io.get("heads", 0),
                                         io.get("head_dim", 0))
        self.prefix_cache = False

    def encode(self, sid, frames):
        f = np.ascontiguousarray(frames.detach().cpu().numpy() if hasattr(frames, "detach") else frames, dtype=np.float32)
        self._ck(self.L.gc_encode(self.h, sid, f.ctypes.data_as(C.POINTER(C.c_float)), f.shape[0], f.shape[1]))


class _Encoded:
    """The backend as the Engine sees it: a sequence reaching its first step (or its first after a preemption) is encoded first."""

    def __init__(self, be, frames_of):
        self._be, self._frames_of, self._done = be, frames_of, set()

    def __getattr__(self, k):
        return getattr(self._be, k)

    def step(self, work):
        for w in work:
            if w.sid not in self._done:
                self._be.encode(w.sid, self._frames_of(w.sid))
                self._done.add(w.sid)
        return self._be.step(work)

    def release(self, sid):
        self._done.discard(sid)
        self._be.release(sid)


class Seq2SeqRunner(Engine):
    """Engine over an encoder-decoder backend; add_request takes the request's frames [T, F] beside its decoder prompt."""

    def __init__(self, backend, cfg=None, **kw):
        from dataclasses import replace
        cfg = replace(cfg or Config(), prefix_cache=False)
        self.frames = {}
        super().__init__(_Encoded(backend, self._frames_of), cfg, **kw)

    def _frames_of(self, sid):
        for rid, seqs in self.live.items():
            if any(s.sid == sid for s in seqs):
                return self.frames[rid]
        raise RuntimeError(f"sequence {sid}: no live request")

    def add_request(self, rid, prompt, params=None, frames=None):
        be = self.backend._be
        if frames is None or frames.ndim != 2 or frames.shape[1] != be.enc_feat or not 1 <= frames.shape[0] <= be.enc_frames:
            raise ValueError(f"an encoder-decoder request needs frames [1 .. {be.enc_frames}, {be.enc_feat}]")
        super().add_request(rid, prompt, params)
        self.frames[rid] = frames

    def step(self):
        out = super().step()
        for rid in [r for r in self.frames if r not in self.live]:
            del self.frames[rid]
        return out
