"""Encoder-decoder models (model.type encoder-decoder: whisper-style ASR, nemotron speech): the encoder runs once per sequence on float
frames [T, F] and writes that sequence's cross-attention KV into a pool of its own (one cross slot per sequence slot, geometry
cross_geometry), read-only; the decoder is the ordinary step plus cross attention, so the Engine loop runs unchanged.

Seq2SeqRunner  Engine whose requests carry frames; a sequence is encoded the first time it reaches a step (also after a preemption:
               release frees both pools, the encoder is deterministic), and the prefix cache is off (a decoder prefix depends on its audio)
GcSeq2Seq      backend over a gc-c pack whose library reports GC_L_SEQ2SEQ: gc.h 1.2 gc_encode, then gc_step as for a decoder
RefSeq2Seq     CPU reference model (RefModel's decoder with cross attention, an encoder over projected frames) and its backend"""
import ctypes as C
import math

import torch

from .backend import KVGeometry
from .engine import Engine
from .embed import Layout12
from .gc_c import GcBackend, Layout
from .ref import RefModel, TorchBackend, rms
from .scheduler import Config

L_SEQ2SEQ = 1 << 5


class RefSeq2Seq(RefModel):
    """frames [T, feat] -> enc_layers non-causal layers -> per decoder layer cross K, V (kv_heads x dh); decoder token = RefModel's
    self attention, then cross attention over the frames, then the MLP."""

    def __init__(self, feat=8, enc_layers=2, max_frames=32, **kw):
        super().__init__(**kw)
        g = torch.Generator().manual_seed(kw.get("seed", 0) + 101)
        w = lambda *s: (torch.randn(*s, generator=g) / math.sqrt(s[-1])).to(self.device)
        d, kvd = self.d, self.KVH * self.dh
        self.F, self.max_frames = feat, max_frames
        self.inp = w(d, feat)
        self.enc = [dict(q=w(d, d), k=w(d, d), v=w(d, d), o=w(d, d), g=w(2 * d, d), u=w(2 * d, d), dn=w(d, 2 * d)) for _ in range(enc_layers)]
        for l in self.layers:
            l.update(cq=w(self.H * self.dh, d), ck=w(kvd, d), cv=w(kvd, d), co=w(d, self.H * self.dh))

    def encode(self, frames):
        x = frames.float().to(self.device) @ self.inp.T
        for w in self.enc:
            h = torch.stack([rms(r) for r in x])
            a = torch.softmax((h @ w["q"].T) @ (h @ w["k"].T).T / math.sqrt(self.d), -1) @ (h @ w["v"].T)
            x = x + a @ w["o"].T
            h = torch.stack([rms(r) for r in x])
            x = x + (torch.nn.functional.silu(h @ w["g"].T) * (h @ w["u"].T)) @ w["dn"].T
        e = torch.stack([rms(r) for r in x])
        return [((e @ l["ck"].T).view(-1, self.KVH, self.dh), (e @ l["cv"].T).view(-1, self.KVH, self.dh)) for l in self.layers]

    def token(self, t, p, kc, vc, row, rows, cross):
        x = self.emb[t]
        for l, w in enumerate(self.layers):
            h = rms(x)
            q = self._rope((w["q"] @ h).view(self.H, self.dh), p)
            k = self._rope((w["k"] @ h).view(self.KVH, self.dh), p)
            kc[l, row], vc[l, row] = k.to(kc.dtype), ((w["v"] @ h).view(self.KVH, self.dh)).to(vc.dtype)
            K, V = kc[l, rows].float(), vc[l, rows].float()
            sc = torch.einsum("kgd,tkd->kgt", q.view(self.KVH, -1, self.dh), K) / math.sqrt(self.dh)
            x = x + w["o"] @ torch.einsum("kgt,tkd->kgd", torch.softmax(sc, -1), V).reshape(-1)
            h = rms(x)
            Kc, Vc = cross[l]
            cq = (w["cq"] @ h).view(self.KVH, -1, self.dh)
            sc = torch.einsum("kgd,tkd->kgt", cq, Kc) / math.sqrt(self.dh)
            x = x + w["co"] @ torch.einsum("kgt,tkd->kgd", torch.softmax(sc, -1), Vc).reshape(-1)
            h = rms(x)
            x = x + w["dn"] @ (torch.nn.functional.silu(w["g"] @ h) * (w["u"] @ h))
        return self.head @ rms(x)

    def dense(self, frames, tokens, steps):
        """Greedy continuation of tokens given frames, private dense caches: the oracle."""
        cross, n = self.encode(frames), len(tokens) + steps
        kc, vc = self.cache(n)
        toks, out = list(tokens), []
        for p in range(n - 1):
            lg = self.token(toks[p], p, kc, vc, p, torch.arange(p + 1), cross)
            if p >= len(tokens) - 1:
                toks.append(int(lg.argmax()))
                out.append(toks[-1])
        return out


class RefSeq2SeqBackend(TorchBackend):
    """Paged self KV as TorchBackend; the cross KV of a sequence in one of max_seqs cross slots, written by encode, freed by release."""

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.cross_geometry = KVGeometry(model.max_frames, self.max_seqs, model.L, model.KVH, model.dh)
        self.cross, self.enc_frames, self.enc_feat = {}, model.max_frames, model.F
        self.caps = dict(self.caps, tree=False, taps={})
        self.stats = dict(encodes=0)

    def encode(self, sid, frames):
        if sid in self.cross:
            raise RuntimeError(f"sequence {sid} is already encoded")
        if len(self.cross) == self.cross_geometry.num_blocks:
            raise RuntimeError("no free cross slot")
        if frames.ndim != 2 or frames.shape[1] != self.enc_feat or not 1 <= frames.shape[0] <= self.enc_frames:
            raise ValueError(f"frames {tuple(frames.shape)}: [1 .. {self.enc_frames}, {self.enc_feat}]")
        self.cross[sid] = self.m.encode(frames)
        self.stats["encodes"] += 1

    def step(self, work):
        rows = []
        for w in work:
            if w.sid not in self.cross:
                raise RuntimeError(f"sequence {w.sid} has no cross KV: encode first")
            sl = self._slots(w.blocks)
            for j, t in enumerate(w.tokens):
                lg = self.m.token(t, w.pos + j, self.kc, self.vc, sl[w.pos + j], sl[:w.pos + j + 1], self.cross[w.sid])
                if j >= len(w.tokens) - w.want:
                    rows.append(lg)
        return torch.stack(rows) if rows else torch.empty(0, self.vocab_size)

    def release(self, sid):
        super().release(sid)
        self.cross.pop(sid, None)


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
        f = frames.float().contiguous()
        self._ck(self.L.gc_encode(self.h, sid, C.cast(f.data_ptr(), C.POINTER(C.c_float)), f.shape[0], f.shape[1]))


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
