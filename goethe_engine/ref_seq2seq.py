"""CPU reference encoder-decoder model (torch, the optional extra): RefModel's decoder with cross attention, an encoder over projected
frames, and its paged backend."""
import math

import torch

from .backend import KVGeometry
from .ref import RefModel, TorchBackend, rms


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
        x = torch.as_tensor(frames).float().to(self.device) @ self.inp.T
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
