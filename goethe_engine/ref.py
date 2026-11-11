"""Pure torch reference: a tiny GQA transformer (optionally with recurrent layers) run one token at a time, so a row is
bitwise independent of batching, chunking, paging and tree shape; a backend over it with tree verify, path commit and
taps; and drafters as ordinary small models (draft LM, EAGLE head / MTP module, Medusa heads)."""
import math

import torch

from .backend import KVGeometry
from .spec import depths, path

KV_DT = {"f16": torch.float16, "fp8": torch.float8_e4m3fn}


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean() + 1e-6)


class RefModel:
    """rec: layers whose mixer is a gated linear recurrence (state [d]) instead of attention (a hybrid model)."""

    def __init__(self, vocab=64, d=32, layers=2, heads=4, kv_heads=2, ffn=64, seed=0, device="cpu", rec=()):
        g = torch.Generator().manual_seed(seed)
        self.V, self.d, self.L, self.H, self.KVH, self.dh = vocab, d, layers, heads, kv_heads, d // heads
        w = lambda *s: (torch.randn(*s, generator=g) / math.sqrt(s[-1] if len(s) > 1 else 1)).to(device)
        self.emb, self.head = w(vocab, d) * 2, w(vocab, d) * 2
        self.layers = [dict(q=w(heads * self.dh, d), k=w(kv_heads * self.dh, d), v=w(kv_heads * self.dh, d),
                            o=w(d, heads * self.dh), g=w(ffn, d), u=w(ffn, d), dn=w(d, ffn)) for _ in range(layers)]
        self.rec, self.device = set(rec), device
        for l in self.rec:
            self.layers[l].update(a=w(d, d), b=w(d, d))
        self.inv = (10000.0 ** (-torch.arange(0, self.dh, 2).float() / self.dh)).to(device)

    def _rope(self, x, p):
        a = self.inv * p
        c, s = a.cos(), a.sin()
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def token(self, t, p, kc, vc, row, rows, st=None, x0=None):
        """Token t at RoPE position p: writes KV row `row` of kc, vc [layers, rows, kv_heads, dh] and attends to rows
        (position order, row last). st: recurrent state per layer before the token; x0 overrides the embedding.
        Returns (logits, taps {l_out-l, result_norm}, state after the token)."""
        x, taps, new = (self.emb[t] if x0 is None else x0), {}, {}
        for l, w in enumerate(self.layers):
            h = rms(x)
            if l in self.rec:
                s = (st or {}).get(l)
                new[l] = torch.sigmoid(w["a"] @ h) * (s if s is not None else torch.zeros_like(h)) + w["b"] @ h
                x = x + new[l]
            else:
                q = self._rope((w["q"] @ h).view(self.H, self.dh), p)
                k = self._rope((w["k"] @ h).view(self.KVH, self.dh), p)
                v = (w["v"] @ h).view(self.KVH, self.dh)
                kc[l, row], vc[l, row] = k.to(kc.dtype), v.to(vc.dtype)
                K, V = kc[l, rows].float(), vc[l, rows].float()
                sc = torch.einsum("kgd,tkd->kgt", q.view(self.KVH, -1, self.dh), K) / math.sqrt(self.dh)
                x = x + w["o"] @ torch.einsum("kgt,tkd->kgd", torch.softmax(sc, -1), V).reshape(-1)
            h = rms(x)
            x = x + w["dn"] @ (torch.nn.functional.silu(w["g"] @ h) * (w["u"] @ h))
            taps[f"l_out-{l}"] = x
        taps["result_norm"] = rms(x)
        return self.head @ taps["result_norm"], taps, new

    def cache(self, n, kv="f16"):
        kc = torch.zeros(self.L, n, self.KVH, self.dh, dtype=KV_DT[kv], device=self.device)
        return kc, torch.zeros_like(kc)

    def dense(self, tokens, steps, kv="f16"):
        """Greedy continuation with a private dense cache: the oracle that never touches paging or trees."""
        n = len(tokens) + steps
        kc, vc = self.cache(n, kv)
        slots = torch.arange(n, device=self.device)
        toks, out, st = list(tokens), [], None
        for p in range(n - 1):
            lg, _, st = self.token(toks[p], p, kc, vc, p, slots[:p + 1], st)
            if p >= len(tokens) - 1:
                toks.append(int(lg.argmax()))
                out.append(toks[-1])
        return out


class TorchBackend:
    def __init__(self, model, num_blocks=64, block_size=4, kv="f16", max_seqs=64, taps=()):
        self.m, self.vocab_size, self.max_seqs, self.taps = model, model.V, max_seqs, tuple(taps)
        self.geometry = KVGeometry(block_size, num_blocks, model.L, model.KVH, model.dh, kv)
        self.kc, self.vc = model.cache(num_blocks * block_size, kv)
        self.caps = dict(tree=True, taps={t: model.d for t in taps}, recurrent="tree" if model.rec else None)
        self.state, self.nodes, self.blk = {}, {}, {}  # committed state; state after each token of the step; table

    def _slots(self, blocks):
        bs = self.geometry.block_size
        return (torch.tensor(blocks, device=self.m.device)[:, None] * bs + torch.arange(bs, device=self.m.device)).flatten()

    def step(self, work):
        rows, feats = [], []
        for w in work:
            sl = self._slots(w.blocks)
            st0 = None if w.pos == 0 else self.state.get(w.sid)
            dep, sts = depths(w.parents, len(w.tokens)), []
            for j, t in enumerate(w.tokens):
                par = w.parents[j] if w.parents is not None else j - 1
                vis = torch.cat([sl[:w.pos], sl[w.pos + torch.tensor(path(w.parents, j), device=sl.device)]])
                lg, tp, st = self.m.token(t, w.pos + dep[j], self.kc, self.vc, sl[w.pos + j], vis,
                                          st0 if par < 0 else sts[par])
                sts.append(st)
                if w.feats:
                    feats.append(torch.cat([tp[k] for k in self.taps]) if self.taps else torch.zeros(0))
                if j >= len(w.tokens) - w.want:
                    rows.append(lg)
            self.nodes[w.sid], self.state[w.sid], self.blk[w.sid] = sts, sts[-1], w.blocks
        out = torch.stack(rows) if rows else torch.empty(0, self.vocab_size, device=self.m.device)
        return (out, torch.stack(feats)) if any(w.feats for w in work) else out

    def commit(self, sid, base, pth):
        self.state[sid] = self.nodes[sid][pth[-1]]
        sl = self._slots(self.blk[sid])
        for j, i in enumerate(pth):
            if i != j:
                self.kc[:, sl[base + j]] = self.kc[:, sl[base + i]]
                self.vc[:, sl[base + j]] = self.vc[:, sl[base + i]]

    def copy_blocks(self, pairs):
        bs = self.geometry.block_size
        for s, d in pairs:
            self.kc[:, d * bs:(d + 1) * bs] = self.kc[:, s * bs:(s + 1) * bs]
            self.vc[:, d * bs:(d + 1) * bs] = self.vc[:, s * bs:(s + 1) * bs]

    def release(self, sid):
        for x in (self.state, self.nodes, self.blk):
            x.pop(sid, None)


# --- drafters: ordinary models behind spec.TreeDrafter's dist(seq, path) -------------------------------------------

class _Rows:
    """A model's rows over one sequence's inputs, dense; rows are kept for the longest common key prefix."""

    def __init__(self, m):
        self.m, self.seqs = m, {}

    def run(self, sid, key, inputs):
        """key[r]: identity of row r; inputs(r, previous row's output) -> (token, x0 or None, position). Returns the
        last row's (logits, last layer output, result_norm)."""
        c, n = self.seqs.get(sid), len(key)
        if c is None or c["kc"].shape[1] < n:
            kc, vc = self.m.cache(max(64, 2 * n))
            if c:
                kc[:, :c["kc"].shape[1]], vc[:, :c["kc"].shape[1]] = c["kc"], c["vc"]
            c = self.seqs[sid] = dict(c or dict(key=[], out=[], st=[]), kc=kc, vc=vc)
        k = 0
        while k < min(n, len(c["key"])) and c["key"][k] == key[k]:
            k += 1
        del c["key"][k:], c["out"][k:], c["st"][k:]
        rows = torch.arange(n, device=self.m.device)
        for r in range(k, n):
            t, x0, p = inputs(r, c["out"][r - 1] if r else None)
            lg, tp, st = self.m.token(t, p, c["kc"], c["vc"], r, rows[:r + 1], c["st"][r - 1] if r else None, x0)
            c["key"].append(key[r])
            c["out"].append((lg, tp[f"l_out-{self.m.L - 1}"], tp["result_norm"]))
            c["st"].append(st)
        return c["out"][-1]

    def release(self, sid):
        self.seqs.pop(sid, None)


class DraftLM:
    """Any small LM (ar:K, SpecInfer): log-probabilities after seq.tokens + path."""

    taps = ()

    def __init__(self, m):
        self.m, self.rows = m, _Rows(m)

    def dist(self, seq, pth):
        full = seq.tokens + list(pth)
        return torch.log_softmax(self.rows.run(seq.sid, full, lambda r, _: (full[r], None, r))[0].double(), 0)

    def release(self, sid):
        self.rows.release(sid)


class EagleHead:
    """EAGLE-1 / 2 / 3 and MTP as one ordinary model: row p reads the target feature f_p and token t_(p+1) as
    x0 = fc [rms(emb t); rms(fuse f)] (concat order of vllm qwen3_5_mtp.py / deepseek_mtp.py), then its own layers
    and the target's head. f_p: the target's taps at committed positions (EAGLE-3: three l_out taps through a linear
    fuse, eagle/model/cnets.py fc; EAGLE-1 / MTP: result_norm), past them the head's own previous output. The target's
    embedding and head are reused (same tensors). Features are stored by token prefix, so prefix-cache hits and
    preemption replays find the rows they need."""

    def __init__(self, target, taps=("result_norm",), layers=1, seed=1, own="result_norm"):
        self.t, self.taps, self.own = target, tuple(taps), own
        self.m = RefModel(target.V, target.d, layers, target.H, target.KVH, 2 * target.d, seed, target.device)
        self.m.emb, self.m.head = target.emb, target.head
        g = torch.Generator().manual_seed(seed + 7)
        w = lambda *s: (torch.randn(*s, generator=g) / math.sqrt(s[-1])).to(target.device)
        self.fuse = w(target.d, target.d * len(taps)) if len(taps) > 1 else None
        self.fc = w(target.d, 2 * target.d)
        self.feat, self.rows = {}, _Rows(self.m)

    def feed(self, seq, pos, rows):
        for j in range(len(rows)):
            self.feat[tuple(seq.tokens[:pos + j + 1])] = rows[j]

    def dist(self, seq, pth):
        full, L = seq.tokens + list(pth), len(seq.tokens)

        def inputs(r, prev):
            if r < L - 1:
                f = self.feat[tuple(full[:r + 1])]
                f = self.fuse @ f if self.fuse is not None else f
            else:
                f = prev[2] if self.own == "result_norm" else prev[1]
            return full[r + 1], self.fc @ torch.cat([rms(self.t.emb[full[r + 1]]), rms(f)]), r

        key = [(full[r + 1], r < L - 1) for r in range(len(full) - 1)]
        return torch.log_softmax(self.rows.run(seq.sid, key, inputs)[0].double(), 0)

    def release(self, sid):
        self.rows.release(sid)


class MedusaHeads:
    """Medusa-1 heads on the target's last committed result_norm: head k predicts the k-th token after the anchor,
    logits = target.head (f + silu(W_k f)) (Medusa section 2.1.1, ResBlock heads)."""

    taps = ("result_norm",)

    def __init__(self, target, heads=4, seed=3):
        g = torch.Generator().manual_seed(seed)
        self.t, self.feat = target, {}
        self.W = [(torch.randn(target.d, target.d, generator=g) / math.sqrt(target.d)).to(target.device)
                  for _ in range(heads)]

    def feed(self, seq, pos, rows):
        for j in range(len(rows)):
            self.feat[tuple(seq.tokens[:pos + j + 1])] = rows[j]

    def dist(self, seq, pth):
        f = self.feat[tuple(seq.tokens[:-1])]
        W = self.W[min(len(pth), len(self.W) - 1)]
        return torch.log_softmax((self.t.head @ (f + torch.nn.functional.silu(W @ f))).double(), 0)


class ModelDrafter:
    """A smaller RefModel as a stateless greedy chain drafter (dense recompute, for tests)."""

    needs = dict(tree=False, taps=())

    def __init__(self, model, kv="f16"):
        self.m, self.kv = model, kv

    def propose(self, seqs, k):
        return [self.m.dense(s.tokens, k, self.kv) for s in seqs]
