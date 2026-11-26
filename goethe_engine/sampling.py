from dataclasses import dataclass

import numpy as np

M64 = (1 << 64) - 1


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    max_tokens: int = 16
    stop_token_ids: tuple = ()
    stop_sequences: tuple = ()
    ignore_eos: bool = False
    n: int = 1
    grammar: tuple | None = None  # (kind, spec), see grammar.py
    priority: int = 0
    accept: str = "exact"  # speculative acceptance at temperature > 0: exact (seeded sample match) | rejection

    def validate(self):
        if not (self.temperature >= 0 and self.temperature < float("inf")):
            raise ValueError("temperature must be finite and >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.n > 1 and self.temperature == 0:
            raise ValueError("n > 1 needs temperature > 0")
        if any(len(s) == 0 for s in self.stop_sequences):
            raise ValueError("empty stop sequence")
        if self.accept not in ("exact", "rejection"):
            raise ValueError("accept must be exact or rejection")


def mix(seed, i):
    """splitmix64 of (seed, i): the child seed or the uniform stream of a request."""
    z = (seed + (i + 1) * 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def uniform(seed, idx):
    return (mix(seed, idx) >> 11) / float(1 << 53)


def host(x):
    """A numpy view of logits / masks / probabilities from any backend (numpy, or a torch tensor of the reference backends)."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _sorted_p(x, t):
    """(order, probabilities in that order): the row divided by t (float32), sorted descending (stable), softmax in float64."""
    z = x / np.float32(t)
    order = np.argsort(-z, kind="stable")
    zs = z[order].astype(np.float64)
    e = np.exp(zs - zs[0])
    return order, e / e.sum()


def _sample_full(x, p, u):
    order, q = _sorted_p(x, p.temperature)
    keep = (np.arange(len(q)) < (p.top_k or len(q))) & (np.cumsum(q) - q < p.top_p)
    c = np.cumsum(q * keep)
    return int(order[max(min(int((c < u * c[-1]).sum()), int(keep.sum()) - 1), 0)])


def _sample_row(x, p, u):
    """Exact top-k/top-p draw without sorting the full vocabulary when a small head covers the support."""
    n = len(x)
    z = x / np.float32(p.temperature)
    zmax = float(z.max())
    if not np.isfinite(zmax) or n <= 512 or (p.top_k == 0 and p.top_p >= 1.0):
        return _sample_full(x, p, u)
    if p.top_k == 1:
        return int(z.argmax())
    k = min(p.top_k, n) if p.top_k else min(256, n)
    idx = np.argpartition(-z, k - 1)[:k]
    thresh = float(z[idx].min())
    cand = np.flatnonzero(z > -np.inf if thresh == -np.inf else z >= thresh)  # ascending ids keep stable ties
    sub = cand[np.argsort(-z[cand], kind="stable")]
    if p.top_k:
        sub = sub[:p.top_k]
    zmax64 = np.float64(zmax)
    e_sub = np.exp(z[sub].astype(np.float64) - zmax64)
    if p.top_p >= 1.0:
        c = np.cumsum(e_sub)
        return int(sub[max(min(int((c < u * c[-1]).sum()), len(sub) - 1), 0)])
    e_sum = float(e_sub.sum()) if thresh == -np.inf else float(np.exp(z.astype(np.float64) - zmax64).sum())
    q = e_sub / e_sum
    cs = np.cumsum(q)
    if not p.top_k and float(cs[-1]) < p.top_p and len(sub) < n:
        return _sample_full(x, p, u)
    keep = (cs - q) < p.top_p
    m = max(int(keep.sum()), 1)
    c = np.cumsum(q[:m])
    return int(sub[max(min(int((c < u * c[-1]).sum()), m - 1), 0)])


def sample(logits, params, seeds, idxs, masks):
    """Token per row, -1 for a row without a finite choice. A row depends only on its logits, params, seed and token index."""
    raw = host(logits)
    has_mask = any(m is not None for m in masks)
    x = np.array(raw, dtype=np.float32) if has_mask else np.asarray(raw, dtype=np.float32)
    if not len(x):
        return []
    out = [0] * len(x)
    for r, p in enumerate(params):
        row = x[r]
        m = masks[r]
        if m is not None:
            if np.isnan(float(row.max())):  # NaN in raw logits fails closed even when masked out
                out[r] = -1
                continue
            row[~host(m).astype(bool)] = -np.inf
        rmax = float(row.max())
        if np.isnan(rmax) or rmax == -np.inf or (rmax == np.inf and not np.isfinite(row).any()):
            out[r] = -1
        elif p.temperature > 0:
            out[r] = _sample_row(row, p, uniform(seeds[r], idxs[r]))
        else:
            out[r] = int(row.argmax())
    return out


def dist(logits, p, mask=None):
    """float64 probabilities over the vocabulary that sample() draws a temperature > 0 row from (top-k, top-p, mask)."""
    x = np.array(host(logits), dtype=np.float32)
    if mask is not None:
        x[~host(mask).astype(bool)] = -np.inf
    order, q = _sorted_p(x, p.temperature)
    q = q * ((np.arange(len(q)) < (p.top_k or len(q))) & (np.cumsum(q) - q < p.top_p))
    out = np.zeros_like(q)
    out[order] = q / q.sum()
    return out


def draw(q, u):
    """Inverse CDF of probabilities q (vocabulary order) at u in [0, 1)."""
    q = host(q)
    c = np.cumsum(q)
    return int(min(int((c < u * c[-1]).sum()), int(np.nonzero(q > 0)[0][-1])))
