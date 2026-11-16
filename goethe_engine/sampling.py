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


def sample(logits, params, seeds, idxs, masks):
    """Token per row, -1 for a row without a finite choice. A row depends only on its logits, params, seed and token index."""
    x = np.array(host(logits), dtype=np.float32)
    bad = np.isnan(x).any(1)
    for r, m in enumerate(masks):
        if m is not None:
            x[r][~host(m).astype(bool)] = -np.inf
    bad |= ~np.isfinite(x).any(1)
    out = np.where(bad, 0, np.nan_to_num(x, nan=-np.inf).argmax(1)) if len(x) else np.zeros(0, np.int64)
    for r, p in enumerate(params):
        if p.temperature > 0 and not bad[r]:
            order, q = _sorted_p(x[r], p.temperature)
            keep = (np.arange(len(q)) < (p.top_k or len(q))) & (np.cumsum(q) - q < p.top_p)
            c = np.cumsum(q * keep)
            j = max(min(int((c < uniform(seeds[r], idxs[r]) * c[-1]).sum()), int(keep.sum()) - 1), 0)
            out[r] = order[j]
    out[bad] = -1
    return [int(t) for t in out]


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
