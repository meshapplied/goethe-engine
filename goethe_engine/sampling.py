from dataclasses import dataclass

import torch

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


def sample(logits, params, seeds, idxs, masks):
    """Token per row, -1 for a row without a finite choice. A row depends only on its logits, params, seed and token index."""
    x = logits.float().clone()
    bad = torch.isnan(x).any(1)
    for r, m in enumerate(masks):
        if m is not None:
            x[r] = x[r].masked_fill(~m.to(x.device), float("-inf"))
    bad |= ~torch.isfinite(x).any(1)
    out = x.argmax(1)
    hot = [r for r, p in enumerate(params) if p.temperature > 0]
    if hot:
        V = x.shape[1]
        ix = torch.tensor(hot, device=x.device)
        t = torch.tensor([params[r].temperature for r in hot], dtype=torch.float32, device=x.device)
        z, order = torch.sort(x[ix] / t[:, None], dim=1, descending=True, stable=True)
        p = torch.softmax(z.double(), dim=1)
        rank = torch.arange(V, device=x.device)[None]
        k = torch.tensor([params[r].top_k or V for r in hot], device=x.device)[:, None]
        tp = torch.tensor([params[r].top_p for r in hot], dtype=torch.float64, device=x.device)[:, None]
        keep = (rank < k) & (p.cumsum(1) - p < tp)
        c = (p * keep).cumsum(1)
        u = torch.tensor([uniform(seeds[r], idxs[r]) for r in hot], dtype=torch.float64, device=x.device)
        j = (c < (u * c[:, -1])[:, None]).sum(1).minimum(keep.sum(1) - 1).clamp(min=0)
        out[ix] = order.gather(1, j[:, None])[:, 0]
    out[bad] = -1
    return out.tolist()


def dist(logits, p, mask=None):
    """float64 probabilities over the vocabulary that sample() draws a temperature > 0 row from (top-k, top-p, mask)."""
    x = logits.float()
    if mask is not None:
        x = x.masked_fill(~mask.to(x.device), float("-inf"))
    z, order = torch.sort(x / p.temperature, descending=True, stable=True)
    q = torch.softmax(z.double(), 0)
    rank = torch.arange(len(q), device=x.device)
    q = q * ((rank < (p.top_k or len(q))) & (q.cumsum(0) - q < p.top_p))
    return torch.zeros_like(q).scatter_(0, order, q / q.sum())


def draw(q, u):
    """Inverse CDF of probabilities q (vocabulary order) at u in [0, 1)."""
    c = q.cumsum(0)
    return int(min(int((c < u * c[-1]).sum()), int((q > 0).nonzero()[-1, 0])))
