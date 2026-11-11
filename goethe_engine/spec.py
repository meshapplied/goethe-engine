"""Speculative decoding: one abstraction, every algorithm a policy that builds its proposal (SPEC.md).

proposal  Tree: the anchor (node 0, the last committed token, not yet run) and drafted tokens, parent of node i < i
verify    one target pass over [anchor] + drafts, node i at position base + depth(i) attending to the cached prefix and
          its ancestors (Work.parents; a chain is the linear case)
accept    per path from the anchor (engine.py): exact (the target's seeded sample equals a child; greedy: the argmax)
          or rejection (mss: SpecInfer multi-step speculative sampling; Leviathan et al. / Chen et al. for one child)
commit    the accepted path's rows to positions base .. base + len - 1, recurrent state after its last node (Backend.commit)

Policies: NgramDrafter (chain) and TreeDrafter over any next-token model: chain (ar:K, MTP), static rank paths (Medusa,
EAGLE-1), dynamic expand + rerank (EAGLE-2 / 3), sampled children (SpecInfer); device rounds (DFlash walk) live in
the backend. check() / compatible() decide what a backend's capabilities run."""
from dataclasses import dataclass

import torch

from .sampling import draw, mix, uniform

SALT_ACC, SALT_RES, SALT_DRAFT = 1 << 40, 1 << 41, 1 << 42  # uniform streams: acceptance coins, residual, drafter


@dataclass
class Tree:
    tokens: list  # drafted tokens: node i + 1
    parents: list  # node of each one's parent: 0 is the anchor, j is tokens[j - 1]
    q: list | None = None  # per drafted token the distribution it was drawn from (float64 [V]); None: deterministic

    @staticmethod
    def chain(tokens, q=None):
        return Tree(list(tokens), list(range(len(tokens))), q)

    def __len__(self):
        return len(self.tokens)

    def cut(self, n):
        return Tree(self.tokens[:n], self.parents[:n], self.q[:n] if self.q else None)

    def fit(self, n, depth):
        """The first n nodes no deeper than depth, parents kept before children."""
        d, idx, toks, par, qs = [0], {0: 0}, [], [], []
        for i, (t, p) in enumerate(zip(self.tokens, self.parents)):
            d.append(d[p] + 1)
            if len(toks) < n and p in idx and d[-1] <= depth:
                idx[i + 1] = len(toks) + 1
                toks.append(t)
                par.append(idx[p])
                qs.append(self.q[i] if self.q else None)
        return Tree(toks, par, qs if self.q else None)

    @property
    def is_chain(self):
        return self.parents == list(range(len(self.parents)))

    def kids(self):
        """Children of every node (node 0 the anchor) in draft order."""
        out = [[] for _ in range(len(self.tokens) + 1)]
        for i, p in enumerate(self.parents):
            out[p].append(i + 1)
        return out

    def ok(self, vocab):
        return (len(self.parents) == len(self.tokens) and all(0 <= p <= i for i, p in enumerate(self.parents))
                and all(0 <= t < vocab for t in self.tokens) and (self.q is None or len(self.q) == len(self.tokens)))


def as_tree(x):
    return x if isinstance(x, Tree) else Tree.chain(x)


def depths(parents, n):
    """Depth of each of n work tokens; parents[i] < i is the parent's index, -1 the cached prefix; None: a chain."""
    if parents is None:
        return list(range(n))
    d = []
    for p in parents:
        d.append(0 if p < 0 else d[p] + 1)
    return d


def path(parents, i):
    """Work token indices from the root to i (inclusive)."""
    out = [i]
    while parents is not None and parents[out[-1]] >= 0:
        out.append(parents[out[-1]])
    return out[::-1] if parents is not None else list(range(i + 1))


def mss(p, kids, seed, i):
    """Multi-step speculative sampling at one node (SpecInfer Algorithm 2 VerifyStochastic): p the target distribution,
    kids [(token, q or None: one-hot)] in draft order. Returns (token, index of the accepted kid or None)."""
    for r, (x, q) in enumerate(kids):
        qx = 1.0 if q is None else float(q[x])
        if qx > 0 and uniform(mix(seed, SALT_ACC + r), i) * qx < float(p[x]):
            return x, r
        res = p.clone()
        if q is None:
            res[x] = 0
        else:
            res = (p - q.to(p)).clamp(min=0)
        if res.sum() > 0:
            p = res / res.sum()
    return draw(p, uniform(mix(seed, SALT_RES), i)), None


class NgramDrafter:
    """Prompt lookup: the tokens that followed the latest earlier occurrence of the longest matching suffix."""

    needs = dict(tree=False, taps=())

    def __init__(self, max_n=3):
        self.max_n = max_n

    def propose(self, seqs, k):
        return [self._one(s.tokens, k) for s in seqs]

    def _one(self, t, k):
        for n in range(min(self.max_n, len(t) - 1), 0, -1):
            suf = t[-n:]
            for i in range(len(t) - n - 1, -1, -1):
                if t[i:i + n] == suf:
                    return t[i + n:i + n + k]
        return []


class TreeDrafter:
    """A token tree over a next-token model m: m.dist(seq, path) -> float64 log-probabilities of the token after
    seq.tokens + path (path: drafted tokens after the anchor seq.tokens[-1]); m.taps the target features it reads,
    m.feed(seq, pos, rows) / m.release(sid) optional. Shapes: chain; static rank paths (Medusa medusa_choices.py,
    EAGLE-1); dynamic (EAGLE-2 eagle/model/cnets.py topK_genrate: expand the topk best of each level by cumulative
    log-probability, rerank every candidate, keep the k best). temperature > 0: children are drawn from the model at
    that temperature and carry q (SpecInfer; rejection acceptance)."""

    def __init__(self, m, shape="chain", topk=4, depth=6, choices=None, temperature=0.0):
        assert shape in ("chain", "static", "dynamic") and (shape != "static" or choices)
        self.m, self.shape, self.topk, self.depth, self.t = m, shape, topk, depth, temperature
        self.choices = sorted({tuple(c) for c in choices or ()}, key=lambda c: (len(c), c))
        self.needs = dict(tree=shape != "chain", taps=tuple(getattr(m, "taps", ())))

    def propose(self, seqs, k):
        return [self.grow(s, k) if k > 0 else Tree([], []) for s in seqs]

    def feed(self, seq, pos, rows):
        if hasattr(self.m, "feed"):
            self.m.feed(seq, pos, rows)

    def release(self, sid):
        if hasattr(self.m, "release"):
            self.m.release(sid)

    def _pick(self, s, lp, n, at):
        """n children of a node whose next token sits at absolute index at: (token, q, log-prob) each."""
        if self.t == 0:
            top = torch.sort(lp, descending=True, stable=True).indices[:n].tolist()
            return [(x, None, float(lp[x])) for x in top]
        q = torch.softmax(lp.double() / self.t, 0)
        xs = [draw(q, uniform(mix(s.seed, SALT_DRAFT + r), at)) for r in range(n)]
        return [(x, q, float(q[x].log())) for x in xs]

    def grow(self, s, k):
        m, at = self.m, len(s.tokens)
        if self.shape == "chain":
            toks, qs = [], []
            for d in range(k):
                x, q, _ = self._pick(s, m.dist(s, toks), 1, at + d)[0]
                toks.append(x)
                qs.append(q)
            return Tree.chain(toks, qs if self.t else None)
        if self.shape == "static":
            node, toks, par, lps = {(): 0}, [], [], {}
            for c in self.choices:
                if len(toks) == k or c[:-1] not in node:
                    continue
                p = node[c[:-1]]
                if p not in lps:
                    lps[p] = torch.sort(m.dist(s, self._path(toks, par, p)), descending=True, stable=True).indices
                toks.append(int(lps[p][c[-1]]))
                par.append(p)
                node[c] = len(toks)
            return Tree(toks, par)
        cand = []  # (score, token, parent candidate index or -1, q)
        beam = [-1]
        for d in range(self.depth):
            new = []
            for b in beam:
                pt = self._cpath(cand, b)
                for x, q, lp in self._pick(s, m.dist(s, pt), self.topk, at + len(pt)):
                    new.append((lp + (cand[b][0] if b >= 0 else 0.0), x, b, q))
            base = len(cand)
            cand += new
            beam = sorted(range(base, len(cand)), key=lambda i: (-cand[i][0], i))[:self.topk]
        keep, sel = {}, []
        for i in sorted(range(len(cand)), key=lambda i: (-cand[i][0], i)):
            if len(sel) == k:
                break
            if cand[i][2] < 0 or cand[i][2] in keep:
                keep[i] = None
                sel.append(i)
        sel.sort()
        idx = {i: j + 1 for j, i in enumerate(sel)}
        return Tree([cand[i][1] for i in sel], [idx.get(cand[i][2], 0) for i in sel],
                    [cand[i][3] for i in sel] if self.t else None)

    @staticmethod
    def _path(toks, par, n):
        out = []
        while n:
            out.append(toks[n - 1])
            n = par[n - 1]
        return out[::-1]

    @staticmethod
    def _cpath(cand, b):
        out = []
        while b >= 0:
            out.append(cand[b][1])
            b = cand[b][2]
        return out[::-1]


# --- capabilities ---------------------------------------------------------------------------------------------------

CAPS = dict(verify=None, tree=False, taps={}, recurrent=None, logits=True, round=None, drafters=None)
# verify: most tokens one verify item may carry (None: any); tree: ancestor-mask verify; taps: target features
# exported per row {name: width}; recurrent: None | "chain" (rolls back along a chain) | "tree" (state per node);
# "step" (advances every step, no rollback); logits: host logits for sampled rows; round: device policy of a backend
# that drafts and verifies on its own; drafters: drafter kinds a pack carries (None: drafters are host objects)

# algorithm -> what it needs; "layers": n distinct l_out taps (EAGLE-3: low, mid, high); drafter: the pack role kind
ALGOS = {
    "ngram": dict(tree=False, taps=(), drafter=None),
    "ar": dict(tree=False, taps=(), drafter="ar"),
    "mtp": dict(tree=False, taps=("result_norm",), drafter="mtp"),
    "medusa": dict(tree=True, taps=("result_norm",), drafter="medusa"),
    "eagle1": dict(tree=True, taps=("result_norm",), drafter="eagle"),
    "eagle2": dict(tree=True, taps=("result_norm",), drafter="eagle"),
    "eagle3": dict(tree=True, taps=("layers", 3), drafter="eagle3"),
    "specinfer": dict(tree=True, taps=(), drafter="ar"),
    "dflash": dict(round="walk", drafter="walk"),
    "ar-device": dict(round="ar", drafter="ar"),
    "mtp-device": dict(round="mtp", drafter="mtp"),
}


def caps_of(backend):
    c = dict(CAPS, **getattr(backend, "caps", {}))
    if getattr(backend, "spec_width", 0):
        c.update(round=c["round"] or "walk", verify=backend.spec_width, logits=False)
    return c


def _taps_ok(have, need):
    if need[:1] == ("layers",):
        n = len([t for t in have if t.startswith("l_out-")])
        return None if n >= need[1] else f"needs {need[1]} layer taps (l_out-*), backend exports {sorted(have) or 'none'}"
    miss = [t for t in need if t not in have]
    return f"needs target taps {miss}, backend exports {sorted(have) or 'none'}" if miss else None


def check(caps, needs, spec_k=0, prefix_cache=False, accept=()):
    """None if a policy with needs runs on a backend with caps, else the reason it cannot."""
    if needs.get("round"):
        return None if caps["round"] == needs["round"] else f"needs a device round ({needs['round']}), backend has " \
                                                             f"{caps['round'] or 'none'}"
    if caps["round"]:
        return f"backend drafts and verifies on the device (round {caps['round']}): a host drafter would never run"
    if caps["recurrent"] == "step":
        return "backend's recurrent state advances every step and cannot roll back a rejected draft"
    if needs.get("tree") and not caps["tree"]:
        return "needs tree verify (ancestor mask), backend verifies chains only"
    if needs.get("tree") and caps["recurrent"] == "chain":
        return "needs tree verify, backend's recurrent state commits along a chain only"
    if caps["verify"] is not None and spec_k + 1 > caps["verify"]:
        return f"spec_k {spec_k} + anchor exceeds the backend's verify width {caps['verify']}"
    r = _taps_ok(caps["taps"], tuple(needs.get("taps", ())))
    if r:
        return r
    if "rejection" in accept and not caps["logits"]:
        return "rejection acceptance needs target logits, backend returns device tokens only"
    return None


def compatible(caps, spec_k=1):
    """{algorithm: None if it runs, else the reason} for a backend's capabilities."""
    out = {}
    for a, nd in ALGOS.items():
        r = check(caps, nd, spec_k if not nd.get("round") else 0)
        if r is None and caps["drafters"] is not None and nd["drafter"] and nd["drafter"] not in caps["drafters"]:
            r = f"pack carries no {nd['drafter']} drafter (has {sorted(caps['drafters']) or 'none'})"
        out[a] = r
    return out
