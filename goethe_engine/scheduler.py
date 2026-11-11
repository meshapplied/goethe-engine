"""Continuous batching with chunked prefill, preemption by recompute and cache-aware admission.

Budgeted mixed steps follow vLLM v1 core/sched/scheduler.py (running first, preempt the lowest priority on pressure,
admit waiting only if nothing was preempted); the waiting order follows SGLang schedule_policy.py (longest prefix
match, in-batch prefix deferral). A step item is a prefill chunk, a decode, or a verify (decode plus a draft tree)."""
from dataclasses import dataclass

from .kv import KVManager
from .spec import Tree, as_tree


@dataclass
class Config:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    policy: str = "lpm"  # "fcfs" | "lpm"
    prefix_cache: bool = True
    spec_k: int = 0
    kv_dtype: str = "f16"  # "fp8" only as an explicit opt-in


class Sequence:
    def __init__(self, sid, rid, index, prompt, params, arrival, seed, grammar, limit):
        self.sid, self.rid, self.index, self.params, self.seed, self.grammar = sid, rid, index, params, seed, grammar
        self.tokens, self.nprompt, self.arrival, self.limit = list(prompt), len(prompt), arrival, limit
        self.blocks, self.computed, self.tpages = [], 0, 0
        self.hit = None  # prompt tokens served by the prefix cache at first admission
        self.prefilled = 0  # rows computed for the prompt, recompute included
        self.preemptions, self.preempted = 0, []
        self.forks = params.n - 1 if index == 0 else 0
        self.finished = None

    @property
    def key(self):
        return self.params.priority, self.arrival

    @property
    def nout(self):
        return len(self.tokens) - self.nprompt


@dataclass
class Item:
    seq: Sequence
    n: int
    tree: Tree  # drafts after the anchor (decodes only)
    spec: bool = False

    @property
    def tokens(self):
        s = self.seq
        return s.tokens[s.computed:s.computed + self.n - len(self.tree)] + self.tree.tokens

    @property
    def want(self):
        s = self.seq
        return len(self.tree) + 1 if s.computed + self.n - len(self.tree) >= len(s.tokens) else 0

    @property
    def parents(self):
        t = self.tree
        return None if t.is_chain else [-1] + t.parents


def strict(s):
    """Rejection acceptance at temperature > 0: drafts change the sampled tokens, so a tree is never cut by budget or
    pool pressure, and a recompute chunk stops before the anchor (the token after it comes from a verify, as before)."""
    return s.params.accept == "rejection" and s.params.temperature > 0


def chunk(s, need):
    return need - 1 if need > 1 and s.nout and strict(s) else need


class Scheduler:
    def __init__(self, cfg, kv: KVManager, drafter=None, spec_round=False, vocab=1 << 30):
        self.cfg, self.kv, self.drafter, self.spec_round, self.vocab = cfg, kv, drafter, spec_round, vocab
        self.waiting, self.running = [], []
        self.preemptions = 0

    def add(self, seq):
        self.waiting.append(seq)

    def remove(self, seq):
        """Drop a sequence from the queues and free its blocks."""
        for q in (self.waiting, self.running):
            if seq in q:
                q.remove(seq)
        self.kv.release(seq)

    def _preempt(self, seq):
        self.running.remove(seq)
        self.kv.release(seq)
        seq.preemptions += 1
        self.preemptions += 1
        self.preempted.append(seq.sid)
        self.waiting.append(seq)

    def _order(self):
        if self.cfg.policy == "fcfs" or len(self.waiting) > 128:
            return sorted(self.waiting, key=lambda s: s.key)
        hit = {s.sid: len(self.kv.match(s.tokens)) for s in self.waiting}
        return sorted(self.waiting, key=lambda s: (s.params.priority, -hit[s.sid], s.arrival))

    def _chain(self, tokens, npages):
        bs, h, out = self.kv.bs, 0, []
        for i in range(npages):
            h = hash((h, tuple(tokens[i * bs:(i + 1) * bs])))
            out.append(h)
        return out

    def schedule(self):
        """(items, copies): the work of one step and the CoW block copies to run before it."""
        cfg, kv = self.cfg, self.kv
        budget, items, copies, preempted = cfg.max_num_batched_tokens, [], [], False
        self.preempted = []
        self.running.sort(key=lambda s: s.key)
        trees, self.invalid = {}, 0
        if self.drafter and cfg.spec_k and not self.spec_round:
            dec = [s for s in self.running if len(s.tokens) - s.computed == 1 and s.limit - s.nout > 1
                   and (s.nout or not strict(s))]  # token 0 of a strict request: the prompt's own sample
            if dec:
                for s, t in zip(dec, self.drafter.propose(dec, cfg.spec_k)):
                    t = as_tree(t)
                    if not t.ok(self.vocab):
                        self.invalid += 1
                        continue
                    trees[s.sid] = t.fit(cfg.spec_k, s.limit - s.nout - 1)
        i, empty = 0, Tree([], [])
        while i < len(self.running) and budget > 0:
            s = self.running[i]
            need = chunk(s, len(s.tokens) - s.computed)
            t = trees.get(s.sid, empty)
            if need + len(t) > budget:
                if len(t) and strict(s):
                    i += 1
                    continue
                t = t.cut(budget - need)
            n = min(need + len(t), budget)
            while (pairs := kv.reserve(s, s.computed + n)) is None:
                if len(t) and not strict(s):
                    t, n = empty, min(need, budget)
                    continue
                victim = self.running[-1]
                self._preempt(victim)
                preempted = True
                if victim is s:
                    break
            if pairs is None:
                continue
            copies += pairs
            items.append(Item(s, n, t, self.spec_round and need == 1))
            budget -= n
            i += 1
        if preempted:
            return items, copies
        seen = set()
        for s in self._order():
            if budget <= 0 or len(self.running) >= cfg.max_num_seqs:
                break
            if len(self.running) + s.params.n > cfg.max_num_seqs:
                continue
            if cfg.policy == "lpm" and kv.cache and len(kv.match(s.tokens)) * kv.bs < 32:
                lead = 0
                for h in self._chain(s.tokens, (len(s.tokens) - 1) // kv.bs):
                    if h not in seen:
                        break
                    lead += 1
                if lead * kv.bs >= 32:
                    continue
            kv.attach(s)
            n = min(chunk(s, len(s.tokens) - s.computed), budget)
            if len(s.tokens) - s.computed == 1 and strict(s) and s.nout:
                n = 0  # the anchor alone: next step's decode carries its drafts
            pairs = kv.reserve(s, s.computed + n)
            if pairs is None:
                kv.release(s)
                break
            if s.hit is None:
                s.hit = s.computed
            copies += pairs
            seen.update(self._chain(s.tokens, (s.computed + n) // kv.bs))
            self.waiting.remove(s)
            self.running.append(s)
            if n:
                items.append(Item(s, n, Tree([], [])))
            budget -= n
        return items, copies
