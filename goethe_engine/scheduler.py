"""Continuous batching with chunked prefill, preemption by recompute and cache-aware admission.

Budgeted mixed steps follow vLLM v1 core/sched/scheduler.py (running first, preempt the lowest priority on pressure,
admit waiting only if nothing was preempted); the waiting order follows SGLang schedule_policy.py (longest prefix
match, in-batch prefix deferral). A step item is a prefill chunk, a decode, or a verify (decode plus a draft tree)."""
from dataclasses import dataclass
from operator import attrgetter

from .kv import KVManager
from .spec import Tree, as_tree

_KEY = attrgetter("key")
_EMPTY_TREE = Tree([], [])


@dataclass
class Config:
    max_num_seqs: int | None = None  # None: the backend's capability (up to 64 sequences; 1 on a linear stream)
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    policy: str = "lpm"  # "fcfs" | "lpm"
    prefix_cache: bool | None = None  # None: on where the backend can share KV pages (paged, no recurrent state), else off
    spec_k: int = 0
    kv_dtype: str = "f16"  # api.Model sets the backend's: fp8 when asked, or with kv=auto on a pack that has only fp8 KV


class Sequence:
    __slots__ = ("sid", "rid", "index", "params", "seed", "grammar", "tokens", "nprompt", "arrival", "limit",
                 "stop_len", "key", "blocks", "blocks_ver", "computed", "tpages", "restore", "hit", "prefilled",
                 "preemptions", "preempted", "forks", "finished", "compiled")

    def __init__(self, sid, rid, index, prompt, params, arrival, seed, grammar, limit):
        self.sid, self.rid, self.index, self.params, self.seed, self.grammar = sid, rid, index, params, seed, grammar
        self.tokens, self.nprompt, self.arrival, self.limit = list(prompt), len(prompt), arrival, limit
        self.stop_len = self.nprompt + limit
        self.key = (params.priority, arrival)
        self.blocks, self.blocks_ver, self.computed, self.tpages = [], 0, 0, 0
        self.restore = None  # snapshot page to copy into the sequence's page when it gets one (page packs)
        self.hit = None  # prompt tokens served by the prefix cache at first admission
        self.prefilled = 0  # rows computed for the prompt, recompute included
        self.preemptions, self.preempted = 0, []
        self.forks = params.n - 1 if index == 0 else 0
        self.finished = None
        self.compiled = None

    @property
    def nout(self):
        return len(self.tokens) - self.nprompt


@dataclass(slots=True)
class Item:
    seq: Sequence
    n: int
    tree: Tree  # drafts after the anchor (decodes only)
    spec: bool = False

    @property
    def tokens(self):
        s, tt = self.seq, self.tree.tokens
        if not tt:
            return [s.tokens[s.computed]] if self.n == 1 else s.tokens[s.computed:s.computed + self.n]
        return s.tokens[s.computed:s.computed + self.n - len(tt)] + tt

    @property
    def want(self):
        s, nt = self.seq, len(self.tree.tokens)
        return nt + 1 if s.computed + self.n - nt >= len(s.tokens) else 0

    @property
    def parents(self):
        t = self.tree
        return None if not t.parents or t.is_chain else [-1] + t.parents


def strict(s):
    """Rejection acceptance at temperature > 0: drafts change the sampled tokens, so a tree is never cut by budget or
    pool pressure, and a recompute chunk stops before the anchor (the token after it comes from a verify, as before)."""
    return s.params.accept == "rejection" and s.params.temperature > 0


def chunk(s, need, snaps=False):
    """Tokens of the next chunk: a page pack's prompt stops before its last token, where the state is snapshotted."""
    if snaps and s.computed < s.nprompt - 1 < s.computed + need:
        return s.nprompt - 1 - s.computed
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
            return sorted(self.waiting, key=_KEY)
        hit = {s.sid: self.kv.hit(s.tokens) for s in self.waiting}
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
        running = self.running
        if len(running) > 1:
            running.sort(key=_KEY)
        trees, self.invalid = None, 0
        if self.drafter and cfg.spec_k and not self.spec_round:
            dec = [s for s in running if len(s.tokens) - s.computed == 1 and s.limit - s.nout > 1
                   and (s.nout or not strict(s))]  # token 0 of a strict request: the prompt's own sample
            if dec:
                trees = {}
                for s, t in zip(dec, self.drafter.propose(dec, cfg.spec_k)):
                    t = as_tree(t)
                    if not t.ok(self.vocab):
                        self.invalid += 1
                        continue
                    trees[s.sid] = t.fit(cfg.spec_k, s.limit - s.nout - 1)
        snaps = bool(kv.snaps or kv.bsnaps)
        i, empty, spec_round = 0, _EMPTY_TREE, self.spec_round
        while i < len(running) and budget > 0:
            s = running[i]
            need = len(s.tokens) - s.computed
            if need > 1:
                need = chunk(s, need, snaps)
            t = trees.get(s.sid, empty) if trees else empty
            nt = len(t.tokens)
            if need + nt > budget:
                if nt and strict(s):
                    i += 1
                    continue
                t = t.cut(budget - need)
                nt = len(t.tokens)
            n = need + nt if need + nt < budget else budget
            while (pairs := kv.reserve(s, s.computed + n)) is None:
                if nt and not strict(s):
                    t, nt, n = empty, 0, need if need < budget else budget
                    continue
                victim = running[-1]
                self._preempt(victim)
                preempted = True
                if victim is s:
                    break
            if pairs is None:
                continue
            if pairs:
                copies += pairs
            items.append(Item(s, n, t, spec_round and need == 1))
            budget -= n
            i += 1
        if preempted or not self.waiting:
            return items, copies
        seen = set()
        for s in self._order():
            if budget <= 0 or len(running) >= cfg.max_num_seqs:
                break
            if len(running) + s.params.n > cfg.max_num_seqs:
                continue
            if cfg.policy == "lpm" and kv.cache and kv.hit(s.tokens) < 32:
                lead = 0
                for h in self._chain(s.tokens, (len(s.tokens) - 1) // kv.bs):
                    if h not in seen:
                        break
                    lead += 1
                if lead * kv.bs >= 32:
                    continue
            kv.attach(s)
            need = len(s.tokens) - s.computed
            n = min(chunk(s, need, snaps) if need > 1 else need, budget)
            if need == 1 and strict(s) and s.nout:
                n = 0  # the anchor alone: next step's decode carries its drafts
            pairs = kv.reserve(s, s.computed + n)
            if pairs is None:
                kv.release(s)
                break
            if s.hit is None:
                s.hit = s.computed
            if pairs:
                copies += pairs
            seen.update(self._chain(s.tokens, (s.computed + n) // kv.bs))
            self.waiting.remove(s)
            running.append(s)
            if n:
                items.append(Item(s, n, empty))
            budget -= n
        return items, copies
