"""The engine: add_request / abort / step, token level. Sampling, stop rules, grammar masks and speculative
acceptance live here; scheduling in scheduler.py, KV bookkeeping in kv.py, GPU work behind Backend."""
import random
from collections import deque
from dataclasses import dataclass, field, replace

import torch

from .backend import Work
from .kv import KVManager
from .sampling import SamplingParams, dist, mix, sample
from .scheduler import Config, Scheduler, Sequence, strict
from .spec import Tree, caps_of, check, compatible, mss


@dataclass
class RequestOutput:
    rid: str
    index: int
    token_ids: list  # tokens new in this step
    finished: bool = False
    finish_reason: str | None = None  # stop | length | abort | error
    error: str | None = None
    stats: dict = field(default_factory=dict)  # on finish: prompt_tokens, completion_tokens, cached_tokens, computed_prefill_tokens, preemptions, seed


class Engine:
    def __init__(self, backend, cfg=None, drafter=None, grammar=None, eos_ids=()):
        cfg = cfg or Config()
        g = backend.geometry
        linear = bool(getattr(backend, "spec_width", 0)) or not getattr(backend, "paged", True)
        if cfg.max_num_seqs is None or cfg.prefix_cache is None:   # unset knobs follow what the backend can do; explicit ones are checked below
            cfg = replace(cfg, max_num_seqs=(1 if linear else min(64, backend.max_seqs)) if cfg.max_num_seqs is None else cfg.max_num_seqs,
                          prefix_cache=(not linear and not caps_of(backend)["recurrent"]) if cfg.prefix_cache is None else cfg.prefix_cache)
        if g.dtype != cfg.kv_dtype:
            raise ValueError(f"backend KV dtype {g.dtype} != config {cfg.kv_dtype} (fp8 is an explicit opt-in)")
        if cfg.max_num_seqs > backend.max_seqs:
            raise ValueError("max_num_seqs exceeds the backend's")
        if cfg.max_num_batched_tokens < 1 + cfg.spec_k:
            raise ValueError("token budget below one verify step")
        self.backend, self.cfg, self.grammar, self.eos = backend, cfg, grammar, set(eos_ids)
        self.width = getattr(backend, "spec_width", 0)
        pages = 0 if self.width or getattr(backend, "paged", True) else getattr(backend, "state_pages", 0)
        if cfg.prefix_cache and not pages and (self.width or not getattr(backend, "paged", True)):
            raise ValueError("backend has no prefix cache: no paged KV and no snapshot pages")
        if cfg.max_model_len > getattr(backend, "max_len", cfg.max_model_len):
            raise ValueError(f"max_model_len exceeds the backend's {backend.max_len}")
        self.caps = caps_of(backend)
        if self.caps["recurrent"] and cfg.prefix_cache and not pages:
            raise ValueError("backend has recurrent state: a prefix-cache hit would skip the rows that build it")
        if drafter is not None:
            r = check(self.caps, getattr(drafter, "needs", {}), cfg.spec_k, cfg.prefix_cache)
            if r:
                raise ValueError(f"speculative policy refused: {r}")
        need = tuple(getattr(drafter, "needs", {}).get("taps", ())) if drafter is not None else ()
        self.drafter, self.feats, self.cols = drafter, bool(need), None
        if need and need[:1] != ("layers",):
            off, at = 0, {}
            for k, n in self.caps["taps"].items():
                at[k], off = range(off, off + n), off + n
            self.cols = torch.tensor([c for k in need for c in at[k]])
        self.kv = KVManager(g.num_blocks, g.block_size, cfg.prefix_cache, pages)
        self.sched = Scheduler(cfg, self.kv, drafter, bool(self.width), backend.vocab_size)
        self.inbox, self.live, self.counter = deque(), {}, 0
        self.m = dict(steps=0, prompt_tokens=0, generation_tokens=0, prefix_hit_tokens=0, aborted=0, errors=0,
                      spec_drafted=0, spec_accepted=0, spec_invalid=0)
        self._out = {}

    def add_request(self, rid, prompt, params=None):
        """Validate now (ValueError), enqueue for the next step. Safe to call from another thread."""
        params = params or SamplingParams()
        params.validate()
        if rid in self.live:
            raise ValueError(f"duplicate request id {rid}")
        if not prompt or not all(0 <= t < self.backend.vocab_size for t in prompt):
            raise ValueError("prompt must be non-empty token ids inside the vocabulary")
        if len(prompt) >= self.cfg.max_model_len:
            raise ValueError("prompt does not fit max_model_len")
        if params.accept == "rejection" and params.temperature > 0:
            if not self.caps["logits"]:
                raise ValueError("rejection acceptance refused: needs target logits, backend returns device tokens only")
        if params.n > self.cfg.max_num_seqs:
            raise ValueError("n exceeds max_num_seqs")
        limit = min(params.max_tokens, self.cfg.max_model_len - len(prompt))
        bs, nb = self.backend.geometry.block_size, self.backend.geometry.num_blocks
        if -(-(len(prompt) + limit + (self.cfg.spec_k if self.drafter else 0)) // bs) > nb:
            raise ValueError("request needs more KV blocks than the pool has")
        compiled = None
        if params.grammar:
            if not self.grammar:
                raise ValueError("no grammar compiler configured")
            compiled = self.grammar.compile(*params.grammar)
        self.counter += 1
        seed = params.seed if params.seed is not None else random.getrandbits(63)
        seq = Sequence(self.counter, rid, 0, prompt, params, self.counter, seed,
                       compiled.new_matcher() if compiled else None, limit)
        seq.compiled = compiled
        self.live[rid] = [seq]
        self.inbox.append(("add", seq))

    def abort(self, rid):
        """Idempotent; unknown or finished ids are ignored. Safe to call from another thread."""
        self.inbox.append(("abort", rid))

    def has_work(self):
        return bool(self.inbox or self.sched.waiting or self.sched.running)

    def _drain(self):
        while self.inbox:
            op, x = self.inbox.popleft()
            if op == "add":
                self.sched.add(x)
            else:
                for s in list(self.live.get(x, [])):
                    self._finish(s, "abort")

    def _emit(self, s, toks, reason=None, error=None):
        o = self._out.get(s.sid)
        if o is None:
            o = self._out[s.sid] = RequestOutput(s.rid, s.index, [])
        o.token_ids += toks
        if reason:
            o.finished, o.finish_reason, o.error = True, reason, error
            o.stats = dict(prompt_tokens=s.nprompt, completion_tokens=s.nout, cached_tokens=s.hit or 0,
                           computed_prefill_tokens=s.prefilled, preemptions=s.preemptions, seed=s.seed)

    def _finish(self, s, reason, error=None):
        if s.finished:
            return
        s.finished = reason
        if reason in ("stop", "length"):
            self._copy(self.kv.snapshot(s))
        self.sched.remove(s)
        self.backend.release(s.sid)
        if hasattr(self.drafter, "release"):
            self.drafter.release(s.sid)
        self.live[s.rid].remove(s)
        if not self.live[s.rid]:
            del self.live[s.rid]
        self.m["aborted"] += reason == "abort"
        self.m["prompt_tokens"] += s.nprompt
        self.m["prefix_hit_tokens"] += s.hit or 0
        self.m["errors"] += reason == "error"
        self._emit(s, [], reason, error)

    def _copy(self, pairs):
        if pairs:
            self.backend.copy_blocks(pairs)

    def _flush(self):
        out, self._out = list(self._out.values()), {}
        return out

    def step(self):
        """One scheduler step and one backend call. Returns the outputs (token deltas, finishes) of the step."""
        self._drain()
        items, copies = self.sched.schedule()
        self.m["spec_invalid"] += self.sched.invalid
        for sid in self.sched.preempted:
            self.backend.release(sid)
            if hasattr(self.drafter, "release"):
                self.drafter.release(sid)
        if not items:
            self._copy(copies)
            if self.sched.waiting and not self.sched.running:
                for s in list(self.sched.waiting):
                    self._finish(s, "error", "cannot be scheduled")
            return self._flush()
        self.m["steps"] += 1
        work = [Work(it.seq.sid, it.tokens, it.seq.computed, list(it.seq.blocks), it.want,
                     it.seq.params.temperature == 0 and not it.seq.grammar, it.spec, it.parents, self.feats)
                for it in items]
        try:
            self._copy(copies)
            logits, feats = self.backend.step(work), None
            if self.feats:
                logits, feats = logits
                if tuple(feats.shape[:1]) != (sum(len(w.tokens) for w in work),):
                    raise RuntimeError(f"backend returned features of shape {tuple(feats.shape)}")
            if isinstance(logits, list):
                ok = len(logits) == len(work) and all(w.greedy and (len(t) == w.want or w.spec and t) or not w.want and not t
                                                      for w, t in zip(work, logits))
                if not ok:
                    raise RuntimeError("backend returned tokens that do not match the work")
            elif tuple(logits.shape) != (sum(w.want for w in work), self.backend.vocab_size):
                raise RuntimeError(f"backend returned logits of shape {tuple(logits.shape)}")
        except Exception as e:
            for it in items:
                self._finish(it.seq, "error", f"{type(e).__name__}: {e}")
            return self._flush()
        self._post(items, logits, feats)
        return self._flush()

    def _fork(self, s):
        out = []
        for i in range(1, s.params.n):
            self.counter += 1
            c = Sequence(self.counter, s.rid, i, s.tokens, s.params, self.counter, mix(s.seed, i),
                         s.compiled.new_matcher() if s.compiled else None, s.limit)
            c.compiled, c.hit, c.prefilled = s.compiled, s.hit, s.prefilled
            self.kv.fork(s, c)
            self.sched.running.append(c)
            self.live[s.rid].append(c)
            out.append(c)
        s.forks = 0
        return out

    def _append(self, s, t):
        s.tokens.append(t)
        p = s.params
        if s.grammar and not s.grammar.accept(t):
            return "error"
        if (t in self.eos and not p.ignore_eos) or t in p.stop_token_ids:
            return "stop"
        if any(s.tokens[-len(q):] == list(q) for q in p.stop_sequences):
            return "stop"
        if s.grammar and s.grammar.done():
            return "stop"
        return "length" if s.nout >= s.limit else None

    def _feed(self, s, pos, rows):
        if rows is not None and hasattr(self.drafter, "feed"):
            self.drafter.feed(s, pos, rows if self.cols is None else rows[:, self.cols.to(rows.device)])

    def _post(self, items, res, feats):
        """Acceptance: walk each item's tree from the anchor, one node per round across the batch. A node's token is
        the target's seeded sample (exact) or, for a strict request at a node with children, multi-step speculative
        sampling; the walk follows the child holding it. Then commit the path (rows, recurrent state, KV tree)."""
        samp, off, fo = [], 0, 0
        for ri, it in enumerate(items):
            s, n, base = it.seq, it.n, it.seq.computed
            rows = feats[fo:fo + n] if feats is not None else None
            fo += n
            s.prefilled += max(0, min(s.computed + n, s.nprompt) - s.computed)
            want = it.want
            s.computed += n
            if not want:
                self._copy(self.kv.commit(s))
                self._feed(s, base, rows)
                continue
            if it.spec:
                it.tree = Tree.chain(res[ri][:-1])
                self.m["spec_drafted"] += self.width - 1
            else:
                self.m["spec_drafted"] += len(it.tree)
            e = [it, off, ri, 0, [0], it.tree.kids(), base, rows]
            samp.append(e)
            if s.forks:
                samp += [[type(it)(c, 0, Tree([], [])), off, ri, 0, [0], [[]], c.computed, None] for c in self._fork(s)]
            off += want
        reason, active = {}, samp
        while active:
            ss = [e[0].seq for e in active]
            if isinstance(res, list):
                toks = [res[e[2]][e[3]] for e in active]
            else:
                ex = [k for k, e in enumerate(active) if not (strict(e[0].seq) and e[5][e[3]])]
                toks = [None] * len(active)
                if ex:
                    idx = torch.tensor([active[k][1] + active[k][3] for k in ex], device=res.device)
                    got = sample(res[idx], [ss[k].params for k in ex], [ss[k].seed for k in ex],
                                 [ss[k].nout for k in ex], [ss[k].grammar.mask() if ss[k].grammar else None for k in ex])
                    for k, t in zip(ex, got):
                        toks[k] = t
            nxt = []
            for e, s, t in zip(active, ss, toks):
                it, o, ri, u, pth, kids = e[:6]
                hit = None
                if t is None:
                    p = dist(res[o + u], s.params, s.grammar.mask() if s.grammar else None)
                    q = it.tree.q
                    t, r = mss(p, [(it.tree.tokens[c - 1], q[c - 1] if q else None) for c in kids[u]], s.seed, s.nout)
                    hit = kids[u][r] if r is not None else None
                elif t >= 0:
                    hit = next((c for c in kids[u] if it.tree.tokens[c - 1] == t), None)
                if t < 0:
                    reason[s.sid] = "error"
                    continue
                r = self._append(s, t)
                self._emit(s, [t])
                self.m["generation_tokens"] += 1
                if r:
                    reason[s.sid] = r
                elif hit is not None:
                    e[3] = hit
                    pth.append(hit)
                    nxt.append(e)
            active = nxt
        commit = getattr(self.backend, "commit", None)
        for it, _, _, _, pth, _, base, rows in samp:
            s = it.seq
            self.m["spec_accepted"] += len(pth) - 1
            if len(it.tree) and not it.spec and commit:
                commit(s.sid, base, pth)
            s.computed = len(s.tokens) - 1
            r = reason.get(s.sid)
            if r != "error":
                self._copy(self.kv.commit(s))
                self._feed(s, base, rows if rows is None or not len(it.tree) else rows[pth])
            if r:
                self._finish(s, r, "no valid token" if r == "error" else None)

    def generate(self, reqs):
        """Blocking helper: [(prompt, params)] -> per request, per index, the generated tokens."""
        for i, (p, sp) in enumerate(reqs):
            self.add_request(i, p, sp)
        out = [dict() for _ in reqs]
        while self.has_work():
            for o in self.step():
                out[o.rid].setdefault(o.index, []).extend(o.token_ids)
        return [[v[k] for k in sorted(v)] for v in out]

    def spec_algorithms(self):
        """{algorithm: None if this backend runs it, else why not} (spec.ALGOS)."""
        return compatible(self.caps, max(1, self.cfg.spec_k))

    def numerics(self):
        """The backend's numerics: {contract, bodies, desc} (PackBackend.numerics), None if it does not say."""
        return getattr(self.backend, "numerics", None)

    def metrics(self):
        g, p = self.backend.geometry, self.kv.pool
        return dict(self.m, running=len(self.sched.running), waiting=len(self.sched.waiting),
                    preemptions=self.sched.preemptions, kv_blocks_total=g.num_blocks, kv_blocks_free=len(p.free),
                    kv_tree_blocks=sum(len(n.blocks) for n in self.kv.cache.nodes()) if self.kv.cache else 0)

    def check(self):
        self.kv.check(self.sched.running + self.sched.waiting)
