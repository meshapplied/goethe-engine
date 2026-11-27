"""The engine: add_request / abort / step, token level. Sampling, stop rules, grammar masks and speculative
acceptance live here; scheduling in scheduler.py, KV bookkeeping in kv.py, GPU work behind Backend."""
import random
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from .backend import Work
from .kv import KVManager
from .sampling import SamplingParams, dist, mix, sample
from .scheduler import Config, Item, Scheduler, Sequence, _EMPTY_TREE, strict
from .spec import Tree, caps_of, check, compatible, mss


@dataclass(slots=True)
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
        linear = bool(getattr(backend, "spec_width", 0))   # a spec pair runs one stream; page packs run many (a block is a sequence page)
        bounded = bool(getattr(backend, "bounded", False))
        shut = linear and not caps_of(backend)["state_restore"]   # a pair keeps snapshot pages only when its pack restores them
        snaps = 0 if shut or (getattr(backend, "paged", True) and not bounded) else getattr(backend, "state_pages", 0)
        if cfg.max_num_seqs is None or cfg.prefix_cache is None:   # unset knobs follow what the backend can do; explicit ones are checked below
            cfg = replace(cfg, max_num_seqs=(1 if linear else min(64, backend.max_seqs)) if cfg.max_num_seqs is None else cfg.max_num_seqs,
                          prefix_cache=(snaps > 0 if bounded or not getattr(backend, "paged", True) else not linear and not caps_of(backend)["recurrent"] and getattr(backend, "prefix_cache", True)) if cfg.prefix_cache is None else cfg.prefix_cache)
        if g.dtype != cfg.kv_dtype:
            raise ValueError(f"backend KV dtype {g.dtype} != config {cfg.kv_dtype} (fp8 is an explicit opt-in)")
        if cfg.max_num_seqs > backend.max_seqs:
            raise ValueError("max_num_seqs exceeds the backend's")
        if cfg.max_num_batched_tokens < 1 + cfg.spec_k:
            raise ValueError("token budget below one verify step")
        self.backend, self.cfg, self.grammar, self.eos = backend, cfg, grammar, set(eos_ids)
        self.width = getattr(backend, "spec_width", 0)
        pages = 0 if shut or (getattr(backend, "paged", True) and not bounded) else getattr(backend, "state_pages", 0)
        if cfg.prefix_cache and not pages and (self.width or bounded or not getattr(backend, "paged", True)):
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
            self.cols = np.array([c for k in need for c in at[k]])
        self.kv = KVManager(g.num_blocks, g.block_size, cfg.prefix_cache, pages, bounded)
        self.sched = Scheduler(cfg, self.kv, drafter, bool(self.width), backend.vocab_size)
        self.inbox, self.live, self.counter = deque(), {}, 0
        self.m = dict(steps=0, prompt_tokens=0, generation_tokens=0, prefix_hit_tokens=0, aborted=0, errors=0,
                      spec_drafted=0, spec_accepted=0, spec_invalid=0, spec_rounds=0)
        self._out = {}
        self._sp, self.spec_window = {}, deque(maxlen=64)    # per live sequence [rounds, accepted, drafted]; the last 64 finished requests
        self._async = bool(getattr(backend, "async_step", False))
        self._spec_chain = bool(getattr(backend, "spec_prev_token", False))   # a spec round with GC_W_PREV_TOKEN takes its token and position from the device
        self._dev_sample = bool(getattr(backend, "device_sample", False))
        self._dev_bitmask = bool(getattr(backend, "device_bitmask", False))
        self._inflight = None        # (items, work, handle) currently executing on device
        self._inflight_sids = set()
        self._discard = {}           # sid -> Sequence deferred release while still in _inflight

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
        return bool(self._inflight or self.inbox or self.sched.waiting or self.sched.running)

    def warm(self):
        """Warm each advertised graph_bucket rung once (safe when idle; borrows one pool block if paged)."""
        fn = getattr(self.backend, "warm_buckets", None)
        if not fn or not getattr(self.backend, "graph_buckets", ()):
            return
        if self._inflight:
            self._drain_inflight()
        blk = 0
        took = False
        if getattr(self.backend, "paged", True) and self.kv.pool.free:
            blk, = self.kv.pool.take(1)
            took = True
        try:
            fn(self.cfg.max_num_batched_tokens, dummy_block=blk)
        finally:
            if took:
                self.kv.pool.decref(blk)

    def _drain(self):
        while self.inbox:
            op, x = self.inbox.popleft()
            if op == "add":
                self.sched.add(x)
            elif op == "warm":
                try:
                    self.warm()
                finally:
                    x.set()
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
        if reason in ("stop", "length") and s.sid not in self._inflight_sids:
            self._copy(self.kv.snapshot(s))
        if s.sid in self._inflight_sids:
            for q in (self.sched.waiting, self.sched.running):
                if s in q:
                    q.remove(s)
            self._discard[s.sid] = s
        else:
            self.sched.remove(s)
            self.backend.release(s.sid)
            if hasattr(self.drafter, "release"):
                self.drafter.release(s.sid)
        w = self._sp.pop(s.sid, None)
        if w:
            self.spec_window.append(tuple(w))
        self.live[s.rid].remove(s)
        if not self.live[s.rid]:
            del self.live[s.rid]
        self.m["aborted"] += reason == "abort"
        self.m["prompt_tokens"] += s.nprompt
        self.m["prefix_hit_tokens"] += s.hit or 0
        self.m["errors"] += reason == "error"
        self._emit(s, [], reason, error)

    def _release_discarded(self):
        if not self._discard:
            return
        for s in self._discard.values():
            self.kv.release(s)
            self.backend.release(s.sid)
            if hasattr(self.drafter, "release"):
                self.drafter.release(s.sid)
        self._discard.clear()

    def _drain_inflight(self):
        if not self._inflight:
            return
        _, _, _, handle = self._inflight
        self._inflight = None
        self._inflight_sids = set()
        try:
            self.backend.step_wait(handle)
        finally:
            self._release_discarded()

    def _build_work(self, items):
        feats_on, ds, db = self.feats, self._dev_sample, self._dev_bitmask
        work = []
        total_want = 0
        for it in items:
            s, p, want = it.seq, it.seq.params, it.want
            total_want += want
            greedy = p.temperature == 0 and s.grammar is None
            bm, use_samp = None, False
            if want and not greedy and not s.forks and not (strict(s) and it.tree.tokens):
                if s.grammar is not None:
                    if want == 1 and db and hasattr(s.grammar, "bitmask") and (p.temperature == 0 or ds):
                        bm = s.grammar.bitmask()
                        use_samp = True
                elif ds:
                    use_samp = True
            work.append(Work(s.sid, it.tokens, s.computed, s.blocks, want, greedy, it.spec, it.parents, feats_on,
                             use_samp, p.temperature, p.top_p, p.top_k, s.seed, s.nout, bm, False, s.blocks_ver))
        return work, total_want

    def _validate(self, work, total_want, logits):
        feats = None
        if self.feats:
            logits, feats = logits
            if tuple(feats.shape[:1]) != (sum(len(w.tokens) for w in work),):
                raise RuntimeError(f"backend returned features of shape {tuple(feats.shape)}")
        if isinstance(logits, list):
            if len(logits) != len(work):
                raise RuntimeError("backend returned tokens that do not match the work")
            for w, t in zip(work, logits):
                if not (((w.greedy or w.sample) and (len(t) == w.want or (w.spec and t))) or (not w.want and not t)):
                    raise RuntimeError("backend returned tokens that do not match the work")
        elif tuple(logits.shape) != (total_want, self.backend.vocab_size):
            raise RuntimeError(f"backend returned logits of shape {tuple(logits.shape)}")
        return logits, feats

    def _can_chain(self, items, work, dev_tok):
        if not (self._async and dev_tok and items) or self.inbox or self.sched.waiting or self._discard:
            return False
        if self.drafter is not None or self.feats:
            return False
        if self.width and not (self._spec_chain and len(items) == 1):
            return False
        bs, pool, cache = self.kv.bs, self.kv.pool, self.kv.cache
        need_free = 0
        for it, w in zip(items, work):
            s = it.seq
            if it.want != 1 or it.spec != bool(self.width) or it.tree.tokens or s.finished or s.grammar is not None or s.forks or s.restore is not None:
                return False
            if not (w.greedy or w.sample) or s.nout + 1 >= s.limit:
                return False
            if self.width and (not (w.greedy or (w.sample and self.caps["round_step"])) or s.params.stop_sequences):
                return False
            pos, rows = self._chain_rows(s, w)
            if rows > self.cfg.max_model_len:
                return False
            if cache and (pos // bs) > s.tpages:
                return False
            nb = len(s.blocks)
            need = -(-rows // bs)
            if any(pool.ref[s.blocks[i]] > 1 for i in range(s.computed // bs, min(nb, need))):
                return False
            if need > nb:
                need_free += need - nb
        return need_free <= len(pool.free)

    def _chain_rows(self, s, w):
        """(placeholder position, rows to reserve); a spec round's commit count is device-side, so bound it by width."""
        if self.width:
            return s.computed + 1, s.computed + 2 * self.width
        npos = w.pos + len(w.tokens)
        return npos, npos + 1

    def _chain_next(self, items, work):
        next_items, next_work = [], []
        for it, w in zip(items, work):
            s = it.seq
            pos, rows = self._chain_rows(s, w)
            pairs = self.kv.reserve(s, rows)
            assert pairs == [], "chained reserve must not copy or fail"
            next_items.append(Item(s, 1, _EMPTY_TREE, it.spec))
            next_work.append(Work(s.sid, [0], pos, s.blocks, 1, w.greedy, it.spec, None, False,
                                  w.sample, w.temperature, w.top_p, w.top_k, w.seed,
                                  pos - s.nprompt + 1 if self.width else w.step_idx + 1,   # a pair's step follows its placeholder position
                                  None, True, s.blocks_ver))
        return next_items, next_work

    def _complete(self, items, work, total_want, handle, skip_rids=None, skip_sids=None):
        try:
            logits, feats = self._validate(work, total_want, self.backend.step_wait(handle))
        except Exception as e:
            if self._inflight:
                try:
                    self._drain_inflight()
                except Exception:
                    pass
            self._release_discarded()
            for it in items:
                self._finish(it.seq, "error", f"{type(e).__name__}: {e}")
            return
        if self._discard or skip_rids or skip_sids:
            keep_items, keep_res = [], []
            for ri, it in enumerate(items):
                if (it.seq.sid in self._discard or (skip_rids and it.seq.rid in skip_rids)
                        or (skip_sids and it.seq.sid in skip_sids)):
                    continue
                keep_items.append(it)
                if isinstance(logits, list):
                    keep_res.append(logits[ri])
            self._release_discarded()
            if keep_items:
                self.m["steps"] += 1
                self._post(keep_items, keep_res if isinstance(logits, list) else logits, feats)
        else:
            self.m["steps"] += 1
            self._post(items, logits, feats)
        if self._inflight and self._inflight_sids.issubset(self._discard):
            self._drain_inflight()

    def _copy(self, pairs):
        if not pairs:
            return
        ops = [p for p in pairs if p[0] == "save"]
        pairs = [p for p in pairs if p[0] != "save"]
        if pairs:
            self.backend.copy_blocks(pairs)
        for _, sid, snap in ops:
            self.backend.save_state(sid, snap)

    def _flush(self):
        if not self._out:
            return []
        out, self._out = list(self._out.values()), {}
        return out

    def step(self):
        """One scheduler step and one backend call. Returns the outputs (token deltas, finishes) of the step."""
        if self._inflight is not None:
            items, work, total_want, handle = self._inflight
            abort_rids = {x for op, x in self.inbox if op == "abort"} if self.inbox else None
            running_set = set(self.sched.running)
            ext_drop = {it.seq.sid for it in items if it.seq not in running_set and it.seq.sid not in self._discard}
            if not abort_rids and not ext_drop and self._can_chain(items, work, handle[1]):
                try:
                    n_items, n_work = self._chain_next(items, work)
                    n_handle = self.backend.step_begin(n_work)
                    self._inflight = (n_items, n_work, len(n_work), n_handle)
                    self._inflight_sids = {it.seq.sid for it in n_items}
                except Exception as e:
                    self._inflight, self._inflight_sids = None, set()
                    try:
                        self.backend.step_wait(handle)
                    except Exception:
                        pass
                    self._release_discarded()
                    for it in items:
                        self._finish(it.seq, "error", f"{type(e).__name__}: {e}")
                    return self._flush()
            else:
                self._inflight, self._inflight_sids = None, set()
            self._complete(items, work, total_want, handle, skip_rids=abort_rids, skip_sids=ext_drop)
            if abort_rids and self.inbox:
                self._drain()
            return self._flush()
        if self.inbox:
            self._drain()
        items, copies = self.sched.schedule()
        self.m["spec_invalid"] += self.sched.invalid
        if self.sched.preempted:
            if self._inflight:
                self._drain_inflight()
            for sid in self.sched.preempted:
                self.backend.release(sid)
                if hasattr(self.drafter, "release"):
                    self.drafter.release(sid)
        if not items:
            if copies:
                self._copy(copies)
            if self.sched.waiting and not self.sched.running:
                for s in list(self.sched.waiting):
                    self._finish(s, "error", "cannot be scheduled")
            return self._flush()
        work, total_want = self._build_work(items)
        try:
            if copies:
                self._copy(copies)
            if self.kv.bsnaps:
                for it in items:
                    if it.seq.restore is not None:
                        self.backend.restore_state(it.seq.restore, it.seq.sid)
                        it.seq.restore = None
            if self._async and "step" not in self.backend.__dict__:
                handle = self.backend.step_begin(work)
                if self._can_chain(items, work, handle[1]):
                    try:
                        n_items, n_work = self._chain_next(items, work)
                        n_handle = self.backend.step_begin(n_work)
                        self._inflight = (n_items, n_work, len(n_work), n_handle)
                        self._inflight_sids = {it.seq.sid for it in n_items}
                    except Exception:
                        self._inflight, self._inflight_sids = None, set()
                self._complete(items, work, total_want, handle)
                return self._flush()
            self.m["steps"] += 1
            logits, feats = self._validate(work, total_want, self.backend.step(work))
        except Exception as e:
            if self._inflight:
                try:
                    self._drain_inflight()
                except Exception:
                    pass
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
        toks = s.tokens
        toks.append(t)
        p = s.params
        if s.grammar is not None and not s.grammar.accept(t):
            return "error"
        if (self.eos and not p.ignore_eos and t in self.eos) or (p.stop_token_ids and t in p.stop_token_ids):
            return "stop"
        if p.stop_sequences:
            for q in p.stop_sequences:
                if toks[-len(q):] == list(q):
                    return "stop"
        if s.grammar is not None and s.grammar.done():
            return "stop"
        return "length" if len(toks) >= s.stop_len else None

    def _feed(self, s, pos, rows):
        if rows is not None and hasattr(self.drafter, "feed"):
            self.drafter.feed(s, pos, rows if self.cols is None else rows[:, self.cols])

    def _post(self, items, res, feats):
        """Acceptance: walk each item's tree from the anchor, one node per round across the batch. A node's token is
        the target's seeded sample (exact) or, for a strict request at a node with children, multi-step speculative
        sampling; the walk follows the child holding it. Then commit the path (rows, recurrent state, KV tree)."""
        if isinstance(res, list) and feats is None and self.drafter is None:
            m, kv, out = self.m, self.kv, self._out
            for ri, it in enumerate(items):
                s, n = it.seq, it.n
                if s.computed < s.nprompt:
                    s.prefilled += min(s.computed + n, s.nprompt) - s.computed
                s.computed += n
                toks = res[ri]
                if not toks:
                    c = kv.commit(s)
                    if c:
                        self._copy(c)
                    continue
                if not it.spec:
                    t = toks[0]
                    if t < 0:
                        s.computed = len(s.tokens) - 1
                        self._finish(s, "error", "no valid token")
                        continue
                    r = self._append(s, t)
                    o = out.get(s.sid)
                    if o is None:
                        out[s.sid] = RequestOutput(s.rid, s.index, [t])
                    else:
                        o.token_ids.append(t)
                    m["generation_tokens"] += 1
                    s.computed = len(s.tokens) - 1
                    if r != "error":
                        c = kv.commit(s)
                        if c:
                            self._copy(c)
                    if r:
                        self._finish(s, r, "no valid token" if r == "error" else None)
                else:
                    d = self.width - 1
                    m["spec_drafted"] += d
                    emitted, r, acc = [], None, 0
                    for j, t in enumerate(toks):
                        acc = j
                        if t < 0:
                            r = "error"
                            break
                        r = self._append(s, t)
                        emitted.append(t)
                        if r:
                            break
                    if emitted:
                        o = out.get(s.sid)
                        if o is None:
                            out[s.sid] = RequestOutput(s.rid, s.index, emitted)
                        else:
                            o.token_ids += emitted
                        m["generation_tokens"] += len(emitted)
                    m["spec_accepted"] += acc
                    if d:
                        w = self._sp.get(s.sid)
                        if w is None:
                            self._sp[s.sid] = [1, acc, d]
                        else:
                            w[0] += 1
                            w[1] += acc
                            w[2] += d
                        m["spec_rounds"] += 1
                    s.computed = len(s.tokens) - 1
                    if r != "error":
                        c = kv.commit(s)
                        if c:
                            self._copy(c)
                    if r:
                        self._finish(s, r, "no valid token" if r == "error" else None)
            return
        samp, off, fo = [], 0, 0
        for ri, it in enumerate(items):
            s, n, base = it.seq, it.n, it.seq.computed
            rows = feats[fo:fo + n] if feats is not None else None
            fo += n
            if s.computed < s.nprompt:
                s.prefilled += min(s.computed + n, s.nprompt) - s.computed
            want = it.want
            s.computed += n
            if not want:
                c = self.kv.commit(s)
                if c:
                    self._copy(c)
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
                    idx = [active[k][1] + active[k][3] for k in ex]
                    rows_in = res if len(idx) == len(res) and all(a == b for a, b in enumerate(idx)) else res[idx]
                    got = sample(rows_in, [ss[k].params for k in ex], [ss[k].seed for k in ex],
                                 [ss[k].nout for k in ex], [ss[k].grammar.mask() if ss[k].grammar else None for k in ex])
                    for k, t in zip(ex, got):
                        toks[k] = t
            nxt = []
            for e, s, t in zip(active, ss, toks):
                it, o, ri, u, pth, kids = e[:6]
                hit = None
                ku = kids[u]
                if t is None:
                    p = dist(res[o + u], s.params, s.grammar.mask() if s.grammar else None)
                    q = it.tree.q
                    t, r = mss(p, [(it.tree.tokens[c - 1], q[c - 1] if q else None) for c in ku], s.seed, s.nout)
                    hit = ku[r] if r is not None else None
                elif t >= 0 and ku:
                    hit = next((c for c in ku if it.tree.tokens[c - 1] == t), None)
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
                c = self.kv.commit(s)
                if c:
                    self._copy(c)
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
        self.kv.check(self.sched.running + self.sched.waiting + list(self._discard.values()))
