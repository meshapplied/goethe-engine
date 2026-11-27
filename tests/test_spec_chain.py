"""Async chaining of speculative rounds: round N+1 is queued with GC_W_PREV_TOKEN before round N is read, and the
engine's bookkeeping stays exact when each round commits a different number of tokens."""
from dataclasses import replace

import pytest

from conftest import FakeSpec, alone, fake_greedy, fold, row
from goethe_engine import Config, Engine, SamplingParams as P
from goethe_engine.sampling import sample

W = 4
MML = 100


class AsyncSpec(FakeSpec):
    """FakeSpec with a FIFO queue of two: a step runs at step_begin on the device state (a PREV_TOKEN round takes its
    token and position from that state), step_wait pops the oldest result. Every round commits ks[i] tokens."""

    async_step, device_sample, spec_prev_token, max_seqs = True, True, True, 2

    def __init__(self, ks=(3, 1, 4)):
        super().__init__(W)
        self.ks, self.round, self.q, self.begun, self.last, self.bufs, self.sizes = ks, 0, [], [], {}, {}, []

    def step_begin(self, work):
        assert len(self.q) < 2, "more than two steps queued"
        res = []
        self.sizes += [len(work)] * len(work)
        for w in work:
            buf, tok, pos = self.bufs.setdefault(w.sid, []), w.tokens, w.pos
            if w.prev_token:
                assert w.spec and w.tokens == [0] and w.sid in self.last
                tok, pos = [self.last[w.sid]], len(buf)
                assert w.pos <= pos, "placeholder position above the device position"
                assert pos + self.spec_width <= MML, "chained round writes rows past max_model_len"
            assert pos <= len(buf)
            step = w.step_idx + (pos - w.pos)     # the step a round samples at, from the device position (spec_caps.round_step)
            self.begun.append((w, pos))
            buf[pos:] = tok
            if not w.want:
                res.append([])
                continue
            k = self.ks[self.round % len(self.ks)] if w.spec else 1
            self.round += w.spec
            out = fake_sample(buf, k, P(temperature=w.temperature, top_p=w.top_p, top_k=w.top_k), w.seed, step) if w.sample else fake_greedy(buf, k)
            buf += out[:-1] if w.spec else []
            self.last[w.sid] = out[-1]
            res.append(out)
        self.q.append(res)
        return (len(self.begun), True)

    def step_wait(self, handle):
        return self.q.pop(0)


def fake_sample(tokens, k, p, seed, step):
    """k tokens sampled from the fake model's rows at steps step .. step + k - 1, as the logits path draws them."""
    h, out, toks = 7, [], list(tokens)
    for q, t in enumerate(toks):
        h = fold(h, t, q)
    for i in range(k):
        t = sample(row(h)[None], [p], [seed], [step + i], [None])[0]
        out.append(t)
        h = fold(h, t, len(toks))
        toks.append(t)
    return out


class StepSpec(AsyncSpec):
    caps = {"round_step": True}


def test_a_pack_that_steps_on_the_device_chains_sampled_rounds_exactly():
    sp = P(temperature=1.0, seed=5, max_tokens=30)
    for be, chained in ((StepSpec(), True), (AsyncSpec(), False)):
        e = engine(be)
        e.add_request("a", [1, 2, 3], sp)
        assert run_all(e, be)[0]["a"] == alone([1, 2, 3], sp)[("r", 0)]
        assert any(w.prev_token and w.sample for w, _ in be.begun) == chained, "sampled rounds chain only on a pack that steps on the device"


def engine(be=None, **kw):
    return Engine(be or AsyncSpec(), Config(max_num_seqs=kw.pop("seqs", 1), prefix_cache=False, max_model_len=MML, **kw))


def run_all(e, be, check=True):
    """(tokens per rid, finish reason per rid, emitted counts per step) with the engine checked after every step."""
    got, why, per = {}, {}, []
    while e.has_work():
        outs = e.step()
        per.append(sum(len(o.token_ids) for o in outs))
        for o in outs:
            got.setdefault(o.rid, []).extend(o.token_ids)
            why[o.rid] = o.finish_reason or why.get(o.rid)
        e.check()
        if check and not be.q and (s := next(iter(e.live.get("a", [])), None)):
            assert s.computed == len(be.bufs[s.sid]) and s.computed == len(s.tokens) - 1, "host position drifted from the device"
    return got, why, per


def test_round_n_plus_1_is_queued_before_round_n_is_read():
    be = AsyncSpec()
    e = engine(be)
    e.add_request("a", [1, 2, 3], P(max_tokens=40))
    e.step()                                  # the prompt: not a spec round, nothing to chain
    assert [w.prev_token for w, _ in be.begun] == [False] and e._inflight is None
    e.step()
    (r1, _), (r2, _) = be.begun[1:]
    assert (r1.spec, r1.prev_token) == (True, False) and (r2.spec, r2.prev_token) == (True, True)
    assert r2.tokens == [0] and e._inflight is not None and len(be.q) == 1
    run_all(e, be)
    assert be.q == [] and e._inflight is None


def test_variable_acceptance_is_exact():
    be = AsyncSpec((3, 1, 4))
    e = engine(be)
    e.add_request("a", [1, 2, 3], P(max_tokens=17))
    s = e.live["a"][0]
    got, why, per = run_all(e, be)
    assert got["a"] == fake_greedy([1, 2, 3], 17) and why["a"] == "length"
    assert per == [1, 3, 1, 4, 3, 1, 4] and s.nout == 17 and len(s.tokens) == 20
    assert sum(w.prev_token for w, _ in be.begun) > 3
    assert len(be.begun) > len(per), "the last chained round was drained, not read"


def test_limit_inside_a_round_truncates_and_drains():
    be = AsyncSpec((4,))
    e = engine(be)
    e.add_request("a", [1, 2, 3], P(max_tokens=6))
    s = e.live["a"][0]
    got, why, _ = run_all(e, be)
    assert got["a"] == fake_greedy([1, 2, 3], 6) and why["a"] == "length" and s.nout == 6
    assert be.q == [] and e._inflight is None and not e._discard


@pytest.mark.parametrize("cut", [0, 1, 4, 5, 8])
def test_eos_in_round_n_drains_round_n_plus_1(cut):
    ref = fake_greedy([1, 2, 3], 20)
    i = [j for j, t in enumerate(ref) if ref.index(t) == j][cut + 1]   # a token's first occurrence, so the stop is at i
    be = AsyncSpec((3, 1, 4))
    e = engine(be)
    e.eos = {ref[i]}
    e.add_request("a", [1, 2, 3], P(max_tokens=20))
    s = e.live["a"][0]
    got, why, per = run_all(e, be)
    assert got["a"] == ref[:i + 1] and why["a"] == "stop" and s.nout == i + 1
    assert be.q == [] and e._inflight is None and not e._discard and len(be.begun) > len(per)
    assert e.kv.pool.free and len(e.kv.pool.free) == e.backend.geometry.num_blocks and be.released == [s.sid]


def test_max_model_len_boundary_never_chains_past_it():
    be = AsyncSpec((3, 1, 4))
    e = engine(be)
    e.add_request("a", [1, 2, 3], P(max_tokens=500))
    got, why, _ = run_all(e, be)           # step_begin asserts every chained round fits max_model_len
    assert got["a"] == fake_greedy([1, 2, 3], MML - 3) and why["a"] == "length"
    assert any(w.prev_token for w, _ in be.begun) and not be.begun[-1][0].prev_token


def spec_round(seqs=1, **params):
    """(engine, items, work) of the first spec round of `seqs` requests, after their prompts ran."""
    be = AsyncSpec()
    e = engine(be, seqs=seqs)
    for i in range(seqs):
        e.add_request("abc"[i], [1, 2, 3 + i], P(max_tokens=40, **params))
    while not (items := e.sched.schedule()[0]) or not all(it.spec for it in items):
        e.step()
    return e, items, e._build_work(items)[0]


def test_can_chain_table():
    e, items, work = spec_round()
    s, nout, c = items[0].seq, items[0].seq.nout, items[0].seq.computed
    assert e._can_chain(items, work, True)
    assert not e._can_chain(items, work, False)                      # the device did not return tokens
    s.limit = nout + 2
    assert e._can_chain(items, work, True)
    s.limit = nout + 1
    assert not e._can_chain(items, work, True)                       # this round can finish it
    s.limit = 40
    e.cfg.max_model_len = c + 2 * W
    assert e._can_chain(items, work, True)
    e.cfg.max_model_len = c + 2 * W - 1
    assert not e._can_chain(items, work, True)                       # the chained round would write past it
    e.cfg.max_model_len = MML
    s.params = replace(s.params, stop_sequences=((1, 2),))
    assert not e._can_chain(items, work, True)
    s.params = replace(s.params, stop_sequences=())
    s.grammar = object()
    assert not e._can_chain(items, work, True)
    s.grammar = None
    work[0].greedy, work[0].sample = False, True
    assert not e._can_chain(items, work, True)                       # sampled: greedy only
    work[0].greedy, work[0].sample = True, False
    e.backend.spec_prev_token = e._spec_chain = False
    assert not e._can_chain(items, work, True)                       # the backend does not take a device position
    e._spec_chain = True
    assert e._can_chain(items, work, True)


def test_chain_next_bounds_blocks_and_keeps_spec():
    e, items, work = spec_round()
    s = items[0].seq
    (it,), (w,) = e._chain_next(items, work)
    assert (it.spec, w.spec, w.prev_token, w.tokens, w.pos) == (True, True, True, [0], s.computed + 1)
    assert len(s.blocks) * e.kv.bs >= s.computed + 2 * W, "rows of a chained round are reserved"


def test_batch_and_non_greedy_run_unchained_and_exact():
    e, items, work = spec_round(seqs=2)
    assert len(items) == 2 and not e._can_chain(items, work, True)
    for kw, seqs in ((dict(), 2), (dict(temperature=0.7, seed=5), 1), (dict(stop_sequences=[[0, 0, 0]]), 1)):
        be = AsyncSpec()
        e = engine(be, seqs=seqs)
        for i in range(seqs):
            e.add_request("abc"[i], [1, 2, 3 + i], P(max_tokens=12, **kw))
        got, _, _ = run_all(e, be, check=False)
        want = lambda p: alone(p, P(max_tokens=12, **kw))[("r", 0)] if kw.get("temperature") else fake_greedy(p, 12)
        assert all(got["abc"[i]] == want([1, 2, 3 + i]) for i in range(seqs)), kw
        chained = [n for (w, _), n in zip(be.begun, be.sizes) if w.prev_token]
        assert chained == ([1] * len(chained) if seqs == 2 else []), kw   # a lone survivor of a batch may chain, a pair never
        assert (2 in be.sizes) == (seqs == 2), kw


def test_a_backend_without_device_position_never_chains_spec():
    be = AsyncSpec()
    be.spec_prev_token = False
    e = engine(be)
    e.add_request("a", [1, 2, 3], P(max_tokens=12))
    got, _, _ = run_all(e, be)
    assert got["a"] == fake_greedy([1, 2, 3], 12) and not any(w.prev_token for w, _ in be.begun)
