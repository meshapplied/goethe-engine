"""Conformance of a gc-c pack runtime behind GcBackend and the Engine. Backend-agnostic properties (batch invariance,
chunking, prefix cache, small pool and preemption, seeded sampling, tree verify + commit, error codes) run against
the mock libgck by default. Against real packs (on a GPU box):
  GC_CONF_PACK=<signed pack dir> GC_CONF_WEIGHTS=<file>[:<file>...] [GC_CONF_SKU=<sku>] [GC_CONF_CONTRACT=K1]
  [GC_CONF_KEYID=<key id>]  python -m pytest tests/test_gc_conformance.py
(GC_CONF_KEYID trusts that key for this run; without it the pinned allow-list applies). Mock-only checks compare
against the closed-form oracle of conftest.py."""
import os
import random

import pytest

from conftest import fake_greedy
from goethe_engine import Config, Engine, SamplingParams
from goethe_engine.backend import Work
from goethe_engine.gc_c import GcBackend
from toypack import toy_pack

REAL = os.environ.get("GC_CONF_PACK")


@pytest.fixture
def make(tmp_path, key, mocklib):
    made = []

    def factory(**kw):
        if REAL:
            be = GcBackend(REAL, os.environ["GC_CONF_WEIGHTS"].split(":"), sku=os.environ.get("GC_CONF_SKU"),
                           contract=os.environ.get("GC_CONF_CONTRACT"),
                           trusted=(os.environ["GC_CONF_KEYID"],) if "GC_CONF_KEYID" in os.environ else None,
                           **{k: v for k, v in kw.items() if k in ("B", "ctx", "pool_blocks")})
        else:
            d = tmp_path / f"p{len(made)}"
            d.mkdir()
            pack, w, _ = toy_pack(str(d), key, mocklib)
            be = GcBackend(pack, w, sku="mock", trusted=(key.kid,), **kw)
        made.append(be)
        return be
    yield factory
    for b in made:
        b.close()


def prompts(be, n=6, seed=3):
    rng, hi = random.Random(seed), max(8, min(be.vocab_size - 1, 900))
    base = [rng.randrange(1, hi) for _ in range(40)]
    return [base[:rng.choice([1, 5, 9, 17, 33, 40])] + [rng.randrange(1, hi) for _ in range(rng.randint(1, 6))]
            for _ in range(n)]


def gen(be, ps, k=12, params=None, **cfg):
    eng = Engine(be, Config(max_model_len=min(be.max_len, 200), **cfg))
    out = eng.generate([(p, params or SamplingParams(max_tokens=k)) for p in ps])
    eng.check()
    return [o[0] for o in out]


def batch_only(be):
    """A single stream pack (linear KV, one slot) has no batch to invariance-test: these properties are about paged multi sequence buckets."""
    if not be.paged or be.max_seqs < 2:
        pytest.skip("single stream pack: no multi sequence batch")


def test_abi_and_layout(make):
    be = make()
    assert be.L.gc_abi_version() >> 16 == 1
    lo = be.layout
    assert lo.vocab > 1 and lo.max_seqs >= 1 and lo.num_blocks >= 1 and lo.block_size >= 1 and lo.max_ctx >= lo.block_size
    assert be.geometry.dtype in ("f16", "fp8")


def test_batch_invariance_and_chunking(make):
    be = make()
    batch_only(be)
    ps = prompts(be)
    alone = [gen(be, [p], 12, max_num_seqs=1, prefix_cache=False)[0] for p in ps]
    for cfg in (dict(max_num_seqs=8), dict(max_num_seqs=8, max_num_batched_tokens=7, prefix_cache=False),
                dict(max_num_seqs=3, max_num_batched_tokens=64)):
        assert gen(be, ps, **cfg) == alone, cfg


def test_prefix_cache_and_copy_blocks(make):
    be = make()
    batch_only(be)
    base = prompts(be, 1)[0] * 3
    ps = [base[:30] + [5, 6], base[:30] + [7], base[:30] + [5, 6], base[:19]]
    assert gen(be, ps, max_num_seqs=4) == gen(be, ps, max_num_seqs=4, prefix_cache=False)
    # a page pack (window ring, recurrent state) caches whole-page snapshots: the repeat of a finished prompt hits
    eng = Engine(be, Config(max_model_len=min(be.max_len, 200), max_num_seqs=4))
    cold = eng.generate([(ps[0], SamplingParams(max_tokens=12))])
    warm = eng.generate([(ps[0], SamplingParams(max_tokens=12)), (ps[0] + cold[0][0], SamplingParams(max_tokens=12))])
    assert warm[0] == cold[0]
    assert (eng.metrics()["prefix_hit_tokens"] > 0) == getattr(be, "prefix_cache", True)   # a bounded paged pack shares no blocks
    eng.check()


def test_small_pool_preempts_and_matches(make):
    big = make()
    batch_only(big)
    ps = prompts(big, 6)
    need = [-(-(len(p) + 20) // big.layout.block_size) for p in ps]
    assert sum(need) > max(need) + 1
    be = make(pool_blocks=max(need) + 1)  # the longest request fits, all six together do not
    cfg = dict(max_model_len=be.max_len, prefix_cache=False)
    ref = [o[0] for o in Engine(big, Config(max_num_seqs=1, **cfg)).generate([(p, SamplingParams(max_tokens=20)) for p in ps])]
    eng = Engine(be, Config(max_num_seqs=6, **cfg))
    got = eng.generate([(p, SamplingParams(max_tokens=20)) for p in ps])
    # a page pack's page is a sequence's whole reservation: a short pool queues the rest and never preempts
    assert [o[0] for o in got] == ref and (eng.metrics()["preemptions"] > 0 or not be.paged)


def test_forced_preemption_recomputes_to_the_same_tokens(make):
    be = make()
    ps = prompts(be, 5)
    ref = gen(be, ps, 14, max_num_seqs=1, prefix_cache=False)
    for cache in (False, True):
        eng = Engine(be, Config(max_model_len=min(be.max_len, 200), max_num_seqs=5, prefix_cache=cache))
        for i, p in enumerate(ps):
            eng.add_request(i, p, SamplingParams(max_tokens=14))
        out, k = {}, 0
        while eng.has_work():
            for o in eng.step():
                out.setdefault(o.rid, []).extend(o.token_ids)
            eng.check()
            k += 1
            if k % 3 == 0 and eng.sched.running:
                v = eng.sched.running[-1]
                eng.sched._preempt(v)
                be.release(v.sid)
        assert [out[i] for i in range(len(ps))] == ref and eng.metrics()["preemptions"] > 2


def test_seeded_sampling_logits_path_is_batch_independent(make):
    be = make()
    batch_only(be)
    ps = prompts(be)
    sp = SamplingParams(temperature=0.8, top_k=20, seed=11, max_tokens=10)
    alone = [gen(be, [p], params=sp, max_num_seqs=1, prefix_cache=False)[0] for p in ps]
    assert gen(be, ps, params=sp, max_num_seqs=8) == alone


def test_tree_verify_equals_chain_and_commit(make):
    be = make()
    if not be.caps["tree"]:
        pytest.skip("pack does not advertise tree verify")
    hi = max(8, min(be.vocab_size - 1, 900))
    pre, a, b, c = [3, 9, 4, 1], 7 % hi, 11 % hi, 13 % hi
    blocks = list(range(0, 8))

    def step(sid, pos, toks, par, want):
        w = Work(sid, toks, pos, blocks, want, False, False, par, False)
        return be.step([w])

    step(1, 0, pre, None, 0)
    tree = step(1, 4, [a, b, c], [-1, 0, 0], 3)              # b and c are siblings under a
    step(2, 0, pre, None, 0)
    chain = step(2, 4, [a, c], None, 2)                      # a then c
    assert (tree[2] == chain[1]).all() and (tree[0] == chain[0]).all()
    be.commit(1, 4, [0, 2])                                  # keep a, c as rows 4, 5
    after_tree = step(1, 6, [a], None, 1)
    after_chain = step(2, 6, [a], None, 1)
    assert (after_tree == after_chain).all()


def test_errors_are_codes_not_crashes(make):
    be = make()
    with pytest.raises(RuntimeError, match="gc call failed"):
        be.step([Work(1, [1, 2], 0, [10_000], 1)])
    be.release(999)   # unknown sid is not an error


@pytest.mark.skipif(bool(REAL), reason="oracle is the mock's")
def test_mock_matches_closed_form_oracle(make):
    be = make()
    ps = [[1, 2, 3, 4, 5], [9], [4, 4, 4, 4, 4, 4, 4]]
    assert gen(be, ps, 9, max_num_seqs=4) == [fake_greedy(p, 9) for p in ps]


@pytest.mark.skipif(bool(REAL), reason="mock spec pack")
def test_mock_spec_round_matches_oracle(tmp_path, key, mocklib):
    pack, w, dw = toy_pack(str(tmp_path), key, mocklib, spec=True)
    be = GcBackend(pack, w, drafter_weights=dw, sku="mock", trusted=(key.kid,))
    assert be.spec and be.spec_width == 4 and not be.paged
    eng = Engine(be, Config(max_num_seqs=1, prefix_cache=False, max_model_len=100))
    got = eng.generate([([1, 2, 3], SamplingParams(max_tokens=11))])
    assert got[0][0] == fake_greedy([1, 2, 3], 11)
    be.close()


@pytest.mark.skipif(bool(REAL), reason="mock library")
def test_abi_1_0_struct_sizes_still_load(make, tmp_path, key, mocklib):
    import ctypes as C
    from goethe_engine import gc_c
    be = make()
    assert be.L.gc_abi_version() == (1 << 16 | 3) and be.state_pages == 0

    class Open10(C.Structure):
        _fields_ = gc_c.OpenParams._fields_[:-1]

    class Layout10(C.Structure):
        _fields_ = gc_c.Layout._fields_[:17]
    pack, w, _ = toy_pack(str(tmp_path / "old"), key, mocklib)
    stage = gc_c.P.stage(pack, trusted=(key.kid,))
    ws, bk = gc_c.arr(C.c_char_p, [w.encode()]), gc_c.arr(C.c_char_p, [b"target.mock.K1.2.256"])
    p = Open10(C.sizeof(Open10), gc_c.OPEN_NO_CHECK, os.fsencode(stage.dir), ws, 1, -1, bk, 1, 0, None, None)
    h = C.c_void_p()
    assert be.L.gc_open(C.cast(C.byref(p), C.POINTER(gc_c.OpenParams)), C.byref(h)) == 0
    lo = Layout10(size=C.sizeof(Layout10))
    assert be.L.gc_layout_get(h, C.cast(C.byref(lo), C.POINTER(gc_c.Layout))) == 0 and lo.num_blocks > 0
    be.L.gc_close(h)
    stage.close()


@pytest.mark.skipif(bool(REAL), reason="mock ABI 1.2 compatibility")
def test_abi_1_2_backward_compatibility(tmp_path, key, mocklib, mocklib12):
    d12, d13 = tmp_path / "p12", tmp_path / "p13"
    d12.mkdir()
    d13.mkdir()
    p12, w12, _ = toy_pack(str(d12), key, mocklib12)
    p13, w13, _ = toy_pack(str(d13), key, mocklib)
    be12 = GcBackend(p12, w12, sku="mock", trusted=(key.kid,))
    be13 = GcBackend(p13, w13, sku="mock", trusted=(key.kid,))
    try:
        assert be12.L.gc_abi_version() == (1 << 16 | 2)
        assert not be12.device_sample and not be12.device_bitmask and not be12.async_step and not be12.flat_batch
        assert be12.graph_buckets == ()
        assert be13.device_sample and be13.device_bitmask and be13.async_step and be13.flat_batch
        assert be13.graph_buckets == (1, 2, 4, 8)
        ps = prompts(be12, 6)
        assert gen(be12, ps, 12, max_num_seqs=6) == gen(be13, ps, 12, max_num_seqs=6)
        sp = SamplingParams(temperature=0.8, top_p=0.875, top_k=15, seed=42, max_tokens=12)
        assert gen(be12, ps, params=sp, max_num_seqs=6) == gen(be13, ps, params=sp, max_num_seqs=6)
    finally:
        be12.close()
        be13.close()


@pytest.mark.skipif(bool(REAL), reason="mock ABI 1.3 device sampling parity")
def test_device_sample_matches_cpu_bit_for_bit(tmp_path, key, mocklib, mocklib12, monkeypatch):
    d12, d13 = tmp_path / "s12", tmp_path / "s13"
    d12.mkdir()
    d13.mkdir()
    p12, w12, _ = toy_pack(str(d12), key, mocklib12)
    p13, w13, _ = toy_pack(str(d13), key, mocklib)
    be12 = GcBackend(p12, w12, sku="mock", trusted=(key.kid,))
    be13 = GcBackend(p13, w13, sku="mock", trusted=(key.kid,))
    try:
        ps = prompts(be13, 6)
        for sp in (
            SamplingParams(temperature=0.5, top_p=1.0, top_k=0, seed=1, max_tokens=14),
            SamplingParams(temperature=0.75, top_p=0.875, top_k=0, seed=7, max_tokens=14),
            SamplingParams(temperature=1.25, top_p=1.0, top_k=10, seed=19, max_tokens=14),
            SamplingParams(temperature=0.9, top_p=0.5, top_k=1, seed=99, max_tokens=10),
            SamplingParams(temperature=1.0, top_p=0.95, top_k=20, seed=1234567, max_tokens=14),
        ):
            assert gen(be13, ps, params=sp, max_num_seqs=6) == gen(be12, ps, params=sp, max_num_seqs=6), sp
        # NaN fails closed (-1 -> error)
        monkeypatch.setenv("GCK_MOCK_FORCE_NAN", "1")
        eng = Engine(be13, Config(max_num_seqs=2, max_model_len=100))
        eng.add_request("nan", [1, 2, 3], SamplingParams(temperature=0.7, seed=3, max_tokens=4))
        outs = eng.step()
        assert len(outs) == 1 and outs[0].finish_reason == "error"
        eng.check()
    finally:
        be12.close()
        be13.close()


@pytest.mark.skipif(bool(REAL), reason="mock ABI 1.3 device bitmask parity")
def test_device_bitmask_grammar_matches_cpu_mask(tmp_path, key, mocklib, mocklib12):
    import numpy as np
    import torch

    class BitmaskMatcher:
        def __init__(self, allowed, vocab):
            self.allowed, self.vocab, self.step_n = allowed, vocab, 0

        def _set(self):
            return self.allowed[self.step_n % len(self.allowed)]

        def mask(self):
            m = torch.zeros(self.vocab, dtype=torch.bool)
            for t in self._set():
                m[t] = True
            return m

        def bitmask(self):
            words = np.zeros((self.vocab + 31) // 32, dtype=np.uint32)
            for t in self._set():
                words[t >> 5] |= np.uint32(1 << (t & 31))
            return words

        def accept(self, tok):
            ok = tok in self._set()
            self.step_n += 1
            return ok

        def done(self):
            return False

    class BitmaskCompiler:
        def __init__(self, vocab):
            self.vocab = vocab

        def compile(self, kind, spec):
            return type("C", (), {"new_matcher": lambda _: BitmaskMatcher(spec, self.vocab)})()

    d12, d13 = tmp_path / "b12", tmp_path / "b13"
    d12.mkdir()
    d13.mkdir()
    p12, w12, _ = toy_pack(str(d12), key, mocklib12)
    p13, w13, _ = toy_pack(str(d13), key, mocklib)
    be12 = GcBackend(p12, w12, sku="mock", trusted=(key.kid,))
    be13 = GcBackend(p13, w13, sku="mock", trusted=(key.kid,))
    try:
        spec = ((2, 5, 11, 31, 32, 52), (0, 1, 7, 33, 51), (10, 20, 30, 40, 50))
        ps = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        for sp in (
            SamplingParams(temperature=0.0, max_tokens=10, grammar=("custom", spec)),
            SamplingParams(temperature=0.8, top_p=0.9, top_k=4, seed=13, max_tokens=10, grammar=("custom", spec)),
        ):
            e12 = Engine(be12, Config(max_num_seqs=4, max_model_len=100), grammar=BitmaskCompiler(be12.vocab_size))
            e13 = Engine(be13, Config(max_num_seqs=4, max_model_len=100), grammar=BitmaskCompiler(be13.vocab_size))
            assert e13.generate([(p, sp) for p in ps]) == e12.generate([(p, sp) for p in ps])
            e12.check()
            e13.check()
        # Empty allowed set fails closed (-1 -> error)
        e13 = Engine(be13, Config(max_num_seqs=2, max_model_len=100), grammar=BitmaskCompiler(be13.vocab_size))
        e13.add_request("empty", [1, 2], SamplingParams(temperature=0.0, max_tokens=3, grammar=("custom", ((),))))
        outs = e13.step()
        assert len(outs) == 1 and outs[0].finish_reason == "error"
        e13.check()
    finally:
        be12.close()
        be13.close()


@pytest.mark.skipif(bool(REAL), reason="mock ABI 1.3 async overlap")
def test_async_overlap_matches_sync_and_catches_broken_prev_token(make, monkeypatch):
    be = make()
    assert be.async_step
    ps = prompts(be, 6)
    # 1. Greedy & sampled parity: async vs sync
    for sp in (SamplingParams(max_tokens=14), SamplingParams(temperature=0.7, top_p=0.9, seed=17, max_tokens=14)):
        be.async_step = False
        sync_out = gen(be, ps, params=sp, max_num_seqs=6)
        be.async_step = True
        async_out = gen(be, ps, params=sp, max_num_seqs=6)
        assert async_out == sync_out
    # 2. Early EOS mid-batch (one sequence stops early while others continue)
    first_toks = gen(be, ps[:3], 10, max_num_seqs=3)
    eos = (first_toks[0][2],)
    be.async_step = False
    e_sync = Engine(be, Config(max_num_seqs=3, max_model_len=100), eos_ids=eos)
    ref_eos = e_sync.generate([(p, SamplingParams(max_tokens=10)) for p in ps[:3]])
    be.async_step = True
    e_async = Engine(be, Config(max_num_seqs=3, max_model_len=100), eos_ids=eos)
    got_eos = e_async.generate([(p, SamplingParams(max_tokens=10)) for p in ps[:3]])
    e_async.check()
    assert got_eos == ref_eos
    # 3. Mid-flight abort: 0 extra tokens emitted after abort, clean KV check
    eng = Engine(be, Config(max_num_seqs=2, max_model_len=100, prefix_cache=False))
    eng.add_request("a", ps[0], SamplingParams(max_tokens=12))
    eng.add_request("b", ps[1], SamplingParams(max_tokens=12))
    o0 = eng.step()
    assert eng._inflight is not None and len(o0) == 2
    eng.abort("a")
    o1 = eng.step()
    by_rid = {o.rid: o for o in o1}
    assert by_rid["a"].finish_reason == "abort" and by_rid["a"].token_ids == []
    while eng.has_work():
        eng.step()
        eng.check()
    # 4. Break GC_W_PREV_TOKEN and confirm the test catches it
    be.async_step = False
    ref_solo = gen(be, [ps[0]], 10, max_num_seqs=1)
    be.async_step = True
    monkeypatch.setenv("GCK_MOCK_BREAK_PREV_TOKEN", "1")
    broken = gen(be, [ps[0]], 10, max_num_seqs=1)
    assert broken != ref_solo


@pytest.mark.skipif(bool(REAL), reason="mock ABI 1.3 flat CSR and graph buckets")
def test_flat_csr_and_graph_buckets_batch_invariance(make):
    be = make()
    assert be.flat_batch and be.graph_buckets == (1, 2, 4, 8)
    ps = prompts(be, 5)
    be.flat_batch = False
    ref = gen(be, ps, 12, max_num_seqs=5, max_num_batched_tokens=11)
    be.flat_batch = True
    got = gen(be, ps, 12, max_num_seqs=5, max_num_batched_tokens=11)
    assert got == ref
    # Tree verify under flat CSR vs gc_work**
    blocks = list(range(8))
    be.step([Work(10, [3, 9, 4, 1], 0, blocks, 0)])
    t_flat = be.step([Work(10, [7, 11, 13], 4, blocks, 3, parents=[-1, 0, 0])]).copy()
    be.release(10)
    be.flat_batch = False
    be.step([Work(11, [3, 9, 4, 1], 0, blocks, 0)])
    t_work = be.step([Work(11, [7, 11, 13], 4, blocks, 3, parents=[-1, 0, 0])]).copy()
    be.release(11)
    be.flat_batch = True
    assert (t_flat == t_work).all()
    # Warm graph buckets via Engine.warm() and check pad_cols invariance + bounds error
    eng = Engine(be, Config(max_num_seqs=4, max_model_len=100))
    eng.warm()
    eng.check()
    r1 = be.step([Work(20, [1, 2, 3], 0, [0], 1, greedy=True)], pad_cols=1)
    be.release(20)
    r8 = be.step([Work(21, [1, 2, 3], 0, [0], 1, greedy=True)], pad_cols=8)
    be.release(21)
    assert r1 == r8
    with pytest.raises(RuntimeError, match="gc call failed"):
        be.step([Work(22, [1], 0, [0], 1, greedy=True)], pad_cols=99999)


def test_unsatisfiable_numerics_request_names_the_missing_capability(tmp_path, key, mocklib):
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib)
    with pytest.raises(Exception, match="kv=fp8.*the pack has"):
        GcBackend(pack, w, sku="mock", trusted=(key.kid,), kv="fp8")
    with pytest.raises(Exception, match="numerics=exact"):
        GcBackend(pack, w, sku="mock", trusted=(key.kid,), numerics="exact", B=8)


@pytest.mark.skipif(bool(REAL), reason="the mock's flag")
def test_bounded_state_pack_shares_no_prefix_by_block(make, monkeypatch):
    be = make()
    assert not be.bounded and be.prefix_cache
    monkeypatch.setenv("GCK_MOCK_BOUNDED", "1")
    b = make()
    assert b.paged and b.bounded and b.state_pages == 0 and not b.prefix_cache   # the mock opens no snapshot slots
    assert b.L.gc_state_save(b.h, 1, 0) == 0 and b.L.gc_state_restore(b.h, 0, 1) == 0
    monkeypatch.delenv("GCK_MOCK_BOUNDED")
    assert make().L.gc_state_save(b.h, 1, 0) != 0   # without the flag a library refuses


@pytest.mark.skipif(bool(REAL), reason="mock ABI 1.3 robustness: NaN fail-closed, state_restore+spec, preempt drain")
def test_robustness_nan_state_restore_and_preempt_async_drain(make, monkeypatch):
    be = make()
    # 1. NaN verify fails closed (error finish, clean KV accounting)
    monkeypatch.setenv("GCK_MOCK_FORCE_NAN", "1")
    eng = Engine(be, Config(max_num_seqs=2, max_model_len=100))
    eng.add_request("nan_greedy", [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=4))
    outs = eng.step()
    assert len(outs) == 1 and outs[0].finish_reason == "error"
    eng.check()
    monkeypatch.delenv("GCK_MOCK_FORCE_NAN")

    # 2. Preempt/release while async step is in flight drains cleanly without leaking KV blocks or slots
    eng_p = Engine(be, Config(max_num_seqs=2, max_model_len=100, prefix_cache=False))
    eng_p.add_request("r1", [1, 2, 3], SamplingParams(max_tokens=8))
    eng_p.add_request("r2", [4, 5, 6], SamplingParams(max_tokens=8))
    eng_p.step()
    assert eng_p._inflight is not None
    # Simulate external drop/preempt of r2 while r1+r2 are in flight
    s2 = eng_p.live["r2"][0]
    eng_p.sched.running.remove(s2)
    eng_p.kv.release(s2)
    eng_p.backend.release(s2.sid)
    eng_p.live.pop("r2", None)
    while eng_p.has_work():
        eng_p.step()
        eng_p.check()

    # 3. state_save / state_restore followed by single-work fast path step succeeds cleanly
    if be.state_pages > 0:
        be.step([Work(501, [1, 2, 3], 0, [0], 1, greedy=True)])
        be.save_state(501, 0)
        be.restore_state(0, 502)
        out_r = be.step([Work(502, [4], 3, [0], 1, greedy=True)])
        assert len(out_r) == 1 and len(out_r[0]) == 1 and out_r[0][0] >= 0
        be.release(501)
        be.release(502)


@pytest.mark.skipif(bool(REAL), reason="mock unified multi-variant + multi-numerics branching")
def test_unified_pack_numerics_branching_ar_and_spec(tmp_path, key, mocklib):
    import copy
    import os
    from goethe_engine import pack as P

    def add_tiers(m):
        m["contracts"] = ["K1", "K1_fast"]
        base_v = m["buckets"][0]
        draft_b = m["buckets"][1]
        base_v["act"] = "a16"
        base_v["step_us"] = 5240.0
        # AR buckets for K1 (batch_invariant, a16) and K1_fast (tolerance, a8)
        b_k1_1 = copy.deepcopy(base_v)
        b_k1_1.update(id="target.mock.K1.1.256", ncol=1, batch=False, io=None, step_us=5240.0)
        b_k1_2 = copy.deepcopy(base_v)
        b_k1_2.update(id="target.mock.K1.2.256", ncol=2, batch=True, io=None, step_us=6280.0)
        b_fast_1 = copy.deepcopy(b_k1_1)
        b_fast_1.update(id="target.mock.K1_fast.1.256", contract="K1_fast", determinism="tolerance", act="a8", step_us=4080.0)
        b_fast_2 = copy.deepcopy(b_k1_2)
        b_fast_2.update(id="target.mock.K1_fast.2.256", contract="K1_fast", determinism="tolerance", act="a8", step_us=5000.0)
        b_fast_v = copy.deepcopy(base_v)
        b_fast_v.update(id="target.mock.K1_fast.4v.256", contract="K1_fast", determinism="tolerance", act="a8", step_us=7870.0)
        m["buckets"].extend([b_k1_1, b_k1_2, b_fast_1, b_fast_2, b_fast_v])
        m["spec"].append(dict(name="fast_ar", target=b_fast_v["id"], drafter=draft_b["id"], policy=dict(kind="ar", K="3")))

    pack, w, dw = toy_pack(str(tmp_path), key, mocklib, spec=True, alt=True, mutate=add_tiers)
    aw = os.path.join(str(tmp_path), "alt.gguf")
    for num, want_c in (("invariant", "K1"), ("tolerance", "K1_fast")):
        be_ar = GcBackend(pack, w, B=1, sku="mock", numerics=num, trusted=(key.kid,))
        be_sp = GcBackend(pack, w, B=1, sku="mock", numerics=num, drafter_weights=dw, trusted=(key.kid,))
        try:
            assert be_ar.bucket["contract"] == want_c, f"AR {num}: {be_ar.bucket['contract']} != {want_c}"
            assert be_sp.spec and be_sp.bucket["contract"] == want_c, f"Spec {num}: {be_sp.bucket['contract']} != {want_c}"
        finally:
            be_ar.close()
            be_sp.close()
    be_alt = GcBackend(pack, aw, B=2, sku="mock", trusted=(key.kid,))
    try:
        assert be_alt.bucket["variant"] == "alt"
    finally:
        be_alt.close()
    # Direct spec_pairs unit check: invariant filters out tolerance; tolerance ranks tolerance first
    st = P.stage(pack, trusted=(key.kid,))
    try:
        m_v = P.for_variant(st.manifest, "q4", "d")
        inv_pairs = P.spec_pairs(m_v, sku="mock", want=P.wanted("invariant", single=True))
        tol_pairs = P.spec_pairs(m_v, sku="mock", want=P.wanted("tolerance", single=True))
        assert [t["contract"] for t, *_ in inv_pairs] == ["K1"]
        assert [t["contract"] for t, *_ in tol_pairs] == ["K1_fast", "K1"]
    finally:
        st.close()
