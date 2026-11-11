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


def test_abi_and_layout(make):
    be = make()
    assert be.L.gc_abi_version() >> 16 == 1
    lo = be.layout
    assert lo.vocab > 1 and lo.max_seqs >= 1 and lo.num_blocks >= 1 and lo.block_size >= 1 and lo.max_ctx >= lo.block_size
    assert be.geometry.dtype in ("f16", "fp8")


def test_batch_invariance_and_chunking(make):
    be = make()
    ps = prompts(be)
    alone = [gen(be, [p], 12, max_num_seqs=1, prefix_cache=False)[0] for p in ps]
    for cfg in (dict(max_num_seqs=8), dict(max_num_seqs=8, max_num_batched_tokens=7, prefix_cache=False),
                dict(max_num_seqs=3, max_num_batched_tokens=64)):
        assert gen(be, ps, **cfg) == alone, cfg


def test_prefix_cache_and_copy_blocks(make):
    be = make()
    base = prompts(be, 1)[0] * 3
    ps = [base[:30] + [5, 6], base[:30] + [7], base[:30] + [5, 6], base[:19]]
    assert gen(be, ps, max_num_seqs=4) == gen(be, ps, max_num_seqs=4, prefix_cache=False)


def test_small_pool_preempts_and_matches(make):
    big = make()
    ps = prompts(big, 6)
    need = [-(-(len(p) + 20) // big.layout.block_size) for p in ps]
    assert sum(need) > max(need) + 1
    be = make(pool_blocks=max(need) + 1)  # the longest request fits, all six together do not
    cfg = dict(max_model_len=be.max_len, prefix_cache=False)
    ref = [o[0] for o in Engine(big, Config(max_num_seqs=1, **cfg)).generate([(p, SamplingParams(max_tokens=20)) for p in ps])]
    eng = Engine(be, Config(max_num_seqs=6, **cfg))
    got = eng.generate([(p, SamplingParams(max_tokens=20)) for p in ps])
    assert [o[0] for o in got] == ref and eng.metrics()["preemptions"] > 0


def test_seeded_sampling_logits_path_is_batch_independent(make):
    be = make()
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
