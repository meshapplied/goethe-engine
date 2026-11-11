import pytest
import torch

from goethe_engine import Config, Engine, SamplingParams as P
from goethe_engine.ref import ModelDrafter, RefModel, TorchBackend
from goethe_engine.spec import NgramDrafter
from conftest import run, toks

DEV = [pytest.param("cpu"), pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no cuda")])]


def prompts():
    base = [3, 9, 27, 14, 1, 22, 5, 40, 33, 8, 19, 2]
    return [base[:n] + [t] for n, t in ((12, 7), (12, 8), (9, 7), (5, 1), (12, 7))]


@pytest.mark.parametrize("dev", DEV)
@pytest.mark.parametrize("kv", ["f16", "fp8"])
def test_greedy_engine_equals_dense_reference(dev, kv):
    m = RefModel(device=dev)
    be = TorchBackend(m, num_blocks=20, block_size=2, kv=kv)
    eng = Engine(be, Config(max_num_seqs=4, max_num_batched_tokens=8, kv_dtype=kv))
    for i, p in enumerate(prompts()):
        eng.add_request(i, p, P(max_tokens=9))
    got = toks(run(eng))
    for i, p in enumerate(prompts()):
        assert got[(i, 0)] == m.dense(p, 9, kv), i
    assert eng.metrics()["prefix_hit_tokens"] > 0


@pytest.mark.parametrize("dev", DEV)
def test_spec_decode_with_model_and_ngram_drafters_is_token_exact(dev):
    m, small = RefModel(device=dev), RefModel(d=16, layers=1, heads=2, kv_heads=1, ffn=16, seed=0, device=dev)
    ps = [[3, 9, 27, 14, 3, 9, 27, 14, 3, 9], [5, 1, 5, 1, 5, 1], [30, 31, 2]]
    for drafter in (ModelDrafter(m), ModelDrafter(small), NgramDrafter()):
        eng = Engine(TorchBackend(m, 30, 2), Config(max_num_seqs=4, max_num_batched_tokens=12, spec_k=4), drafter=drafter)
        for i, p in enumerate(ps):
            eng.add_request(i, p, P(max_tokens=12))
        got = toks(run(eng))
        for i, p in enumerate(ps):
            assert got[(i, 0)] == m.dense(p, 12), (type(drafter).__name__, i)
        assert eng.metrics()["spec_drafted"] > 0
    assert eng.metrics()["spec_accepted"] >= 0
    exact = Engine(TorchBackend(m, 30, 2), Config(max_num_seqs=4, spec_k=4), drafter=ModelDrafter(m))
    exact.add_request(0, ps[0], P(max_tokens=12))
    run(exact)
    mm = exact.metrics()
    assert mm["spec_accepted"] == mm["spec_drafted"] > 0 and mm["steps"] < 12  # a perfect drafter cuts the steps


@pytest.mark.parametrize("dev", DEV)
def test_sampled_tokens_independent_of_batch_chunking_and_preemption(dev):
    m = RefModel(device=dev)
    ps = prompts()
    sp = lambda i: P(temperature=0.9, top_p=0.9, top_k=20, seed=100 + i, max_tokens=8)
    eng = Engine(TorchBackend(m, 16, 2), Config(max_num_seqs=5, max_num_batched_tokens=6))
    for i, p in enumerate(ps):
        eng.add_request(i, p, sp(i))
    got = toks(run(eng))
    assert eng.metrics()["preemptions"] > 0
    for i, p in enumerate(ps):
        solo = Engine(TorchBackend(m, 40, 4), Config(max_num_seqs=1, prefix_cache=False))
        solo.add_request("s", p, sp(i))
        assert toks(run(solo))[("s", 0)] == got[(i, 0)], i


def test_fp8_geometry_is_explicit():
    m = RefModel()
    with pytest.raises(ValueError):
        Engine(TorchBackend(m, 8, 2, kv="fp8"), Config())
    assert TorchBackend(m, 8, 2, kv="fp8").geometry.bytes_per_block * 2 == TorchBackend(m, 8, 2).geometry.bytes_per_block
