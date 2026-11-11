import pytest

from conftest import FakeBackend, alone, make, run, toks
from goethe_engine import Config, Engine, SamplingParams as P


def prompt(n, base=1):
    return [(base + i) % 50 + 1 for i in range(n)]


def test_token_budget_and_chunked_prefill_mixed_with_decode():
    eng, be = make(blocks=64, bs=4, max_num_batched_tokens=8)
    eng.add_request("a", prompt(5), P(max_tokens=6))
    outs = eng.step()
    eng.add_request("b", prompt(20, 7), P(max_tokens=3))
    outs += run(eng)
    assert all(sum(len(w.tokens) for w in st) <= 8 for st in be.steps)
    assert any(len({w.sid for w in st}) == 2 and any(len(w.tokens) > 1 for w in st) for st in be.steps)
    assert toks(outs)[("b", 0)] == alone(prompt(20, 7), P(max_tokens=3))[("r", 0)]
    assert toks(outs)[("a", 0)] == alone(prompt(5), P(max_tokens=6))[("r", 0)]


def test_prompt_chunk_wants_logits_only_at_the_end():
    eng, be = make(max_num_batched_tokens=4)
    eng.add_request("a", prompt(10), P(max_tokens=2))
    run(eng)
    ws = [w for st in be.steps for w in st]
    assert [w.want for w in ws] == [0, 0, 1, 1] and [len(w.tokens) for w in ws] == [4, 4, 2, 1]


def test_prefix_cache_hit_at_block_boundary_and_stats():
    eng, _ = make(blocks=64, bs=4)
    eng.add_request("a", prompt(9), P(max_tokens=2))
    run(eng)
    eng.add_request("b", prompt(8) + [52], P(max_tokens=2))
    o = {x.rid: x for x in run(eng) if x.finished}
    assert o["b"].stats["cached_tokens"] == 8 and o["b"].stats["computed_prefill_tokens"] == 1
    eng.add_request("c", prompt(8), P(max_tokens=2))  # exactly 2 blocks: the last one is recomputed
    o = {x.rid: x for x in run(eng) if x.finished}
    assert o["c"].stats["cached_tokens"] == 4


def test_stop_rules_finish_inside_the_step():
    t = alone(prompt(6), P(max_tokens=8))[("r", 0)]
    eng, _ = make(blocks=64, bs=4)
    eng.eos = {t[2]}
    eng.add_request("a", prompt(6), P(max_tokens=8))
    eng.add_request("b", prompt(6), P(max_tokens=8, stop_sequences=((t[1], t[2]),)))
    eng.add_request("c", prompt(6), P(max_tokens=8, stop_token_ids=(t[0],)))
    eng.add_request("d", prompt(6), P(max_tokens=8, ignore_eos=True))
    out = {x.rid: x for x in run(eng) if x.finished}
    assert out["a"].finish_reason == "stop" and out["a"].stats["completion_tokens"] == 3
    assert out["b"].stats["completion_tokens"] == 3 and out["c"].stats["completion_tokens"] == 1
    assert out["d"].finish_reason == "length" and out["d"].stats["completion_tokens"] == 8


def test_preemption_resume_gives_identical_tokens_and_no_leak():
    reqs = [(prompt(11, i), P(max_tokens=10)) for i in range(6)]
    eng, be = make(blocks=12, bs=2, max_num_batched_tokens=16)
    for i, (p, sp) in enumerate(reqs):
        eng.add_request(i, p, sp)
    outs = run(eng)
    assert eng.metrics()["preemptions"] > 0
    got = toks(outs)
    for i, (p, sp) in enumerate(reqs):
        assert got[(i, 0)] == alone(p, sp)[("r", 0)], i
    assert len(eng.kv.pool.free) + eng.metrics()["kv_tree_blocks"] == 12
    assert any(o.stats["preemptions"] for o in outs if o.finished)
    assert len(be.released) == 6 + eng.metrics()["preemptions"]


def test_preemption_victim_is_lowest_priority_youngest():
    eng, _ = make(blocks=10, bs=2, max_num_batched_tokens=64)
    eng.add_request("old", prompt(8), P(max_tokens=12))
    eng.add_request("young", prompt(8, 9), P(max_tokens=12))
    eng.add_request("vip", prompt(8, 20), P(max_tokens=12, priority=-1))
    first = None
    for _ in range(200):
        if not eng.has_work():
            break
        for o in eng.step():
            if o.finished and first is None:
                first = o.rid
        eng.check()
    assert first == "vip"


def test_abort_mid_prefill_waiting_and_unknown_frees_everything():
    eng, be = make(blocks=32, bs=4, max_num_batched_tokens=4)
    eng.add_request("a", prompt(30), P(max_tokens=4))
    eng.add_request("w", prompt(30, 3), P(max_tokens=4))
    eng.step()
    assert eng.sched.running
    eng.abort("a")
    eng.abort("w")
    eng.abort("nope")
    outs = eng.step()
    assert {o.rid for o in outs} == {"a", "w"} and all(o.finish_reason == "abort" for o in outs)
    assert not eng.has_work() and len(be.released) >= 1
    eng.check()
    eng.abort("a")
    eng.step()
    assert len(eng.kv.pool.free) + eng.metrics()["kv_tree_blocks"] == 32


def test_lpm_prefers_cached_prompts_and_defers_in_batch_twins():
    shared = prompt(40)
    eng, be = make(blocks=64, bs=4, max_num_seqs=1, max_num_batched_tokens=8)
    eng.add_request("seed", shared + [51], P(max_tokens=2))
    run(eng)
    eng.add_request("cold", prompt(40, 30), P(max_tokens=2))
    eng.add_request("warm", shared + [52], P(max_tokens=2))
    eng.step()
    assert [w.sid for w in be.steps[-1]] == [3]  # warm arrived second but matches 40 cached tokens
    eng2, be2 = make(blocks=64, bs=4, max_num_seqs=4, max_num_batched_tokens=256)
    eng2.add_request("t1", shared + [1], P(max_tokens=2))
    eng2.add_request("t2", shared + [2], P(max_tokens=2))
    eng2.step()
    assert {w.sid for w in be2.steps[-1]} == {1}  # the twin waits one step and then hits the cache
    outs = run(eng2)
    assert [o.stats["cached_tokens"] for o in outs if o.finished and o.rid == "t2"] == [40]


def test_validation_and_backend_failure():
    eng, be = make(blocks=8, bs=4)
    for bad in ([], [99], list(range(1, 10))):
        with pytest.raises(ValueError):
            eng.add_request("x", bad, P(max_tokens=100))
    eng.add_request("a", prompt(4), P(max_tokens=2))
    with pytest.raises(ValueError):
        eng.add_request("a", prompt(4))
    be.fail = True
    outs = eng.step()
    assert outs[0].finish_reason == "error" and "boom" in outs[0].error
    be.fail = False
    eng.add_request("b", prompt(4), P(max_tokens=2))
    assert toks(run(eng))[("b", 0)] == alone(prompt(4), P(max_tokens=2))[("r", 0)]
    eng.check()


def test_fp8_is_opt_in():
    with pytest.raises(ValueError):
        Engine(FakeBackend(), Config(kv_dtype="fp8"))


def test_parallel_sampling_forks_with_cow():
    eng, be = make(blocks=64, bs=4, max_num_seqs=8)
    p = P(temperature=1.0, seed=3, max_tokens=6, n=3)
    eng.add_request("a", prompt(6), p)
    got = toks(run(eng))
    assert len({tuple(v) for v in got.values()}) > 1 and set(got) == {("a", 0), ("a", 1), ("a", 2)}
    ref = alone(prompt(6), p)
    assert got == {("a", i): ref[("r", i)] for i in range(3)}
    assert len(eng.kv.pool.free) + eng.metrics()["kv_tree_blocks"] == 64
