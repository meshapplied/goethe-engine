"""A page pack (window ring, recurrent state: no paged KV) serves many sequences: pages are allocated like KV blocks, a
preempted sequence recomputes, the prefix cache is whole-page snapshots keyed by the exact tokens they followed."""
import pytest

from conftest import FakePages, fake_greedy, run, toks
from goethe_engine import Config, Engine, SamplingParams as P


def prompt(n, base=1):
    return [(base + 3 * i) % 50 + 1 for i in range(n)]


def go(be, reqs, **cfg):
    e = Engine(be, Config(max_model_len=60, **cfg))
    for i, (p, sp) in enumerate(reqs):
        e.add_request(i, p, sp)
    out = run(e)
    return e, toks(out), {o.rid: o.stats for o in out if o.finished}


def test_many_sequences_equal_alone_and_chunk_edges_do_not_matter():
    ps = [prompt(n, b) for n, b in ((1, 1), (5, 2), (17, 3), (30, 4), (9, 5), (2, 6))]
    reqs = [(p, P(max_tokens=11)) for p in ps]
    for cfg in (dict(max_num_seqs=4), dict(max_num_seqs=4, max_num_batched_tokens=7), dict(max_num_seqs=2, prefix_cache=False)):
        _, got, _ = go(FakePages(), reqs, **cfg)
        for i, p in enumerate(ps):
            assert got[(i, 0)] == fake_greedy(p, 11), cfg


def test_short_pool_queues_and_a_forced_preemption_recomputes():
    ps = [prompt(n, b) for n, b in ((12, 1), (20, 2), (7, 3), (15, 4), (25, 5))]
    reqs = [(p, P(max_tokens=14)) for p in ps]
    # a page holds a sequence's whole reservation: a pool shorter than the batch queues the rest, it never preempts
    e, got, _ = go(FakePages(seqs=5, pages=2), reqs, max_num_seqs=5, prefix_cache=False)
    assert e.metrics()["preemptions"] == 0 and all(got[(i, 0)] == fake_greedy(p, 14) for i, p in enumerate(ps))
    for cache in (False, True):
        be = FakePages(seqs=5, pages=3)
        e = Engine(be, Config(max_model_len=60, max_num_seqs=5, prefix_cache=cache))
        for i, (p, sp) in enumerate(reqs):
            e.add_request(i, p, sp)
        out = []
        for k in range(1000):
            if not e.has_work():
                break
            out += e.step()
            e.check()
            if k % 4 == 3 and e.sched.running:
                v = e.sched.running[-1]
                e.sched._preempt(v)
                be.release(v.sid)
        got = toks(out)
        assert e.metrics()["preemptions"] > 2 and all(got[(i, 0)] == fake_greedy(p, 14) for i, p in enumerate(ps))


def test_snapshot_hit_is_exact_and_saves_the_prefix():
    base = prompt(26)
    e = Engine(FakePages(seqs=2, spare=3), Config(max_model_len=60, max_num_seqs=2))
    e.add_request("cold", base, P(max_tokens=6))
    run(e)
    for rid, p, want in (("same", base, 25), ("ext", base + [9, 8, 7], 25), ("fork", base[:10] + [1, 1, 1], 0)):
        e.add_request(rid, p, P(max_tokens=6))
        out = run(e)
        assert [o for o in out if o.finished][0].stats["cached_tokens"] == want, rid
        assert toks(out)[(rid, 0)] == fake_greedy(p, 6)


def test_snapshot_runs_equal_cold_runs_token_for_token():
    base = prompt(26)
    reqs = [(base + [i + 1] * (i % 3), P(max_tokens=9)) for i in range(7)] + [(base, P(max_tokens=9))] * 2
    _, warm, st = go(FakePages(seqs=3, spare=2), reqs, max_num_seqs=3, max_num_batched_tokens=9)
    assert sum(s["cached_tokens"] for s in st.values()) > 0
    for i, (p, _) in enumerate(reqs):
        assert warm[(i, 0)] == fake_greedy(p, 9)


def test_next_turn_resumes_from_the_end_of_the_last_turn():
    be = FakePages(seqs=2, spare=2, max_len=120)
    e = Engine(be, Config(max_model_len=100, max_num_seqs=2))
    p1 = prompt(20)
    e.add_request("a", p1, P(max_tokens=8))
    a = toks(run(e))[("a", 0)]
    p2 = p1 + a + [4, 5]
    e.add_request("b", p2, P(max_tokens=8))
    out = run(e)
    st = [o for o in out if o.finished][0].stats
    assert st["cached_tokens"] == len(p1) + len(a) - 1 and toks(out)[("b", 0)] == fake_greedy(p2, 8)


def test_forks_get_their_own_page_on_first_write():
    _, got, _ = go(FakePages(seqs=4), [(prompt(9), P(n=3, temperature=0.9, seed=4, max_tokens=7))], max_num_seqs=4)
    assert len({tuple(v) for v in got.values()}) > 1 and len(got) == 3


def test_snapshot_pages_are_evicted_least_recently_used_and_never_leak():
    be = FakePages(seqs=2, spare=1)
    reqs = [(prompt(14, b), P(max_tokens=5)) for b in range(1, 9)]
    e, got, _ = go(be, reqs, max_num_seqs=2)
    assert len(e.kv.snaps.keys) == 1 and all(got[(i, 0)] == fake_greedy(p, 5) for i, (p, _) in enumerate(reqs))


def test_no_snapshot_pages_means_no_prefix_cache_not_one_stream():
    with pytest.raises(ValueError, match="no prefix cache"):
        Engine(FakePages(spare=0), Config(max_model_len=60, max_num_seqs=4))
    reqs = [(prompt(8, b), P(max_tokens=5)) for b in range(1, 6)]
    _, got, _ = go(FakePages(spare=0), reqs, max_num_seqs=4, prefix_cache=False)
    assert all(got[(i, 0)] == fake_greedy(p, 5) for i, (p, _) in enumerate(reqs))
