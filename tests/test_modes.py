import pytest

from conftest import FakeBackend, FakeSpec, FakeTokens, alone, fake_greedy, run, toks
from goethe_engine import Config, Engine, SamplingParams as P


def prompt(n, base=1):
    return [(base + i) % 50 + 1 for i in range(n)]


def test_device_argmax_tokens_equal_logits_path_and_sampled_rows_still_use_logits():
    reqs = [(prompt(9, i), P(max_tokens=7)) for i in range(5)] + [(prompt(6), P(temperature=1.0, seed=3, max_tokens=6))]
    a = Engine(FakeTokens(14, 2), Config(max_num_seqs=4, max_num_batched_tokens=10))
    b = Engine(FakeBackend(14, 2), Config(max_num_seqs=4, max_num_batched_tokens=10))
    for e in (a, b):
        for i, (p, sp) in enumerate(reqs):
            e.add_request(i, p, sp)
    assert toks(run(a)) == toks(run(b))
    assert a.metrics()["preemptions"] > 0


def test_verify_rows_in_token_mode_accept_like_logits_mode():
    from goethe_engine.spec import NgramDrafter
    p = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
    a = Engine(FakeTokens(40, 2), Config(spec_k=3), drafter=NgramDrafter())
    a.add_request("a", p, P(max_tokens=10))
    assert toks(run(a))[("a", 0)] == alone(p, P(max_tokens=10))[("r", 0)]


@pytest.mark.parametrize("width", [2, 4, 8])
def test_spec_round_backend_tokens_equal_greedy_and_truncate_at_stops(width):
    be = FakeSpec(width)
    e = Engine(be, Config(max_num_seqs=1, prefix_cache=False, max_num_batched_tokens=7, max_model_len=150))
    ps = [prompt(5), prompt(20, 9), prompt(3, 30)]
    for i, p in enumerate(ps):
        e.add_request(i, p, P(max_tokens=23))
    got = toks(run(e))
    for i, p in enumerate(ps):
        assert got[(i, 0)] == fake_greedy(p, 23)
    m = e.metrics()
    assert m["spec_drafted"] > 0 and 0 < m["spec_accepted"] <= m["spec_drafted"] and m["steps"] < 3 * 23
    g = fake_greedy(ps[0], 8)
    e.eos = {g[2]}
    e.add_request("s", ps[0], P(max_tokens=23))
    out = [o for o in run(e) if o.finished][0]
    assert out.stats["completion_tokens"] == 3 and out.finish_reason == "stop"
    e.eos = set()
    e.add_request("l", ps[0], P(max_tokens=5))
    assert len(toks(run(e))[("l", 0)]) == 5


def test_linear_stream_backend_config_is_checked():
    for cfg in (Config(max_num_seqs=2, prefix_cache=False), Config(max_num_seqs=1, prefix_cache=True)):
        with pytest.raises(ValueError):
            Engine(FakeSpec(), cfg)
    e = Engine(FakeSpec(), Config(max_model_len=150))   # unset knobs follow the backend: one stream, no prefix cache
    assert (e.cfg.max_num_seqs, e.cfg.prefix_cache) == (1, False)
    with pytest.raises(ValueError):
        Engine(FakeSpec(), Config(max_num_seqs=1, prefix_cache=False, max_model_len=500))


def test_bad_token_reply_fails_the_step_not_the_engine():
    class Bad(FakeTokens):
        def step(self, work):
            return [[1, 2, 3] for _ in work]

    e = Engine(Bad(16, 4), Config())
    e.add_request("a", prompt(4), P(max_tokens=3))
    assert e.step()[0].finish_reason == "error"
