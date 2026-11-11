import random

import pytest

from conftest import FakeBackend, V, alone, fake_greedy, toks
from goethe_engine import Config, Engine, SamplingParams as P
from goethe_engine.spec import NgramDrafter


class NoisyOracle:
    """Drafts the true greedy continuation, corrupted at random positions: exercises partial and full acceptance."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def propose(self, seqs, k):
        out = []
        for s in seqs:
            d = fake_greedy(s.tokens, k)
            out.append([x if self.rng.random() > 0.2 else self.rng.randrange(V) for x in d][:self.rng.randint(0, k)])
        return out


def make_trace(rng, n=36):
    tmpl = [[rng.randrange(1, V) for _ in range(rng.randint(10, 30))] for _ in range(4)]
    reqs = []
    for i in range(n):
        t = rng.choice(tmpl)
        prompt = t[:rng.randint(3, len(t))] + [rng.randrange(1, V) for _ in range(rng.randint(0, 6))]
        if rng.random() < 0.5:
            p = P(max_tokens=rng.randint(1, 14), stop_token_ids=(rng.randrange(V),) if rng.random() < .3 else ())
        else:
            p = P(temperature=rng.uniform(0.3, 1.5), top_p=rng.choice([1, .9, .5]), top_k=rng.choice([0, 3, 10]),
                  seed=rng.randrange(1 << 30), max_tokens=rng.randint(1, 14), n=rng.choice([1, 1, 1, 2, 3]))
        reqs.append((i, prompt, p, rng.randint(0, 50), rng.random() < 0.15))
    return reqs


CFGS = [
    dict(bs=2, blocks=40, policy="lpm", prefix_cache=True, budget=12, spec=0),
    dict(bs=4, blocks=24, policy="fcfs", prefix_cache=True, budget=7, spec=0),
    dict(bs=1, blocks=90, policy="lpm", prefix_cache=True, budget=32, spec=0),
    dict(bs=2, blocks=40, policy="lpm", prefix_cache=False, budget=9, spec=0),
    dict(bs=2, blocks=44, policy="lpm", prefix_cache=True, budget=16, spec=3),
    dict(bs=4, blocks=24, policy="fcfs", prefix_cache=True, budget=11, spec=4),
]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("c", range(len(CFGS)))
def test_random_trace_equals_unbatched_reference_and_leaks_nothing(seed, c):
    cf = CFGS[c]
    rng = random.Random(seed * 100 + c)
    reqs = make_trace(rng)
    drafter = (NoisyOracle(seed) if seed % 2 else NgramDrafter()) if cf["spec"] else None
    eng = Engine(FakeBackend(cf["blocks"], cf["bs"]),
                 Config(max_num_seqs=8, max_num_batched_tokens=cf["budget"], policy=cf["policy"],
                        prefix_cache=cf["prefix_cache"], spec_k=cf["spec"]), drafter=drafter)
    outs, t = [], 0
    while t <= 50 or eng.has_work():
        for rid, prompt, p, at, ab in reqs:
            if at == t:
                eng.add_request(rid, prompt, p)
            if ab and at + 3 == t:
                eng.abort(rid)
        outs += eng.step()
        eng.check()
        t += 1
        assert t < 20000
    aborted = {r[0] for r in reqs if r[4]}
    got = toks(outs)
    assert not [o for o in outs if o.finish_reason == "error"]
    for rid, prompt, p, _, _ in reqs:
        if rid in aborted:
            continue
        ref = alone(prompt, p, bs=cf["bs"])
        assert {k[1]: v for k, v in got.items() if k[0] == rid} == {k[1]: v for k, v in ref.items()}, (rid, p)
    assert not eng.live and not eng.sched.running and not eng.sched.waiting
    if eng.kv.cache:
        eng.kv.cache.evict(10 ** 6)
    assert len(eng.kv.pool.free) == cf["blocks"] and all(r == 0 for r in eng.kv.pool.ref)
    if cf["spec"]:
        m = eng.metrics()
        assert m["spec_drafted"] > 0 and (m["spec_accepted"] > 0 or not seed % 2)


def test_tiny_pool_liveness_every_request_finishes():
    rng = random.Random(9)
    eng = Engine(FakeBackend(14, 2), Config(max_num_seqs=8, max_num_batched_tokens=6))
    for i in range(30):
        eng.add_request(i, [rng.randrange(1, V) for _ in range(rng.randint(2, 14))], P(max_tokens=rng.randint(1, 12)))
    done = set()
    for _ in range(50000):
        if not eng.has_work():
            break
        done |= {o.rid for o in eng.step() if o.finished and o.finish_reason in ("length", "stop")}
        eng.check()
    assert done == set(range(30)) and eng.metrics()["preemptions"] > 0
