"""Speculative decoding over the one abstraction: every policy greedy-exact against the dense oracle, tree commit with
recurrent state, multi-stream traces under pool pressure / preemption / aborts, rejection sampling distribution-equal
to the target (with a broken-residual control), seeded reproducibility, capability detection."""
import itertools
import math
import random

import pytest
import torch

from conftest import FakeBackend, V, alone, fold, row, run, toks
from goethe_engine import Config, Engine, SamplingParams as P
from goethe_engine import engine as E
from goethe_engine.ref import DraftLM, EagleHead, MedusaHeads, RefModel, TorchBackend
from goethe_engine.sampling import dist, sample
from goethe_engine.spec import ALGOS, NgramDrafter, Tree, TreeDrafter, caps_of, check, compatible, depths, mss, path

MEDUSA = [(0,), (1,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (2,), (0, 0, 1)]


def test_tree_helpers():
    t = Tree([5, 6, 7, 8, 9], [0, 0, 1, 3, 2])
    assert t.kids() == [[1, 2], [3], [5], [4], [], []] and not t.is_chain and Tree.chain([1, 2]).is_chain
    assert depths([-1] + t.parents, 6) == [0, 1, 1, 2, 3, 2] and path([-1] + t.parents, 4) == [0, 1, 3, 4]
    f = t.fit(4, 2)
    assert f.tokens == [5, 6, 7, 9] and f.parents == [0, 0, 1, 2]
    assert not Tree([1], [1]).ok(V) and not Tree([V], [0]).ok(V) and t.ok(V)


def ref_prompts():
    return [[3, 9, 27, 14, 3, 9, 27, 14, 3, 9], [5, 1, 5, 1, 5, 1, 7], [30, 31, 2], [3, 9, 27, 14, 3, 9, 27, 11]]


class Commits:
    """Counts commits whose path is not the identity (the KV rows had to move)."""

    def __init__(self, be):
        self.be, self.moved, inner = be, 0, be.commit

        def commit(sid, base, pth):
            self.moved += pth != list(range(len(pth)))
            inner(sid, base, pth)

        be.commit = commit


def policies(m):
    small = RefModel(d=16, layers=1, heads=2, kv_heads=1, ffn=16, seed=5)
    return {
        "ngram": (NgramDrafter(), ()),
        "ar:3 small": (TreeDrafter(DraftLM(small), "chain"), ()),
        "ar perfect": (TreeDrafter(DraftLM(m), "chain"), ()),
        "specinfer top-k (perfect LM)": (TreeDrafter(DraftLM(m), "dynamic", topk=2, depth=3), ()),
        "specinfer sampled": (TreeDrafter(DraftLM(small), "dynamic", topk=2, depth=3, temperature=1.0), ()),
        "eagle1 static": (TreeDrafter(EagleHead(m), "static", choices=MEDUSA), ("result_norm",)),
        "eagle2 dynamic": (TreeDrafter(EagleHead(m), "dynamic", topk=3, depth=3), ("result_norm",)),
        "eagle3 dynamic": (TreeDrafter(EagleHead(m, taps=("l_out-0", "l_out-1", "result_norm"), own="l_out"),
                                       "dynamic", topk=2, depth=4), ("l_out-0", "l_out-1", "result_norm")),
        "mtp chain": (TreeDrafter(EagleHead(m), "chain"), ("result_norm",)),
        "medusa static": (TreeDrafter(MedusaHeads(m, heads=3), "static", choices=MEDUSA), ("result_norm",)),
    }


@pytest.mark.parametrize("name", list(policies(RefModel())))
def test_every_policy_greedy_equals_dense_oracle_multistream_with_preemption(name):
    m = RefModel()
    drafter, taps = policies(m)[name]
    be = TorchBackend(m, 26, 2, taps=taps)
    cm = Commits(be)
    eng = Engine(be, Config(max_num_seqs=4, max_num_batched_tokens=14, spec_k=6), drafter=drafter)
    for i, p in enumerate(ref_prompts()):
        eng.add_request(i, p, P(max_tokens=14))
    outs = []
    while eng.has_work():
        outs += eng.step()
        eng.check()
    got = toks(outs)
    for i, p in enumerate(ref_prompts()):
        assert got[(i, 0)] == m.dense(p, 14), (name, i)
    mt = eng.metrics()
    assert mt["spec_drafted"] > 0 and mt["errors"] == 0 and mt["spec_invalid"] == 0
    if "perfect" in name:
        assert mt["spec_accepted"] > mt["spec_drafted"] / 4
    if name == "specinfer top-k (perfect LM)":
        assert cm.moved > 0


def test_recurrent_target_tree_verify_commits_the_state_of_the_accepted_node():
    m = RefModel(rec=(0,), seed=4)
    for drafter in (TreeDrafter(DraftLM(m), "dynamic", topk=2, depth=3), TreeDrafter(DraftLM(m), "chain"),
                    NgramDrafter()):
        be = TorchBackend(m, 40, 2)
        cm = Commits(be)
        eng = Engine(be, Config(max_num_seqs=3, max_num_batched_tokens=9, spec_k=5, prefix_cache=False),
                     drafter=drafter)
        for i, p in enumerate(ref_prompts()):
            eng.add_request(i, p, P(max_tokens=12))
        got = toks(run(eng))
        for i, p in enumerate(ref_prompts()):
            assert got[(i, 0)] == m.dense(p, 12), (type(drafter.m if hasattr(drafter, "m") else drafter), i)
    with pytest.raises(ValueError, match="recurrent"):
        Engine(TorchBackend(m, 40, 2), Config(prefix_cache=True))


# --- multi-stream property traces on the fake backend ---------------------------------------------------------------

def target_path(s, k):
    """The tokens the fake target emits after s (greedy: argmax; sampled: its seeded draws)."""
    toks_, out = list(s.tokens), []
    for j in range(k):
        h = 7
        for p_, t in enumerate(toks_):
            h = fold(h, t, p_)
        x = row(h)[None]
        t = sample(x, [s.params], [s.seed], [s.nout + j], [None])[0] if s.params.temperature else int(x.argmax())
        out.append(t)
        toks_.append(t)
    return out


class TreeOracle:
    """A random tree around the target's own continuation, deterministic in the history: siblings before the true
    child (so the accepted path moves KV rows), corrupted nodes, random depth."""

    needs = dict(tree=True, taps=())

    def __init__(self, seed):
        self.seed = seed

    def propose(self, seqs, k):
        out = []
        for s in seqs:
            rng = random.Random(hash((self.seed, tuple(s.tokens))))
            good = target_path(s, rng.randint(1, 5)) if rng.random() < 0.8 else []
            toks_, par, at = [], [], 0
            for g in good:
                for _ in range(rng.randint(0, 2)):
                    toks_.append(rng.randrange(V))
                    par.append(at)
                toks_.append(g if rng.random() > 0.15 else rng.randrange(V))
                par.append(at)
                at = len(toks_)
            out.append(Tree(toks_, par))
        return out


def trace(rng, n=30):
    tmpl = [[rng.randrange(1, V) for _ in range(rng.randint(8, 24))] for _ in range(3)]
    reqs = []
    for i in range(n):
        t = rng.choice(tmpl)
        prompt = t[:rng.randint(3, len(t))] + [rng.randrange(1, V) for _ in range(rng.randint(0, 5))]
        kind = rng.choice(["greedy", "exact", "rejection"])
        p = P(max_tokens=rng.randint(1, 16)) if kind == "greedy" else P(
            temperature=rng.uniform(0.5, 1.5), top_k=rng.choice([0, 5]), top_p=rng.choice([1, .9]),
            seed=rng.randrange(1 << 30), max_tokens=rng.randint(1, 16), accept=kind, n=rng.choice([1, 1, 2]))
        reqs.append((i, prompt, p, rng.randint(0, 40), rng.random() < 0.12))
    return reqs


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("cfg", [dict(bs=2, blocks=48, budget=11, cache=True), dict(bs=4, blocks=30, budget=9, cache=True),
                                 dict(bs=1, blocks=120, budget=24, cache=False)])
def test_multistream_tree_traces_equal_references_and_leak_nothing(seed, cfg):
    rng = random.Random(seed * 7 + cfg["bs"])
    reqs = trace(rng)
    mk = lambda blocks, seqs, budget, cache: Engine(
        FakeBackend(blocks, cfg["bs"]), Config(max_num_seqs=seqs, max_num_batched_tokens=budget, prefix_cache=cache,
                                               spec_k=8), drafter=TreeOracle(seed))
    eng, outs, t = mk(cfg["blocks"], 8, cfg["budget"], cfg["cache"]), [], 0
    while t <= 40 or eng.has_work():
        for rid, prompt, p, at, ab in reqs:
            if at == t:
                eng.add_request(rid, prompt, p)
            if ab and at + 2 == t:
                eng.abort(rid)
        outs += eng.step()
        eng.check()
        t += 1
        assert t < 20000
    got, aborted = toks(outs), {r[0] for r in reqs if r[4]}
    assert not [o for o in outs if o.finish_reason == "error"]
    for rid, prompt, p, _, _ in reqs:
        if rid in aborted:
            continue
        mine = {k[1]: v for k, v in got.items() if k[0] == rid}
        if p.accept == "rejection" and p.temperature:
            solo = mk(400, max(1, p.n), 64, False)
            solo.add_request("r", prompt, p)
            ref = {k[1]: v for k, v in toks(run(solo)).items()}
        else:
            ref = {k[1]: v for k, v in alone(prompt, p, bs=cfg["bs"]).items()}
        assert mine == ref, (rid, p)
    m = eng.metrics()
    assert m["spec_accepted"] > 0 and m["preemptions"] + m["aborted"] > 0
    if eng.kv.cache:
        eng.kv.cache.evict(10 ** 6)
    assert len(eng.kv.pool.free) == cfg["blocks"] and all(r == 0 for r in eng.kv.pool.ref)


# --- rejection sampling: distribution equal to the target -----------------------------------------------------------

class FakeLM:
    """A drafter LM unlike the target: logits of a perturbed hash."""

    taps = ()

    def __init__(self, v):
        self.v = v

    def dist(self, s, pth):
        h = 3
        for p_, t in enumerate(list(s.tokens) + list(pth)):
            h = fold(h, t, p_)
        return torch.log_softmax(row(h, self.v).double() / 2, 0)


def target_joint(prompt, p, v, n):
    out = {}
    for seq in itertools.product(range(v), repeat=n):
        pr, toks_ = 1.0, list(prompt)
        for t in seq:
            h = 7
            for p_, x in enumerate(toks_):
                h = fold(h, x, p_)
            pr *= float(dist(row(h, v), p)[t])
            toks_.append(t)
        out[seq] = pr
    return out


def chi2(counts, probs, n):
    """Pearson statistic and degrees of freedom, bins with expectation < 5 pooled."""
    stat, df, pe, po = 0.0, -1, 0.0, 0
    for k, pr in probs.items():
        e = pr * n
        if e < 5:
            pe, po = pe + e, po + counts.get(k, 0)
            continue
        stat, df = stat + (counts.get(k, 0) - e) ** 2 / e, df + 1
    if pe > 0:
        stat, df = stat + (po - pe) ** 2 / pe, df + 1
    return stat, df


def sampled_joint(drafter, p, prompt, v, n, runs):
    eng = Engine(FakeBackend(4000, 4, vocab=v), Config(max_num_seqs=256, max_num_batched_tokens=2048, spec_k=6,
                                                      prefix_cache=False), drafter=drafter)
    for i in range(runs):
        eng.add_request(i, prompt, P(temperature=p.temperature, seed=1000 + i, max_tokens=n, accept=p.accept,
                                     ignore_eos=True))
    got = toks(run(eng, check=False))
    c = {}
    for i in range(runs):
        k = tuple(got[(i, 0)])
        c[k] = c.get(k, 0) + 1
    return c, eng.metrics()


DRAFTERS = {
    "chain sampled q (Leviathan / Chen)": lambda v: TreeDrafter(FakeLM(v), "chain", temperature=1.0),
    "tree sampled q (SpecInfer MSS)": lambda v: TreeDrafter(FakeLM(v), "dynamic", topk=2, depth=3, temperature=1.0),
    "tree top-k one-hot (EAGLE)": lambda v: TreeDrafter(FakeLM(v), "dynamic", topk=3, depth=2),
}


@pytest.mark.parametrize("name", list(DRAFTERS))
def test_rejection_sampling_is_distribution_equal_to_the_target(name):
    v, n, runs, prompt = 5, 3, 6000, [1, 2, 3]
    p = P(temperature=3.0, accept="rejection")
    want = target_joint(prompt, p, v, n)
    c, m = sampled_joint(DRAFTERS[name](v), p, prompt, v, n, runs)
    stat, df = chi2(c, want, runs)
    assert m["spec_accepted"] > runs / 4, m
    assert stat < df + 6 * math.sqrt(2 * df), (stat, df)


def test_chi2_has_power_a_residual_free_sampler_is_caught(monkeypatch):
    def broken(p, kids, seed, i):
        for r, (x, q) in enumerate(kids):
            if q is None or float(q[x]) > 0:
                return x, r
        return mss(p, [], seed, i)

    monkeypatch.setattr(E, "mss", broken)
    v, n, runs, prompt = 5, 3, 3000, [1, 2, 3]
    p = P(temperature=3.0, accept="rejection")
    c, _ = sampled_joint(DRAFTERS["chain sampled q (Leviathan / Chen)"](v), p, prompt, v, n, runs)
    stat, df = chi2(c, target_joint(prompt, p, v, n), runs)
    assert stat > 10 * (df + 6 * math.sqrt(2 * df)), (stat, df)


def test_rejection_seeded_outputs_reproduce_across_batches_and_budgets():
    v, prompt = 7, [4, 1, 2]
    sp = [P(temperature=1.5, seed=77 + i, max_tokens=12, accept="rejection", ignore_eos=True) for i in range(6)]
    res = []
    for seqs, budget, blocks in ((6, 64, 400), (1, 7, 400), (3, 9, 22)):
        eng = Engine(FakeBackend(blocks, 2, vocab=v), Config(max_num_seqs=seqs, max_num_batched_tokens=budget, spec_k=6),
                     drafter=DRAFTERS["tree sampled q (SpecInfer MSS)"](v))
        for i, s in enumerate(sp):
            eng.add_request(i, prompt + [i % v], s)
        res.append(toks(run(eng)))
    assert res[0] == res[1] == res[2]


# --- capability detection --------------------------------------------------------------------------------------------

def test_capabilities_decide_and_refuse_with_reasons():
    m = RefModel()
    full = caps_of(TorchBackend(RefModel(layers=3), 8, 2, taps=("l_out-0", "l_out-1", "l_out-2", "result_norm")))
    ok = compatible(full, 6)
    assert {a for a, r in ok.items() if r is None} == set(ALGOS) - {"dflash", "ar-device", "mtp-device"}
    assert "roll back" in check(dict(full, recurrent="step"), ALGOS["ar"], 6)
    assert "device round" in ok["dflash"]
    chain = dict(full, tree=False, taps={})
    r = compatible(chain, 6)
    assert r["ngram"] is None and r["ar"] is None and "tree verify" in r["eagle2"] and "taps" in r["mtp"]
    assert "along a chain only" in check(dict(full, recurrent="chain"), ALGOS["eagle2"], 6)
    assert "verify width 4" in check(dict(full, verify=4), ALGOS["ar"], 6)
    assert "layer taps" in compatible(dict(full, taps={"result_norm": 32}))["eagle3"]
    assert "no medusa drafter" in compatible(dict(full, drafters=["ar", "walk"]))["medusa"]
    with pytest.raises(ValueError, match="tree verify"):
        be = TorchBackend(m, 8, 2)
        be.caps = dict(be.caps, tree=False)
        Engine(be, Config(spec_k=4), drafter=TreeDrafter(DraftLM(m), "dynamic"))
    with pytest.raises(ValueError, match="taps"):
        Engine(TorchBackend(m, 8, 2), Config(spec_k=4), drafter=TreeDrafter(EagleHead(m), "chain"))


def test_device_round_backend_refuses_host_drafters_and_rejection():
    from conftest import FakeSpec
    with pytest.raises(ValueError, match="device"):
        Engine(FakeSpec(), Config(max_num_seqs=1, prefix_cache=False, max_model_len=150, spec_k=3),
               drafter=NgramDrafter())
    e = Engine(FakeSpec(), Config(max_num_seqs=1, prefix_cache=False, max_model_len=150))
    assert e.spec_algorithms()["dflash"] is None and e.spec_algorithms()["eagle2"]
    with pytest.raises(ValueError, match="device tokens"):
        e.add_request("a", [1, 2], P(temperature=1.0, accept="rejection"))
