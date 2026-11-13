import random

import pytest
import torch

from goethe_engine import Config, Engine, KVGeometry, SamplingParams
from goethe_engine.ref import RefModel, TorchBackend
from goethe_engine.spec import depths, path

V = 53


def fold(h, tok, p):
    return (h * 1000003 + tok * 31 + p) % (1 << 61)


def row(h, v=V):
    return ((torch.arange(v) * 2654435761 + (h % 1000003) * 40503) % 1009).float() / 90


def fake_greedy(tokens, k):
    h, out = 7, []
    toks = list(tokens)
    for p, t in enumerate(toks):
        h = fold(h, t, p)
    for _ in range(k):
        t = int(row(h).argmax())
        out.append(t)
        h = fold(h, t, len(toks))
        toks.append(t)
    return out


class FakeBackend:
    """Logits depend on every KV row the block table and the tree path point at, so a wrong table, a stale block, a
    wrong ancestor or an uncommitted path changes tokens."""

    max_seqs = 1 << 20
    caps = dict(tree=True, taps={"result_norm": 4})

    def __init__(self, num_blocks=64, bs=4, vocab=V):
        self.geometry = KVGeometry(bs, num_blocks, 1, 1, 1)
        self.vocab_size, self.kv, self.steps, self.released, self.blk = vocab, {}, [], [], {}
        self.fail = False

    def step(self, work):
        if self.fail:
            raise RuntimeError("boom")
        bs, rows, feats = self.geometry.block_size, [], []
        self.steps.append(work)
        for w in work:
            slot = lambda p: w.blocks[p // bs] * bs + p % bs
            dep = depths(w.parents, len(w.tokens))
            self.blk[w.sid] = w.blocks
            for j, t in enumerate(w.tokens):
                self.kv[slot(w.pos + j)] = (t, w.pos + dep[j])
            for j in range(len(w.tokens)):
                h = 7
                for p in range(w.pos):
                    tok, pos = self.kv[slot(p)]
                    assert pos == p, "stale or foreign KV row"
                    h = fold(h, tok, p)
                for a in path(w.parents, j):
                    tok, pos = self.kv[slot(w.pos + a)]
                    assert pos == w.pos + dep[a], "tree row at the wrong position"
                    h = fold(h, tok, pos)
                if w.feats:
                    feats.append(row(h, 4))
                if j >= len(w.tokens) - w.want:
                    rows.append(row(h, self.vocab_size))
        out = torch.stack(rows) if rows else torch.empty(0, self.vocab_size)
        return (out, torch.stack(feats)) if any(w.feats for w in work) else out

    def commit(self, sid, base, pth):
        bs, b = self.geometry.block_size, self.blk[sid]
        slot = lambda p: b[p // bs] * bs + p % bs
        for j, i in enumerate(pth):
            if i != j:
                tok, pos = self.kv[slot(base + i)]
                self.kv[slot(base + j)] = (tok, base + j)

    def copy_blocks(self, pairs):
        bs = self.geometry.block_size
        for s, d in pairs:
            for i in range(bs):
                if s * bs + i in self.kv:
                    self.kv[d * bs + i] = self.kv[s * bs + i]

    def release(self, sid):
        self.released.append(sid)


def make(blocks=64, bs=4, **cfg):
    be = FakeBackend(blocks, bs)
    return Engine(be, Config(**cfg), eos_ids=cfg.pop("eos", ())), be


def run(eng, check=True, limit=100000):
    outs = []
    for _ in range(limit):
        if not eng.has_work():
            break
        outs += eng.step()
        if check:
            eng.check()
    else:
        raise AssertionError("engine did not drain")
    return outs


def toks(outs):
    d = {}
    for o in outs:
        d.setdefault((o.rid, o.index), []).extend(o.token_ids)
    return d


def alone(prompt, params, bs=4, blocks=256, backend=None, **cfg):
    """Unbatched reference: one request, no prefix cache, no spec, one sequence at a time."""
    eng = Engine(backend or FakeBackend(blocks, bs), Config(max_num_seqs=max(1, params.n), prefix_cache=False,
                                                            max_num_batched_tokens=4096, **cfg))
    eng.add_request("r", prompt, params)
    return toks(run(eng, check=False))


@pytest.fixture
def rng():
    return random.Random(1)


class FakeTokens(FakeBackend):
    """Greedy steps return device-argmax tokens instead of logits."""

    def step(self, work):
        out = super().step([w for w in work])
        if not all(w.greedy for w in work) or isinstance(out, tuple):
            return out
        res, i = [], 0
        for w in work:
            res.append(out[i:i + w.want].argmax(1).tolist())
            i += w.want
        return res


class FakeSpec(FakeBackend):
    """One linear stream that decodes by rounds of 1 .. width tokens, like a pack's target and drafter buckets."""

    paged, spec_width, max_len, max_seqs, caps = False, 4, 200, 1, {}

    def __init__(self, width=4):
        super().__init__(200, 1)
        self.spec_width, self.buf, self.n = width, [], 0

    def step(self, work):
        assert len(work) == 1 and work[0].greedy
        w, = work
        self.buf = self.buf[:w.pos] + list(w.tokens)
        assert len(self.buf) == w.pos + len(w.tokens)
        if not w.want:
            return [[]]
        self.n += 1
        k = 1 + self.n % self.spec_width if w.spec else 1
        out = fake_greedy(self.buf, k)
        self.buf += out[:-1] if w.spec else []
        return [out]


class FakePages:
    """A page pack: a block is a whole sequence page (here its running hash, which a real page holds as recurrent state and
    window ring), valid at one position only. A step on a page whose state is not at w.pos fails, so a stale, shared or
    wrongly restored page changes the run."""

    paged, caps = False, {}

    def __init__(self, seqs=4, spare=2, max_len=64, vocab=V, pages=None):
        self.geometry = KVGeometry(max_len, pages or seqs, 1, 1, 1)
        self.max_seqs, self.max_len, self.state_pages, self.vocab_size = seqs, max_len, spare, vocab
        self.page, self.steps, self.copies = {}, 0, []

    def step(self, work):
        rows = []
        self.steps += 1
        assert len({w.blocks[0] for w in work}) == len(work), "two sequences on one page"
        for w in work:
            pg = w.blocks[0]
            assert pg < self.geometry.num_blocks, "running sequence on a snapshot page"
            if w.pos == 0:
                self.page[pg] = (7, 0)
            h, n = self.page[pg]
            assert n == w.pos, f"page {pg} is at position {n}, step at {w.pos}"
            for j, t in enumerate(w.tokens):
                h = fold(h, t, w.pos + j)
                if j >= len(w.tokens) - w.want:
                    rows.append(row(h, self.vocab_size))
            self.page[pg] = (h, w.pos + len(w.tokens))
        return torch.stack(rows) if rows else torch.empty(0, self.vocab_size)

    def copy_blocks(self, pairs):
        self.copies += pairs
        for s, d in pairs:
            self.page[d] = self.page[s]

    def release(self, sid):
        pass


class FakeBounded:
    """A bounded paged pack: KV rows live in blocks, and a slot per sequence holds bounded state (here the running hash a real
    pack keeps as ring and recurrent state), valid at one position. Logits depend on both, so a stale, shared or wrongly
    restored state, or a wrong block table, changes the run."""

    paged, bounded, prefix_cache, caps = True, True, True, {}

    def __init__(self, num_blocks=64, bs=4, seqs=4, spare=2, vocab=V):
        self.geometry = KVGeometry(bs, num_blocks, 1, 1, 1)
        self.vocab_size, self.max_seqs, self.max_len, self.state_pages = vocab, seqs, bs * num_blocks, spare
        self.kv, self.state, self.snap, self.steps, self.log = {}, {}, {}, 0, []

    def step(self, work):
        bs, rows = self.geometry.block_size, []
        self.steps += 1
        assert len(work) <= self.max_seqs
        for w in work:
            slot = lambda p: w.blocks[p // bs] * bs + p % bs
            if w.pos == 0:
                self.state[w.sid] = (7, 0)
            h, n = self.state[w.sid]
            assert n == w.pos, f"state of {w.sid} is at position {n}, step at {w.pos}"
            for p in range(w.pos):
                assert self.kv[slot(p)][1] == p, "stale or foreign KV row"
            for j, t in enumerate(w.tokens):
                self.kv[slot(w.pos + j)] = (t, w.pos + j)
                h = fold(h, t, w.pos + j)
                if j >= len(w.tokens) - w.want:
                    rows.append(row(fold(h, self.kv[slot(0)][0], 0), self.vocab_size))
            self.state[w.sid] = (h, w.pos + len(w.tokens))
        return torch.stack(rows) if rows else torch.empty(0, self.vocab_size)

    def copy_blocks(self, pairs):
        bs = self.geometry.block_size
        for s, d in pairs:
            for i in range(bs):
                if s * bs + i in self.kv:
                    self.kv[d * bs + i] = self.kv[s * bs + i]

    def save_state(self, sid, snap):
        self.log.append(("save", sid, snap))
        self.snap[snap] = self.state[sid]

    def restore_state(self, snap, sid):
        self.log.append(("restore", snap, sid))
        self.state[sid] = self.snap[snap]

    def release(self, sid):
        self.state.pop(sid, None)


# --- signed packs and the mock libgck (tests/toypack.py, tests/mock/gck_mock.c)
import os
import subprocess

from toypack import Key, toy_pack


@pytest.fixture(scope="session")
def key(tmp_path_factory):
    return Key(str(tmp_path_factory.mktemp("key")))


@pytest.fixture(scope="session")
def mocklib(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("mock") / "libgck.so")
    src = os.path.join(os.path.dirname(__file__), "mock", "gck_mock.c")
    subprocess.run(["cc", "-std=c99", "-Wall", "-O1", "-shared", "-fPIC", "-o", out, src], check=True)
    return out


@pytest.fixture
def toy(tmp_path, key, mocklib):
    """(pack dir, weights, drafter weights) of a signed toy gc-c pack."""
    return toy_pack(str(tmp_path), key, mocklib)
