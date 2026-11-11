import pytest

from goethe_engine.kv import KVManager
from goethe_engine.scheduler import Sequence
from goethe_engine.sampling import SamplingParams


def seq(tokens, sid=1):
    return Sequence(sid, "r", 0, tokens, SamplingParams(), sid, 0, None, 100)


def test_attach_reserve_commit_release_no_leak():
    kv = KVManager(8, 2)
    a = seq(list(range(7)))
    kv.attach(a)
    assert a.computed == 0 and kv.reserve(a, 5) == []
    assert len(a.blocks) == 3
    a.computed = 5
    kv.commit(a)
    assert a.tpages == 2
    kv.check([a])
    kv.release(a)
    kv.check([])
    assert len(kv.pool.free) == 8 - 2  # the tree keeps two blocks


def test_prefix_hit_leaves_last_token_and_shares_blocks():
    kv = KVManager(16, 2)
    a = seq([1, 2, 3, 4, 5, 6])
    kv.attach(a)
    kv.reserve(a, 6)
    a.computed = 6
    kv.commit(a)
    b = seq([1, 2, 3, 4, 5, 6], 2)
    kv.attach(b)  # 6 tokens: at most (6 - 1) // 2 = 2 pages, the last token is recomputed
    assert b.computed == 4 and b.blocks == a.blocks[:2]
    assert [kv.pool.ref[x] for x in a.blocks] == [3, 3, 2]
    kv.check([a, b])
    c = seq([1, 2, 3, 4, 5, 6, 7], 3)
    kv.attach(c)
    assert c.computed == 6 and c.blocks == a.blocks
    kv.check([a, b, c])


def test_cow_on_forked_partial_block():
    kv = KVManager(8, 4)
    a = seq([1, 2, 3, 4, 5, 6])
    kv.attach(a)
    kv.reserve(a, 6)
    a.computed = 6
    b = seq([1, 2, 3, 4, 5, 6], 2)
    kv.fork(a, b)
    kv.check([a, b])
    assert kv.pool.ref[a.blocks[1]] == 2
    pairs = kv.reserve(b, 7)
    assert pairs == [(a.blocks[1], b.blocks[1])] and b.blocks[0] == a.blocks[0] and b.blocks[1] != a.blocks[1]
    assert kv.reserve(a, 7) == []  # sole owner now: in place
    kv.check([a, b])
    kv.release(a)
    kv.release(b)
    kv.check([])
    assert len(kv.pool.free) == 8


def test_reserve_is_atomic_when_short():
    kv = KVManager(3, 2, prefix_cache=False)
    a, b = seq([1] * 4), seq([2] * 8, 2)
    kv.attach(a)
    kv.reserve(a, 4)
    kv.attach(b)
    assert kv.reserve(b, 8) is None and b.blocks == []
    kv.check([a, b])


def test_evict_lru_leaf_and_never_in_use_blocks():
    kv = KVManager(6, 2)
    old, new = seq([1, 1, 1, 1], 1), seq([2, 2, 2, 2], 2)
    for s in (old, new):
        kv.attach(s)
        kv.reserve(s, 4)
        s.computed = 4
        kv.commit(s)
        kv.release(s)
    kv.cache.match([2, 2, 2, 2, 9], 2)  # touch: [1,1,1,1] is now least recent
    held = seq([2, 2, 2, 2, 9], 3)
    kv.attach(held)
    assert kv.cache.evict(2) == 2
    assert kv.cache.match([1, 1, 1, 1, 9], 2, touch=False) == []
    assert kv.cache.match([2, 2, 2, 2, 9], 2, touch=False) == held.blocks
    assert kv.cache.evict(10) == 0  # the rest is pinned by `held`
    kv.check([held])


def test_commit_adopts_existing_block_of_a_concurrent_twin():
    kv = KVManager(8, 2)
    a, b = seq([1, 2, 3, 4, 5], 1), seq([1, 2, 3, 4, 6], 2)
    for s in (a, b):
        kv.attach(s)
        kv.reserve(s, 5)
        s.computed = 5
    kv.commit(a)
    mine = list(b.blocks)
    kv.commit(b)
    assert b.blocks[:2] == a.blocks[:2] and b.blocks[2] == mine[2]
    assert kv.pool.ref[mine[0]] == 0  # the duplicate went back to the pool
    kv.check([a, b])


def test_double_free_is_caught():
    kv = KVManager(2, 2)
    with pytest.raises(AssertionError):
        kv.pool.decref(0)
