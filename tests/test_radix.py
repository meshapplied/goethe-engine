from goethe_engine.kv import BlockPool, RadixCache


def build(n=32, bs=2):
    pool = BlockPool(n)
    return pool, RadixCache(pool, bs)


def put(pool, rc, tokens, start=0):
    blocks = pool.take(len(tokens) // rc.bs)
    canon = rc.insert(tokens, blocks, start)
    for b, c in zip(blocks, canon):
        if b != c:
            pool.decref(b)
    return blocks


def test_insert_match_split():
    pool, rc = build()
    a = put(pool, rc, [1, 2, 3, 4, 5, 6])
    assert rc.match([1, 2, 3, 4, 5, 6], 3) == a
    b = put(pool, rc, [1, 2, 3, 4, 9, 9])  # splits the edge after two pages
    assert rc.match([1, 2, 3, 4, 9, 9], 3)[:2] == a[:2] and rc.match([1, 2, 3, 4, 9, 9], 3)[2] == b[2]
    assert rc.match([1, 2, 3, 4, 5, 6], 3) == a
    assert rc.match([1, 2, 3, 4, 7, 7], 3) == a[:2]
    assert rc.match([1, 2, 3], 5) == a[:1]  # a partial page never matches
    assert rc.match([8, 8], 5) == []
    assert rc.match([1, 2, 3, 4, 5, 6], 2) == a[:2]  # max_blocks respected
    assert len(rc.root.children) == 1
    assert [len(n.blocks) for n in rc.nodes()].count(2) == 1


def test_insert_dedupes_and_start_skips_known_pages():
    pool, rc = build()
    a = put(pool, rc, [1, 2, 3, 4])
    own = pool.take(3)
    canon = rc.insert([1, 2, 3, 4, 5, 6], own, 0)
    assert canon[:2] == a and canon[2] == own[2]
    own2 = pool.take(4)
    canon = rc.insert([1, 2, 3, 4, 5, 6, 7, 8], own2, 3)
    assert canon == [own2[3]]
    assert pool.ref[own2[3]] == 2 and pool.ref[own2[2]] == 1


def test_evict_lru_leaves_then_parents_and_pinned_tail():
    pool, rc = build(8)
    a = put(pool, rc, [1, 2, 3, 4])
    b = put(pool, rc, [1, 2, 5, 6])
    rc.match([1, 2, 5, 6], 2)  # a-branch is older
    for x in a + b:
        pool.decref(x) if pool.ref[x] > 1 else None  # drop the caller refs, the tree keeps its own
    assert rc.evict(1) == 1 and rc.match([1, 2, 3, 4], 2, touch=False) == a[:1]
    assert rc.evict(10) == 2 and rc.match([1, 2, 5, 6], 2, touch=False) == []
    assert len(pool.free) == 8 and rc.nodes() == [rc.root]
    c = put(pool, rc, [1, 2, 3, 4, 5, 6])
    for x in c:
        pool.decref(x)
    pool.incref(c[0])  # a user holds the first page
    assert rc.evict(5) == 2
    assert rc.match([1, 2, 3, 4, 5, 6], 3, touch=False) == c[:1]
