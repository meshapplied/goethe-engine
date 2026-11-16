"""pack.fit: weights + slots + pool of a ladder against the device's free bytes."""
import pytest
from goethe_engine import pack as P

G = 10 ** 9
def bucket(ncol, wset="a", w=15 * G, slot=G // 10, block=2 * 10 ** 6, fixed=10 ** 8, rows=32):
    return dict(ncol=ncol, maxctx=4096, resident=dict(weights=w, wset=wset, fixed=fixed, slot=slot, block=block, slots=ncol, block_rows=rows))

def test_default_pool_shrinks_to_the_device():
    lad = [bucket(2), bucket(64)]
    # base 15 + 0.2 + 0.1 * (64 + 4) = 22.0 GB; a block is 2 MB, a full pool 64 * 128 blocks = 16.4 GB
    pb, need = P.fit(lad, 4, 32 * G, margin=G)
    assert pb is not None and 0 < pb < 64 * 128 and need <= 31 * G
    pb2, _ = P.fit(lad, 4, 80 * G, margin=G)
    assert pb2 is None                                    # the full pool fits: the library default

def test_refuses_what_cannot_fit_with_numbers():
    with pytest.raises(P.PackError, match=r"needs .* GB .*weights 15.00 GB.*free"):
        P.fit([bucket(64, slot=G)], 0, 40 * G, margin=G)   # 15 + 64 GB of slots

def test_explicit_pool_blocks_is_checked():
    with pytest.raises(P.PackError, match="pool_blocks 8000"):
        P.fit([bucket(8)], 0, 20 * G, pool_blocks=8000, margin=G)
    assert P.fit([bucket(8)], 0, 20 * G, pool_blocks=100, margin=G)[0] == 100

def test_shared_weight_set_counted_once_distinct_sets_added():
    same = P.fit([bucket(8), bucket(32)], 0, 200 * G, margin=G)[1]
    diff = P.fit([bucket(8, wset="a"), bucket(32, wset="b")], 0, 200 * G, margin=G)[1]
    assert diff - same == 15 * G

def test_old_pack_or_unknown_device_is_not_checked():
    b = bucket(8); del b["resident"]
    assert P.fit([b], 0, 1, pool_blocks=7) == (7, None)
    assert P.fit([bucket(8)], 0, None) == (None, None)


def test_a_prefill_bucket_after_the_ladder_does_not_hide_the_pool():
    pf = bucket(128, block=0, rows=0, slot=0)
    pf["resident"]["slots"] = 1
    pb, need = P.fit([bucket(2), bucket(64), pf], 4, 32 * G, margin=G)
    assert pb is not None and 0 < pb < 64 * 128 and need <= 31 * G   # sized from the paged top, not "unpaged: the library default"


def test_open_steps_down_to_the_widest_ladder_that_fits(tmp_path, key, mocklib, monkeypatch):
    """Fit is decided before any launch: a device whose free memory holds the 2 column ladder but not the 8 column one opens the 2
    column one (max_seqs 2); one that holds neither is refused (code oom) with each width's numbers."""
    from goethe_engine.errors import OutOfMemory
    from goethe_engine.gc_c import GcBackend
    from toypack import toy_pack
    G = 1 << 30

    def mut(m):
        for b in m["buckets"]:
            b["resident"] = dict(weights=G, wset="w", fixed=0, scratch=G // 10, slot=G // 10, slots=b["ncol"], block=G // 100, block_rows=16)
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib, mutate=mut)
    monkeypatch.setattr(P, "device_free", lambda i=None: int(3.0 * G))
    be = GcBackend(pack, w, B=8, sku="mock", trusted=(key.kid,))
    assert be.bucket["ncol"] == 2 and be.resident < 3.0 * G - (1 << 30)
    be.close()
    monkeypatch.setattr(P, "device_free", lambda i=None: int(2.5 * G))
    with pytest.raises(OutOfMemory) as e:
        GcBackend(pack, w, B=8, sku="mock", trusted=(key.kid,))
    assert e.value.code == "oom" and "B 8:" in str(e.value) and "B 2:" in str(e.value)


def test_weights_are_the_union_of_parts_and_the_margin_is_the_measured_overhead():
    """A handle relays each weight part once: two buckets sharing part a count it once (resident.wparts); without wparts each weight set
    counts once. The kept margin is the manifest's runtime.overhead for the SKU, else 1 GiB."""
    G = 1 << 30
    r = lambda parts, w, ws: dict(ncol=2, maxctx=4096, resident=dict(weights=w, wset=ws, wparts=parts, fixed=0, slot=0, block=0,
                                                                      slots=2, block_rows=0))
    lad = [r(dict(a=2 * G, b=G), 3 * G, "x"), r(dict(a=2 * G, c=G), 3 * G, "y")]
    assert P.fit(lad, 0, 10 * G, margin=0)[1] == 4 * G                     # union a + b + c, not 3 + 3
    lad2 = [dict(b, resident={k: v for k, v in b["resident"].items() if k != "wparts"}) for b in lad]
    assert P.fit(lad2, 0, 10 * G, margin=0)[1] == 6 * G                    # two weight sets without parts
    m = dict(runtime=dict(kind="gc-c", version=1, overhead=dict(s1=2 * G)))
    assert P.overhead(m, "s1") == 2 * G and P.overhead(m, "s2") == G and P.overhead({"runtime": {}}, "s1") == G
    with pytest.raises(P.PackError):
        P.fit(lad, 0, 5 * G, margin=P.overhead(m, "s1"))                   # 4 GiB parts + 2 GiB overhead > 5 GiB free
    assert P.fit(lad, 0, 6 * G, margin=P.overhead(m, "s1"))[1] == 4 * G
