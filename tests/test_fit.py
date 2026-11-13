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
