"""Open warms every loaded bucket: a ladder whose launch cannot get its device memory (a fit the manifest's resident cannot see:
local memory, launch workspace) drops to the next narrower ladder with one note; a forced bucket and the narrowest ladder refuse."""
import pytest

from goethe_engine import api, pack as P
from goethe_engine.gc_c import GcBackend
from conftest import fake_greedy
from toypack import toy_pack


def test_warm_passes_and_keeps_the_ladder(tmp_path, key, mocklib):
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib)
    be = GcBackend(pack, w, B=8, sku="mock", trusted=(key.kid,))
    try:
        assert be.note is None and [b["ncol"] for b in be.ladder] == [2, 8] and be.stats["steps"] == 0
    finally:
        be.close()


def test_launch_oom_drops_to_the_narrower_ladder(tmp_path, key, mocklib, monkeypatch, capsys):
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib)
    monkeypatch.setenv("GCK_MOCK_LAUNCH_MAX", "4")
    with api.open(pack, w, B=8, sku="mock", trusted=(key.kid,)) as m:
        be = m.backend
        assert [b["ncol"] for b in be.ladder] == [2] and be.bucket["ncol"] == 2
        assert be.note.startswith("the 8-column ladder cannot launch (gc call failed (-4): launch: out of memory)") and be.launch_ws == 0
        assert m.generate([[1, 2, 3], [4, 5]], 5) == [fake_greedy([1, 2, 3], 5), fake_greedy([4, 5], 5)]
    assert "goethe-engine: the 8-column ladder cannot launch" in capsys.readouterr().err


def test_forced_bucket_and_narrowest_ladder_refuse(tmp_path, key, mocklib, monkeypatch):
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib)
    monkeypatch.setenv("GCK_MOCK_LAUNCH_MAX", "4")
    with pytest.raises(P.OutOfMemory, match="launch: out of memory"):
        GcBackend(pack, w, sku="mock", trusted=(key.kid,), bucket="target.mock.K1.8.256")
    monkeypatch.setenv("GCK_MOCK_LAUNCH_MAX", "1")
    with pytest.raises(P.OutOfMemory, match="launch: out of memory"):
        GcBackend(pack, w, B=8, sku="mock", trusted=(key.kid,))


def test_measured_launch_memory_retries_the_wider_ladder_once(tmp_path, key, mocklib, monkeypatch, capsys):
    """Launch memory measured on the narrower ladder: the wider one is reopened once with room for it (here it still cannot launch)."""
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib)
    monkeypatch.setenv("GCK_MOCK_LAUNCH_MAX", "4")
    free = iter(range(100 * 10 ** 9, 0, -10 ** 9))          # every read 1 GB lower: the warm takes 1 GB
    monkeypatch.setattr(P, "device_free", lambda idx=None: next(free))
    be = GcBackend(pack, w, B=8, sku="mock", trusted=(key.kid,))
    try:
        assert [b["ncol"] for b in be.ladder] == [2] and be.launch_ws == 10 ** 9
        err = capsys.readouterr().err
        assert err.count("the 8-column ladder cannot launch") == 2 and "launch memory measured 1.00 GB on the 2-column ladder" in err
    finally:
        be.close()
