"""The one engine API: the task kind comes from the manifest (generate, embed, transcribe), a call of another kind is refused, a
drafter opens the pack's speculative pair with output equal to plain decode."""
import numpy as np
import pytest

from goethe_engine import api, pack as P
from conftest import fake_greedy
from test_embed import encode_pack, mockgce  # noqa: F401 (fixture)
from test_media import media_pack, mockgcm  # noqa: F401 (fixture)
from toypack import toy_pack


def test_generate_and_capabilities(tmp_path, key, mocklib):
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib)
    c = api.capabilities(pack)
    assert (c["type"], c["kind"], c["ladder"], c["spec"], c["media"]) == ("decoder", "generate", [2, 8], [], [])
    ps = [[1, 2, 3], [9, 9]]
    with api.open(pack, w, B=2, sku="mock", trusted=(key.kid,)) as m:
        assert m.kind == "generate" and m.generate(ps, 7) == [fake_greedy(p, 7) for p in ps]
        with pytest.raises(P.PackError, match=r"\(task generate\), not embed"):
            m.embed([[1]])


def test_spec_pair_equals_plain(tmp_path, key, mocklib):
    pack, w, dw = toy_pack(str(tmp_path), key, mocklib, spec=True)
    assert api.capabilities(pack)["spec"] == ["ar"]
    with api.open(pack, w, drafter=dw, sku="mock", trusted=(key.kid,)) as m:
        assert m.backend.spec_width == 4 and m.generate([[1, 2, 3]], 11) == [fake_greedy([1, 2, 3], 11)]


def test_embed_kind(tmp_path, key, mockgce):  # noqa: F811
    pack, w, _ = encode_pack(tmp_path, key, mockgce)
    with api.open(pack, w, device=0, sku="mock", trusted=(key.kid,)) as m:
        assert m.kind == "embed" and m.embed([[1, 2, 3]], norm=False).tolist() == [[6, 3, 1, 3]]


def test_transcribe_kind(tmp_path, key, mockgcm):  # noqa: F811
    pack, ws = media_pack(tmp_path, key, mockgcm)
    assert api.capabilities(pack)["media"] == ["audio"]
    with api.open(pack, ws, device=0, sku="mock", trusted=(key.kid,)) as m:
        assert m.kind == "transcribe" and m.transcribe([np.ones(4, np.float32)], "<m>", 2) == [14, 1000]


def test_gc_c_paths_never_import_torch(tmp_path, key, mocklib, mockgce):  # noqa: F811
    """torch is the reference backends' optional extra: generate and embed on gc-c packs run on numpy buffers (torch would also preload
    its wheels' cuBLAS ahead of a pinned body's)."""
    import os
    import subprocess
    import sys
    pack, w, _ = toy_pack(str(tmp_path / "d"), key, mocklib)
    epack, ew, _ = encode_pack(tmp_path / "e", key, mockgce)
    code = (f"import sys; from goethe_engine import api\n"
            f"m = api.open({pack!r}, {w!r}, B=2, sku='mock', trusted=({key.kid!r},)); m.generate([[1, 2, 3]], 4); m.close()\n"
            f"m = api.open({epack!r}, {ew!r}, device=0, sku='mock', trusted=({key.kid!r},)); m.embed([[1, 2]]); m.close()\n"
            "assert 'torch' not in sys.modules, 'torch imported'")
    root = os.path.join(os.path.dirname(__file__), "..")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env=dict(os.environ, PYTHONPATH=root))
