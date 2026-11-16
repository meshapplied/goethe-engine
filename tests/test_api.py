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
