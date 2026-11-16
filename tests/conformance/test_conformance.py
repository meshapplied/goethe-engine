"""Conformance of a pack against the engine's C ABIs (gc.h for generate / embed, gcm.h for media): open, step determinism, refusals.

Any pack: CONF_PACK=FILE CONF_WEIGHTS=a[:b..] CONF_KEYID=hex [CONF_SKU=..] python -m pytest tests/conformance (a GPU box for a real
pack). Without CONF_PACK the toy packs over the mock libraries run (CPU). Each test states the property; a property the pack cannot
show (no limits declared, no pinned body) is skipped with the reason."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from goethe_engine import api, pack as P  # noqa: E402
from goethe_engine.errors import Refused  # noqa: E402
from test_embed import mockgce  # noqa: E402,F401 (fixture)
from test_media import mockgcm  # noqa: E402,F401 (fixture)
from cf_run import run  # noqa: E402

REAL = os.environ.get("CONF_PACK")


@pytest.fixture
def subject(tmp_path, key, mocklib, mockgce, mockgcm):
    """(pack, weights, trusted, sku, kind, mutate-able: True for toy packs) for each task kind."""
    if REAL:
        return [(REAL, os.environ["CONF_WEIGHTS"].split(":"), (os.environ["CONF_KEYID"],), os.environ.get("CONF_SKU"),
                 api.capabilities(REAL)["kind"], False)]
    from test_embed import encode_pack
    from test_media import media_pack
    from toypack import toy_pack
    (tmp_path / "m").mkdir()
    d, e, m = toy_pack(str(tmp_path / "d"), key, mocklib), encode_pack(tmp_path / "e", key, mockgce), media_pack(tmp_path / "m", key, mockgcm)
    return [(d[0], d[1], (key.kid,), "mock", "generate", True), (e[0], e[1], (key.kid,), "mock", "embed", True),
            (m[0], m[1], (key.kid,), "mock", "transcribe", True)]


ISOLATED = r"""
import json, sys
import numpy as np
sys.path.insert(0, {root!r})
from goethe_engine import api
sys.path.insert(0, {here!r})
from cf_run import run
pack, ws, tr, sku, kind = json.loads({args!r})
with api.open(pack, ws, trusted=tuple(tr), sku=sku, device=0) as m:
    assert m.kind == kind
    a, b = run(m, kind), run(m, kind)
    assert np.isfinite(np.asarray(a, dtype=np.float64)).all() and np.asarray(a).tobytes() == np.asarray(b).tobytes(), kind
assert "torch" not in sys.modules
print("OK")
"""


def test_open_and_steps_are_deterministic(subject):
    """In a fresh interpreter for a real pack: the test process imports torch (the reference fixtures), and torch would bind its
    wheels' cuBLAS ahead of a pinned body's (refused, correctly)."""
    import json
    import subprocess
    if REAL:
        here = os.path.dirname(os.path.abspath(__file__))
        for pack, ws, tr, sku, kind, _ in subject:
            code = ISOLATED.format(root=os.path.join(here, "..", ".."), here=here, args=json.dumps([pack, ws, list(tr), sku, kind]))
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, errors="replace")
            assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]
        return
    for pack, ws, tr, sku, kind, _ in subject:
        with api.open(pack, ws, trusted=tr, sku=sku, device=0) as m:
            assert m.kind == kind
            a, b = run(m, kind), run(m, kind)
            assert np.isfinite(np.asarray(a, dtype=np.float64)).all() and np.asarray(a).tobytes() == np.asarray(b).tobytes(), kind


def test_refusals_are_named(subject, tmp_path):
    """An untrusted signer, and weights of another structure, are refused (code refused) before any device work."""
    for pack, ws, tr, sku, kind, _ in subject:
        with pytest.raises(Refused, match="allow-list"):
            api.open(pack, ws, trusted=("0" * 64,), sku=sku, device=0)
        other = tmp_path / f"other_{kind}.gguf"
        from toypack import gguf
        gguf(str(other), [("x.weight", [4, 4], 0)])
        with pytest.raises(Refused, match="not a model this pack was compiled for"):
            api.open(pack, [str(other)] * len(ws), trusted=tr, sku=sku, device=0)


def test_limits_refusal(subject, tmp_path, key, mocklib):
    """A context beyond model.limits is refused with the limit's reason."""
    for pack, ws, tr, sku, kind, toy in subject:
        if kind != "generate":
            continue
        if toy:
            from toypack import toy_pack

            def mut(mm):
                mm["abi"] = "3.1.0"
                mm["model"].update(name="toy", limits=[dict(maxctx=64, why="conformance limit")])
            pack, ws, _ = toy_pack(str(tmp_path / "lim"), key, mocklib, mutate=mut)
            ws = [ws]
        lim = P.ctx_limit(api.manifest(pack))
        if lim is None:
            pytest.skip("the pack declares no model.limits")
        with pytest.raises(Refused, match=lim[1]):
            api.open(pack, ws, trusted=tr, sku=sku, device=0, ctx=lim[0] + 1)


def test_pinned_provenance_mismatch_refusal(subject, monkeypatch):
    """A pinned body (provenance ggml-pinned) refuses library versions other than its provenance's (toy: the mock library reports a
    different cuBLAS; a real pack: run where another cuBLAS is bound first and expect the same refusal)."""
    hit = [s for s in subject if s[4] == "transcribe" and s[5]]
    if not hit:
        pytest.skip("needs the toy media pack (a real pack shows it only in a process with another cuBLAS bound)")
    pack, ws, tr, sku, _, _ = hit[0]
    monkeypatch.setenv("GCM_MOCK_CUBLAS", "0.0.1")
    with pytest.raises(Refused, match="cublas"):
        api.open(pack, ws, trusted=tr, sku=sku, device=0)
