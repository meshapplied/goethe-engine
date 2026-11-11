import json
from datetime import datetime, timezone

import pytest

from goethe_engine import index as I
from goethe_engine import sign

H = lambda c: c * 64
FILES = {"m.gguf": H("a")}
NOW = datetime(2026, 11, 11, tzinfo=timezone.utc)


def entry(**kw):
    e = dict(model=dict(hf_repo="org/m", revision="r1", format="gguf", files=FILES), arch="sm_120a", abi_major=1,
             runtime=dict(kind="gc-c", version=1), tier="base", manifest_sha256=H("b"), manifest_sig_sha256=H("c"),
             size=10, min_engine="0.1.0")
    e.update(kw)
    return e


def index(serial=5, expires="2026-12-01T00:00:00Z", entries=None):
    return dict(schema="gc-pack-index", version=1, serial=serial, issued="2026-11-01T00:00:00Z", expires=expires,
                entries=[entry()] if entries is None else entries)


def signed(key, ix):
    raw = json.dumps(ix).encode()
    return raw, key.sign(raw)


def test_verify_match(key):
    raw, sig = signed(key, index(entries=[entry(), entry(tier="a", manifest_sha256=H("d")), entry(runtime=dict(kind="other", version=1)),
                                         entry(arch="sm_90a"), entry(min_engine="9.0.0"), entry(abi_major=2)]))
    ix, kid = I.verify(raw, sig, 5, NOW, (key.kid,))
    assert kid == key.kid and ix["serial"] == 5
    hit = I.match(ix, FILES, "sm_120a", "0.1.0")
    assert [e["tier"] for e in hit] == ["a", "base"]   # runtime, arch, min_engine and abi filtered; tier orders
    assert I.match(ix, {"m.gguf": H("d")}, "sm_120a") == []
    assert I.match(ix, FILES, "sm_120a", hf_repo="org/other") == []
    assert I.match(ix, FILES, "sm_120a", tier="base")[0]["tier"] == "base"


def test_rollback_and_freeze_refused(key):
    raw, sig = signed(key, index(serial=4))
    with pytest.raises(I.IndexError_, match="rollback"):
        I.verify(raw, sig, 5, NOW, (key.kid,))
    I.verify(raw, sig, 4, NOW, (key.kid,))   # equal serial is fine
    with pytest.raises(I.IndexError_, match="expired"):
        I.verify(raw, sig, 0, datetime(2026, 12, 2, tzinfo=timezone.utc), (key.kid,))


def test_bad_signature_and_schema_refused(key):
    raw, sig = signed(key, index())
    with pytest.raises(I.IndexError_, match="does not verify"):
        I.verify(raw + b" ", sig, 0, NOW, (key.kid,))
    with pytest.raises(I.IndexError_, match="allow-list"):
        I.verify(raw, sig, 0, NOW, ())
    for bad in (dict(serial=0), dict(schema="x"), dict(expires="tomorrow"), dict(entries=[entry(arch="x")]),
                dict(entries=[entry(tier="Bad Tier")]), dict(version=2)):
        raw, sig = signed(key, dict(index(), **bad))
        with pytest.raises(I.IndexError_):
            I.verify(raw, sig, 0, NOW, (key.kid,))


def test_cli(key, tmp_path, capsys):
    raw, sig = signed(key, index(expires="2999-01-01T00:00:00Z"))
    (tmp_path / "index.json").write_bytes(raw)
    (tmp_path / "index.sig").write_bytes(sig)
    base = ["--index", str(tmp_path / "index.json"), "--sig", str(tmp_path / "index.sig")]
    t = (key.kid,)
    assert I.main(["verify", *base], t) == 0
    assert json.loads(capsys.readouterr().out)["serial"] == 5
    m = ["match", *base, "--arch", "sm_120a", "--file", f"m.gguf={H('a')}"]
    assert I.main(m, t) == 0
    assert json.loads(capsys.readouterr().out)["manifest_sha256"] == H("b")
    assert I.main(["verify", *base, "--last-serial", "6"], t) == 1
    assert "rollback" in capsys.readouterr().err
    assert I.main(m[:-1] + [f"m.gguf={H('d')}"], t) == 3
    assert I.main(["verify", *base], ()) == 1                      # no pinned key
    assert I.main(["verify", *base, "--sig", str(tmp_path / "none")], t) == 1
    assert sign.TRUSTED_KEY_IDS != (key.kid,)


def test_version_matches_package_metadata():
    import re
    import goethe_engine
    assert re.search(r'^version = "(.+)"', open(__file__.replace("tests/test_index.py", "pyproject.toml")).read(), re.M)[1] == goethe_engine.__version__
