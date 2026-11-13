import json
from datetime import datetime, timezone

import pytest

from goethe_engine import index as I
from goethe_engine import sign, structure
from toypack import ALT, TARGET, gguf

H = lambda c: c * 64
NOW = datetime(2026, 11, 11, tzinfo=timezone.utc)
KEYS = ["general.architecture"]


def entry(variants=(("q4", H("a")),), **kw):
    e = dict(model=dict(hf_repo="org/m", revision="r1", format="gguf",
                        structure=[dict(variant=v, sha256=h, keys=KEYS) for v, h in variants]),
             arch="sm_120a", abi_major=3, runtime=dict(kind="gc-c", version=1), tier="base", file="m.gcpack", manifest_sha256=H("b"),
             sig=dict(schema="gc-sig-1"), size=10, min_engine="0.2.0")
    e.update(kw)
    return e


def index(serial=5, expires="2026-12-01T00:00:00Z", entries=None):
    return dict(schema="gc-pack-index", version=3, serial=serial, issued="2026-11-01T00:00:00Z", expires=expires,
                entries=[entry()] if entries is None else entries)


def signed(key, ix):
    raw = json.dumps(ix).encode()
    return raw, key.sign(raw)


def model_dir(d, tensors, name="m.gguf"):
    d.mkdir(exist_ok=True)
    gguf(str(d / name), tensors)
    return str(d)


def sha(d):
    return structure.structure([str(p) for p in sorted(__import__("pathlib").Path(d).glob("*.gguf"))], KEYS)["sha256"]


def test_verify_and_match_by_structure(key, tmp_path):
    ref = model_dir(tmp_path / "ref", TARGET)
    other = model_dir(tmp_path / "other", ALT)
    raw, sig = signed(key, index(entries=[
        entry(variants=(("q4", sha(ref)),)), entry(variants=(("alt", sha(other)), ("q4", sha(ref))), tier="a", manifest_sha256=H("d")),
        entry(variants=(("q4", sha(ref)),), runtime=dict(kind="other", version=1)), entry(variants=(("q4", sha(ref)),), arch="sm_90a"),
        entry(variants=(("q4", sha(ref)),), min_engine="9.0.0")]))
    ix, kid = I.verify(raw, sig, 5, NOW, (key.kid,))
    assert kid == key.kid and ix["serial"] == 5
    hit = I.match_dir(ix, ref, "sm_120a", "0.2.0")
    assert [(e["tier"], e["variant"]) for e in hit] == [("a", "q4"), ("base", "q4")]       # runtime, arch and min_engine filtered; tier orders
    assert [(e["tier"], e["variant"]) for e in I.match_dir(ix, other, "sm_120a")] == [("a", "alt")]
    assert I.match_dir(ix, ref, "sm_120a", tier="base")[0]["tier"] == "base"


def test_a_finetune_matches_and_changed_structure_does_not(key, tmp_path):
    ref = model_dir(tmp_path / "ref", TARGET)
    ix = dict(index(entries=[entry(variants=(("q4", sha(ref)),))]))
    ft = tmp_path / "ft"
    ft.mkdir()
    gguf(str(ft / "finetune-renamed.gguf"), list(reversed(TARGET)), align=64)      # another name, order and alignment
    assert I.match_dir(ix, str(ft), "sm_120a", "0.2.0")
    for bad in (ALT, [("a.weight", [64, 8], 12), TARGET[1]], TARGET + [("c.weight", [4], 0)], TARGET[:1]):
        assert I.match_dir(ix, model_dir(tmp_path / f"bad{len(str(bad))}", bad), "sm_120a", "0.2.0") == []
    (tmp_path / "empty").mkdir()
    assert I.match_dir(ix, str(tmp_path / "empty"), "sm_120a", "0.2.0") == []


def test_rollback_and_freeze_refused(key):
    raw, sig = signed(key, index(serial=4))
    with pytest.raises(I.IndexError_, match="rollback"):
        I.verify(raw, sig, 5, NOW, (key.kid,))
    I.verify(raw, sig, 4, NOW, (key.kid,))   # equal serial is fine
    with pytest.raises(I.IndexError_, match="expired"):
        I.verify(raw, sig, 0, datetime(2026, 12, 2, tzinfo=timezone.utc), (key.kid,))


def test_bad_signature_and_entries_refused(key):
    raw, sig = signed(key, index())
    with pytest.raises(I.IndexError_, match="does not verify"):
        I.verify(raw + b" ", sig, 0, NOW, (key.kid,))
    with pytest.raises(I.IndexError_, match="allow-list"):
        I.verify(raw, sig, 0, NOW, ())
    e = entry()
    for bad in (dict(serial=0), dict(schema="x"), dict(expires="tomorrow"), dict(entries=[dict(e, arch="x")]),
                dict(entries=[dict(e, tier="Bad Tier")]), dict(version=2), dict(version=4),
                dict(entries=[dict(e, file="../m.gcpack")]), dict(entries=[dict(e, file="m.bin")]), dict(entries=[dict(e, file="mödel.gcpack")]),
                dict(entries=[dict(e, manifest_sha256="zz")]), dict(entries=[dict(e, abi_major=2)]),
                dict(entries=[dict(e, model=dict(e["model"], structure=[]))]), dict(entries=[{k: v for k, v in e.items() if k != "size"}])):
        raw, sig = signed(key, dict(index(), **bad))
        with pytest.raises(I.IndexError_):
            I.verify(raw, sig, 0, NOW, (key.kid,))


def test_cli(key, tmp_path, capsys):
    ref = model_dir(tmp_path / "ref", TARGET)
    raw, sig = signed(key, index(expires="2999-01-01T00:00:00Z", entries=[entry(variants=(("q4", sha(ref)),))]))
    (tmp_path / "index.json").write_bytes(raw)
    (tmp_path / "index.sig").write_bytes(sig)
    base = ["--index", str(tmp_path / "index.json"), "--sig", str(tmp_path / "index.sig")]
    t = (key.kid,)
    assert I.main(["verify", *base], t) == 0
    assert json.loads(capsys.readouterr().out)["serial"] == 5
    m = ["match", *base, "--arch", "sm_120a", "--model-dir", ref]
    assert I.main(m, t) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["manifest_sha256"] == H("b") and j["variant"] == "q4" and j["file"] == "m.gcpack"
    assert I.main(["verify", *base, "--last-serial", "6"], t) == 1
    assert "rollback" in capsys.readouterr().err
    assert I.main(["match", *base, "--arch", "sm_90a", "--model-dir", ref], t) == 3
    assert I.main(["match", *base, "--arch", "sm_120a", "--model-dir", model_dir(tmp_path / "x", ALT)], t) == 3
    assert I.main(["verify", *base], ()) == 1                      # no pinned key
    assert I.main(["verify", *base, "--sig", str(tmp_path / "none")], t) == 1
    assert sign.TRUSTED_KEY_IDS != (key.kid,)


def test_version_matches_package_metadata():
    import re
    import goethe_engine
    assert re.search(r'^version = "(.+)"', open(__file__.replace("tests/test_index.py", "pyproject.toml")).read(), re.M)[1] == goethe_engine.__version__
