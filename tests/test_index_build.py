"""index build: one entry per (arch, target variant) whose model is that variant's source and files, the paired drafter under model.with;
the written index verifies and matches; a pack without per-variant sources is refused."""
import json
import os
import shutil

import pytest

from goethe_engine import build as B, index as I, pack as P
from test_build import spec_dir
from toypack import parts, toy_pack

EXPIRES = "2099-01-01T00:00:00Z"


def spec_pack(tmp_path, key, lib, arches=(("mock", "sm_120a", "12.0"),), unpaired_target=False, attested=True):
    """A built, signed pack with target q4 (a verify bucket accepting drafts), drafter d and one speculative pair, on arches;
    unpaired_target adds target q8 with no pair."""
    sp = spec_dir(str(tmp_path / "s"), lib)
    spec = json.load(open(sp))
    spec["skus"] = [dict(spec["skus"][0], sku=s, arch=a, cc=cc) for s, a, cc in arches]
    spec["buckets"] = [dict(spec["buckets"][1], suffix="8v.256", io=dict(accept=True), batch=False),
                       dict(spec["buckets"][0], suffix="1.256", role="drafter", variant="d", ncol=1, batch=False)]
    spec["spec"] = [dict(name="ar", target="K1.8v.256", drafter="K1.1.256", policy=dict(kind="ar", K="3"))]
    if unpaired_target:
        q4 = next(v for v in spec["variants"] if v["id"] == "q4")
        spec["variants"].append(dict(q4, id="q8", quant="q8_0", files=[q4["files"][0].replace("w.gguf", "w8.gguf")]))
        open(os.path.join(tmp_path / "s", spec["variants"][-1]["files"][0]), "wb").write(open(os.path.join(tmp_path / "s", q4["files"][0]), "rb").read() + b"\0")
        spec["buckets"].append(dict(spec["buckets"][0], suffix="8v.256.q8", variant="q8"))
    json.dump(spec, open(sp, "w"))
    out = str(tmp_path / "toy.gcpack")
    B.build(sp, out, key.priv, cuda="12.8")
    if attested:
        attest(out)
    return out, tmp_path / "s"


def test_entries_per_arch_and_target_with_the_drafter_and_the_index_verifies_and_matches(tmp_path, key, mocklib):
    pack, d = spec_pack(tmp_path, key, mocklib, arches=(("a", "sm_120a", "12.0"), ("b", "sm_89", "8.9")))
    ix = I.build([pack], 9, EXPIRES, key.priv, str(tmp_path / "site"), trusted=(key.kid,))
    assert [(e["arch"], e["model"]["structure"][0]["variant"]) for e in ix["entries"]] == [("sm_120a", "q4"), ("sm_89", "q4")]
    m = ix["entries"][0]["model"]
    assert (m["hf_repo"], m["format"], [f["name"] for f in m["files"]]) == ("org/toy-GGUF", "gguf", ["w.gguf"])
    assert [(w["role"], w["variant"], [f["name"] for f in w["files"]]) for w in m["with"]] == [("drafter", "d", ["d.gguf"])]
    raw, sig = (open(tmp_path / "site" / n, "rb").read() for n in ("index.json", "index.sig"))
    vix, _ = I.verify(raw, sig, trusted=(key.kid,))
    hit = I.match_dir(vix, str(d), "sm_120a")
    assert hit and hit[0]["variant"] == "q4" and hit[0]["file"] == "toy.gcpack"


def test_a_target_lists_only_the_drafters_paired_with_it(tmp_path, key, mocklib):
    pack, _ = spec_pack(tmp_path, key, mocklib, unpaired_target=True)
    assert {e["model"]["structure"][0]["variant"]: [w["variant"] for w in e["model"].get("with", [])] for e in I.entries(pack, trusted=(key.kid,))} == \
        {"q4": ["d"], "q8": []}


def test_a_pack_without_sources_is_refused_and_nothing_is_written(tmp_path, key, mocklib):
    m, data = parts(toy_pack(str(tmp_path / "src"), key, mocklib, sign=False)[0])
    old = str(tmp_path / "old.gcpack")
    P.write(old, m, data.values())
    key.sign_pack(old)
    with pytest.raises(I.IndexError_, match="no source"):
        I.build([old], 9, EXPIRES, key.priv, str(tmp_path / "site"), trusted=(key.kid,))
    assert not os.path.exists(tmp_path / "site")


def test_model_with_must_name_drafters_with_their_files(tmp_path, key, mocklib):
    pack, _ = spec_pack(tmp_path, key, mocklib)
    e = I.entries(pack, trusted=(key.kid,))[0]
    I.validate_entry(e, 0)
    bad = dict(e, model=dict(e["model"], **{"with": [dict(e["model"]["with"][0], role="projector")]}))
    with pytest.raises(I.IndexError_, match="model.with"):
        I.validate_entry(bad, 0)


def test_index_build_cli_then_verify(tmp_path, key, mocklib, capsys):
    pack, _ = spec_pack(tmp_path, key, mocklib)
    site = str(tmp_path / "site")
    assert I.main(["build", "--pack", pack, "--serial", "9", "--expires", EXPIRES, "--key", key.priv, "--out", site], trusted=(key.kid,)) == 0
    assert json.loads(capsys.readouterr().out)["entries"] == 1
    assert I.main(["verify", "--index", os.path.join(site, "index.json"), "--sig", os.path.join(site, "index.sig")], trusted=(key.kid,)) == 0


def test_a_pack_not_signed_by_a_trusted_key_is_refused(tmp_path, key, mocklib):
    pack, _ = spec_pack(tmp_path, key, mocklib)
    with pytest.raises(P.PackError, match="key"):
        I.entries(pack, trusted=("0" * 16,))
    open(pack + ".sig", "w").write("{}")
    with pytest.raises(P.PackError):
        I.entries(pack, trusted=(key.kid,))



def test_a_current_index_keeps_its_other_entries_and_the_serial_must_go_up(tmp_path, key, mocklib):
    toy, _ = spec_pack(tmp_path, key, mocklib)
    other = str(tmp_path / "other.gcpack")
    for ext in ("", ".sig", ".check.json"):
        shutil.copy(toy + ext, other + ext)
    I.build([other, toy], 9, EXPIRES, key.priv, str(tmp_path / "v9"), trusted=(key.kid,))
    base = tuple(open(tmp_path / "v9" / n, "rb").read() for n in ("index.json", "index.sig"))
    ix = I.build([toy], 10, EXPIRES, key.priv, str(tmp_path / "v10"), trusted=(key.kid,), base=base)
    assert sorted(e["file"] for e in ix["entries"]) == ["other.gcpack", "toy.gcpack"], "the rebuilt pack's entry is replaced, the other kept"
    with pytest.raises(I.IndexError_, match="not above"):
        I.build([toy], 9, EXPIRES, key.priv, str(tmp_path / "v9b"), trusted=(key.kid,), base=base)
    with pytest.raises(I.IndexError_):
        I.build([toy], 11, EXPIRES, key.priv, str(tmp_path / "v11"), trusted=(key.kid,), base=(base[0] + b" ", base[1]))
    assert not os.path.exists(tmp_path / "v9b") and not os.path.exists(tmp_path / "v11")


def test_pack_sign_writes_a_signature_that_verifies(tmp_path, key, mocklib, capsys):
    toy, _ = spec_pack(tmp_path, key, mocklib)
    os.remove(toy + ".sig")
    P.main(["sign", toy, "--key", key.priv])
    assert json.loads(capsys.readouterr().out)["keyid"] == key.kid
    P.verify(toy, trusted=(key.kid,))



def attest(pack, failures=(), arches=None):
    """A pack check record for each arch of the pack (or those given), as pack check writes it on each arch's device."""
    import hashlib
    raw = P.read_container_v(open(pack, "rb"), os.path.getsize(pack))[3]
    m = json.loads(raw)
    rec = {b["requires"]["arch"]: dict(pairs=[s["target"] for s in m["spec"]], failures=list(failures)) for b in m["buckets"]
           if arches is None or b["requires"]["arch"] in arches}
    open(pack + ".check.json", "w").write(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), arches=rec)))


def test_a_spec_pack_is_listed_only_on_the_arches_its_check_passed(tmp_path, key, mocklib):
    pack, _ = spec_pack(tmp_path, key, mocklib, arches=(("a", "sm_120a", "12.0"), ("b", "sm_89", "8.9")), attested=False)
    for fails in (None, ["t: seeds give the same tokens"]):
        if fails:
            attest(pack, fails)
        with pytest.raises(I.IndexError_, match="no target has a passing pack check"):
            I.entries(pack, trusted=(key.kid,))
    attest(pack, arches=("sm_120a",))
    left = []
    assert [e["arch"] for e in I.entries(pack, trusted=(key.kid,), left=left)] == ["sm_120a"] and left == [("q4", "sm_89")]


def test_a_dropped_pack_leaves_the_index_and_a_drop_must_name_one_of_it(tmp_path, key, mocklib):
    old, _ = spec_pack(tmp_path / "o", key, mocklib)
    new, _ = spec_pack(tmp_path / "n", key, mocklib)
    os.rename(new, str(tmp_path / "n" / "new.gcpack"))
    for ext in (".sig", ".check.json"):
        os.rename(new + ext, str(tmp_path / "n" / "new.gcpack") + ext)
    new = str(tmp_path / "n" / "new.gcpack")
    I.build([old], 9, EXPIRES, key.priv, str(tmp_path / "v9"), trusted=(key.kid,))
    base = tuple(open(tmp_path / "v9" / n, "rb").read() for n in ("index.json", "index.sig"))
    ix = I.build([new], 10, EXPIRES, key.priv, str(tmp_path / "v10"), trusted=(key.kid,), base=base, drop=["toy.gcpack"])
    assert {e["file"] for e in ix["entries"]} == {"new.gcpack"}
    with pytest.raises(I.IndexError_, match="not in the current index"):
        I.build([new], 10, EXPIRES, key.priv, str(tmp_path / "v10b"), trusted=(key.kid,), base=base, drop=["nope.gcpack"])


@pytest.mark.parametrize("mode,ok", [("sample", True), ("argmax", False), (None, False)])
def test_pack_check_passes_a_round_that_samples_and_fails_one_that_ignores_or_refuses_it(tmp_path, key, mocklib, monkeypatch, mode, ok):
    if mode:
        monkeypatch.setenv("GCK_MOCK_SPEC", mode)
    alone = lambda m: m["buckets"].append(dict(m["buckets"][0], id="target.mock.K1.2.256", ncol=2, batch=True, io=None))
    p, w, dw = toy_pack(str(tmp_path), key, mocklib, spec=True, mutate=alone)
    r = B.check(p, w, dw, trusted=(key.kid,), sku="mock")
    assert r["pairs"] and (not r["failures"]) == ok, r
