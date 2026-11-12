"""Fail-closed load of a signed pack: signature, pinned key, hashes, ABI, arch, runtime."""
import json
import os

import pytest

from goethe_engine import pack as P
from goethe_engine import sign
from goethe_engine.pack_backend import caps_of_bucket, numerics_of
from toypack import toy_pack


def test_loader_is_the_pinned_abi():
    assert P.ABI == "2.0.0" and P.SCHEMA == "gc-pack" and list(P.RUNTIMES) == ["gc-c"]


def test_caps_and_numerics_of_a_bucket(toy, key):
    m = P.verify(toy[0], trusted=(key.kid,))
    b = P.select(m, sku="mock", B=8)
    assert numerics_of(m, b) == dict(contract="K1", bodies={}, desc=None)
    assert caps_of_bucket(b)["tree"] and caps_of_bucket(b)["taps"] == {"result_norm": 4}


def test_other_majors_refused_newer_minor_read():
    with pytest.raises(P.PackError, match="major"):
        P.validate(dict(schema=P.SCHEMA, abi="1.1.0"))
    with pytest.raises(P.PackError):
        P.validate(dict(schema="other", abi="2.0.0"))
    with pytest.raises(P.PackError, match="lacks"):
        P.validate(dict(schema=P.SCHEMA, abi="2.7.0"))  # a newer minor parses; this one lacks its fields


# --- fail-closed load: signature, key, hashes, ABI, arch, runtime

def test_toy_pack_loads_and_verifies(toy, key):
    m = P.verify(toy[0], trusted=(key.kid,), arch="sm_120a")
    assert m["runtime"]["kind"] == "gc-c"


def test_default_allow_list_is_pinned_and_refuses(toy):
    assert isinstance(sign.TRUSTED_KEY_IDS, tuple)
    with pytest.raises(P.PackError, match="allow-list"):
        P.verify(toy[0])


def test_missing_and_bad_signature_refused(toy, key):
    sig = toy[0] + ".sig"
    good = open(sig, "rb").read()
    j = json.loads(good)
    j["sig"] = j["sig"][:-8] + "AAAAAAA="
    open(sig, "wb").write(json.dumps(j).encode())
    with pytest.raises(P.PackError, match="does not verify|malformed"):
        P.verify(toy[0], trusted=(key.kid,))
    os.remove(sig)
    with pytest.raises(P.PackError, match="no signature"):
        P.verify(toy[0], trusted=(key.kid,))


def test_any_tampered_byte_refused(toy, key):
    raw = bytearray(open(toy[0], "rb").read())
    for i in (0, len(raw) // 2, len(raw) - 1):
        t = bytearray(raw)
        t[i] ^= 1
        open(toy[0], "wb").write(t)
        with pytest.raises(P.PackError, match="does not verify"):
            P.verify(toy[0], trusted=(key.kid,))
    open(toy[0], "ab").write(b" ")
    with pytest.raises(P.PackError, match="does not verify"):
        P.verify(toy[0], trusted=(key.kid,))


def test_unpinned_key_refused(toy, key):
    with pytest.raises(P.PackError, match="allow-list"):
        P.verify(toy[0], trusted=("0" * 64,))


def test_blob_not_hashing_to_its_name_refused_even_if_signed(tmp_path, key, mocklib):
    from toypack import gcpack
    d = str(tmp_path)
    pack = toy_pack(d, key, mocklib)[0]
    raw = open(pack, "rb").read()
    n = int.from_bytes(raw[8:16], "little")
    hdr = json.loads(raw[16:16 + n])
    base = 16 + n
    m = raw[base:base + hdr["manifest"][1]]
    blobs = {h: raw[base + o:base + o + l] for h, (o, l) in hdr["blobs"].items()}
    h = sorted(blobs)[0]
    blobs[h] = blobs[h] + b"x"
    gcpack(pack, m, blobs)
    key.sign_file(pack)
    with pytest.raises(P.PackError, match="does not hash"):
        P.verify(pack, trusted=(key.kid,))
    gcpack(pack, m, {k: v for k, v in blobs.items() if k != h})
    key.sign_file(pack)
    with pytest.raises(P.PackError, match="missing"):
        P.verify(pack, trusted=(key.kid,))


def test_not_a_gcpack_refused(tmp_path, key):
    f = str(tmp_path / "x.gcpack")
    open(f, "wb").write(b"not a pack at all")
    key.sign_file(f)
    with pytest.raises(P.PackError, match="magic"):
        P.verify(f, trusted=(key.kid,))


def test_arch_abi_runtime_refused(tmp_path, key, mocklib):
    t = (key.kid,)
    d = str(tmp_path / "a")
    os.makedirs(d)
    pack = toy_pack(d, key, mocklib)[0]
    with pytest.raises(P.PackError, match="arch"):
        P.verify(pack, trusted=t, arch="sm_90a")
    for kw, pat in ((dict(abi="1.1.0"), "major"), (dict(runtime="mystery-rt"), "runtime")):
        d = str(tmp_path / pat)
        os.makedirs(d)
        with pytest.raises(P.PackError, match=pat):
            P.verify(toy_pack(d, key, mocklib, **kw)[0], trusted=t)


def test_dev_override_is_loud_and_off_by_default(toy, key, monkeypatch, capsys):
    os.remove(toy[0] + ".sig")
    with pytest.raises(P.PackError):
        P.verify(toy[0])
    monkeypatch.setenv(sign.ALLOW_UNSIGNED_ENV, "1")
    assert P.verify(toy[0])["arch"] == "sm_120a"
    assert "NOT authenticated" in capsys.readouterr().err


def test_reader_has_no_writer():
    assert not any(hasattr(P, n) for n in ("write", "add_blob", "bucket_entry", "bucket_files", "RUNTIME", "load", "blobs"))


# --- staging: verify then load the same bytes

def test_stage_is_private_and_independent_of_the_source(toy, key):
    import stat, hashlib
    st = P.stage(toy[0], trusted=(key.kid,))
    assert stat.S_IMODE(os.stat(st.dir).st_mode) == 0o700
    assert st.sha256 == hashlib.sha256(open(toy[0], "rb").read()).hexdigest() and st.keyid == key.kid
    assert not os.path.exists(os.path.join(st.dir, "pack.gcpack"))
    open(toy[0], "ab").write(b"tamper")   # the source changes after staging
    h = sorted(os.listdir(os.path.join(st.dir, "blobs")))[0]
    assert hashlib.sha256(open(os.path.join(st.dir, "blobs", h), "rb").read()).hexdigest() == h
    d = P.materialize(st, P.select(st.manifest, sku="mock", B=2))
    assert d.startswith(st.dir)
    st.close()
    assert not os.path.exists(st.dir or "/nonexistent-dir")


def test_failed_stage_leaves_no_dir(toy, key, tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile
    tempfile.tempdir = None
    os.remove(toy[0] + ".sig")
    with pytest.raises(P.PackError):
        P.stage(toy[0], trusted=(key.kid,))
    assert [d for d in os.listdir(tmp_path) if d.startswith("goethe-engine-")] == []
    tempfile.tempdir = None


def test_symlinks_refused(toy, key, tmp_path):
    os.symlink(toy[0], str(tmp_path / "l.gcpack"))
    os.symlink(toy[0] + ".sig", str(tmp_path / "l.gcpack.sig"))
    with pytest.raises(P.PackError, match="symlink"):
        P.stage(str(tmp_path / "l.gcpack"), trusted=(key.kid,))
    os.rename(toy[0] + ".sig", str(tmp_path / "s"))
    os.symlink(str(tmp_path / "s"), toy[0] + ".sig")
    with pytest.raises(P.PackError, match="symlink"):
        P.stage(toy[0], trusted=(key.kid,))


@pytest.mark.parametrize("name", ["../x", "/abs/x", "a/../b", "a//b", "./a", "a/./b", "a\\b", "", "a\0b"])
def test_traversal_names_refused(tmp_path, key, mocklib, name):
    def bad(m):
        for b in m["buckets"]:
            b["files"][name] = b["files"]["libgck.so"]
    with pytest.raises(P.PackError, match="bad file entry"):
        P.verify(toy_pack(str(tmp_path), key, mocklib, mutate=bad)[0], trusted=(key.kid,))


def test_oversized_json_refused_before_parsing(toy, key, monkeypatch):
    monkeypatch.setattr(P, "MAX_JSON", 100)
    with pytest.raises(P.PackError, match="larger than"):
        P.stage(toy[0], trusted=(key.kid,))
    monkeypatch.undo()
    open(toy[0] + ".sig", "ab").write(b" " * sign.MAX_SIG)
    with pytest.raises(P.PackError, match="larger than|too large"):
        P.stage(toy[0], trusted=(key.kid,))


def test_cli_verify_prints_json(toy, key, monkeypatch, capsys):
    monkeypatch.setattr(sign, "TRUSTED_KEY_IDS", (key.kid,))
    P.main(["verify", toy[0], "--arch", "sm_120a"])
    j = json.loads(capsys.readouterr().out)
    assert j["ok"] and j["keyid"] == key.kid and j["buckets"] == 2 and len(j["sha256"]) == 64 and j["weights"] == ["w.gguf", "d.gguf"]
    with pytest.raises(SystemExit, match="REFUSED"):
        P.main(["verify", toy[0], "--arch", "sm_90a"])


def test_weights_must_not_be_world_writable(toy, key):
    m = P.verify(toy[0], trusted=(key.kid,))
    P.check_weights(m, [toy[1], toy[2]])
    os.chmod(toy[1], 0o666)
    with pytest.raises(P.PackError, match="world-writable"):
        P.check_weights(m, [toy[1], toy[2]])


def test_pinned_key_is_the_release_key():
    pem = ("-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEcvBeilUyHdQbvn9eHVjUZkn0Og4n\n"
           "QvHAoaZr1KzcHGsBptYLZktabraS2V1K6eC6lDlagNe/6mOmPCcYLkK4kg==\n-----END PUBLIC KEY-----\n")
    assert sign.key_id(pem.encode()) in sign.TRUSTED_KEY_IDS


def test_contract_and_body_labels_are_opaque():
    def bk(i, c, **kw):
        return dict(id=i, role="target", sku="s", contract=c, ncol=1, batch=False, maxctx=8, bodies=kw.get("bodies"))
    m = dict(contracts=["Zq", "Yp"], runtime=dict(kind="gc-c", version=1), buckets=[bk("a", "Yp"), bk("b", "Zq", bodies={"f": "w9"})],
             contract_desc={"Zq": "text"})
    b = P.select(m, sku="s")
    assert b["id"] == "b"                                    # default = first label the pack lists, any string
    assert P.select(m, sku="s", contract="Yp")["id"] == "a"  # exact-string match
    assert numerics_of(m, b) == dict(contract="Zq", bodies={"f": "w9"}, desc="text")
    assert P.usable(m, b) == ""


def test_pair_target_advertises_device_acceptance_by_io_or_spec_caps(tmp_path, key, mocklib):
    """gc-c buckets carry no io: the pair's target says it through spec_caps.accept (pack ABI 1.1, additive)."""
    def via(how):
        def mut(m):
            t = m["buckets"][0]
            t["io"] = None
            t["spec_caps"] = dict(t["spec_caps"], accept=how)
        return mut
    for i, (how, ok) in enumerate((([], False), (["exact"], True))):
        pack, _, _ = toy_pack(str(tmp_path / str(i)), key, mocklib, spec=True, mutate=via(how))
        if ok:
            assert P.verify(pack, trusted=(key.kid,))["spec"]
        else:
            with pytest.raises(P.PackError, match="device acceptance"):
                P.verify(pack, trusted=(key.kid,))


def test_prefill_bucket_is_beside_the_ladder_never_in_it(tmp_path, key, mocklib):
    """A non-batch W column bucket marked "prefill" is found by P.prefill (widest wins) and is not a ladder step or a
    selectable decode bucket."""
    def mut(m):
        b = dict(m["buckets"][1], id="target.mock.K1.8v.256", batch=False, prefill=8)
        w = dict(m["buckets"][1], id="target.mock.K1.32v.256", ncol=32, batch=False, prefill=32)
        m["buckets"] += [b, w]
    pack, _, _ = toy_pack(str(tmp_path), key, mocklib, mutate=mut)
    m = P.verify(pack, trusted=(key.kid,))
    b = P.select(m, sku="mock", B=8)
    assert b["id"] == "target.mock.K1.8.256"
    assert [x["id"] for x in P.ladder(m, b)] == ["target.mock.K1.2.256", "target.mock.K1.8.256"]
    assert P.prefill(m, b)["id"] == "target.mock.K1.32v.256"
    assert P.prefill(m, dict(b, maxctx=512)) is None


def test_single_stream_bucket_is_one_non_batch_column_of_any_contract():
    """B 1 takes the pack's 1 column non-batch bucket, whatever its contract is called; a batch bucket is never one, and an
    explicit contract is honored."""
    def bk(i, c, n, batch):
        return dict(id=i, role="target", sku="s", contract=c, ncol=n, batch=batch, maxctx=8)
    m = dict(contracts=["Zq", "Yp"], runtime=dict(kind="gc-c", version=1),
             buckets=[bk("z2", "Zq", 2, True), bk("z8", "Zq", 8, True), bk("y1", "Yp", 1, False)])
    assert P.single_stream(m, sku="s")["id"] == "y1"
    assert P.single_stream(m, sku="s", contract="Zq") is None
    assert P.select(m, sku="s", B=2)["id"] == "z2"           # the default contract still serves several sequences
    m["buckets"].pop()
    assert P.single_stream(m, sku="s") is None
