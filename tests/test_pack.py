"""Fail-closed load of a signed v3 pack: manifest signature, pinned key, blob hashes, variants, ABI, arch, runtime, partial files."""
import json
import os

import pytest

from goethe_engine import pack as P
from goethe_engine import sign
from goethe_engine.pack_backend import caps_of_bucket, numerics_of
from toypack import toy_pack


def test_loader_is_the_pinned_abi():
    assert P.ABI == "3.0.0" and P.MAGIC == b"GCPACK3\n" and P.SCHEMA == "gc-pack" and list(P.RUNTIMES) == ["gc-c"]


def test_caps_and_numerics_of_a_bucket(toy, key):
    m = P.verify(toy[0], trusted=(key.kid,))
    b = P.select(m, sku="mock", B=8)
    assert numerics_of(m, b) == dict(contract="K1", bodies={}, desc=None)
    assert caps_of_bucket(b)["tree"] and caps_of_bucket(b)["taps"] == {"result_norm": 4}


def test_other_majors_refused_newer_minor_read():
    with pytest.raises(P.PackError, match="major"):
        P.validate(dict(schema=P.SCHEMA, abi="2.1.0"))
    with pytest.raises(P.PackError):
        P.validate(dict(schema="other", abi="3.0.0"))
    with pytest.raises(P.PackError, match="lacks"):
        P.validate(dict(schema=P.SCHEMA, abi="3.7.0"))  # a newer minor parses; this one lacks its fields


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


def _rewrite(pack, edit):
    """Re-write the pack file with (header, manifest bytes, blobs) edited by edit(hdr, manifest, blobs) -> (hdr or None, manifest, blobs)."""
    from toypack import gcpack
    raw = open(pack, "rb").read()
    n = int.from_bytes(raw[8:16], "little")
    hdr = json.loads(raw[16:16 + n])
    base = 16 + n
    man = raw[base + hdr["manifest"][0]:base + hdr["manifest"][0] + hdr["manifest"][1]]
    blobs = {h: raw[base + o:base + o + l] for h, (o, l, *_) in hdr["blobs"].items()}
    man, blobs = edit(man, blobs)
    gcpack(pack, man, blobs)


def test_altered_manifest_refused(toy, key):
    _rewrite(toy[0], lambda man, blobs: (man.replace(b'"ncol": 8', b'"ncol": 9'), blobs))
    with pytest.raises(P.PackError, match="does not verify"):
        P.verify(toy[0], trusted=(key.kid,))


def test_swapped_manifest_signature_refused(tmp_path, key, mocklib):
    a = toy_pack(str(tmp_path / "a"), key, mocklib)[0]
    b = toy_pack(str(tmp_path / "b"), key, mocklib, mutate=lambda m: m["compiler"].update(commit="1" * 40))[0]
    os.replace(b + ".sig", a + ".sig")                 # a valid signature of another manifest
    with pytest.raises(P.PackError, match="does not verify"):
        P.verify(a, trusted=(key.kid,))


def test_altered_container_header_refused(toy, key):
    raw = bytearray(open(toy[0], "rb").read())
    n = int.from_bytes(raw[8:16], "little")
    hdr = json.loads(bytes(raw[16:16 + n]))
    h = sorted(hdr["blobs"])[0]
    hdr["blobs"][h][0] += 1                            # an altered offset: the bytes read no longer hash to the blob's name
    hb = json.dumps(hdr).encode()
    open(toy[0], "wb").write(raw[:8] + len(hb).to_bytes(8, "little") + hb + raw[16 + n:])
    with pytest.raises(P.PackError, match="does not hash|out of range|truncated"):
        P.verify(toy[0], trusted=(key.kid,))


def test_a_v2_container_is_not_a_pack(tmp_path, key):
    f = str(tmp_path / "x.gcpack")
    open(f, "wb").write(b"GCPACK2\n" + (2).to_bytes(8, "little") + b"{}")
    with pytest.raises(P.PackError, match="magic"):
        P.verify(f, trusted=(key.kid,))


def test_unpinned_key_refused(toy, key):
    with pytest.raises(P.PackError, match="allow-list"):
        P.verify(toy[0], trusted=("0" * 64,))


def test_blob_not_hashing_to_its_name_refused_even_if_signed(tmp_path, key, mocklib):
    pack = toy_pack(str(tmp_path), key, mocklib)[0]

    def flip(man, blobs):
        h = sorted(blobs)[0]
        return man, dict(blobs, **{h: blobs[h] + b"x"})
    _rewrite(pack, flip)
    with pytest.raises(P.PackError, match="does not hash"):
        P.verify(pack, trusted=(key.kid,))

    def drop(man, blobs):
        return man, {k: v for k, v in blobs.items() if k != sorted(blobs)[0]}
    _rewrite(pack, drop)
    with pytest.raises(P.PackError, match="missing"):
        P.verify(pack, trusted=(key.kid,))


def test_not_a_gcpack_refused(tmp_path, key):
    f = str(tmp_path / "x.gcpack")
    open(f, "wb").write(b"not a pack at all")
    with pytest.raises(P.PackError, match="magic"):
        P.verify(f, trusted=(key.kid,))


def test_arch_abi_runtime_refused(tmp_path, key, mocklib):
    t = (key.kid,)
    d = str(tmp_path / "a")
    os.makedirs(d)
    pack = toy_pack(d, key, mocklib)[0]
    with pytest.raises(P.PackError, match="arch"):
        P.verify(pack, trusted=t, arch="sm_90a")
    for kw, pat in ((dict(abi="2.0.0"), "major"), (dict(runtime="mystery-rt"), "runtime")):
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
    assert st.keyid == key.kid and len(st.manifest_sha256) == 64
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
    assert j["ok"] and j["keyid"] == key.kid and j["buckets"] == 2 and len(j["manifest_sha256"]) == 64 and j["variants"] == ["alt", "d", "q4"]
    with pytest.raises(SystemExit, match="REFUSED"):
        P.main(["verify", toy[0], "--arch", "sm_90a"])


def test_weights_must_not_be_world_writable(toy, key):
    m = P.for_variant(P.verify(toy[0], trusted=(key.kid,)), "q4")
    P.check_weights(m, [toy[1]])
    os.chmod(toy[1], 0o666)
    with pytest.raises(P.PackError, match="world-writable"):
        P.check_weights(m, [toy[1]])


def test_the_weights_choose_the_variant_by_structure(tmp_path, key, mocklib):
    pack, w, dw = toy_pack(str(tmp_path), key, mocklib, alt=True)
    m = P.verify(pack, trusted=(key.kid,))
    alt = str(tmp_path / "alt.gguf")
    assert P.pick_variant(m, [w]) == "q4" and P.pick_variant(m, [alt]) == "alt"
    from toypack import gguf
    gguf(str(tmp_path / "other.gguf"), [("a.weight", [64, 9], 12), ("b.weight", [64], 0)])      # another tensor type: no variant
    with pytest.raises(P.PackError, match="a.weight"):
        P.pick_variant(m, [str(tmp_path / "other.gguf")])
    mv = P.for_variant(m, "alt")
    assert {b["variant"] for b in mv["buckets"]} == {"alt"} and P.select(mv, sku="mock", B=2)["id"].endswith(".alt")
    with pytest.raises(P.PackError, match="not the structure"):                      # the other variant's model under this variant
        P.check_weights(P.for_variant(m, "q4"), [alt])


def test_a_pair_has_a_variant_per_role(tmp_path, key, mocklib):
    pack, w, dw = toy_pack(str(tmp_path), key, mocklib, spec=True)
    m = P.verify(pack, trusted=(key.kid,))
    assert P.pick_variant(m, [w]) == "q4" and P.pick_variant(m, [dw], "drafter") == "d"
    mv = P.for_variant(m, "q4", "d")
    assert len(mv["buckets"]) == 2 and len(mv["spec"]) == 1
    P.check_weights(mv, [w], [dw])
    with pytest.raises(P.PackError, match="drafter weight files are needed"):
        P.check_weights(mv, [w])


# --- partial files: only the chosen branch's blobs are read

def _hole(pack, blob_hash):
    """Zero the bytes of one blob in place (a download hole)."""
    raw = bytearray(open(pack, "rb").read())
    n = int.from_bytes(raw[8:16], "little")
    o, l = json.loads(bytes(raw[16:16 + n]))["blobs"][blob_hash]
    raw[16 + n + o:16 + n + o + l] = b"\0" * l
    open(pack, "wb").write(raw)


def test_partial_file_serves_the_branch_whose_blobs_are_present(tmp_path, key, mocklib):
    import hashlib
    pack, w, dw = toy_pack(str(tmp_path), key, mocklib, alt=True)
    _hole(pack, hashlib.sha256(b"alt-plan").hexdigest())                 # the alt branch's own blob is a hole
    with pytest.raises(P.PackError, match="does not hash"):              # the full check refuses the file
        P.verify(pack, trusted=(key.kid,))
    st = P.stage(pack, trusted=(key.kid,), defer=True)                   # the q4 branch does not touch it
    m = P.for_variant(st.manifest, "q4")
    st.ensure(P.ladder(m, P.select(m, sku="mock", B=8)))
    with pytest.raises(P.PackError, match="does not hash"):
        st.ensure([b for b in st.manifest["buckets"] if b["variant"] == "alt"])
    st.close()


def test_pack_need_lists_exactly_the_branch_ranges(tmp_path, key, mocklib, capsys):
    pack, w, dw = toy_pack(str(tmp_path), key, mocklib, alt=True)
    raw = open(pack, "rb").read()
    n = int.from_bytes(raw[8:16], "little")
    hdr = json.loads(raw[16:16 + n])
    end = 16 + n + hdr["manifest"][0] + hdr["manifest"][1]
    prefix = tmp_path / "prefix"
    prefix.write_bytes(raw[:end])                                       # only the header and the manifest were downloaded
    d = tmp_path / "model"
    d.mkdir()
    os.link(w, d / "w.gguf")
    P.main(["need", str(prefix), "--model-dir", str(d), "--sku", "mock"])
    j = json.loads(capsys.readouterr().out)
    import hashlib
    assert j["prefix"] == dict(off=0, len=end) and len(j["manifest_sha256"]) == 64
    assert all(hashlib.sha256(raw[r["off"]:r["off"] + r["len"]]).hexdigest() == r["sha256"] for r in j["ranges"]) and j["ranges"]
    # a model whose structure no variant has is refused
    from toypack import gguf
    gguf(str(tmp_path / "x.gguf"), [("zzz", [1], 0)])
    (tmp_path / "bad").mkdir()
    os.link(tmp_path / "x.gguf", tmp_path / "bad" / "x.gguf")
    with pytest.raises(SystemExit, match="not a model this pack was compiled for"):
        P.main(["need", str(prefix), "--model-dir", str(tmp_path / "bad"), "--sku", "mock"])


def test_verify_partial_checks_the_manifest_signature_and_the_branch(tmp_path, key, mocklib, monkeypatch, capsys):
    monkeypatch.setattr(sign, "TRUSTED_KEY_IDS", (key.kid,))
    pack, w, dw = toy_pack(str(tmp_path), key, mocklib)
    d = tmp_path / "m"
    d.mkdir()
    os.link(w, d / "w.gguf")
    P.main(["verify", pack, "--partial", "--model-dir", str(d), "--sku", "mock"])
    j = json.loads(capsys.readouterr().out)
    assert j["ok"] and j["weights"] == ["w.gguf"]
    h = sorted(P.stage(pack, trusted=(key.kid,), defer=True).blobs)[0]
    _hole(pack, h)
    with pytest.raises(SystemExit, match="does not hash"):
        P.main(["verify", pack, "--partial", "--model-dir", str(d), "--sku", "mock"])


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


def test_single_stream_bucket_is_the_cheapest_predicted_step_of_any_contract():
    """B 1 takes the bucket of least step_us among the contracts' B 1 buckets (names opaque); without prices a 1 column non-batch
    bucket; a batch bucket is never one by shape alone."""
    def bk(i, c, n, batch, us=None):
        return dict(id=i, role="target", sku="s", contract=c, ncol=n, batch=batch, maxctx=8, **({"step_us": us} if us else {}))
    m = dict(contracts=["Zq", "Yp"], runtime=dict(kind="gc-c", version=1),
             buckets=[bk("z2", "Zq", 2, True, 900.0), bk("z8", "Zq", 8, True, 950.0), bk("y1", "Yp", 1, False, 700.0)])
    assert P.single_stream(m, sku="s")["id"] == "y1"
    m["buckets"][2]["step_us"] = 1000.0
    assert P.single_stream(m, sku="s")["id"] == "z2"           # priced: the cheaper contract wins, whatever its shape
    m["buckets"][0].pop("step_us")
    assert P.single_stream(m, sku="s")["id"] == "y1"           # an unpriced bucket is not ranked
    m["buckets"][0]["step_us"] = 900.0
    assert P.single_stream(m, sku="s", contract="Yp")["id"] == "y1"
    for b in m["buckets"]:
        b.pop("step_us")
    assert P.single_stream(m, sku="s")["id"] == "y1"           # unpriced: the 1 column bucket
    m["buckets"].pop()
    assert P.single_stream(m, sku="s") is None
    assert P.select(m, sku="s", B=2)["id"] == "z2"


def test_numerics_request_selects_by_capability_records():
    """--numerics / --act / --kv pick buckets from their determinism, act and kv records (opaque contract names); no match names the
    missing capability."""
    def bk(i, c, n, det, act, batch=True, kv="f16"):
        return dict(id=i, role="target", sku="s", contract=c, ncol=n, batch=batch, maxctx=8, determinism=det, act=act, kv=kv)
    m = dict(contracts=["Hh", "Aa", "Ee"], runtime=dict(kind="gc-c", version=1),
             buckets=[bk("h2", "Hh", 2, "batch_invariant", "a16"), bk("a2", "Aa", 2, "batch_invariant", "a8"),
                      bk("e1", "Ee", 1, "exact", None, batch=False)])
    pick = lambda B, **kw: P.select(m, sku="s", B=B, want=P.wanted(single=B == 1, **kw))["id"]
    assert pick(4 - 2) == "h2"                      # default: deterministic, a16 where a bucket has it
    assert pick(2, act="a8") == "a2"
    assert P.single_stream(m, sku="s", want=P.wanted("exact", single=True))["id"] == "e1"
    assert P.select(m, sku="s", B=2, want=P.wanted("fast"))["id"] == "h2"   # fast falls back to deterministic
    with pytest.raises(P.PackError, match="kv=fp8.*the pack has"):
        P.need_bucket(m, "s", P.wanted(kv="fp8"), "kv=fp8")
    with pytest.raises(P.PackError, match="act"):
        P.wanted(act="a4")



def test_backend_runs_the_variant_the_weights_choose_and_refuses_other_weights(tmp_path, key, mocklib):
    from goethe_engine.gc_c import GcBackend
    from toypack import gguf
    pack, w, dw = toy_pack(str(tmp_path), key, mocklib, alt=True)
    for weights, want in ((w, "q4"), (str(tmp_path / "alt.gguf"), "alt")):
        be = GcBackend(pack, weights, B=2, sku="mock", trusted=(key.kid,))
        assert be.bucket["variant"] == want and be.m["variants"][want]["weights"]
        be.close()
    gguf(str(tmp_path / "other.gguf"), [("a.weight", [64, 8], 12), ("b.weight", [64], 0)])
    with pytest.raises(P.PackError, match="not a model this pack was compiled for"):
        GcBackend(pack, str(tmp_path / "other.gguf"), B=2, sku="mock", trusted=(key.kid,))


def test_weight_files_may_be_resharded_reordered_and_are_found_by_tensor_name(tmp_path, key, mocklib):
    """The same tensors in two files, in the other order, give the structure the pack was compiled for; a missing, an extra and a duplicated
    tensor are refused naming it."""
    from toypack import TARGET, gguf
    pack, w, dw = toy_pack(str(tmp_path), key, mocklib)
    m = P.for_variant(P.verify(pack, trusted=(key.kid,)), "q4")
    a, b = str(tmp_path / "s-b.gguf"), str(tmp_path / "s-a.gguf")
    gguf(a, TARGET[1:])                                       # b.weight alone
    gguf(b, TARGET[:1])                                       # a.weight alone
    P.check_weights(m, [a, b])                                # shard count and order do not matter
    P.check_weights(m, [b, a])
    with pytest.raises(P.PackError, match="b.weight.*missing from this model"):
        P.check_weights(m, [b])
    gguf(str(tmp_path / "extra.gguf"), [("c.weight", [4], 0)])
    with pytest.raises(P.PackError, match="c.weight.*missing from the model the pack was compiled for"):
        P.check_weights(m, [a, b, str(tmp_path / "extra.gguf")])
    with pytest.raises(P.PackError, match="occurs twice"):
        P.check_weights(m, [a, b, w])


# --- xz blobs: [off, len, "xz"], name = sha256 of the raw bytes

def _xz(pack, key, how):
    """Re-write the toy pack with its first blob stored by how(raw) -> (stored bytes, codec)."""
    h0 = []

    def edit(man, blobs):
        h0.append(sorted(blobs)[0])
        return man, dict(blobs, **{h0[0]: how(blobs[h0[0]])})
    _rewrite(pack, edit)
    return h0[0]


def test_xz_blob_loads_and_stored_still_does(tmp_path, key, mocklib):
    import lzma
    pack = toy_pack(str(tmp_path), key, mocklib)[0]
    _xz(pack, key, lambda b: (lzma.compress(b), "xz"))
    st = P.stage(pack, trusted=(key.kid,))
    st.ensure(None)
    assert P.verify(pack, trusted=(key.kid,))
    st.close()


def test_xz_bomb_refused_with_bounded_memory(tmp_path, key, mocklib, monkeypatch):
    import lzma, resource
    pack = toy_pack(str(tmp_path), key, mocklib)[0]
    z = lzma.LZMACompressor(lzma.FORMAT_XZ, preset=1)
    zero, bomb = bytes(1 << 24), []
    for _ in range(64):                                  # 1 GiB of zeros in a few KiB
        bomb.append(z.compress(zero))
    bomb.append(z.flush())
    monkeypatch.setattr(P, "MAX_RAW", 256 << 20)
    _xz(pack, key, lambda b: (b"".join(bomb), "xz"))
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with pytest.raises(P.PackError, match="inflates past"):
        P.verify(pack, trusted=(key.kid,))
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r0 < 200 * 1024     # KiB: far below the 1 GiB output


def test_xz_memlimit_truncated_and_wrong_codec_refused(tmp_path, key, mocklib, monkeypatch):
    import lzma
    big = lzma.compress(b"x" * 100, filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 1 << 24}])
    for i, (how, err) in enumerate([(lambda b: (big, "xz"), "xz"), (lambda b: (lzma.compress(b)[:-9], "xz"), "xz"),
                                    (lambda b: (b, "gzip"), "codecs")]):
        pack = toy_pack(str(tmp_path / str(i)), key, mocklib)[0]
        _xz(pack, key, how)
        monkeypatch.setattr(P, "XZ_MEM", 1 << 16 if i == 0 else 64 << 20)    # i = 0: the dictionary needs more than the limit
        with pytest.raises(P.PackError, match=err):
            P.verify(pack, trusted=(key.kid,))
