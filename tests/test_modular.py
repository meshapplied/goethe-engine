"""Pack axes, signed overlays (a new arch or format without re-signing the base), context sized at load, and the plugin registry."""
import hashlib
import json
import os

import pytest

from goethe_engine import pack as P, registry as R, structure
from goethe_engine.gc_c import GcBackend
from toypack import gcpack, gguf, toy_pack

Q8 = [("a.weight", [64, 8], 8), ("b.weight", [64], 0)]     # the base target's tensors in another format (q8_0)


def base_raw(pack):
    with open(pack, "rb") as f:
        return P.read_container(f, os.path.getsize(pack))[2]


def overlay(d, key, pack, lib, *, name="ovl", base=None, ids=("target.mock2.K1.2.256", "target.mock2.K1.8.256"), sign=True,
            variant="q4", arch="sm_100a"):
    """A signed overlay of pack: buckets of a second SKU (another arch) for variant (q4: the base's; alt: a new format whose
    structure and weights the overlay carries)."""
    raw = base_raw(pack)
    m = json.loads(raw)
    blobs = {}

    def blob(data):
        blobs[hashlib.sha256(data).hexdigest()] = data
        return hashlib.sha256(data).hexdigest()
    files = {"libgck.so": blob(open(lib, "rb").read()), "plan.json": blob(b"plan for " + arch.encode())}
    nb = len(m["model"]["weights"])
    o = dict(schema=P.OVERLAY_SCHEMA, abi=m["abi"], base=base or hashlib.sha256(raw).hexdigest(), hw={"mock2": "1" * 64},
             arches={arch: dict(platform="cuda", cc="10.0")}, contracts=["K1"], runtime=m["runtime"],
             blob_raw={h: len(b) for h, b in blobs.items()}, buckets=[], variants={}, weights=[])
    wi = 0
    if variant == "alt":
        aw = os.path.join(d, "alt2.gguf")
        gguf(aw, Q8)
        o["weights"] = [dict(file="alt2.gguf", bytes=os.path.getsize(aw), sha256=hashlib.sha256(open(aw, "rb").read()).hexdigest())]
        o["variants"] = {"alt2": dict(structure=structure.describe([aw], ["general.architecture"]), weights=[nb])}
        variant, wi = "alt2", nb
    for i, nc in zip(ids, (2, 8)):
        b = dict(next(x for x in m["buckets"] if x["ncol"] == nc), id=i, sku="mock2", variant=variant, weights=wi, files=files,
                 requires=dict(arch=arch, cc="10.0", sms=148, name="toy2", smem_block=1))
        o["buckets"].append(b)
    path = os.path.join(d, name + ".gcovl")
    raw_o = (json.dumps(o, indent=1, sort_keys=True) + "\n").encode()
    gcpack(path, raw_o, blobs)
    if sign:
        open(path + ".sig", "wb").write(key.sign_digest(hashlib.sha256(P.OVERLAY_DOMAIN + raw_o).hexdigest()))
    return path


def test_axes_derive_from_a_3_0_bucket():
    b = dict(id="x", ncol=1, batch=False)
    assert P.axes(b) == ("cuda", "stream", ())
    assert P.axes(dict(b, ncol=8, batch=True))[1] == "decode"
    assert P.axes(dict(b, ncol=128, prefill=128))[1] == "prefill"
    assert P.axes(dict(b, ncol=4))[1] == "verify"
    assert P.axes(dict(b, platform="rocm", kind="decode", mesh={"tp": 2, "dp": 1})) == ("rocm", "decode", (("tp", 2),))
    with pytest.raises(P.PackError):
        P.mesh_key({"xp": 2})


def test_restrict_keeps_one_platform_and_mesh():
    bs = [dict(id="a", ncol=2, batch=True, contract="C"), dict(id="b", ncol=2, batch=True, contract="C", mesh={"tp": 2}),
          dict(id="c", ncol=2, batch=True, contract="D", platform="rocm")]
    m = dict(buckets=bs, contracts=["C", "D"], spec=[])
    assert [b["id"] for b in P.restrict(m)["buckets"]] == ["a"] and P.restrict(m)["contracts"] == ["C"]
    assert [b["id"] for b in P.restrict(m, mesh={"tp": 2})["buckets"]] == ["b"]
    with pytest.raises(P.PackError, match="platform vulkan"):
        P.restrict(m, platform="vulkan")


def test_overlay_adds_an_arch_without_resigning_the_base(tmp_path, key, mocklib):
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib)
    sig0 = open(pack + ".sig", "rb").read()
    ovl = overlay(str(tmp_path), key, pack, mocklib)
    st = P.stage([pack, ovl], trusted=(key.kid,), defer=True)
    try:
        m = st.manifest
        assert {b["sku"] for b in m["buckets"]} == {"mock", "mock2"} and P.arches(m) == {"sm_120a", "sm_100a"}
        assert st.manifest_sha256 == hashlib.sha256(base_raw(pack)).hexdigest()       # the index still matches the base
        raw = open(os.path.join(st.dir, "manifest.json"), "rb").read()
        assert open(os.path.join(st.dir, "manifest.sha256")).read().strip() == hashlib.sha256(raw).hexdigest()
        st.ensure([b for b in m["buckets"] if b["sku"] == "mock2"])                   # the overlay's own blobs, hash-checked
    finally:
        st.close()
    assert open(pack + ".sig", "rb").read() == sig0
    be = GcBackend([pack, ovl], w, sku="mock2", trusted=(key.kid,), B=8)              # the mock runtime opens the merged manifest
    assert be.bucket["sku"] == "mock2"
    be.close()


def test_overlay_adds_a_format_with_its_weights(tmp_path, key, mocklib):
    pack, _, _ = toy_pack(str(tmp_path), key, mocklib)
    ovl = overlay(str(tmp_path), key, pack, mocklib, variant="alt")
    st = P.stage([pack, ovl], trusted=(key.kid,), defer=True)
    try:
        m = st.manifest
        assert m["variants"]["alt2"]["weights"] == [3] and m["model"]["weights"][3]["file"] == "alt2.gguf"
        assert P.pick_variant(m, [os.path.join(str(tmp_path), "alt2.gguf")]) == "alt2"
    finally:
        st.close()


REFUSALS = {"base": "base manifest", "collide": "already in the pack", "unsigned": "no signature", "untrusted": "not in the pinned",
            "hw": "hw mock differs", "variant": "variant q4 differs", "flipped": "signature|verif", "symlink": "symlinks are refused",
            "notfile": "not a regular file|symlinks"}


@pytest.mark.parametrize("bad", sorted(REFUSALS))
def test_overlay_refusals(tmp_path, key, mocklib, bad):
    """Every way an overlay could change or forge the base is refused."""
    pack, _, _ = toy_pack(str(tmp_path), key, mocklib)
    kw = dict(base="0" * 64) if bad == "base" else dict(ids=("target.mock.K1.2.256", "x")) if bad == "collide" else \
        dict(sign=False) if bad == "unsigned" else {}
    ovl = overlay(str(tmp_path), key, pack, mocklib, **kw)
    o = json.loads(base_raw(ovl))
    if bad in ("hw", "variant"):        # a base field under the same key with another value, signed by the trusted key
        if bad == "hw":
            o["hw"]["mock"] = "2" * 64
        else:
            o["variants"]["q4"] = dict(json.loads(base_raw(pack))["variants"]["q4"], weights=[1])
        raw = (json.dumps(o, indent=1, sort_keys=True) + "\n").encode()
        gcpack(ovl, raw, {})            # refused at the merge, before any blob is read
        open(ovl + ".sig", "wb").write(key.sign_digest(hashlib.sha256(P.OVERLAY_DOMAIN + raw).hexdigest()))
    elif bad == "untrusted":
        from toypack import Key
        os.makedirs(tmp_path / "other")
        other = Key(str(tmp_path / "other"))           # a valid signature by a key that is not pinned
        open(ovl + ".sig", "wb").write(other.sign_digest(hashlib.sha256(P.OVERLAY_DOMAIN + base_raw(ovl)).hexdigest()))
    elif bad == "flipped":
        b = bytearray(open(ovl, "rb").read())
        i = b.index(b'"base"') + 2
        b[i] ^= 1                        # one byte of the signed manifest
        open(ovl, "wb").write(bytes(b))
    elif bad == "symlink":
        os.symlink(ovl, ovl + ".ln")
        os.symlink(ovl + ".sig", ovl + ".ln.sig")
        ovl += ".ln"
    elif bad == "notfile":
        os.rename(ovl, ovl + ".f")
        os.mkdir(ovl)
    with pytest.raises(P.PackError, match=REFUSALS[bad]):
        P.stage([pack, ovl], trusted=(key.kid,), defer=True)


def test_ctx_is_sized_at_load_to_the_largest_maxctx_that_fits():
    m = dict(buckets=[dict(role="target", maxctx=c) for c in (1024, 4096, 16384)])
    seen = []

    def fits(bl):
        seen.append(bl[0]["maxctx"])
        P.need(bl[0]["maxctx"] <= 4096, f"{bl[0]['maxctx']} does not fit")
    pick = lambda cx: (dict(maxctx=cx), [dict(maxctx=cx)], None)
    assert P.open_ctx(pick, m, None, fits)[0]["maxctx"] == 4096 and seen == [16384, 4096]
    assert P.open_ctx(pick, m, 1024, fits)[0]["maxctx"] == 1024
    with pytest.raises(P.PackError, match="1024 does not fit"):
        P.open_ctx(pick, m, None, lambda bl: P.need(False, f"{bl[0]['maxctx']} does not fit"))


def test_registry_stubs_refuse_with_a_clear_message():
    for p in ("rocm", "vulkan", "metal"):
        with pytest.raises(NotImplementedError, match=f"platform {p}: not implemented yet"):
            R.open_backend("x.gcpack", [], platform=p)
    with pytest.raises(NotImplementedError, match="multi-device parallelism"):
        R.open_backend("x.gcpack", [], mesh=R.Mesh((0, 1), {"tp": 2}))
    with pytest.raises(ValueError):
        R.Mesh((0,), {"tp": 2})
    assert R.Mesh.single(3).devices == (3,) and R.Mesh((0, 1, 2, 3), {"dp": 2, "tp": 2, "pp": 1}).key == {"dp": 2, "tp": 2}
    for t in ("encoder-decoder", "embedding"):
        be = type("B", (), {"m": dict(model=dict(type=t))})()
        with pytest.raises(NotImplementedError, match=f"model type {t}"):
            R.runner(be, None)
    with pytest.raises(P.PackError, match="model type"):
        R.model_type(dict(model=dict(type="diffusion")))


def test_cuda_plugin_opens_the_toy_pack_on_a_mesh_of_one(toy, key):
    pack, w, _ = toy
    be = R.open_backend(pack, w, mesh=R.Mesh.single(0), sku="mock", trusted=(key.kid,), B=8)
    assert be.bucket["ncol"] == 8
    be.close()
