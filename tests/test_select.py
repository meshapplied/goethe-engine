"""Variant selection by structural fit (arch rule, SMs, smem per block) and price, never by SKU name; a device nothing fits is
refused with every variant's reason and the fallback. Mock devices only: no driver, no GPU."""
import pytest

from goethe_engine import pack as P

V = {"a170": ("sm_120a", "12.0", 170, 101376), "a84": ("sm_120a", "12.0", 84, 101376), "c148": ("sm_100a", "10.0", 148, 232448)}


def man(variants=V, cuda="12.8"):
    return dict(cuda=cuda, buckets=[dict(sku=s, requires=dict(arch=a, cc=c, sms=n, smem_block=sm, name=f"card {s}"))
                                    for s, (a, c, n, sm) in variants.items()])


def dev(cc, sms, smem=101376, name="unknown card", **kw):
    return dict(platform="cuda", name=name, cc=cc, sms=sms, smem_block=smem, cuda="12.8", arch_ok=P.cuda_arch_ok, **kw)


def refused(m, d, *want):
    with pytest.raises(P.PackError) as e:
        P.pick_sku(m, d)
    for w in ("runs no variant", "Fallback: serve this model with llama.cpp", "--overlay") + want:
        assert w in str(e.value), (w, str(e.value))


def test_same_arch_unknown_sku_takes_the_fitting_variant():
    assert P.pick_sku(man(), dev("12.0", 188)) == "a170"          # more SMs than any variant: the largest grid that fits
    assert P.pick_sku(man(), dev("12.0", 100)) == "a84"           # fewer than 170: only the 84 SM plan is co-resident
    assert P.pick_sku(man(), dev("12.0", 170, name="card a84")) == "a170"   # a name is no match without its SM count
    assert P.pick_sku(man(), dev("12.0", 84, name="card a84")) == "a84"


def test_price_ranks_the_fitting_variants():
    price = lambda s, d: {"a170": 9.0, "a84": 5.0}[s]
    assert P.pick_sku(man(), dev("12.0", 188, price=price)) == "a84"


def test_same_arch_other_smem():
    refused(man(), dev("12.0", 188, smem=65536), "101376 B shared memory per block, the device allows 65536")
    assert P.pick_sku(man(), dev("12.0", 188, smem=232448)) == "a170"


def test_other_arches():
    assert P.pick_sku(man(), dev("10.0", 148, smem=232448)) == "c148"
    refused(man(), dev("10.3", 148, smem=232448), "sm_100a SASS runs only on cc 10.0", "for sm_103")
    for cc, sms, smem in (("9.0", 132, 232448), ("8.9", 128, 101376), ("8.6", 82, 101376), ("8.0", 108, 166912), ("7.5", 40, 65536)):
        refused(man(), dev(cc, sms, smem), f"device cc {cc}", "sm_" + cc.replace(".", ""))


def test_cc_12_1():
    refused(man(), dev("12.1", 48), "sm_120a SASS runs only on cc 12.0")
    assert P.pick_sku(man({"f": ("sm_120f", "12.0", 48, 101376)}), dev("12.1", 48)) == "f"   # a family target covers 12.x, x >= 0


def test_arch_rule():
    assert P.arch_runs("compute_80", "8.9") == "" and P.arch_runs("compute_90", "8.9")
    assert P.arch_runs("sm_86", "8.9") == "" and P.arch_runs("sm_86", "8.0") and P.arch_runs("sm_89", "9.0")
    assert P.arch_runs("sm_90a", "9.0") == "" and P.arch_runs("sm_90a", "10.0")
    assert P.arch_runs("bogus", "9.0")


def test_driver_and_explicit_sku():
    refused(man(), dict(dev("12.0", 188), cuda="12.4"), "driver CUDA 12.4 < the libraries' CUDA 12.8")
    with pytest.raises(P.PackError, match="--sku a170 does not run on device unknown card: the plan's grid needs 170 SMs"):
        P.pick_sku(man(), dev("12.0", 100), sku="a170")
    assert P.pick_sku(man(), dev("12.0", 188), sku="a84") == "a84"


def test_one_signed_pack_file_holds_every_arch(tmp_path, key, mocklib):
    """One pack per model: buckets of two arches in one signed file (manifest arches); the selector picks per device over the union."""
    from toypack import toy_pack

    def mut(m):
        req = dict(m["buckets"][0]["requires"], arch="sm_100a", cc="10.0", sms=148, smem_block=232448, name="b200")
        m["buckets"] += [dict(b, id=b["id"].replace("mock", "b200"), sku="b200", requires=req) for b in m["buckets"]]
        m["arches"] = {"sm_100a": dict(platform="cuda", cc="10.0")}
        m["hw"]["b200"] = m["hw"]["mock"]
    pack, _, _ = toy_pack(str(tmp_path), key, mocklib, mutate=mut)
    m = P.verify(pack, trusted=(key.kid,))
    assert P.arches(m) == {"sm_120a", "sm_100a"}
    assert P.pick_sku(m, dev("10.0", 148, smem=232448)) == "b200"
    assert P.pick_sku(m, dev("12.0", 170)) == "mock"
    assert P.select(m, dev=dev("10.0", 148, smem=232448), B=8)["sku"] == "b200"


def test_pack_cost_model_ranks_the_fitting_skus(tmp_path, key, mocklib, monkeypatch):
    """gc_probe / gc_price of the pack's library rank the fitting SKUs (no exact match); perturbing a probed constant flips the pick;
    a library that declines (no probe) falls back to the most SMs; the probe blob is cached per device, driver, manifest and library."""
    from toypack import toy_pack

    def mut(m):
        req = dict(m["buckets"][0]["requires"], sms=100, name="small card")
        m["buckets"] += [dict(b, id=b["id"].replace("mock", "b2"), sku="b2", requires=req) for b in m["buckets"]]
        m["hw"]["b2"] = m["hw"]["mock"]
    pack, _, _ = toy_pack(str(tmp_path), key, mocklib, mutate=mut)
    st = P.stage(pack, trusted=(key.kid,), defer=True)
    try:
        d = dev("12.0", 188, uuid="GPU-1")
        assert P.pick_sku(st.manifest, dict(d, price=P.lib_price(st, st.manifest))) == "mock"     # unpriced: most SMs (170 > 100)
        P.PRICE_CACHE.clear()
        monkeypatch.setenv("GCK_MOCK_PROBE", "9.0,5.0")
        assert P.pick_sku(st.manifest, dict(d, price=P.lib_price(st, st.manifest))) == "b2"
        monkeypatch.setenv("GCK_MOCK_PROBE", "4.0,5.0")
        assert P.pick_sku(st.manifest, dict(d, price=P.lib_price(st, st.manifest))) == "b2"          # cached constants
        assert P.pick_sku(st.manifest, dict(d, uuid="GPU-2", price=P.lib_price(st, st.manifest))) == "mock"   # another device: probed again
    finally:
        st.close()
        P.PRICE_CACHE.clear()
