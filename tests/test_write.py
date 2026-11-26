"""pack.write is the one checked way to make a pack: what it writes the reader stages and opens (pack 3 and 4); what the reader would
refuse it never writes. pick_sku prefers the device's own architecture over older code of the family and PTX."""
import json
import os

import pytest

from goethe_engine import pack as P
from goethe_engine.gc_c import GcBackend
from toypack import parts, toy_pack


def toy_parts(d, key, lib):
    """(manifest dict, [blob bytes], weights path) of the toy pack."""
    pack, w, _ = toy_pack(d, key, lib, sign=False)
    m, data = parts(pack)
    return m, list(data.values()), w


def test_written_packs_stage_and_open(tmp_path, key, mocklib):
    m, blobs, w = toy_parts(str(tmp_path / "src"), key, mocklib)
    p3 = str(tmp_path / "w3.gcpack")
    P.write(p3, m, blobs)
    key.sign_pack(p3)
    be = GcBackend(p3, w, sku="mock", trusted=(key.kid,), B=8)
    assert be.bucket["ncol"] == 8
    be.close()
    p4, store = str(tmp_path / "w4.gcpack"), str(tmp_path / "store")
    raw = P.write(p4, dict(m, abi="4.0.0"), blobs, store=store)
    key.sign_pack(p4)
    pins = json.loads(raw)["blob_raw"]
    assert sorted(pins.values()) == sorted(len(b) for b in blobs)          # raw sizes pinned from the data, never typed in
    small = [h for h, n in pins.items() if n == 2]                          # the plan blob b"{}": xz would be larger, so stored raw
    assert small and all(os.path.exists(os.path.join(store, h)) for h in small)
    assert any(n.endswith(".xz") for n in os.listdir(store))              # the library compresses: stored as xz, read back against its pin
    be = GcBackend(p4, w, sku="mock", trusted=(key.kid,), B=8, store=store)
    assert be.bucket["ncol"] == 8
    be.close()


@pytest.mark.parametrize("bad, why", [
    (lambda m, b: (m, b[:-1]), "blobs missing"),
    (lambda m, b: (m, b + [b"stray"]), "no bucket names"),
    (lambda m, b: ({k: v for k, v in m.items() if k != "cuda"}, b), "lacks cuda"),
    (lambda m, b: (dict(m, abi="5.0.0"), b), "pack 3 and 4"),
    (lambda m, b: (dict(m, abi="4.0.0"), b), "pass store"),
])
def test_write_refuses_and_leaves_nothing(tmp_path, key, mocklib, bad, why):
    m, blobs = toy_parts(str(tmp_path / "src"), key, mocklib)[:2]
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(P.PackError, match=why):
        P.write(str(out / "x.gcpack"), *bad(m, blobs))
    assert os.listdir(out) == []


ARCHES = {"sm120": ("sm_120a", "12.0"), "sm100": ("sm_100a", "10.0"), "sm90": ("sm_90a", "9.0"), "sm89": ("sm_89", "8.9"),
          "sm80": ("sm_80", "8.0")}
DEVICES = [  # (name, cc, SMs, the SKU an arch-level multi-arch pack runs)
    ("NVIDIA GeForce RTX 5090", "12.0", 170, "sm120"),
    ("NVIDIA RTX PRO 6000 Blackwell", "12.0", 188, "sm120"),
    ("NVIDIA GeForce RTX 5080", "12.0", 84, "sm120"),
    ("NVIDIA B200", "10.0", 148, "sm100"),
    ("NVIDIA H100", "9.0", 132, "sm90"),
    ("NVIDIA GeForce RTX 4090", "8.9", 128, "sm89"),
    ("NVIDIA L40S", "8.9", 142, "sm89"),
    ("NVIDIA A100", "8.0", 108, "sm80"),
    ("NVIDIA GeForce RTX 3090", "8.6", 82, "sm80"),          # no sm_86 build: the family's older SASS
]


def test_pick_sku_prefers_the_native_arch_over_older_family_code():
    m = dict(cuda="12.8", buckets=[dict(sku=s, requires=dict(arch=a, cc=cc, sms=1, name=s, smem_block=0))
                                   for s, (a, cc) in ARCHES.items()])
    for name, cc, sms, want in DEVICES:
        dev = dict(name=name, cc=cc, sms=sms, smem_block=101376, cuda="13.0", arch_ok=P.cuda_arch_ok)
        assert P.pick_sku(m, dev) == want, name
    ptx = dict(m, buckets=m["buckets"] + [dict(sku="ptx", requires=dict(arch="compute_80", cc="8.0", sms=1, name="ptx"))])
    assert P.pick_sku(ptx, dict(name="A100", cc="8.0", sms=108, cuda="13.0", arch_ok=P.cuda_arch_ok)) == "sm80"   # SASS before JIT
