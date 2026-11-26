"""Pack 4: the signed manifest names every blob by hash, the blobs live in a content-addressed store; every tampering is refused."""
import hashlib
import json
import lzma
import os

import pytest

from goethe_engine import pack as P
from goethe_engine.gc_c import GcBackend
from test_modular import base_raw, overlay
from toypack import Key, parts, toy_pack


def v4(d, key, lib, *, mutate=None, domain=P.DOMAIN4, magic=P.MAGIC4, signer=None):
    """(v4 pack, store dir, weights): the toy pack written by P.write as abi 4.0.0 (every blob pinned, in the store, xz where smaller),
    then mutate(manifest, store, blobs), framed with magic and signed over domain."""
    pack3, w, _ = toy_pack(d, key, lib, sign=False)
    m3, data = parts(pack3)
    store, pack = os.path.join(d, "store"), os.path.join(d, "toy4.gcpack")
    m = json.loads(P.write(pack, dict(m3, abi="4.0.0"), data.values(), store=store))
    if mutate:
        mutate(m, store, data)
    raw = (json.dumps(m, indent=1, sort_keys=True) + "\n").encode()
    P.frame(pack, raw, {}, major=4 if magic == P.MAGIC4 else 3)
    open(pack + ".sig", "wb").write((signer or key).sign_digest(hashlib.sha256(domain + raw).hexdigest()))
    return pack, store, w


def test_v4_stages_from_the_store_and_opens(tmp_path, key, mocklib):
    pack, store, w = v4(str(tmp_path), key, mocklib)
    st = P.stage(pack, trusted=(key.kid,), store=store)          # every blob, hash-checked against its pin
    assert st.manifest["abi"] == "4.0.0" and len(os.listdir(os.path.join(st.dir, "blobs"))) == 2
    st.close()
    be = GcBackend(pack, w, sku="mock", trusted=(key.kid,), B=8, store=store)
    assert be.bucket["ncol"] == 8
    be.close()


def test_v4_overlay_merges(tmp_path, key, mocklib):
    pack, store, _ = v4(str(tmp_path), key, mocklib)
    ovl = overlay(str(tmp_path), key, pack, mocklib)                # an abi 4 overlay in a v4 container, its blobs in the store
    o = json.loads(base_raw(ovl))
    assert o["abi"] == "4.0.0"
    with open(ovl, "rb") as f:
        _, blobs, base, _, _ = P.read_container_v(f, os.path.getsize(ovl))
        for h, (off, ln, *_) in blobs.items():
            f.seek(base + off)
            open(os.path.join(store, h), "wb").write(f.read(ln))
    raw = base_raw(ovl)
    P.frame(ovl, raw, {}, major=4)
    open(ovl + ".sig", "wb").write(key.sign_digest(hashlib.sha256(P.OVERLAY_DOMAIN + raw).hexdigest()))
    st = P.stage([pack, ovl], trusted=(key.kid,), store=store)
    assert {b["sku"] for b in st.manifest["buckets"]} == {"mock", "mock2"}
    st.close()


def _flip(m, store, data):
    h = next(iter(data))
    name = h + ".xz" if os.path.exists(os.path.join(store, h + ".xz")) else h
    b = bytearray(open(os.path.join(store, name), "rb").read())
    b[-1] ^= 1
    open(os.path.join(store, name), "wb").write(bytes(b))


def _missing(m, store, data):
    for n in os.listdir(store):
        if n.startswith(m["buckets"][0]["files"]["libgck.so"]):
            os.unlink(os.path.join(store, n))


def _wronghash(m, store, data):
    h = m["buckets"][0]["files"]["libgck.so"]
    for n in os.listdir(store):
        if n.startswith(h):
            os.unlink(os.path.join(store, n))
    open(os.path.join(store, h), "wb").write(b"\0" * m["blob_raw"][h])     # pinned size, other bytes


def _overpin(m, store, data):
    h = m["buckets"][0]["files"]["libgck.so"]
    m["blob_raw"][h] -= 1                                                   # the stored bytes exceed the signed pin


def _bomb(m, store, data):
    h = m["buckets"][0]["files"]["plan.json"]
    open(os.path.join(store, h + ".xz"), "wb").write(lzma.compress(b"{}" + b" " * (1 << 20)))   # inflates past its 2 byte pin


def _unpinned(m, store, data):
    m["blob_raw"].pop(m["buckets"][0]["files"]["libgck.so"])


def _symlink(m, store, data):
    h = m["buckets"][0]["files"]["libgck.so"]
    for n in os.listdir(store):
        if n.startswith(h):
            os.rename(os.path.join(store, n), os.path.join(store, n + ".real"))
            os.symlink(os.path.join(store, n + ".real"), os.path.join(store, n))


CASES = {"flipped-store-blob": (dict(mutate=_flip), "hash|xz|pins"), "missing-blob": (dict(mutate=_missing), "missing from the pack and the store"),
         "wrong-hash-blob": (dict(mutate=_wronghash), "does not hash to its name"), "over-pin": (dict(mutate=_overpin), "inflates past|pins"),
         "xz-bomb": (dict(mutate=_bomb), "inflates past"), "unpinned": (dict(mutate=_unpinned), "not pinned"),
         "store-symlink": (dict(mutate=_symlink), "symlinks are refused"), "v3-domain": (dict(domain=P.DOMAIN), "signature|verif"),
         "v4-magic-abi3": (dict(mutate=lambda m, s, d: m.update(abi="3.0.0")), "major 3"),
         "v3-magic-abi4": (dict(magic=P.MAGIC, domain=P.DOMAIN), "major 4")}


@pytest.mark.parametrize("case", sorted(CASES))
def test_v4_refusals(tmp_path, key, mocklib, case):
    kw, msg = CASES[case]
    pack, store, _ = v4(str(tmp_path), key, mocklib, **kw)
    with pytest.raises(P.PackError, match=msg):
        st = P.stage(pack, trusted=(key.kid,), store=store)
        st.close()


@pytest.mark.parametrize("case", ["unsigned", "untrusted", "flipped-manifest", "symlink", "notfile", "no-store"])
def test_v4_container_refusals(tmp_path, key, mocklib, case):
    other = None
    if case == "untrusted":
        os.makedirs(tmp_path / "k2")
        other = Key(str(tmp_path / "k2"))
    pack, store, _ = v4(str(tmp_path), key, mocklib, signer=other)
    if case == "unsigned":
        os.unlink(pack + ".sig")
    elif case == "flipped-manifest":
        b = bytearray(open(pack, "rb").read())
        b[b.index(b'"abi"') + 2] ^= 1
        open(pack, "wb").write(bytes(b))
    elif case == "symlink":
        os.symlink(pack, pack + ".ln")
        os.symlink(pack + ".sig", pack + ".ln.sig")
        pack += ".ln"
    elif case == "notfile":
        os.rename(pack, pack + ".f")
        os.mkdir(pack)
    msg = {"unsigned": "no signature", "untrusted": "not in the pinned", "flipped-manifest": "signature|verif|manifest",
           "symlink": "symlinks are refused", "notfile": "not a regular file|symlinks", "no-store": "missing from the pack"}[case]
    with pytest.raises(P.PackError, match=msg):
        st = P.stage(pack, trusted=(key.kid,), store=None if case == "no-store" else store)
        st.close()
