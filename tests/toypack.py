"""Test support: a throwaway P-256 key (temp dir only), the signer, and a tiny toy pack around the mock libgck."""
import base64
import hashlib
import json
import os
import subprocess

from goethe_engine import sign


def run(*a, **kw):
    return subprocess.run(a, check=True, capture_output=True, **kw)


class Key:
    def __init__(self, d):
        self.priv, self.pub = os.path.join(d, "k.pem"), os.path.join(d, "k_pub.pem")
        run("openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", self.priv)
        run("openssl", "ec", "-in", self.priv, "-pubout", "-out", self.pub)
        self.pem = open(self.pub).read()
        self.kid = sign.key_id(self.pem.encode())

    def sign(self, data):
        """Signature file bytes over data (gc-sig-1)."""
        sig = run("openssl", "dgst", "-sha256", "-sign", self.priv, input=data).stdout
        return json.dumps(dict(schema=sign.SCHEMA, alg=sign.ALG, keyid=self.kid, pub=self.pem,
                               sig=base64.b64encode(sig).decode())).encode()

    def sign_file(self, path):
        """PATH.sig: the gc-sig-1 signature over the exact bytes of path."""
        open(path + ".sig", "wb").write(self.sign(open(path, "rb").read()))


def gcpack(path, manifest, blobs):
    """Write a .gcpack: manifest bytes and {sha256: bytes}."""
    hdr, off, parts = dict(blobs={}), len(manifest), [manifest]
    hdr["manifest"] = [0, len(manifest)]
    for h, b in blobs.items():
        hdr["blobs"][h] = [off, len(b)]
        off += len(b)
        parts.append(b)
    hb = json.dumps(hdr).encode()
    open(path, "wb").write(b"GCPACK2\n" + len(hb).to_bytes(8, "little") + hb + b"".join(parts))


def toy_pack(d, key, lib, *, spec=False, runtime="gc-c", abi="2.0.0", arch="sm_120a", mutate=None):
    """A signed toy.gcpack under d with ladder buckets ncol 2 and 8 (and a spec pair if spec) over the library at lib;
    returns (pack file, weights path, drafter weights path)."""
    pack, blobs = os.path.join(d, "toy.gcpack"), {}
    os.makedirs(d, exist_ok=True)

    def blob(data):
        blobs[hashlib.sha256(data).hexdigest()] = data
        return hashlib.sha256(data).hexdigest()

    w, dw = os.path.join(d, "w.gguf"), os.path.join(d, "d.gguf")
    open(w, "wb").write(b"weights0")
    open(dw, "wb").write(b"drafter0")
    files = {"libgck.so": blob(open(lib, "rb").read()), "plan.json": blob(b"{}")}
    req = dict(arch=arch, cc="12.0", sms=170, name="toy", smem_block=1)
    caps = dict(verify_widths=[], verify=["linear", "tree"], tree_mask=True, taps=[dict(node="result_norm", n=4)],
                recurrent_commit=[], accept=[], drafter_roles=[])

    def bucket(i, role, nc, batch=True, io=None):
        return dict(id=i, role=role, sku="mock", contract="K1", ncol=nc, batch=batch, maxctx=256, weights=int(role == "drafter"),
                    spec_caps=caps, files=files, lib="libgck.so", plan="plan.json", io=io, tune={}, requires=req,
                    variant_of=None, klass=None, variants=[])
    bs = [bucket("target.mock.K1.2.256", "target", 2), bucket("target.mock.K1.8.256", "target", 8)]
    sp = []
    if spec:
        bs = [bucket("target.mock.K1.4v.256", "target", 4, False, dict(accept=True)), bucket("drafter.mock.K1.1.256", "drafter", 1)]
        sp = [dict(target=bs[0]["id"], drafter=bs[1]["id"], policy=dict(kind="ar", K="3"))]
    m = dict(schema="gc-pack", abi=abi, compiler=dict(commit="0" * 40, dirty=False), arch=arch, cuda="12.8",
             hw=dict(mock="0" * 64), runtime=dict(kind=runtime, version=1), contracts=["K1"], buckets=bs, spec=sp,
             model=dict(key="t", stage="t", graph_sha256="0" * 64, weights=[
                 dict(file="w.gguf", bytes=8, sha256=hashlib.sha256(b"weights0").hexdigest()),
                 dict(file="d.gguf", bytes=8, sha256=hashlib.sha256(b"drafter0").hexdigest())]))
    if mutate:
        mutate(m)
    gcpack(pack, (json.dumps(m, indent=1, sort_keys=True) + "\n").encode(), blobs)
    key.sign_file(pack)
    return pack, w, dw
