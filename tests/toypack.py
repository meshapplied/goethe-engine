"""Test support: a throwaway P-256 key (temp dir only), the signer, and a tiny toy pack around the mock libgck."""
import base64
import hashlib
import json
import os
import shutil
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

    def sign_pack(self, pack):
        """manifest.sha256 and manifest.sig for the manifest.json in pack."""
        raw = open(os.path.join(pack, "manifest.json"), "rb").read()
        open(os.path.join(pack, "manifest.sha256"), "w").write(hashlib.sha256(raw).hexdigest() + "\n")
        open(os.path.join(pack, "manifest.sig"), "wb").write(self.sign(raw))


def blob(pack, data):
    h = hashlib.sha256(data).hexdigest()
    os.makedirs(os.path.join(pack, "blobs"), exist_ok=True)
    open(os.path.join(pack, "blobs", h), "wb").write(data)
    return h


def toy_pack(d, key, lib, *, spec=False, runtime="gc-c", abi="1.1.0", arch="sm_120a", mutate=None):
    """A signed pack dir under d with ladder buckets ncol 2 and 8 (and a spec pair if spec) over the library at lib;
    returns (pack dir, weights path, drafter weights path)."""
    pack = os.path.join(d, "pack")
    os.makedirs(pack)
    w, dw = os.path.join(d, "w.gguf"), os.path.join(d, "d.gguf")
    open(w, "wb").write(b"weights0")
    open(dw, "wb").write(b"drafter0")
    files = {"libgck.so": blob(pack, open(lib, "rb").read()), "plan.json": blob(pack, b"{}")}
    req = dict(arch=arch, cc="12.0", sms=170, name="toy", smem_block=1)
    caps = dict(verify_widths=[], verify=["linear", "tree"], tree_mask=True, taps=[dict(node="result_norm", n=4)],
                recurrent_commit=[], accept=[], drafter_roles=[])

    def bucket(i, role, nc, batch=True, io=None):
        return dict(id=i, role=role, sku="mock", contract="BI", ncol=nc, batch=batch, maxctx=256, weights=int(role == "drafter"),
                    spec_caps=caps, files=files, lib="libgck.so", plan="plan.json", io=io, tune={}, requires=req,
                    variant_of=None, klass=None, variants=[])
    bs = [bucket("target.mock.BI.2.256", "target", 2), bucket("target.mock.BI.8.256", "target", 8)]
    sp = []
    if spec:
        bs = [bucket("target.mock.BI.4v.256", "target", 4, False, dict(accept=True)), bucket("drafter.mock.BI.1.256", "drafter", 1)]
        sp = [dict(target=bs[0]["id"], drafter=bs[1]["id"], policy=dict(kind="ar", K="3"))]
    m = dict(schema="gc-pack", abi=abi, compiler=dict(commit="0" * 40, dirty=False), arch=arch, cuda="12.8",
             hw=dict(mock="0" * 64), runtime=dict(kind=runtime, version=1), contracts=["BI"], buckets=bs, spec=sp,
             model=dict(key="t", stage="t", graph_sha256="0" * 64, weights=[
                 dict(file="w.gguf", bytes=8, sha256=hashlib.sha256(b"weights0").hexdigest()),
                 dict(file="d.gguf", bytes=8, sha256=hashlib.sha256(b"drafter0").hexdigest())]))
    if mutate:
        mutate(m)
    open(os.path.join(pack, "manifest.json"), "w").write(json.dumps(m, indent=1, sort_keys=True) + "\n")
    key.sign_pack(pack)
    return pack, w, dw


def signed_copy(src, dst, key):
    """A copy of an unsigned pack dir, signed by key."""
    shutil.copytree(src, dst)
    key.sign_pack(dst)
    return dst
