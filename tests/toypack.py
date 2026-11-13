"""Test support: a throwaway P-256 key (temp dir only), the signer, and a tiny toy pack around the mock libgck."""
import base64
import hashlib
import json
import os
import subprocess

from goethe_engine import pack as P, sign, structure


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

    def sign_digest(self, hexdigest):
        """A gc-sig-1 signature file whose ECDSA signature is over the given sha256 digest (what a pack's manifest is signed by)."""
        sig = run("openssl", "pkeyutl", "-sign", "-inkey", self.priv, "-pkeyopt", "digest:sha256", input=bytes.fromhex(hexdigest)).stdout
        return json.dumps(dict(schema=sign.SCHEMA, alg=sign.ALG, keyid=self.kid, pub=self.pem, sig=base64.b64encode(sig).decode())).encode()

    def sign_pack(self, path):
        """PATH.sig for a v3 pack: the signature over sha256(b"gc-pack-3\\n" + manifest bytes)."""
        with open(path, "rb") as f:
            _, _, raw, _ = P.read_container(f, os.path.getsize(path))
        open(path + ".sig", "wb").write(self.sign_digest(hashlib.sha256(P.DOMAIN + raw).hexdigest()))


def gcpack(path, manifest, blobs):
    """Write a .gcpack: manifest bytes and {sha256: bytes, or (stored bytes, codec)}."""
    hdr, off, parts = dict(blobs={}), len(manifest), [manifest]
    hdr["manifest"] = [0, len(manifest)]
    for h, b in blobs.items():
        b, *codec = b if isinstance(b, tuple) else (b,)
        hdr["blobs"][h] = [off, len(b), *codec]
        off += len(b)
        parts.append(b)
    hb = json.dumps(hdr).encode()
    open(path, "wb").write(b"GCPACK3\n" + len(hb).to_bytes(8, "little") + hb + b"".join(parts))


def _str(b):
    return len(b).to_bytes(8, "little") + b


def gguf(path, tensors, kv=None, align=32):
    """A GGUF v3 file with zero data: tensors [(name, dims, ggml type id)], kv {key: str}."""
    import struct
    kv = kv or {"general.architecture": "toy"}
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv.items():
        out += _str(k.encode()) + struct.pack("<I", 8) + _str(v.encode())
    off = 0
    for n, d, t in tensors:
        out += _str(n.encode()) + struct.pack("<I", len(d)) + b"".join(struct.pack("<Q", x) for x in d) + struct.pack("<IQ", t, off)
        off += 64
    out += b"\0" * (-len(out) % align) + b"\0" * off
    open(path, "wb").write(out)


TARGET = [("a.weight", [64, 8], 2), ("b.weight", [64], 0)]
ALT = [("a.weight", [64, 9], 2), ("b.weight", [64], 0)]
DRAFT = [("d.weight", [32, 8], 8)]


def toy_pack(d, key, lib, *, spec=False, runtime="gc-c", abi="3.0.0", arch="sm_120a", mutate=None, alt=False, sign=True):
    """A signed toy.gcpack under d with ladder buckets ncol 2 and 8 (and a spec pair if spec) over the library at lib; variant "q4"
    (target) and "d" (drafter), plus variant "alt" with its own buckets if alt. Returns (pack file, weights path, drafter weights path)."""
    pack, blobs = os.path.join(d, "toy.gcpack"), {}
    os.makedirs(d, exist_ok=True)

    def blob(data):
        blobs[hashlib.sha256(data).hexdigest()] = data
        return hashlib.sha256(data).hexdigest()

    w, dw, aw = os.path.join(d, "w.gguf"), os.path.join(d, "d.gguf"), os.path.join(d, "alt.gguf")
    gguf(w, TARGET)
    gguf(dw, DRAFT)
    gguf(aw, ALT)
    keys = ["general.architecture"]
    files = {"libgck.so": blob(open(lib, "rb").read()), "plan.json": blob(b"{}")}
    req = dict(arch=arch, cc="12.0", sms=170, name="toy", smem_block=1)
    caps = dict(verify_widths=[], verify=["linear", "tree"], tree_mask=True, taps=[dict(node="result_norm", n=4)],
                recurrent_commit=[], accept=[], drafter_roles=[])

    alt_files = dict(files, **{"plan.json": blob(b"alt-plan")})       # the alt branch has a blob of its own; libgck.so is shared

    def bucket(i, role, nc, batch=True, io=None, variant="q4", wi=None):
        return dict(id=i, role=role, variant=variant, sku="mock", contract="K1", ncol=nc, batch=batch, maxctx=256,
                    weights=int(role == "drafter") if wi is None else wi, spec_caps=caps, files=alt_files if variant == "alt" else files, lib="libgck.so", plan="plan.json",
                    io=io, tune={}, requires=req, variant_of=None, klass=None, variants=[],
                    determinism="batch_invariant", act="a8", kv="f16")
    bs = [bucket("target.mock.K1.2.256", "target", 2), bucket("target.mock.K1.8.256", "target", 8)]
    sp = []
    if spec:
        bs = [bucket("target.mock.K1.4v.256", "target", 4, False, dict(accept=True)), bucket("drafter.mock.K1.1.256", "drafter", 1, variant="d")]
        sp = [dict(target=bs[0]["id"], drafter=bs[1]["id"], policy=dict(kind="ar", K="3"))]
    if alt:
        bs += [bucket("target.mock.K1.2.256.alt", "target", 2, variant="alt", wi=2), bucket("target.mock.K1.8.256.alt", "target", 8, variant="alt", wi=2)]
    ws = [dict(file="w.gguf", bytes=os.path.getsize(w), sha256=hashlib.sha256(open(w, "rb").read()).hexdigest()),
          dict(file="d.gguf", bytes=os.path.getsize(dw), sha256=hashlib.sha256(open(dw, "rb").read()).hexdigest()),
          dict(file="alt.gguf", bytes=os.path.getsize(aw), sha256=hashlib.sha256(open(aw, "rb").read()).hexdigest())]
    variants = {"q4": dict(structure=structure.describe([w], keys), weights=[0]),
                "d": dict(structure=structure.describe([dw], keys), weights=[1]),
                "alt": dict(structure=structure.describe([aw], keys), weights=[2])}
    m = dict(schema="gc-pack", abi=abi, compiler=dict(commit="0" * 40, dirty=False), arch=arch, cuda="12.8",
             hw=dict(mock="0" * 64), runtime=dict(kind=runtime, version=1), contracts=["K1"], buckets=bs, spec=sp,
             model=dict(key="t", stage="t", graph_sha256="0" * 64, weights=ws), variants=variants)
    if mutate:
        mutate(m)
    gcpack(pack, (json.dumps(m, indent=1, sort_keys=True) + "\n").encode(), blobs)
    if sign:
        key.sign_pack(pack)
    return pack, w, dw
