"""Detached signatures, verify only: ECDSA P-256 / SHA-256 through the openssl CLI, key pinned by key id.

Signature file (JSON): {"schema": "gc-sig-1", "alg": "ecdsa-p256-sha256", "keyid": sha256(DER public key) hex,
"pub": PEM public key, "sig": base64 DER signature over the exact bytes of the signed file}.
The embedded key is trusted only if its key id is in the pinned list; every failure raises SignatureError."""
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

# Pinned signer key ids: sha256 of the DER public key. To rotate, release a version that lists the new id.
TRUSTED_KEY_IDS = ("3600e67ccf414f376ef85b529c5132d8dec8bb1c52c13724c856d5ff0631073f",)

ALLOW_UNSIGNED_ENV = "GOETHE_ENGINE_ALLOW_UNSIGNED"   # development only: "1" skips signature and key checks
MAX_SIG = 64 << 10                                    # bytes: larger signature files are refused
ALG, SCHEMA = "ecdsa-p256-sha256", "gc-sig-1"
HEX = re.compile(r"^[0-9a-f]{64}$")


class SignatureError(Exception):
    """A missing, malformed, untrusted or wrong signature: the signed file is not used."""


def _openssl(*args, data=None, timeout=30):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OPENSSL_CONF": os.devnull}   # argument list, no shell
    try:
        r = subprocess.run(["openssl", *args], input=data, capture_output=True, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        raise SignatureError(f"openssl unavailable: {e}")
    return r


def key_id(pub_pem):
    """sha256 of the DER public key; raises SignatureError if pub_pem is not a P-256 public key."""
    r = _openssl("pkey", "-pubin", "-outform", "DER", data=pub_pem)
    if r.returncode or not r.stdout:
        raise SignatureError("public key unreadable")
    t = _openssl("pkey", "-pubin", "-noout", "-text", data=pub_pem).stdout
    if b"prime256v1" not in t and b"P-256" not in t:
        raise SignatureError("public key is not P-256")
    return hashlib.sha256(r.stdout).hexdigest()


def verify(data, sig, trusted=None, digest=None):
    """Checks sig (bytes of the signature file, None if absent) over data (bytes, or the path of a file), or, with digest (hex sha256),
    the ECDSA signature over that digest (data is then unused); returns the signer's key id.
    trusted: key ids to accept (default TRUSTED_KEY_IDS). With the dev override set: returns "UNSIGNED" unchecked."""
    if os.environ.get(ALLOW_UNSIGNED_ENV) == "1":
        print(f"*** {ALLOW_UNSIGNED_ENV}=1: signature check disabled, content is NOT authenticated (development only) ***",
              file=sys.stderr, flush=True)
        return "UNSIGNED"
    if sig is None:
        raise SignatureError("no signature (refused)")
    if len(sig) > MAX_SIG:
        raise SignatureError("signature file too large (refused)")
    try:
        j = json.loads(sig)
        kid, pub, raw = j["keyid"], j["pub"].encode(), base64.b64decode(j["sig"], validate=True)
        ok = j.get("schema") == SCHEMA and j.get("alg") == ALG and HEX.match(kid)
    except (ValueError, KeyError, TypeError, AttributeError):
        raise SignatureError("signature file malformed (refused)")
    if not ok:
        raise SignatureError("signature file has an unknown schema or algorithm (refused)")
    if kid not in (TRUSTED_KEY_IDS if trusted is None else trusted):
        raise SignatureError(f"key id {kid} is not in the pinned allow-list (refused)")
    if key_id(pub) != kid:
        raise SignatureError("embedded public key does not match its key id (refused)")
    if digest is not None:
        with tempfile.TemporaryDirectory() as d:
            for n, b in (("pub.pem", pub), ("sig.bin", raw), ("digest.bin", bytes.fromhex(digest))):
                with open(os.path.join(d, n), "wb") as f:
                    f.write(b)
            r = _openssl("pkeyutl", "-verify", "-pubin", "-inkey", os.path.join(d, "pub.pem"), "-in", os.path.join(d, "digest.bin"),
                         "-sigfile", os.path.join(d, "sig.bin"), "-pkeyopt", "digest:sha256")
        if r.returncode or b"Successfully" not in r.stdout:
            raise SignatureError("signature does not verify (refused)")
        return kid
    with tempfile.TemporaryDirectory() as d:
        for n, b in (("pub.pem", pub), ("sig.bin", raw)) + (() if isinstance(data, (str, os.PathLike)) else (("data", data),)):
            with open(os.path.join(d, n), "wb") as f:
                f.write(b)
        r = _openssl("dgst", "-sha256", "-verify", os.path.join(d, "pub.pem"), "-signature",
                     os.path.join(d, "sig.bin"), os.fspath(data) if isinstance(data, (str, os.PathLike)) else os.path.join(d, "data"), timeout=900)
    if r.returncode or b"Verified OK" not in r.stdout:
        raise SignatureError("signature does not verify (refused)")
    return kid
