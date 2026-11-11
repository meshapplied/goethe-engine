import json

import pytest

from goethe_engine import sign
from toypack import Key


def test_good_signature_verifies(key):
    assert sign.verify(b"data", key.sign(b"data"), (key.kid,)) == key.kid


def test_refusals(key, tmp_path):
    s = key.sign(b"data")
    with pytest.raises(sign.SignatureError, match="does not verify"):
        sign.verify(b"datb", s, (key.kid,))
    with pytest.raises(sign.SignatureError, match="no signature"):
        sign.verify(b"data", None, (key.kid,))
    with pytest.raises(sign.SignatureError, match="malformed"):
        sign.verify(b"data", b"not json", (key.kid,))
    with pytest.raises(sign.SignatureError, match="allow-list"):
        sign.verify(b"data", s, ())
    other = Key(str(tmp_path))
    j = json.loads(s)
    j["pub"] = other.pem   # claims our key id, carries another key
    with pytest.raises(sign.SignatureError, match="does not match"):
        sign.verify(b"data", json.dumps(j).encode(), (key.kid,))
    j = json.loads(other.sign(b"data"))
    with pytest.raises(sign.SignatureError, match="allow-list"):   # a valid signature by an unpinned key
        sign.verify(b"data", json.dumps(j).encode(), (key.kid,))
    j = json.loads(s)
    j["alg"] = "rsa"
    with pytest.raises(sign.SignatureError, match="algorithm"):
        sign.verify(b"data", json.dumps(j).encode(), (key.kid,))


def test_non_p256_key_refused(tmp_path):
    import subprocess
    k, pub = str(tmp_path / "r.pem"), str(tmp_path / "rp.pem")
    subprocess.run(["openssl", "ecparam", "-name", "secp384r1", "-genkey", "-noout", "-out", k], check=True, capture_output=True)
    subprocess.run(["openssl", "ec", "-in", k, "-pubout", "-out", pub], check=True, capture_output=True)
    with pytest.raises(sign.SignatureError, match="P-256"):
        sign.key_id(open(pub, "rb").read())
