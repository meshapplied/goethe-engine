"""Embedding model type: launch packing, batch invariance, the gc.h 1.2 gc_embed binding (mock libgce), encode buckets never serve
a decoder, /v1/embeddings."""
import base64
import json
import os
import subprocess
import threading
import urllib.request

import numpy as np
import pytest
import torch

from goethe_engine import pack as P, registry as R, serve
from goethe_engine.embed import EmbedRunner, GcEmbed, RefEmbed, normalize
from goethe_engine.ref import RefModel
from toypack import toy_pack


@pytest.fixture(scope="module")
def ref():
    return RefEmbed(RefModel(vocab=64, seed=0), pool="mean", max_cols=24, max_seqs=3, max_len=16)


def seqs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 64, (int(torch.randint(1, 13, (1,), generator=g)),), generator=g).tolist() for _ in range(n)]


def test_launches_respect_rows_and_sequences(ref):
    r = EmbedRunner(ref)
    items = [[1] * 10, [2] * 10, [3] * 5, [4], [5], [6], [7] * 16]
    bs = r.batches(items)
    assert bs == [[0, 1], [2, 3, 4], [5, 6]]
    assert all(sum(len(items[i]) for i in b) <= ref.max_cols and len(b) <= ref.max_seqs for b in bs)
    with pytest.raises(P.PackError, match="17 tokens"):
        r.check([0] * 17)


def test_rows_do_not_depend_on_the_batch(ref):
    """Bitwise: each sequence alone, in arrival order packed by rows, and shuffled into other launches."""
    r, items = EmbedRunner(ref), seqs(9)
    alone = np.stack([r.embed([t])[0] for t in items])
    packed = r.embed(items)
    perm = list(reversed(range(len(items))))
    shuffled = r.embed([items[i] for i in perm])
    assert np.array_equal(alone, packed) and np.array_equal(packed[perm], shuffled)
    assert np.allclose(np.linalg.norm(packed, axis=1), 1, atol=1e-6)
    raw = r.embed(items, norm=False)
    assert np.array_equal(normalize(raw), packed) and not np.array_equal(raw, packed)


def test_pool_kinds_differ():
    m = RefModel(vocab=64, seed=0)
    t = [[3, 9, 27, 17, 51]]
    rows = {p: RefEmbed(m, pool=p).embed(t, norm=False)[0] for p in ("mean", "cls", "last")}
    assert not np.array_equal(rows["mean"], rows["cls"]) and not np.array_equal(rows["cls"], rows["last"])


def test_normalize_is_llama_cpp_common_embd_normalize():
    x = np.array([[3.0, 4.0, 0.0], [1e-3, -2.5, 7.25], [0.0, 0.0, 0.0]], np.float32)
    y = normalize(x)
    assert np.array_equal(y[0], np.array([0.6, 0.8, 0.0], np.float32)) and np.array_equal(y[2], x[2])
    s = 0.0
    for v in x[1].tolist():
        s += float(np.float32(v) * np.float32(v))
    assert np.array_equal(y[1], x[1] * np.float32(1.0 / s ** 0.5))


def test_runner_queue_packs_concurrent_callers(ref):
    r, items = EmbedRunner(ref), seqs(6, seed=1)
    want = r.embed(items)
    qs = [r.submit([t]) for t in items]
    got = np.concatenate([q.get(timeout=30) for q in qs])
    assert np.array_equal(got, want)


@pytest.fixture(scope="session")
def mockgce(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("mockgce") / "libgce.so")
    src = os.path.join(os.path.dirname(__file__), "mock", "gce_mock.c")
    subprocess.run(["cc", "-std=c99", "-Wall", "-O1", "-shared", "-fPIC", "-o", out, src, "-lm"], check=True)
    return out


def encode_pack(tmp_path, key, lib, decoder=False):
    """A signed toy pack whose model is an embedding model with one encode bucket (64 rows), plus the decoder ladder if decoder."""
    def mut(m):
        b = dict(m["buckets"][1], id="target.mock.K1.64e.256", kind="encode", ncol=64, batch=False,
                 io=dict(out=dict(dim=4, pool="mean", normalize=True)))
        m["buckets"] = (m["buckets"] if decoder else []) + [b]
        if not decoder:
            m["model"]["type"] = "embedding"
    return toy_pack(str(tmp_path), key, lib, mutate=mut)


def test_encode_bucket_never_serves_a_decoder(tmp_path, key, mocklib):
    pack, w, _ = encode_pack(tmp_path, key, mocklib, decoder=True)
    m = P.stage(pack, trusted=(key.kid,), defer=True).manifest
    for B, verify in ((1, True), (2, False), (8, True), (64, True)):
        try:
            b = P.select(m, sku="mock", B=B, verify=verify)
        except P.PackError:
            continue
        assert b.get("kind") != "encode", (B, verify, b["id"])
    with pytest.raises(P.PackError):
        P.select(m, sku="mock", B=64, verify=True)
    assert P.select(m, sku="mock", verify=True, kind="encode")["id"] == "target.mock.K1.64e.256"
    assert all(x.get("kind") != "encode" for b in m["buckets"] for x in P.ladder(m, b) if b.get("kind") != "encode")
    assert "encode" in P.KINDS and P.axes(m["buckets"][-1])[1] == "encode"


def test_gc_embed_through_the_registry(tmp_path, key, mockgce):
    pack, w, _ = encode_pack(tmp_path, key, mockgce)
    assert R.pack_type(pack) == "embedding"
    be = R.open_backend(pack, w, mesh=R.Mesh.single(0), sku="mock", trusted=(key.kid,))
    try:
        assert isinstance(be, GcEmbed) and (be.dim, be.pool, be.max_cols, be.max_seqs) == (4, "mean", 64, 8)
        r = R.runner(be, None)
        assert isinstance(r, EmbedRunner)
        items = [[1, 2, 3], [7], [4, 4, 4, 4, 9]]
        raw = r.embed(items, norm=False)
        assert raw.tolist() == [[6, 3, 1, 3], [7, 1, 7, 7], [25, 5, 4, 9]]
        assert np.allclose(r.embed(items), normalize(raw), atol=1e-7)
        assert np.array_equal(np.concatenate([r.embed([t]) for t in items]), r.embed(items))
    finally:
        be.close()


def call(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode())
    try:
        return 200, json.loads(urllib.request.urlopen(req, timeout=60).read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def esrv():
    args = serve.parser().parse_args(["--ref", "--ref-type", "embedding", "--port", "0", "--served-model-name", "emb",
                                      "--context-length", "32", "--max-running-requests", "4"])
    httpd, s = serve.make_server(args)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", s
    httpd.shutdown()
    s.stop = True


def test_v1_embeddings(esrv):
    base, s = esrv
    st, r = call(base, "/v1/embeddings", dict(input=["hello", "a longer text"], model="emb"))
    assert st == 200 and r["object"] == "list" and [d["index"] for d in r["data"]] == [0, 1]
    e0 = torch.tensor(r["data"][0]["embedding"])
    assert abs(float(e0.norm()) - 1) < 1e-5 and r["usage"]["prompt_tokens"] == 5 + 13
    st, one = call(base, "/v1/embeddings", dict(input="hello"))
    assert st == 200 and one["data"][0]["embedding"] == r["data"][0]["embedding"]
    st, ids = call(base, "/v1/embeddings", dict(input=list(b"hello")))
    assert ids["data"][0]["embedding"] == r["data"][0]["embedding"]
    st, b64 = call(base, "/v1/embeddings", dict(input="hello", encoding_format="base64"))
    v = torch.frombuffer(bytearray(base64.b64decode(b64["data"][0]["embedding"])), dtype=torch.float32)
    assert torch.equal(v, e0)
    assert call(base, "/v1/embeddings", dict(input="x" * 40))[0] == 400
    st, r = call(base, "/v1/embeddings", dict(input="x" * 40, truncation="right"))
    assert st == 400 and r["error"]["code"] == "unsupported_parameter" and r["error"]["param"] == "truncation"   # not declared: refused
    st, r = call(base, "/v1/embeddings", dict(input="x", foo_bar=1))
    assert st == 400 and r["error"]["code"] == "unsupported_parameter" and r["error"]["param"] == "foo_bar"
    be = getattr(s.engine, "be", None) or s.engine.backend
    old = getattr(be, "bucket", None)
    be.bucket = {"io": {"params": ["normalize", "truncation"]}}
    try:
        st, r = call(base, "/v1/embeddings", dict(input="x" * 40, truncation="right"))
        assert st == 200 and r["usage"]["prompt_tokens"] == 32
    finally:
        be.bucket = old
    assert call(base, "/v1/embeddings", dict(input=[]))[0] == 400
    assert call(base, "/v1/completions", dict(prompt="hi"))[0] == 400
