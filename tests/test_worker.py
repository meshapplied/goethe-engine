"""Worker protocol v1 over a UNIX socket with the toy packs (mock libraries): handshake and protocol range, load, generate (whole and
streamed, equal to the engine), embed, transcribe, parameter errors with codes, cancel, health, metrics, unload; the descriptor set is
the compiled proto. Needs the protobuf runtime (the [worker] extra)."""
import json
import threading

import numpy as np
import pytest

pytest.importorskip("google.protobuf")
from goethe_engine import worker as W  # noqa: E402
from conftest import fake_greedy  # noqa: E402
from test_embed import encode_pack, mockgce  # noqa: E402,F401
from test_media import media_pack, mockgcm  # noqa: E402,F401
from toypack import toy_pack  # noqa: E402


@pytest.fixture
def client(tmp_path):
    path = str(tmp_path / "w.sock")
    ready = threading.Event()
    threading.Thread(target=W.serve, args=(path,), kwargs=dict(ready=ready), daemon=True).start()
    ready.wait(10)
    return W.Client(path)


def test_protocol_session(client, tmp_path, key, mocklib, mockgce, mockgcm):  # noqa: F811
    r = client.call("hello", min_protocol=2, max_protocol=3)[0]
    assert r.WhichOneof("body") == "error" and r.error.code == "refused"
    r = client.call("hello", min_protocol=1, max_protocol=9)[0]
    assert r.welcome.protocol == 1 and r.welcome.engine_version
    pack, w, _ = toy_pack(str(tmp_path / "d"), key, mocklib)
    opts = json.dumps(dict(B=2, sku="mock", trusted=[key.kid]))
    r = client.call("load", model_id="d", pack=[pack], weights=[w], options_json=opts)[0]
    assert r.loaded.task == "generate" and json.loads(r.loaded.capabilities_json)["kind"] == "generate"
    assert client.call("warm", model_id="d")[0].WhichOneof("body") == "ok"
    p = [1, 2, 3]
    r = client.call("generate", model_id="d", prompt=dict(ids=p), params_json='{"max_tokens": 7}')
    assert list(r[-1].chunk.ids) == fake_greedy(p, 7) and r[-1].chunk.done and r[-1].chunk.finish_reason == "length"
    r = client.call("generate", model_id="d", prompt=dict(ids=p), params_json='{"max_tokens": 7}', stream=True)
    assert [t for x in r for t in x.chunk.ids] == fake_greedy(p, 7) and len(r) > 1
    r = client.call("generate", model_id="d", prompt=dict(ids=p), params_json='{"beam": 2}')[0]
    assert r.error.code == "unsupported_parameter" and r.error.param == "beam"
    r = client.call("generate", model_id="d", prompt=dict(ids=p), params_json='{"top_p": 2}')[0]
    assert r.error.code == "invalid_parameter"
    epack, ew, _ = encode_pack(tmp_path / "e", key, mockgce)
    client.call("load", model_id="e", pack=[epack], weights=[ew], options_json=json.dumps(dict(sku="mock", trusted=[key.kid])))
    r = client.call("embed", model_id="e", items=[dict(ids=[1, 2, 3])], params_json='{"normalize": false}')[0]
    assert r.embed.dim == 4 and list(r.embed.rows) == [6, 3, 1, 3]
    (tmp_path / "m").mkdir()
    mpack, mws = media_pack(tmp_path / "m", key, mockgcm)
    client.call("load", model_id="m", pack=[mpack], weights=mws, options_json=json.dumps(dict(sku="mock", trusted=[key.kid])))
    r = client.call("transcribe", model_id="m", pcm=np.ones(4).tolist(), prompt="<m>", params_json='{"max_new_tokens": 2}')[0]
    assert list(r.transcribe.ids) == [14, 1000]
    h = client.call("health")[0].health
    assert h.ok and list(h.model_ids) == ["d", "e", "m"]
    assert "d" in json.loads(client.call("metrics")[0].metrics.json)
    assert client.call("cancel", target=999)[0].WhichOneof("body") == "ok"
    assert client.call("unload", model_id="e")[0].WhichOneof("body") == "ok"
    assert client.call("embed", model_id="e", items=[dict(ids=[1])])[0].error.code == "refused"


def test_first_request_must_be_hello(client):
    assert client.call("health")[0].error.code == "refused"
