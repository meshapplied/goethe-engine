"""goethe-engine serve on the CPU reference model: routes, SSE, auth, errors, concurrent batching."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from goethe_engine import serve


@pytest.fixture(scope="module")
def srv():
    args = serve.parser().parse_args(["--ref", "--port", "0", "--served-model-name", "ref", "--api-key", "k",
                                      "--max-running-requests", "4", "--context-length", "128"])
    httpd, s = serve.make_server(args)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", s
    httpd.shutdown()
    s.stop = True


def call(base, path, body=None, key="k", raw=False):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    data = r.read()
    return r.status, data if raw else json.loads(data)


def sse(base, path, body):
    st, data = call(base, path, dict(body, stream=True), raw=True)
    evs = [ln[6:] for ln in data.decode().split("\n\n") if ln.startswith("data: ")]
    assert st == 200 and evs[-1] == "[DONE]"
    return [json.loads(e) for e in evs[:-1]]


def test_health_models_and_auth(srv):
    base, _ = srv
    assert call(base, "/health", key=None) == (200, {"status": "ok"})
    st, j = call(base, "/v1/models")
    assert st == 200 and j["data"][0]["id"] == "ref" and j["data"][0]["max_model_len"] == 128
    assert call(base, "/v1/models", key="bad")[0] == 401
    assert call(base, "/v1/models", key=None)[0] == 401
    assert call(base, "/v1/completions", dict(prompt="a"), key="bad")[0] == 401
    assert call(base, "/nope")[0] == 404


def test_completion_and_chat_shape(srv):
    base, _ = srv
    st, j = call(base, "/v1/completions", dict(prompt="hello", max_tokens=5, temperature=0))
    c = j["choices"][0]
    assert st == 200 and j["object"] == "text_completion" and j["usage"] == dict(prompt_tokens=5, completion_tokens=5, total_tokens=10)
    assert c["finish_reason"] == "length" and isinstance(c["text"], str)
    st, j = call(base, "/v1/chat/completions", dict(messages=[{"role": "user", "content": "hi"}], max_tokens=4, temperature=0))
    assert st == 200 and j["choices"][0]["message"]["role"] == "assistant" and j["usage"]["completion_tokens"] == 4


def test_greedy_is_deterministic_and_stream_equals_json(srv):
    base, _ = srv
    body = dict(prompt="abc", max_tokens=12, temperature=0)
    t1 = call(base, "/v1/completions", body)[1]["choices"][0]["text"]
    assert call(base, "/v1/completions", body)[1]["choices"][0]["text"] == t1
    evs = sse(base, "/v1/completions", dict(body, stream_options={"include_usage": True}))
    assert "".join(e["choices"][0]["text"] for e in evs if e["choices"]) == t1
    assert evs[-1]["choices"] == [] and evs[-1]["usage"]["completion_tokens"] == 12
    assert [e["choices"][0]["finish_reason"] for e in evs if e["choices"]][-1] == "length"


def test_chat_stream_roles_and_finish(srv):
    base, _ = srv
    evs = sse(base, "/v1/chat/completions", dict(messages=[{"role": "user", "content": "x"}], max_tokens=6, temperature=0))
    assert evs[0]["choices"][0]["delta"]["role"] == "assistant" and evs[-1]["choices"][0]["finish_reason"] == "length"
    assert all(e["object"] == "chat.completion.chunk" for e in evs)


def test_stop_string_truncates(srv):
    base, _ = srv
    full = call(base, "/v1/completions", dict(prompt="abc", max_tokens=30, temperature=0))[1]["choices"][0]["text"]
    assert len(full) > 4
    stop = full[3:5]
    st, j = call(base, "/v1/completions", dict(prompt="abc", max_tokens=30, temperature=0, stop=stop))
    assert j["choices"][0]["text"] == full[:full.index(stop)] and j["choices"][0]["finish_reason"] == "stop"
    evs = sse(base, "/v1/completions", dict(prompt="abc", max_tokens=30, temperature=0, stop=stop))
    assert "".join(e["choices"][0]["text"] for e in evs) == full[:full.index(stop)]


def test_n_and_seed(srv):
    base, _ = srv
    b = dict(prompt="abc", max_tokens=6, temperature=1.0, n=2, seed=3)
    j = call(base, "/v1/completions", b)[1]
    assert sorted(c["index"] for c in j["choices"]) == [0, 1] and j["usage"]["completion_tokens"] == 12
    assert call(base, "/v1/completions", b)[1]["choices"] == j["choices"]


def test_errors(srv):
    base, _ = srv
    for path, body in (("/v1/completions", {}), ("/v1/completions", dict(prompt="a", logprobs=3)),
                       ("/v1/chat/completions", dict(messages=[])), ("/v1/completions", dict(prompt="a", temperature=-1)),
                       ("/v1/completions", dict(prompt="a" * 200)), ("/v1/completions", dict(prompt="a", n=9, temperature=1)),
                       ("/v1/completions", dict(prompt="a", stop=[""])), ("/v1/completions", dict(prompt="a", tools=[1]))):
        st, j = call(base, path, body)
        assert st == 400 and j["error"]["type"] == "invalid_request_error", (body, st, j)


def test_concurrent_requests_batch_and_match_alone(srv):
    base, s = srv
    prompts = [f"prompt number {i}" for i in range(6)]
    body = lambda p: dict(prompt=p, max_tokens=10, temperature=0)
    alone = [call(base, "/v1/completions", body(p))[1]["choices"][0]["text"] for p in prompts]
    peak, got = [0], [None] * 6
    steps0 = s.engine.metrics()["steps"]

    def go(i):
        got[i] = call(base, "/v1/completions", body(prompts[i]))[1]["choices"][0]["text"]
    ts = [threading.Thread(target=go, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert got == alone
    assert s.engine.metrics()["steps"] - steps0 < 6 * 15   # continuous batching: fewer steps than one request at a time
    s.engine.check()


def test_client_disconnect_frees_engine(srv):
    base, s = srv
    import http.client
    c = http.client.HTTPConnection(base[7:])
    c.request("POST", "/v1/completions", json.dumps(dict(prompt="abc", max_tokens=100, temperature=0, stream=True)),
              {"Authorization": "Bearer k"})
    c.getresponse().read(20)
    c.close()
    for _ in range(100):
        if not s.engine.has_work():
            break
        import time
        time.sleep(0.1)
    assert not s.engine.has_work() and not s.pending


def test_cli_refuses_unsigned_pack(tmp_path, capsys):
    from goethe_engine import cli
    (tmp_path / "w").write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", "--pack", str(tmp_path), "--weights", str(tmp_path / "w")])
    assert "REFUSED" in str(e.value)
