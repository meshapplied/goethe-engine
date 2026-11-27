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
    got = [None] * 6
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


def test_hf_chat_accepts_a_mapping_result():
    """transformers 5 returns {"input_ids": [...], ...} from apply_chat_template, older versions a list."""
    from goethe_engine.tokenizer import HFTokenizer
    t = HFTokenizer.__new__(HFTokenizer)
    for r in ({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}, [1, 2, 3]):
        t.t = type("T", (), dict(chat_template="x", apply_chat_template=lambda self, *a, r=r, **k: r))()
        assert t.chat([{"role": "user", "content": "hi"}]) == [1, 2, 3]


def test_stop_ids_come_from_generation_config_then_config(tmp_path):
    """A chat model ends a turn on its own token (gemma <turn|>, qwen <|im_end|>): every declared stop id ends a generation."""
    import json
    from goethe_engine.tokenizer import model_stop_ids
    assert model_stop_ids(str(tmp_path)) == []
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 7}))
    assert model_stop_ids(str(tmp_path)) == [7]
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106, 50]}))
    assert model_stop_ids(str(tmp_path)) == [1, 106, 50]
    (tmp_path / "generation_config.json").write_text("not json")
    assert model_stop_ids(str(tmp_path)) == [7]
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": None}))
    assert model_stop_ids(str(tmp_path)) == []


def test_parameter_refusals_have_one_shape(srv):
    """Every parameter refusal: HTTP 400 {"error": {type invalid_request_error, code (spec/errors.json), param, message}}."""
    import json as J
    import os as O
    codes = J.load(open(O.path.join(O.path.dirname(__file__), "..", "goethe_engine", "spec", "errors.json")))["codes"]
    for body, code, param in ((dict(top_p=2.0), "invalid_parameter", "top_p"), (dict(logprobs=True), "unsupported_parameter", "logprobs"),
                              (dict(presence_penalty=0.5), "unsupported_parameter", "presence_penalty"), (dict(stop=[""]), "invalid_parameter", "stop"),
                              (dict(foo_bar=1), "unsupported_parameter", "foo_bar")):
        st, r = call(srv[0], "/v1/completions", dict(dict(prompt="hi", max_tokens=2), **body))
        e = r["error"]
        assert st == 400 and e["type"] == "invalid_request_error" and e["code"] == code and e["param"] == param and e["message"], body
        assert e["code"] in codes


def test_a_refused_request_leaves_a_kept_alive_connection_usable(srv):
    """A refusal before the body is used (bad key, no route, wrong task) still reads it: the next request on the connection (goethed
    pools them) is parsed from its own start."""
    import http.client
    import urllib.parse
    u = urllib.parse.urlparse(srv[0])
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
    ok = {"Authorization": "Bearer k", "Content-Type": "application/json"}
    for path, h in (("/v1/embeddings", ok), ("/nope", ok), ("/v1/completions", {"Authorization": "Bearer bad"})):
        c.request("POST", path, json.dumps(dict(prompt="hi", input=["a"], max_tokens=2)), h)
        r = c.getresponse()
        assert r.status in (400, 401, 404) and "error" in json.loads(r.read()), path
        c.request("POST", "/v1/completions", json.dumps(dict(prompt="hi", max_tokens=2, temperature=0)), ok)
        r = c.getresponse()
        assert r.status == 200 and json.loads(r.read())["usage"]["completion_tokens"] == 2, path
    c.close()


def test_wrong_task_names_the_model_kinds_customers_see(srv):
    st, r = call(srv[0], "/v1/embeddings", dict(input="hi"))
    assert st == 400 and r["error"]["code"] == "unsupported_task"
    assert "this model is a text model, not an embedding model" in r["error"]["message"]
    st, r = call(srv[0], "/v1/models")
    assert r["data"][0]["kind"] == "text"


class _Chars:
    """Ids below 1000 are characters; 1001 .. are a turn format's markers."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i < 1000)


FMT = dict(parts={1001: ("head", None), 1003: ("reasoning", 1002), 1004: ("content", 1002), 1005: ("tool", 1002)}, args="args")
PAIRS = dict(parts={1006: ("reasoning", 1007), 1008: ("tool", 1009)}, args="arguments")     # open/close pairs, text between is content
_s = lambda x: [ord(c) for c in x]


@pytest.mark.parametrize("fmt,ids,want", [
    (FMT, [1001, 1003, *_s("hm"), 1002, 1001, 1004, *_s("hi"), 1002], [("reasoning", "hm"), ("content", "hi")]),
    (FMT, [1001, *_s("f"), 1005, *_s('{"name":"f","args":{"a":1}}'), 1002], [("tool", {"name": "f", "arguments": '{"a": 1}'})]),
    (FMT, [1001, 1005, *_s('{"args":1}'), 1002], [("content", '{"args":1}')]),        # not a call: kept as text, never dropped
    (FMT, _s("no markers"), [("content", "no markers")]),
    (PAIRS, [1006, *_s("hm"), 1007, *_s("ok "), 1008, *_s('{"name":"g","arguments":{}}'), 1009],
     [("reasoning", "hm"), ("content", "ok "), ("tool", {"name": "g", "arguments": "{}"})]),
])
def test_a_turn_reads_reasoning_content_and_whole_tool_calls_alike_streamed_or_not(fmt, ids, want):
    from goethe_engine.tokenizer import Turn
    join = lambda ps: [(k, "".join(v for _, v in g)) if k != "tool" else (k, next(g)[1])
                       for k, g in __import__("itertools").groupby(ps, key=lambda p: p[0])]
    t = Turn(_Chars(), fmt)
    assert join(Turn(_Chars(), fmt).feed(ids, True)) == join(sum((t.feed([i]) for i in ids), []) + t.feed([], True)) == want


def test_a_choice_stops_on_content_only_and_keeps_one_call_when_asked():
    calls = [1001, 1005, *_s('{"name":"f","args":{}}'), 1002] * 2
    c = serve.Choice(_Chars(), ("hm",), FMT, one_call=True)
    c.feed([1001, 1003, *_s("hm"), 1002, *calls, 1001, 1004, *_s("ahmb"), 1002], final=True)
    assert (c.reasoning, [x["function"]["name"] for x in c.calls], c.text, c.finish) == ("hm", ["f"], "a", "stop")


def test_messages_are_parsed_once_at_the_boundary():
    m = serve.message(dict(role="assistant", content=None, tool_calls=[dict(id="c", function=dict(name="f", arguments='{"x": 1}'))]))
    assert m == dict(role="assistant", content="", tool_calls=[dict(id="c", type="function", function=dict(name="f", arguments={"x": 1}))])
    for bad in (dict(role="user", content=3), dict(role="assistant", tool_calls=[dict(function=dict(name="f", arguments="{x"))])):
        with pytest.raises(serve.ApiError):
            serve.message(bad)


def test_tools_are_refused_where_the_engine_cannot_read_calls_back(srv):
    base, _ = srv
    tool = dict(type="function", function=dict(name="f", parameters={}))
    for path, body in (("/v1/chat/completions", dict(messages=[{"role": "user", "content": "x"}], tools=[tool])),
                       ("/v1/chat/completions", dict(messages=[{"role": "user", "content": "x"}], tool_choice="required")),
                       ("/v1/completions", dict(prompt="x", tools=[tool]))):
        st, j = call(base, path, body)
        assert st == 400 and j["error"]["code"] == "unsupported_parameter", (body, j)
