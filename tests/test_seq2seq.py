"""Encoder-decoder model type: the encoder once per sequence into its cross slot, the Engine loop unchanged, equal to a dense oracle under
continuous batching and preemption; cross slots freed with the sequence; /v1/audio/transcriptions."""
import json
import threading
import urllib.request

import pytest
import torch

from goethe_engine import registry as R, serve
from goethe_engine.sampling import SamplingParams
from goethe_engine.scheduler import Config
from goethe_engine.seq2seq import GcSeq2Seq, RefSeq2Seq, RefSeq2SeqBackend, Seq2SeqRunner


def frames(seed, T):
    return torch.randn(T, 8, generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module")
def model():
    return RefSeq2Seq(vocab=64, seed=0)


def run(eng, reqs):
    for rid, p, f, n in reqs:
        eng.add_request(rid, p, SamplingParams(max_tokens=n, temperature=0), frames=f)
    out = {}
    while eng.has_work():
        for o in eng.step():
            out.setdefault(o.rid, []).extend(o.token_ids)
    return out


def test_matches_the_dense_oracle_under_batching(model):
    be = RefSeq2SeqBackend(model, num_blocks=64, block_size=4, max_seqs=4)
    eng = R.runner(be, Config(max_num_seqs=4, max_model_len=48))
    assert isinstance(eng, Seq2SeqRunner) and not eng.cfg.prefix_cache
    reqs = [(f"r{i}", [1, 2, 3 + i % 2], frames(i, 5 + 3 * i), 12) for i in range(6)]
    got = run(eng, reqs)
    for rid, p, f, n in reqs:
        assert got[rid] == model.dense(f, p, n), rid
    assert got["r0"] != got["r2"] or reqs[0][2].shape != reqs[2][2].shape
    assert be.cross == {} and be.stats["encodes"] == 6


def test_same_prompt_other_audio_gives_other_tokens(model):
    be = RefSeq2SeqBackend(model, num_blocks=64, block_size=4, max_seqs=4)
    got = run(R.runner(be, Config(max_num_seqs=4, max_model_len=48)), [("a", [5, 6], frames(1, 9), 10), ("b", [5, 6], frames(2, 9), 10)])
    assert got["a"] != got["b"]


def test_preempted_sequences_are_encoded_again(model):
    be = RefSeq2SeqBackend(model, num_blocks=9, block_size=4, max_seqs=3)
    eng = R.runner(be, Config(max_num_seqs=3, max_model_len=24, max_num_batched_tokens=8))
    reqs = [(f"p{i}", [1, 2, 3], frames(10 + i, 6), 14) for i in range(3)]
    got = run(eng, reqs)
    for rid, p, f, n in reqs:
        assert got[rid] == model.dense(f, p, n), rid
    assert be.stats["encodes"] > 3 and be.cross == {}


def test_frames_are_checked(model):
    eng = R.runner(RefSeq2SeqBackend(model, num_blocks=16, block_size=4, max_seqs=2), Config(max_num_seqs=2, max_model_len=24))
    for bad in (None, torch.zeros(5, 7), torch.zeros(0, 8), torch.zeros(40, 8)):
        with pytest.raises(ValueError, match="frames"):
            eng.add_request("x", [1], SamplingParams(max_tokens=2), frames=bad)


def test_gc_backend_refuses_a_library_without_seq2seq(toy, key):
    pack, w, _ = toy
    with pytest.raises(RuntimeError, match="GC_L_SEQ2SEQ"):
        GcSeq2Seq(pack, w, sku="mock", trusted=(key.kid,), B=8)


def test_v1_audio_transcriptions():
    args = serve.parser().parse_args(["--ref", "--ref-type", "encoder-decoder", "--port", "0", "--context-length", "40"])
    httpd, s = serve.make_server(args)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode())
        try:
            return 200, json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        f = frames(3, 7)
        st, r = call("/v1/audio/transcriptions", dict(frames=f.tolist(), prompt=[1, 2], max_tokens=8))
        m = s.engine.backend._be.m
        assert st == 200 and r["tokens"] == [t for t in m.dense(f, [1, 2], 8)][:len(r["tokens"])] and r["tokens"]
        assert call("/v1/audio/transcriptions", dict(frames=[[0.0] * 3], prompt=[1]))[0] == 400
        assert call("/v1/completions", dict(prompt="hi"))[0] == 400
        assert call("/v1/embeddings", dict(input="hi"))[0] == 400
    finally:
        httpd.shutdown()
        s.stop = True
