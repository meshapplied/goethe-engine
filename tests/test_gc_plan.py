"""gc-plan (data-only packs run by gc_plan.py): GGUF data offsets, the inputs the engine generates, the runtime kind accepted. CPU only;
the device side (Runtime, PlanEmbed) is checked on a GPU against llama.cpp when a pack is built."""
import json

import numpy as np

from goethe_engine import gc_plan as G
from goethe_engine import pack as P
from goethe_engine.structure import gguf_offsets
from toypack import gguf


def test_gguf_offsets(tmp_path):
    w = tmp_path / "w.gguf"
    gguf(str(w), [("a.weight", [64, 8], 2), ("b.weight", [64], 0)], align=32)
    off = gguf_offsets(str(w))
    raw = w.read_bytes()
    assert set(off) == {"a.weight", "b.weight"} and off["b.weight"] - off["a.weight"] == 64
    assert off["a.weight"] % 32 == 0 and off["b.weight"] + 64 == len(raw)       # the data section ends the file


def test_encoder_mask_pads_keys_and_cells():
    T, n_real = 128, 100                    # 100 real tokens in a 128 chunk; keys padded to 256 cells (f16)
    hp = {"causal": False, "n_swa": 0, "masks": {"self_kq_mask": {"esize": 2, "swa": False}}}
    st = G.Step([7] * T, [0] * T, range(T), range(T), [0] * T, T, T, range(T), n_real=n_real)
    m = np.frombuffer(G.generate("CUDA0#self_kq_mask#0", 2 * T * 256, st, hp), np.uint16).reshape(T, 256)
    assert (m[:, :n_real] == 0).all() and (m[:, n_real:] == G.NEG_INF_F16).all()    # pad queries see the real keys: finite rows


def test_swa_mask_is_symmetric():
    T = 64
    hp = {"causal": False, "n_swa": 16, "masks": {"leaf_3": {"esize": 4, "swa": True}}}
    st = G.Step([1] * T, [0] * T, range(T), range(T), [0] * T, T, T, range(T))
    m = np.frombuffer(G.generate("CUDA0#leaf_3#0", 4 * T * T, st, hp), np.float32).reshape(T, T)
    q, k = np.indices((T, T))
    assert ((m == 0) == (np.abs(q - k) <= 8)).all()


def test_pooling_and_fills():
    st = G.Step([3] * 8, [0] * 8, range(8), range(8), [0] * 8, 8, 8, range(8), n_real=5)
    mean = np.frombuffer(G.generate("CUDA0#mean#0", 32, st, {}), np.float32)
    assert np.allclose(mean[:5], 0.2) and (mean[5:] == 0).all()
    assert np.frombuffer(G.generate("CUDA0#cls#0", 4, st, {}), np.int32).tolist() == [0]
    rows = np.frombuffer(G.generate("CUDA0#inp_scale_rows#0", 32, st, {"fills": {"inp_scale_rows": 2.5}}), np.float32)
    assert (rows == 2.5).all()


def test_view_input_is_a_slice_of_its_base():
    st = G.Step([5] * 4, [0, 1, 2, 3], range(4), range(4), [0] * 4, 4, 4, range(4))     # four sequences, one token each
    hp = {"views": {"CUDA0#rs_s_copy (view)#0@12": 1}}
    v = np.frombuffer(G.generate("CUDA0#rs_s_copy (view)#0", 12, st, hp), np.int32)
    assert v.tolist() == [1, 2, 3]


def test_lerp_plan_interpolates_widths():
    import struct
    fake = lambda v: struct.pack("<Q", G.FAKE0 + v).hex()
    fd = lambda d: struct.pack("<3I", *G._fastdiv(d)).hex()
    host = struct.pack("<Q", 0x7FFF00001000).hex()
    act = lambda w, h: {"launch": "k", "g": [w // 256, 1, 1], "args": [fake(4 * w) + host[:0], fd(w // 128), struct.pack("<i", w).hex(), h],
                        "ptrs": [[0, 0, 4 * w]]}
    p0 = {"io": {"n_kv": 512}, "actions": [act(512, host)], "buffers": [{"role": "compute", "size": 1024 * 512}]}
    p1 = {"io": {"n_kv": 2048}, "actions": [act(2048, struct.pack("<Q", 0x7FFD00002000).hex())],
          "buffers": [{"role": "compute", "size": 1024 * 2048}]}
    p = G.lerp_plan(p0, p1, (1536 - 512, 2048 - 512))
    want = act(1536, host)
    assert p["io"]["n_kv"] == 1536 and p["buffers"][0]["size"] == 1024 * 1536 and p["actions"][0] == want
    try:
        G.lerp_plan({"io": {"n_kv": 1}}, {"io": {"n_kv": 2}}, (1, 3))
    except P.PackError:
        pass
    else:
        raise AssertionError("a non-integer step must be refused")


def test_fit_is_one_typed_rule():
    G_ = 1 << 30
    p = {"nseq": 4, "buffers": [{"role": "weights", "size": 10 * G_}, {"role": "kv", "size": 2 * G_}, {"role": "compute", "size": G_}],
         "caches": {"cache_k_l0": [1, 0, 2 * G_]}, "actions": []}
    e = G.does_not_fit({"b": p}, 13 * G_)
    assert isinstance(e, P.DoesNotFit) and isinstance(e, P.OutOfMemory) and (e.streams, e.needed) == (1, 2) and e.free == 13 * G_
    assert e.need > e.free and "does not fit this GPU: 2 streams need" in str(e)
    assert G.does_not_fit({"b": p}, 14 * G_) is None and G._streams({"b": p}, 14 * G_) == 3
    try:
        G._fit_streams({"b": p}, 14 * G_, min_streams=4)
    except P.DoesNotFit as x:
        assert (x.streams, x.needed) == (3, 4)
    else:
        raise AssertionError("a bucket needing more streams than fit must be refused by type")
    assert G._fit_streams({"b": p}, 14 * G_)["b"]["nseq"] == 3

def test_unknown_input_refused():
    st = G.Step([0], [0], [0], [0], [0], 1, 1, [0])
    try:
        G.generate("CUDA0#mystery#0", 4, st, {})
    except P.PackError as e:
        assert "mystery" in str(e)
    else:
        raise AssertionError("an input of no known kind must be refused")


def test_runtime_kind_accepted():
    assert P.RUNTIMES["gc-plan"] == (1,)
    m = {"runtime": {"kind": "gc-plan", "version": 1}}
    assert json.loads(json.dumps(m))["runtime"]["kind"] in P.RUNTIMES


def _ring_reference(pos_by_stream, S, W, K):
    """a sliding-window ring simulated: every position up to the stream's last is written to cell p % S (newest wins); a query
    at p sees cell j iff it holds p_j with p - W < p_j <= p (llama.cpp's LLAMA_SWA_TYPE_STANDARD)"""
    rows = []
    for ps in pos_by_stream:
        held = {}
        for q in range(max(ps) + 1):
            held[q % S] = q
        for p in ps:
            rows.append([j in held and held[j] <= p and p - held[j] < W for j in range(K)])
    return np.array(rows)


def test_swa_ring_inputs_match_a_simulated_cache():
    S, W, hp = 1536, 1024, {"swa_cells": 1536, "n_swa": 1024}
    cases = [([4096] * 4, 1536), ([2047], 1536), (list(range(3584, 4096)), 1536), (list(range(0, 300)), 512), ([5], 256)]
    for ps, K in cases:
        multi = len(ps) > 1 and len(set(ps)) == 1                  # a decode of several sequences: one token per stream
        streams = list(range(len(ps))) if multi else [0] * len(ps)
        st = G.Step([1] * len(ps), streams, ps, ps, streams, 4352, 0, range(len(ps)))
        m = np.frombuffer(G.generate("CUDA0#attn_inp_kq_mask_swa#0", 2 * len(ps) * K, st, hp), np.uint16).reshape(len(ps), K)
        ref = _ring_reference([[p] for p in ps] if multi else [ps], S, W, K)
        assert ((m == 0) == ref).all(), (ps[:2], K)
        idx = np.frombuffer(G.generate("CUDA0#attn_inp_k_idxs_swa#0", 8 * len(ps), st, hp), np.int64)
        assert (idx == np.array(streams) * S + np.array(ps) % S).all()
