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
