"""Media model type: a pack's media encode bucket (io.media) through abi/gcm.h (mock libgcm): prompt chunks, media rows, in-library
decode, the pinned-body provenance refusal, the WAV reader."""
import os
import struct
import subprocess

import numpy as np
import pytest

from goethe_engine import pack as P, registry as R
from goethe_engine.media import GcMedia, MediaRunner, check_provenance, read_wav
from goethe_engine import structure
from toypack import gguf, toy_pack

PROV = dict(body="ggml-pinned", cublas="13.4.1", cublasLt="13.4.1", pdl="off")


@pytest.fixture(scope="session")
def mockgcm(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("mockgcm") / "libgcm.so")
    src = os.path.join(os.path.dirname(__file__), "mock", "gcm_mock.c")
    subprocess.run(["cc", "-std=c99", "-Wall", "-O1", "-shared", "-fPIC", "-o", out, src], check=True)
    return out


def media_pack(tmp_path, key, lib, prov=PROV, decode=True):
    """(pack, [projector file, text model file]): one media encode bucket over both files (the variant's structure spans them)."""
    mm = str(tmp_path / "mm.gguf")
    gguf(mm, [("mm.proj.weight", [4, 4], 0)])

    def mut(m):
        m["variants"]["q4"]["structure"] = structure.describe([mm, str(tmp_path / "w.gguf")], ["general.architecture"])
        b = dict(m["buckets"][1], id="target.mock.K1.1m.256", kind="encode", ncol=1, batch=False, provenance=prov,
                 io=dict(media="audio", decode=decode, weights=["mmproj", "text"], **{"in": dict(pcm="f32", rate=16000)},
                         out=dict(rows="f32", n_embd=4)))
        m["buckets"] = [b]
        m["model"]["type"] = "media"
    pack, w, _ = toy_pack(str(tmp_path), key, lib, mutate=mut)
    return pack, [mm, w]


def test_transcribe_through_the_registry(tmp_path, key, mockgcm):
    pack, ws = media_pack(tmp_path, key, mockgcm)
    assert R.pack_type(pack) == "media"
    be = R.open_backend(pack, ws, mesh=R.Mesh.single(0), sku="mock", trusted=(key.kid,))
    try:
        assert isinstance(be, GcMedia) and (be.n_embd, be.media, be.rate, be.marker) == (4, "audio", 16000, "<m>")
        assert be.tokenize("ab<m>c") == [(0, [97, 98]), (1, None), (0, [99])]
        pcm = np.arange(10, dtype=np.float32)
        rows = be.encode_audio(pcm)
        assert rows.shape == (3, 4) and rows[:, 0].tolist() == [6, 22, 17] and rows[0, 2] == 10
        r = R.runner(be, None)
        assert isinstance(r, MediaRunner)
        assert r.transcribe([pcm], "ab<m>c", max_tokens=8) == [(97 + 98) % 97, 3 * 10 + 6, 99 % 97, 1000, 1001, 1002]
        with pytest.raises(P.PackError, match="1 media item"):
            r.transcribe([pcm], "no marker")
        with pytest.raises(P.PackError, match="more media markers"):
            r.transcribe([pcm], "<m><m>")
    finally:
        be.close()


def test_pinned_provenance_mismatch_is_refused(tmp_path, key, mockgcm, monkeypatch):
    pack, ws = media_pack(tmp_path, key, mockgcm)
    monkeypatch.setenv("GCM_MOCK_CUBLAS", "13.1.1")
    with pytest.raises(P.PackError, match="cublas 13.4.1, this process bound cublas 13.1.1"):
        R.open_backend(pack, ws, mesh=R.Mesh.single(0), sku="mock", trusted=(key.kid,))
    b = dict(id="x", provenance=dict(PROV, pdl="on"))
    with pytest.raises(P.PackError, match="pdl off"):
        check_provenance(b, dict(PROV))
    with pytest.raises(P.PackError, match="lacks"):
        check_provenance(dict(id="x", provenance=dict(body="ggml-pinned")), {})
    check_provenance(dict(id="x", provenance=dict(PROV, cublas=dict(version="13.4.1"))), dict(PROV, cublasLt=dict(version="13.4.1")))
    check_provenance(dict(id="x", provenance={}), {})          # a compiled body pins no reference library


def test_no_in_library_decode_names_the_missing_piece(tmp_path, key, mockgcm):
    pack, ws = media_pack(tmp_path, key, mockgcm, decode=False)
    be = R.open_backend(pack, ws, mesh=R.Mesh.single(0), sku="mock", trusted=(key.kid,))
    try:
        with pytest.raises(P.PackError, match="supplied-rows prefill"):
            MediaRunner(be).transcribe([np.zeros(4, np.float32)], "<m>")
    finally:
        be.close()


def wav(path, samples, rate=16000, ch=1):
    data = np.asarray(samples, "<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, ch, rate, rate * 2 * ch, 2 * ch, 16)
    open(path, "wb").write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
                           + b"data" + struct.pack("<I", len(data)) + data)


def test_read_wav(tmp_path):
    p = str(tmp_path / "a.wav")
    wav(p, [0, 16384, -32768])
    assert read_wav(p, 16000).tolist() == [0.0, 0.5, -1.0]
    with pytest.raises(P.PackError, match="8000 Hz"):
        wav(p, [0], rate=8000)
        read_wav(p, 16000)
    wav(p, [16384, 0, -16384, -16384], ch=2)
    assert read_wav(p).tolist() == [0.25, -0.5]


def test_media_path_never_imports_torch(tmp_path, key, mockgcm):
    """torch preloads its wheels' cuBLAS copies: a pinned body would bind those, not its provenance's (measured: 12.8.4 for 12.8.3)."""
    import sys
    pack, ws = media_pack(tmp_path, key, mockgcm)
    code = (f"import sys; from goethe_engine import api; m = api.open({pack!r}, {ws!r}, device=0, sku='mock', trusted=({key.kid!r},)); "
            "m.transcribe([__import__('numpy').ones(4, 'float32')], '<m>', 2); m.close(); assert 'torch' not in sys.modules, 'torch imported'")
    root = os.path.join(os.path.dirname(__file__), "..")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env=dict(os.environ, PYTHONPATH=root))
