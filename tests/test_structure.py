import json
import os
import struct

import pytest

from goethe_engine import structure as S


def _s(b):
    return struct.pack("<Q", len(b)) + b


def gguf(path, kv, tensors, align=32, pad=0):
    """A GGUF v3 file: kv {key: str | int(u32) | float(f32)}, tensors [(name, dims, type id, nbytes)], zero data."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv.items():
        out += _s(k.encode())
        out += struct.pack("<I", 8) + _s(v.encode()) if isinstance(v, str) else \
            struct.pack("<I", 6) + struct.pack("<f", v) if isinstance(v, float) else struct.pack("<II", 4, v)
    off = 0
    for n, d, t, nb in tensors:
        out += _s(n.encode()) + struct.pack("<I", len(d)) + b"".join(struct.pack("<Q", x) for x in d) + struct.pack("<IQ", t, off)
        off += -(-nb // align) * align
    out += b"\0" * (-len(out) % align) + b"\0" * (off + pad)
    open(path, "wb").write(out)


def safetensors(path, tensors):
    """tensors {name: (dtype, shape)}, zero data."""
    h, off = {}, 0
    for n, (dt, sh) in tensors.items():
        nb = 2 if dt == "BF16" else 4
        for x in sh:
            nb *= x
        h[n] = dict(dtype=dt, shape=sh, data_offsets=[off, off + nb])
        off += nb
    hb = json.dumps(h).encode()
    open(path, "wb").write(struct.pack("<Q", len(hb)) + hb + b"\0" * off)


KV = {"general.architecture": "llama", "llama.rope.freq_base": 10000.0, "llama.block_count": 2, "tokenizer.ggml.model": "gpt2"}
T = [("a.weight", [64, 8], 2, 288), ("b.weight", [64], 0, 256), ("c.weight", [32, 32], 12, 576)]


def test_gguf_structure_ignores_order_alignment_values_and_unread_keys(tmp_path):
    keys = ["llama.rope.freq_base", "llama.block_count", "general.architecture"]
    gguf(tmp_path / "a.gguf", KV, T)
    gguf(tmp_path / "b.gguf", dict(reversed(list(dict(KV, **{"tokenizer.ggml.model": "other", "general.name": "ft"}).items()))),
         list(reversed(T)), align=64, pad=100)
    a, b = S.structure([str(tmp_path / "a.gguf")], keys), S.structure([str(tmp_path / "b.gguf")], keys)
    assert a == b and len(a["sha256"]) == 64 and a["keys"] == sorted(keys)


@pytest.mark.parametrize("what", ["shape", "type", "name", "extra", "rope", "layers"])
def test_changes_that_must_not_match(tmp_path, what):
    keys = ["llama.rope.freq_base", "llama.block_count"]
    kv, t = dict(KV), [list(x) for x in T]
    if what == "shape":
        t[0][1] = [64, 9]
    elif what == "type":
        t[2][2] = 14
    elif what == "name":
        t[1][0] = "b2.weight"
    elif what == "extra":
        t.append(["d.weight", [4], 0, 16])
    elif what == "rope":
        kv["llama.rope.freq_base"] = 500000.0
    else:
        kv["llama.block_count"] = 3
    gguf(tmp_path / "a.gguf", KV, T)
    gguf(tmp_path / "b.gguf", kv, [tuple(x) for x in t])
    assert S.structure([str(tmp_path / "a.gguf")], keys)["sha256"] != S.structure([str(tmp_path / "b.gguf")], keys)["sha256"]
    msg = S.differ(S.table([str(tmp_path / "a.gguf")]), S.table([str(tmp_path / "b.gguf")]), keys)
    assert msg and (("tensor" in msg) or ("key" in msg))


def test_dynamic_quant_type_table_differs(tmp_path):
    """A different per-tensor type table (an unsloth-style dynamic quant) is another structure."""
    gguf(tmp_path / "a.gguf", KV, T)
    gguf(tmp_path / "b.gguf", KV, [("a.weight", [64, 8], 12, 288)] + T[1:])
    assert S.structure([str(tmp_path / "a.gguf")])["sha256"] != S.structure([str(tmp_path / "b.gguf")])["sha256"]


def test_safetensors_dir_shards_config_and_quant(tmp_path):
    for d, split in (("a", 1), ("b", 2)):
        p = tmp_path / d
        p.mkdir()
        tens = {"w1": ("BF16", [4, 4]), "w2": ("F32", [4]), "w3": ("U8", [4, 2])}
        names = list(tens)
        if split == 1:
            safetensors(p / "model.safetensors", tens)
        else:
            safetensors(p / "model-00001-of-00002.safetensors", {n: tens[n] for n in names[:1]})
            safetensors(p / "model-00002-of-00002.safetensors", {n: tens[n] for n in names[1:]})
        json.dump({"text_config": {"rope_theta": 1e6}, "quantization_config": {"quant_method": "compressed-tensors"}}, open(p / "config.json", "w"))
    keys = ["rope_theta"]
    assert S.structure(str(tmp_path / "a"), keys) == S.structure(str(tmp_path / "b"), keys)       # resharded: same structure
    json.dump({"text_config": {"rope_theta": 5e5}, "quantization_config": {"quant_method": "compressed-tensors"}}, open(tmp_path / "b" / "config.json", "w"))
    assert S.structure(str(tmp_path / "a"), keys) != S.structure(str(tmp_path / "b"), keys)
    assert "rope_theta" in S.differ(S.table(str(tmp_path / "a")), S.table(str(tmp_path / "b")), keys)
    json.dump({"text_config": {"rope_theta": 1e6}, "quantization_config": {"quant_method": "awq"}}, open(tmp_path / "b" / "config.json", "w"))
    assert S.structure(str(tmp_path / "a"), keys) != S.structure(str(tmp_path / "b"), keys)       # another quantization config


def test_truncated_and_foreign_files_refused(tmp_path):
    open(tmp_path / "x.gguf", "wb").write(b"GGUF\3\0\0\0")
    with pytest.raises(S.StructureError):
        S.table([str(tmp_path / "x.gguf")])
    open(tmp_path / "y.bin", "wb").write(b"zz")
    with pytest.raises(S.StructureError):
        S.table([str(tmp_path / "y.bin")])


H = lambda c: c * 64


def _entry(sha, keys, files=None, tier="base", fh="b"):
    model = dict(hf_repo="org/m", revision="r1", format="gguf", structure=dict(sha256=sha, keys=keys))
    if files:
        model["files"] = files
    return dict(model=model, arch="sm_120a", abi_major=2, runtime=dict(kind="gc-c", version=1), tier=tier, file_sha256=H(fh),
                sig=dict(schema="gc-sig-1"), size=10, min_engine="0.1.0")


def test_index_matches_structure_and_labels_identity(tmp_path):
    from goethe_engine import index as I
    keys = ["llama.rope.freq_base", "llama.block_count"]
    ref, ft = tmp_path / "ref", tmp_path / "ft"
    ref.mkdir()
    ft.mkdir()
    gguf(ref / "m.gguf", KV, T)
    gguf(ft / "finetune-q4.gguf", dict(KV, **{"general.name": "ft"}), list(reversed(T)), align=64, pad=7)   # renamed, reordered, other values
    st = S.structure([str(ref / "m.gguf")], keys)
    import hashlib
    files = {"m.gguf": hashlib.sha256((ref / "m.gguf").read_bytes()).hexdigest()}
    ix = dict(entries=[_entry(st["sha256"], keys, files, tier="exact", fh="c"), _entry(st["sha256"], keys, None, tier="a", fh="d")])
    hit = I.match_dir(ix, str(ref), "sm_120a", "0.1.0")
    assert [(e["tier"], e["identity"]) for e in hit] == [("exact", "exact"), ("a", "structural")]      # exact outranks, and is labelled
    hit = I.match_dir(ix, str(ft), "sm_120a", "0.1.0")
    assert [(e["tier"], e["identity"]) for e in hit] == [("exact", "structural"), ("a", "structural")]  # no file hash matches: both structural
    bad = tmp_path / "bad"
    bad.mkdir()
    gguf(bad / "m.gguf", dict(KV, **{"llama.rope.freq_base": 500000.0}), T)                            # a read hyperparameter changed
    assert I.match_dir(ix, str(bad), "sm_120a", "0.1.0") == []
    gguf(bad / "m.gguf", KV, [("a.weight", [64, 8], 12, 288)] + T[1:])                                  # another per-tensor type table
    assert I.match_dir(ix, str(bad), "sm_120a", "0.1.0") == []


def test_check_weights_structural_names_the_difference(tmp_path):
    from goethe_engine import pack as P
    keys = ["llama.rope.freq_base"]
    gguf(tmp_path / "a.gguf", KV, T)
    m = dict(model=dict(weights=[dict(file="a.gguf", bytes=1, sha256=H("0"))]), structure=S.describe([str(tmp_path / "a.gguf")], keys))
    gguf(tmp_path / "b.gguf", dict(KV, general_name="x"), list(reversed(T)), align=64)
    P.check_weights(m, [str(tmp_path / "b.gguf")])                       # same structure, other bytes, other size: no hashing
    gguf(tmp_path / "c.gguf", dict(KV, **{"llama.rope.freq_base": 5.0}), T)
    with pytest.raises(P.PackError, match="llama.rope.freq_base"):
        P.check_weights(m, [str(tmp_path / "c.gguf")])
    gguf(tmp_path / "d.gguf", KV, [("a.weight", [64, 8], 2, 288), ("b.weight", [64], 0, 256), ("c.weight", [32, 33], 12, 600)])
    with pytest.raises(P.PackError, match="c.weight"):
        P.check_weights(m, [str(tmp_path / "d.gguf")])
    old = dict(model=m["model"])                                         # an old pack (no structure) stays exact
    with pytest.raises(P.PackError, match="bytes|sha256"):
        P.check_weights(old, [str(tmp_path / "b.gguf")])
