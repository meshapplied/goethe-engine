"""One pack, a GGUF variant and a safetensors checkpoint variant: the user's files choose the branch by structure (header reads)."""
import json
import os
import struct

import pytest

from goethe_engine import pack as P, structure
from toypack import gguf, toy_pack

CK = {"a.weight": ("U8", [8, 32]), "a.weight_scale": ("F32", [8]), "b.weight": ("BF16", [64])}


def safetensors(path, tensors):
    """tensors {name: (dtype, shape)}, zero data."""
    h, off = {}, 0
    for n, (dt, sh) in tensors.items():
        nb = {"BF16": 2, "F32": 4, "U8": 1}[dt]
        for x in sh:
            nb *= x
        h[n] = dict(dtype=dt, shape=sh, data_offsets=[off, off + nb])
        off += nb
    hb = json.dumps(h).encode()
    open(path, "wb").write(struct.pack("<Q", len(hb)) + hb + b"\0" * off)


def checkpoint(d, algo="NVFP4"):
    """A two-shard checkpoint dir: shards, config.json, hf_quant_config.json. Returns its shard files."""
    os.makedirs(d, exist_ok=True)
    names = list(CK)
    safetensors(os.path.join(d, "model-00001-of-00002.safetensors"), {n: CK[n] for n in names[:2]})
    safetensors(os.path.join(d, "model-00002-of-00002.safetensors"), {n: CK[n] for n in names[2:]})
    json.dump({"text_config": {"rope_theta": 1e6}}, open(os.path.join(d, "config.json"), "w"))
    json.dump({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": algo}}, open(os.path.join(d, "hf_quant_config.json"), "w"))
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors"))


def test_gguf_and_checkpoint_variants_of_one_pack(tmp_path, key, mocklib):
    shards = checkpoint(str(tmp_path / "ck"))

    def st_variant(m):          # variant "alt" is the checkpoint's (a gc pack --variants GGUF,ST pack)
        m["variants"]["alt"]["structure"] = structure.describe(shards, ["rope_theta"])
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib, alt=True, mutate=st_variant)
    m = P.verify(pack, trusted=(key.kid,))
    assert P.pick_variant(m, [w]) == "q4"                     # the user holds the GGUF
    assert P.pick_variant(m, shards) == "alt"                 # the user holds the checkpoint
    assert P.pick_variant(m, shards[::-1]) == "alt"           # shard order does not matter
    v, bs = P.branch(m, shards, sku="mock")
    assert v == "alt" and {b["variant"] for b in bs} == {"alt"}
    P.check_weights(P.for_variant(m, "alt"), shards)
    with pytest.raises(P.PackError, match="not the structure"):
        P.check_weights(P.for_variant(m, "alt"), [w])         # the GGUF under the checkpoint's variant
    other = checkpoint(str(tmp_path / "ck2"), algo="FP8")      # same tensors, another quantization config: no variant
    with pytest.raises(P.PackError, match="not a model this pack was compiled for"):
        P.pick_variant(m, other)
    gguf(str(tmp_path / "x.gguf"), [("a.weight", [64, 8], 8), ("b.weight", [64], 0)])   # another GGUF type table: no variant
    with pytest.raises(P.PackError, match="not a model this pack was compiled for"):
        P.pick_variant(m, [str(tmp_path / "x.gguf")])
