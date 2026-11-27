"""pack plan: from a pack's first bytes, what each variant downloads (files, sizes, source), its longest context and whether the device
runs it; from abi minor 2 every variant names its quant and its source."""
import hashlib
import json
import os

import pytest

from goethe_engine import pack as P
from toypack import parts, toy_pack

DEV = dict(name="toy", cc="12.0", sms=170, smem_block=101376, cuda="13.0", arch_ok=P.cuda_arch_ok)


def manifest(tmp_path, key, lib):
    m, data = parts(toy_pack(str(tmp_path / "src"), key, lib, spec=True, alt=True, sign=False)[0])
    return m, list(data.values())


def test_plan_lists_each_variant_with_its_files_context_and_device_fit(tmp_path, key, mocklib):
    m, _ = manifest(tmp_path, key, mocklib)
    by = {v["id"]: v for v in P.plan(m, DEV)}
    assert {k: v["role"] for k, v in by.items()} == {"q4": "target", "d": "drafter", "alt": "target"}
    assert {k: [f["name"] for f in v["files"]] for k, v in by.items()} == {"q4": ["w.gguf"], "d": ["d.gguf"], "alt": ["alt.gguf"]}
    assert all(f["bytes"] > 0 for v in by.values() for f in v["files"])
    assert by["q4"]["max_ctx"] == 256 and by["q4"]["runs"] is True and by["d"]["runs"] is None
    assert by["q4"]["lanes"] == 1, "a variant with a speculative pair: the streams its single-stream verify target serves"
    assert by["alt"]["lanes"] == max(P.lanes(b) for b in m["buckets"] if b["variant"] == "alt"), "else its widest batch"
    slow = next(v for v in P.plan(m, dict(DEV, cc="9.0")) if v["id"] == "q4")
    assert slow["runs"] is False and "cc 9.0" in slow["why"]
    assert all(v["runs"] is None for v in P.plan(m))                       # no device asked: no answer, never a guess


def test_pack_plan_reads_only_the_prefix(tmp_path, key, mocklib, capsys, monkeypatch):
    m, blobs = manifest(tmp_path, key, mocklib)
    pack = str(tmp_path / "x.gcpack")
    raw = P.write(pack, m, blobs)
    whole = open(pack, "rb").read()
    n = int.from_bytes(whole[8:16], "little")
    hdr = json.loads(whole[16:16 + n])
    prefix = tmp_path / "x.prefix"
    prefix.write_bytes(whole[:16 + n + hdr["manifest"][0] + hdr["manifest"][1]])
    monkeypatch.setattr(P, "device", lambda: (_ for _ in ()).throw(OSError("no GPU")))
    P.main(["plan", str(prefix)])
    out = json.loads(capsys.readouterr().out)
    assert out["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert sorted(v["id"] for v in out["variants"]) == ["alt", "d", "q4"] and all(v["runs"] is None for v in out["variants"])


def test_from_minor_2_every_variant_names_its_quant_and_source(tmp_path, key, mocklib):
    m, blobs = manifest(tmp_path, key, mocklib)
    out = str(tmp_path / "y.gcpack")
    with pytest.raises(P.PackError, match="from minor 2"):
        P.write(out, dict(m, abi="3.2.0", model=dict(m["model"], name="toy")), blobs)
    assert not os.path.exists(out)
    src = dict(hf_repo="org/toy", revision="0" * 40)
    vs = {k: dict(v, desc=dict(quant="q4_k_m"), source=src) for k, v in m["variants"].items()}
    raw = P.write(out, dict(m, abi="3.2.0", model=dict(m["model"], name="toy"), variants=vs), blobs)
    q4 = next(v for v in P.plan(json.loads(raw)) if v["id"] == "q4")
    assert (q4["quant"], q4["source"]) == ("q4_k_m", src)
