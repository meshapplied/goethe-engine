"""pack build: a declarative spec becomes a pack the engine opens; what a build can read is read (structure, sizes, cuda), step_us and
the hw record exist only where a bench row measured them, and an unbuilt library is refused."""
import hashlib
import json
import os
import shutil

import pytest

from goethe_engine import build as B, pack as P
from goethe_engine.gc_c import GcBackend
from toypack import DRAFT, TARGET, gguf

CAPS = dict(verify_widths=[], verify=["linear", "tree"], tree_mask=True, taps=[dict(node="result_norm", n=4)], recurrent_commit=[],
            accept=[], drafter_roles=[])
AXES = dict(determinism="batch_invariant", act="a8", kv="f16", spec_caps=CAPS, io=None, variant_of=None, klass=None, variants=[])


def spec_dir(d, lib, measured=None):
    """A spec folder: the library, the model files, a plan blob and spec.json; returns the spec path."""
    os.makedirs(d, exist_ok=True)
    shutil.copy(lib, os.path.join(d, "libgck.so"))
    gguf(os.path.join(d, "w.gguf"), TARGET)
    gguf(os.path.join(d, "d.gguf"), DRAFT)
    open(os.path.join(d, "plan.json"), "w").write("{}")
    spec = dict(abi="3.2.0", model=dict(name="org/toy", type="decoder"), runtime=dict(kind="gc-c", version=1), contracts=["K1"],
                skus=[dict(sku="mock", arch="sm_120a", cc="12.0", lib="libgck.so", requires=dict(sms=170, smem_block=1, name="toy"))],
                variants=[dict(id="q4", quant="q4_k_m", source=dict(hf_repo="org/toy-GGUF", revision="0" * 40), files=["w.gguf"],
                               keys=["general.architecture"]),
                          dict(id="d", quant="q8_0", source=dict(hf_repo="org/toy-GGUF", revision="0" * 40), files=["d.gguf"],
                               keys=["general.architecture"])],
                plans=dict(p="plan.json"),
                buckets=[dict(AXES, suffix="2.256", role="target", variant="q4", contract="K1", ncol=2, batch=True, maxctx=256, plan="p"),
                         dict(AXES, suffix="8.256", role="target", variant="q4", contract="K1", ncol=8, batch=True, maxctx=256, plan="p")],
                **({"measured": measured} if measured else {}))
    path = os.path.join(d, "spec.json")
    json.dump(spec, open(path, "w"))
    return path


def test_a_built_pack_opens_and_carries_only_what_was_measured(tmp_path, key, mocklib):
    sp = spec_dir(str(tmp_path / "s"), mocklib)
    out = str(tmp_path / "toy.gcpack")
    m = json.loads(B.build(sp, out, key.priv, cuda="12.8"))
    assert {b["id"] for b in m["buckets"]} == {"target.mock.K1.2.256", "target.mock.K1.8.256"}
    assert m["hw"] == {} and not any("step_us" in b for b in m["buckets"])
    w = str(tmp_path / "s" / "w.gguf")
    assert m["variants"]["q4"]["desc"]["quant"] == "q4_k_m" and m["model"]["weights"][0] == dict(file="w.gguf", bytes=os.path.getsize(w),
                                                                                               sha256=hashlib.sha256(open(w, "rb").read()).hexdigest())
    be = GcBackend(out, str(tmp_path / "s" / "w.gguf"), sku="mock", trusted=(key.kid,), B=8)
    assert be.bucket["ncol"] == 8
    be.close()
    dev = dict(name="toy", cc="12.0", sms=170, smem_block=101376, uuid="u")
    sp2 = spec_dir(str(tmp_path / "t"), mocklib, measured=dict(mock=dict(device=dev, step_us={"K1.8.256": 950.0})))
    m2 = json.loads(B.build(sp2, str(tmp_path / "t.gcpack"), cuda="12.8"))
    us = {b["id"]: b.get("step_us") for b in m2["buckets"]}
    assert us == {"target.mock.K1.2.256": None, "target.mock.K1.8.256": 950.0}
    assert m2["hw"] == {"mock": hashlib.sha256(json.dumps(dev, sort_keys=True).encode()).hexdigest()}


def test_an_unbuilt_library_is_refused_and_cuda_comes_from_nvcc(tmp_path, mocklib, monkeypatch):
    sp = spec_dir(str(tmp_path / "s"), mocklib)
    os.unlink(tmp_path / "s" / "libgck.so")
    with pytest.raises(P.PackError, match="not built"):
        B.build(sp, str(tmp_path / "x.gcpack"), cuda="12.8")
    assert not os.path.exists(tmp_path / "x.gcpack")

    class Out:
        stdout = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.0, V13.0.88\n"
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: Out())
    assert B.nvcc_version() == "13.0"


def test_pack_build_cli_writes_a_signed_pack(tmp_path, key, mocklib, monkeypatch, capsys):
    sp = spec_dir(str(tmp_path / "s"), mocklib)
    monkeypatch.setattr(B, "nvcc_version", lambda: "12.8")
    out = str(tmp_path / "cli.gcpack")
    P.main(["build", sp, "--out", out, "--key", key.priv])
    j = json.loads(capsys.readouterr().out)
    assert (j["out"], j["variants"], j["buckets"]) == (out, ["d", "q4"], 2) and os.path.exists(out + ".sig")
    assert P.verify(out, trusted=(key.kid,))


def test_templates_are_named_by_contract_and_suffix_and_never_twice(tmp_path, mocklib):
    # Two contracts may share a suffix (8v.4096 under K1 and K1_fast): spec pairs and bench rows name contract.suffix, a repeat is refused.
    sp = spec_dir(str(tmp_path / "s"), mocklib, measured=dict(mock=dict(device={}, step_us={"K2.8.256": 1.0})))
    spec = json.load(open(sp))
    spec["contracts"] = ["K1", "K2"]
    spec["buckets"] += [dict(spec["buckets"][1], contract="K2")]
    spec["buckets"][1].update(io=dict(accept=True))
    spec["buckets"][2].update(io=dict(accept=True))
    spec["buckets"].append(dict(spec["buckets"][0], suffix="1.256", role="drafter", variant="d", ncol=1, batch=False))
    spec["spec"] = [dict(name="ar", target="K2.8.256", drafter="K1.1.256", policy=dict(kind="ar", K="3"))]
    json.dump(spec, open(sp, "w"))
    m = B.manifest(json.load(open(sp)), str(tmp_path / "s"), cuda="12.8")[0]
    P.validate(m, m["abi"])
    assert m["spec"][0]["target"] == "target.mock.K2.8.256"
    assert {b["id"]: b.get("step_us") for b in m["buckets"]}["target.mock.K2.8.256"] == 1.0
    spec["buckets"].append(dict(spec["buckets"][0]))
    json.dump(spec, open(sp, "w"))
    with pytest.raises(P.PackError, match="named twice"):
        B.manifest(json.load(open(sp)), str(tmp_path / "s"), cuda="12.8")
