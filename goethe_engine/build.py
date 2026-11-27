"""Pack builder: one declarative pack spec (JSON, data only) -> the manifest and blobs pack.write makes a pack of.

spec  {abi, model {name, type, ...}, runtime {kind, version}, contracts [..],
       skus [{sku, arch, cc, lib, requires {sms, smem_block, name}, tune}],          one row per built library
       variants [{id, quant, source {hf_repo, revision}, files [paths], keys [..]}],  structure computed from the files
       plans {name: path},                                                            bucket plan blobs
       buckets [{suffix, role, variant, contract, ncol, maxctx, plan, ...}],          per SKU: id role.sku.contract.suffix
       spec [{name, target: name, drafter: name, policy}],                            per SKU; a template's name is contract.suffix
       measured {sku: {device {...}, step_us {name: us}}}}                            bench rows only
Paths are relative to the spec file. Values a build can read are read, never typed: structure from the model files, weight sizes from
the files, weight sizes and sha256 from the files, cuda from nvcc, blob sizes by write; step_us and the hw record hash exist only where
measured has them."""
import hashlib
import json
import os
import re
import subprocess

from . import __version__, pack as P, structure

TEMPLATE_KEYS = ("suffix", "plan")      # bucket template fields the builder turns into id, files and plan


def file_sha256(path):
    """sha256 of a model file, streamed in 16 MiB reads. O(bytes) time, O(1) space."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def nvcc_version():
    """The CUDA version of the nvcc on PATH (what the libraries were built with); refused without nvcc."""
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise P.PackError(f"nvcc: {e} (the pack's cuda is read from the toolchain that built its libraries)")
    m = re.search(r"release (\d+\.\d+)", out)
    P.need(m, "nvcc --version: no release number")
    return m[1]


def manifest(spec, base, cuda=None):
    """(manifest dict, [blob bytes]) of spec; base is the directory its paths are relative to. O(files + skus x buckets)."""
    path = lambda p: os.path.join(base, p)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    blobs, plans = {}, {}
    for name, p in spec.get("plans", {}).items():
        b = open(path(p), "rb").read()
        blobs[sha(b)], plans[name] = b, sha(b)
    weights, variants = [], {}
    for v in spec["variants"]:
        files = [path(f) for f in v["files"]]
        idx = list(range(len(weights), len(weights) + len(files)))
        weights += [dict(file=os.path.basename(f), bytes=os.path.getsize(f), sha256=file_sha256(f)) for f in files]
        variants[v["id"]] = dict(structure=structure.describe(files, v.get("keys", [])), weights=idx, desc=dict(quant=v["quant"]),
                                 source=v["source"])
    windex = {v: variants[v]["weights"][0] for v in variants}
    names = [f"{t['contract']}.{t['suffix']}" for t in spec["buckets"]]
    P.need(len(set(names)) == len(names), f"bucket templates named twice (contract.suffix): {sorted({n for n in names if names.count(n) > 1})}")
    buckets, spec_out, hw = [], [], {}
    for s in spec["skus"]:
        P.need(os.path.isfile(path(s["lib"])), f"sku {s['sku']}: library {s['lib']} is not built")
        lib = open(path(s["lib"]), "rb").read()
        blobs[sha(lib)] = lib
        libname = os.path.basename(s["lib"])
        measured = spec.get("measured", {}).get(s["sku"])
        if measured:
            hw[s["sku"]] = sha(json.dumps(measured["device"], sort_keys=True).encode())
        bid = lambda t: f"{t['role']}.{s['sku']}.{t['contract']}.{t['suffix']}"
        by_name = {}
        for t in spec["buckets"]:
            b = {k: v for k, v in t.items() if k not in TEMPLATE_KEYS}
            b.update(id=bid(t), sku=s["sku"], weights=windex[t["variant"]], lib=libname, requires=dict(arch=s["arch"], cc=s["cc"], **s["requires"]),
                     files={libname: sha(lib), **({"plan.json": plans[t["plan"]]} if t.get("plan") else {})}, tune={**s.get("tune", {}), **t.get("tune", {})})
            if t.get("plan"):
                b["plan"] = "plan.json"
            us = (measured or {}).get("step_us", {}).get(f"{t['contract']}.{t['suffix']}")
            if us is not None:
                b["step_us"] = us
            buckets.append(b)
            by_name[f"{t['contract']}.{t['suffix']}"] = b["id"]
        for x in spec.get("spec", []):
            P.need(x["target"] in by_name and x["drafter"] in by_name, f"spec {x.get('name')}: {x['target']} / {x['drafter']} name no bucket template")
            spec_out.append(dict(x, target=by_name[x["target"]], drafter=by_name[x["drafter"]]))
    m = dict(schema=P.SCHEMA, abi=spec["abi"], compiler=dict(name="goethe-engine pack build", version=__version__), model=dict(spec["model"], weights=weights),
             arch=spec["skus"][0]["arch"], arches={s["arch"]: dict(platform="cuda", cc=s["cc"]) for s in spec["skus"]}, cuda=cuda or nvcc_version(), hw=hw, runtime=spec["runtime"], contracts=spec["contracts"],
             variants=variants, buckets=buckets, spec=spec_out)
    return m, list(blobs.values())


SEEDS = (1, 2, 3)


def check(pack, weights, drafter, trusted=None, n=32, sku=None):
    """{arch, pairs, failures} on this device: each speculative pair the loader serves (per numerics) samples at temperature 1 what its
    target alone samples, seed by seed, and the seeds differ. O(pairs x seeds x n) steps."""
    from .api import Model
    # ponytail: the pairs the loader picks per numerics, not every pair of the arch; check each when a pack serves a pair by policy
    pairs, failures, arch = [], [], None
    for numerics in ("invariant", "tolerance"):
        runs = []
        for d in (drafter, None):
            try:
                with Model(pack, weights, drafter=d, numerics=numerics, trusted=trusted, sku=sku) as mo:
                    if d and not mo.backend.spec:
                        break
                    b = mo.backend.bucket
                    try:
                        runs.append((b["id"], [mo.generate([list(range(1, 9))], n, temperature=1.0, seed=s)[0] for s in SEEDS]))
                    except RuntimeError as e:
                        runs.append((b["id"], e))
                    arch = b["requires"]["arch"]
            except P.PackError:
                break                    # the pack serves nothing at these numerics on this device
        if len(runs) < 2 or runs[0][0] in pairs:
            continue
        (t, got), (_, ref) = runs
        pairs.append(t)
        if isinstance(got, Exception):
            failures.append(f"{t}: a sampled request fails: {got}")
        elif len({tuple(x) for x in got}) < len(SEEDS):
            failures.append(f"{t}: seeds {SEEDS} give the same tokens: the round ignores sampling")
        elif got != ref:
            failures.append(f"{t}: a seeded sample differs from its target alone's (seed {SEEDS[[g == r for g, r in zip(got, ref)].index(False)]})")
    P.need(pairs, "this device serves none of the pack's speculative pairs: nothing to check")
    return dict(arch=arch, pairs=pairs, failures=failures)


def build(spec_path, out, key=None, cuda=None, store=None):
    """Build the pack of the spec at spec_path into out (signed with key when given); returns the manifest bytes."""
    spec = json.load(open(spec_path))
    m, blobs = manifest(spec, os.path.dirname(os.path.abspath(spec_path)), cuda)
    raw = P.write(out, m, blobs, store=store)
    if key:
        P.sign_pack(out, key)
    return raw
