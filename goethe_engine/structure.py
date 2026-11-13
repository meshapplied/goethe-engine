"""Model structure: what a pack was compiled against, independent of the weight values.

structure(paths, keys=None) is the canonical sha256 over (a) the metadata values of `keys` (GGUF KV keys, or config.json keys as
dotted paths) and (b) the tensor table: name, shape and element type (GGUF) or dtype (safetensors), plus for a safetensors checkpoint
its quantization config; the table is sorted, so tensor order, file order and data alignment do not matter. Weight bytes are never
read: GGUF and safetensors headers only. `keys` is the list a pack was compiled with (what the compiler's capture read); None
hashes the tensor table alone.

  paths: a model directory (every *.safetensors in it, plus config.json and hf_quant_config.json) or a list of files
         (.gguf shards, or .safetensors files and the directory's configs)
Returns {"sha256": hex, "keys": [...]}; table(paths) is the parsed form and differ(a, b) names the first difference."""
import glob
import hashlib
import json
import os
import struct

GGUF_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
              22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
              35: "TQ2_0", 39: "MXFP4", 40: "NVFP4"}
MAX_HEADER = 1 << 30


class StructureError(Exception):
    pass


class _Rd:
    def __init__(self, f):
        self.f = f

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise StructureError("truncated header")
        return b

    def get(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def str(self):
        n = self.get("Q")
        if n > MAX_HEADER:
            raise StructureError("bad string length")
        return self.take(n).decode("utf-8", "replace")


_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _value(r, t):
    if t in _SCALARS:
        return r.get(_SCALARS[t])
    if t == 8:
        return r.str()
    if t == 9:
        et, n = r.get("I"), r.get("Q")
        if et == 9 or n > MAX_HEADER:
            raise StructureError("unsupported array")
        return [_value(r, et) for _ in range(n)]
    raise StructureError(f"unknown metadata type {t}")


def gguf_header(path):
    """({key: value}, [(name, dims, type name)]) of a GGUF file: header only."""
    with open(path, "rb") as f:
        r = _Rd(f)
        if r.take(4) != b"GGUF":
            raise StructureError(f"{path}: not a GGUF file")
        if r.get("I") not in (2, 3):
            raise StructureError(f"{path}: GGUF version")
        nt, nkv = r.get("Q"), r.get("Q")
        kv = {}
        for _ in range(nkv):
            k = r.str()
            kv[k] = _value(r, r.get("I"))
        tens = []
        for _ in range(nt):
            name, nd = r.str(), r.get("I")
            if nd > 8:
                raise StructureError(f"{path}: tensor rank")
            dims = [r.get("Q") for _ in range(nd)]
            ty = r.get("I")
            r.get("Q")                                   # the data offset: not structure
            tens.append((name, dims, GGUF_TYPES.get(ty, f"type{ty}")))
    return kv, tens


def st_header(path):
    """[(name, shape, dtype)] of a safetensors file: header only."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n <= MAX_HEADER:
            raise StructureError(f"{path}: safetensors header")
        h = json.loads(f.read(n))
    return [(k, list(v["shape"]), v["dtype"]) for k, v in h.items() if k != "__metadata__"]


def _dig(d, path):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def table(paths):
    """{"kv": {key: value}, "config": {...}, "tensors": sorted [(name, shape, type)], "quant": ...} of a model directory or file list."""
    paths = [paths] if isinstance(paths, str) else list(paths)
    if len(paths) == 1 and os.path.isdir(paths[0]):
        d = paths[0]
        paths = sorted(glob.glob(os.path.join(d, "*.safetensors"))) or sorted(glob.glob(os.path.join(d, "*.gguf")))
        extra = [os.path.join(d, n) for n in ("config.json", "hf_quant_config.json") if os.path.isfile(os.path.join(d, n))]
    else:
        extra = [p for p in paths if os.path.basename(p) in ("config.json", "hf_quant_config.json")]
        paths = [p for p in paths if p not in extra]
        if not extra and any(p.endswith(".safetensors") for p in paths):    # a checkpoint's configs sit beside its shards
            extra = [os.path.join(os.path.dirname(paths[0]), n) for n in ("config.json", "hf_quant_config.json")
                     if os.path.isfile(os.path.join(os.path.dirname(paths[0]), n))]
    kv, tens, cfg = {}, [], {}
    for p in paths:
        if p.endswith(".gguf"):
            k, t = gguf_header(p)
            kv.update(k)
            tens += t
        elif p.endswith(".safetensors"):
            tens += st_header(p)
        else:
            raise StructureError(f"{p}: neither GGUF nor safetensors")
    for p in extra:
        cfg[os.path.basename(p)] = json.load(open(p))
    names = [t[0] for t in tens]
    if len(set(names)) != len(names):
        raise StructureError("a tensor name occurs twice")
    quant = {"config.json": (cfg.get("config.json") or {}).get("quantization_config"),
             "hf_quant_config.json": (cfg.get("hf_quant_config.json") or {}).get("quantization")}
    return dict(kv=kv, config=cfg.get("config.json") or {}, tensors=sorted((n, list(s), t) for n, s, t in tens), quant=quant)


def _read(tab, key):
    """The value of key: a GGUF KV key, else a dotted config.json path (text_config is looked through)."""
    if key in tab["kv"]:
        return tab["kv"][key]
    v = _dig(tab["config"], key)
    return v if v is not None else _dig(tab["config"].get("text_config") or {}, key)


def canonical(tab, keys=None):
    keys = sorted(keys or [])
    return json.dumps(dict(keys={k: _read(tab, k) for k in keys}, tensors=tab["tensors"], quant=tab["quant"]), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)


def structure(paths, keys=None):
    tab = table(paths)
    return dict(sha256=hashlib.sha256(canonical(tab, keys).encode()).hexdigest(), keys=sorted(keys or []))


def differ(a, b, keys=None):
    """The first difference between two tables (as table() returns), named; None if the structures are equal."""
    for k in sorted(keys or []):
        if _read(a, k) != _read(b, k):
            return f"key {k}: {_read(a, k)!r} != {_read(b, k)!r}"
    ta, tb = {t[0]: t for t in a["tensors"]}, {t[0]: t for t in b["tensors"]}
    for n in sorted(set(ta) | set(tb)):
        if n not in ta:
            return f"tensor {n}: missing from the model the pack was compiled for"
        if n not in tb:
            return f"tensor {n}: missing from this model"
        if ta[n] != tb[n]:
            return f"tensor {n}: {ta[n][1:]} != {tb[n][1:]}"
    if a["quant"] != b["quant"]:
        return "quantization config differs"
    return None


def describe(paths, keys=None):
    """The manifest form of a model's structure: {sha256, keys, kv (the values of keys), tensors, quant}; the tensor table is stored so a
    load that finds another model can name the first difference without any weight hashing."""
    tab = table(paths)
    keys = sorted(keys or [])
    return dict(sha256=hashlib.sha256(canonical(tab, keys).encode()).hexdigest(), keys=keys, kv={k: _read(tab, k) for k in keys},
                tensors=tab["tensors"], quant=tab["quant"])


def as_table(st):
    """A manifest structure as a table() dict (for differ and canonical)."""
    return dict(kv=st["kv"], config={}, tensors=[(n, list(s), t) for n, s, t in st["tensors"]], quant=st["quant"])


def check(st, paths):
    """Raise StructureError naming the first difference unless the model at paths has the structure st (a describe() dict)."""
    tab = table(paths)
    got = hashlib.sha256(canonical(tab, st["keys"]).encode()).hexdigest()
    if got != st["sha256"]:
        raise StructureError(differ(as_table(st), tab, st["keys"]) or "structure differs")
