"""Pack index: which signed pack serves which model on which arch. Verify and match only; offline, fail-closed.

index.json (signed by index.sig, sign.py scheme, detached over the exact bytes):
  {"schema": "gc-pack-index", "version": 2, "serial": int >= 1 (monotonic), "issued": "YYYY-MM-DDTHH:MM:SSZ",
   "expires": same format,
   "entries": [{"model": {"hf_repo": str, "revision": str, "format": "gguf" | "safetensors",
                          "files": {file name: sha256 hex},           # optional: the model's whole per-file sha256 set (exact identity)
                          "structure": {"sha256": hex, "keys": [str]}},   # optional: structural identity (structure.py)
                "arch": "sm_120a", "abi_major": 2, "runtime": {"kind": str, "version": int}, "tier": str,
                "file_sha256": hex,                                   # the .gcpack file, named by its sha256 in the store
                "sig": {gc-sig-1 object},                             # detached signature over the exact file bytes
                "size": int bytes of the file, "min_engine": "MAJOR.MINOR.PATCH"}]}
Unknown extra keys are ignored; any other "version" is refused."""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

from . import __version__, pack, sign, structure

SCHEMA, VERSION = "gc-pack-index", 2
HEX = re.compile(r"^[0-9a-f]{64}$")
RT_PREF = {"gc-c": 0}   # when several entries match: ranked by runtime, then the most model files (a speculative pair over its target), tier, file hash


class IndexError_(Exception):
    """The index is refused: bad signature, bad schema, rollback, expiry, or no entry matches."""


def need(c, msg):
    if not c:
        raise IndexError_(msg)


def when(s, what):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        raise IndexError_(f"{what} {s!r} is not YYYY-MM-DDTHH:MM:SSZ")


def vtuple(s, what):
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", s) if isinstance(s, str) else None
    need(m, f"{what} {s!r} is not MAJOR.MINOR.PATCH")
    return tuple(int(x) for x in m.groups())


def validate_entry(e, i):
    w = f"entries[{i}]"
    need(isinstance(e, dict) and isinstance(e.get("model"), dict), f"{w}: not an entry")
    m = e["model"]
    need(isinstance(m.get("hf_repo"), str) and m["hf_repo"] and isinstance(m.get("revision"), str) and m["revision"],
         f"{w}: model needs hf_repo and revision")
    need(m.get("format") in ("gguf", "safetensors"), f"{w}: model format")
    f, st = m.get("files"), m.get("structure")
    need(st is not None or (isinstance(f, dict) and f), f"{w}: model needs files (exact) or a structure")
    need(f is None or (isinstance(f, dict) and all(isinstance(k, str) and k and HEX.match(str(v)) for k, v in f.items())),
         f"{w}: model files must map names to sha256")
    need(st is None or (isinstance(st, dict) and HEX.match(str(st.get("sha256"))) and isinstance(st.get("keys"), list)
                        and all(isinstance(k, str) for k in st["keys"])), f"{w}: model structure needs sha256 and keys")
    need(isinstance(e.get("arch"), str) and re.fullmatch(r"sm_\d+[a-z]?", e["arch"]), f"{w}: arch is not sm_XX")
    need(isinstance(e.get("abi_major"), int) and e["abi_major"] >= 0, f"{w}: abi_major")
    r = e.get("runtime")
    need(isinstance(r, dict) and isinstance(r.get("kind"), str) and isinstance(r.get("version"), int), f"{w}: runtime")
    need(isinstance(e.get("tier"), str) and re.fullmatch(r"[a-z0-9_-]+", e["tier"]), f"{w}: tier")
    need(HEX.match(str(e.get("file_sha256"))), f"{w}: file_sha256")
    need(isinstance(e.get("sig"), dict) and e["sig"].get("schema") == sign.SCHEMA, f"{w}: sig is not a gc-sig-1 object")
    need(isinstance(e.get("size"), int) and e["size"] >= 0, f"{w}: size")
    vtuple(e.get("min_engine"), f"{w}: min_engine")


def verify(raw, sig, last_serial=0, now=None, trusted=None):
    """The checked index dict and the signer's key id. Refuses: signature, schema, serial < last_serial (rollback),
    now >= expires (freeze)."""
    try:
        kid = sign.verify(raw, sig, trusted)
    except sign.SignatureError as e:
        raise IndexError_(str(e))
    try:
        ix = json.loads(raw)
    except ValueError as e:
        raise IndexError_(f"index.json: {e}")
    need(isinstance(ix, dict) and ix.get("schema") == SCHEMA, "not a pack index")
    need(ix.get("version") == VERSION, f"index version {ix.get('version')!r} unknown")
    need(isinstance(ix.get("serial"), int) and ix["serial"] >= 1, "serial must be an integer >= 1")
    iss, exp = when(ix.get("issued"), "issued"), when(ix.get("expires"), "expires")
    need(iss <= exp, "issued after expires")
    need(isinstance(ix.get("entries"), list), "entries must be a list")
    for i, e in enumerate(ix["entries"]):
        validate_entry(e, i)
    need(ix["serial"] >= last_serial, f"serial {ix['serial']} < last seen {last_serial}: rollback (refused)")
    need((now or datetime.now(timezone.utc)) < exp, f"index expired {ix['expires']} (refused)")
    return ix, kid


def dir_files(entry_files, model_dir):
    """{name: sha256} of the files an entry names, as found in model_dir; None if one is missing. Only those files are
    hashed (memoized per user by path, inode, size and mtime); other files in the directory are ignored."""
    out = {}
    for n in entry_files:
        p = os.path.join(model_dir, n)
        if not os.path.isfile(p):
            return None
        out[n] = pack.sha256_file(p, pack.memo_dir())
    return out


def model_sets(model_dir):
    """The file sets of model_dir that can be one model: a checkpoint's shards (all *.safetensors), else each GGUF (a sharded one
    -0000k-of-0000n.gguf together). Header-only readers take these."""
    st = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if st:
        return [st]
    sets = {}
    for p in sorted(glob.glob(os.path.join(model_dir, "*.gguf"))):
        sets.setdefault(re.sub(r"-\d+-of-\d+\.gguf$", ".gguf", p), []).append(p)
    return list(sets.values())


def identity(e, model_dir, sets=None, memo=None):
    """"exact" (every file the entry names is in model_dir with its sha256), "structural" (a file set of model_dir has the
    structure the entry names), else None. The structure is read from headers; the exact check only hashes if the entry has files."""
    m = e["model"]
    st, memo = m.get("structure"), {} if memo is None else memo
    if st is not None:
        key = tuple(st["keys"])
        if key not in memo:
            memo[key] = {}
        ok = False
        for fs in (model_sets(model_dir) if sets is None else sets):
            k2 = (key, tuple(fs))
            if k2 not in memo[key]:
                try:
                    memo[key][k2] = structure.structure(fs, st["keys"])["sha256"]
                except (structure.StructureError, OSError, ValueError):
                    memo[key][k2] = None
            if memo[key][k2] == st["sha256"]:
                ok = True
                break
        if not ok:
            return None
        if m.get("files") and dir_files(m["files"], model_dir) == m["files"]:
            return "exact"
        return "structural"
    return "exact" if dir_files(m["files"], model_dir) == m["files"] else None


def match_dir(ix, model_dir, arch, engine_version=__version__, tier=None):
    """Like match, with the identity read from model_dir: an entry matches when a file set of model_dir has its structure
    (header-only) or every file it names is there with that sha256. Candidates are the entries for this arch, ABI, runtime and
    engine version; best first: exact identity, then see RT_PREF. Each returned entry carries "identity": "exact" | "structural"."""
    ev, la = vtuple(engine_version, "engine version"), pack.semver(pack.ABI)[0]
    memo, sets = {}, model_sets(model_dir)
    hit = []
    for e in ix["entries"]:
        if not (e["arch"] == arch and e["abi_major"] == la and e["runtime"]["version"] in pack.RUNTIMES.get(e["runtime"]["kind"], ())
                and vtuple(e["min_engine"], "min_engine") <= ev and (tier is None or e["tier"] == tier)):
            continue
        idy = identity(e, model_dir, sets, memo)
        if idy:
            hit.append(dict(e, identity=idy))
    return sorted(hit, key=lambda e: (e["identity"] != "exact", RT_PREF.get(e["runtime"]["kind"], 9), -len(e["model"].get("files") or {}),
                                      e["tier"], e["file_sha256"]))


def main(argv, trusted=None):
    """`index verify` and `pack match`: exit 0 ok, 1 refused (signature, key, schema, rollback, expiry), 3 no pack."""
    ap = argparse.ArgumentParser(prog="goethe-engine", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__ + """
Commands (cmd = verify for `index verify`, match for `pack match`); offline, JSON on stdout.
  index verify  -> {"serial", "expires", "keyid", "entries"}
  pack match    -> the best entry whose model.structure (tensor table and the metadata keys the pack was compiled with,
                   read from headers; structure.py) a file set of --model-dir has, or whose model.files all exist in --model-dir
                   with those sha256 (other files in the dir are ignored); an exact hit outranks a structural one and the
                   output adds "identity": "exact" | "structural", for --arch, ABI major 2, a runtime this engine runs and min_engine <= --engine-version; a pair
                   (more files) outranks its target, then tier, then file_sha256. Entry fields: model, arch, abi_major,
                   runtime, tier, file_sha256, size, sig (gc-sig-1 object: write it to FILE.gcpack.sig), min_engine.
Exit: 0 ok; 1 refused (bad signature, key id not pinned, schema, serial < --last-serial, expired), reason on stderr;
      3 pack match only: valid index, no entry fits; 2 usage.
""")
    ap.add_argument("cmd", choices=("verify", "match"))
    ap.add_argument("--index", required=True)
    ap.add_argument("--sig", required=True)
    ap.add_argument("--last-serial", type=int, default=0)
    ap.add_argument("--now", help="RFC3339 UTC (tests)")
    ap.add_argument("--arch")
    ap.add_argument("--model-dir")
    ap.add_argument("--tier")
    ap.add_argument("--engine-version", default=__version__)
    a = ap.parse_args(argv)
    try:
        ix, kid = verify(pack.read_capped(a.index), pack.read_capped(a.sig, sign.MAX_SIG) if os.path.lexists(a.sig) else None,
                         a.last_serial, when(a.now, "--now") if a.now else None, trusted)
        if a.cmd == "verify":
            print(json.dumps(dict(serial=ix["serial"], expires=ix["expires"], keyid=kid, entries=len(ix["entries"]))))
            return 0
        need(a.arch and a.model_dir, "match needs --arch and --model-dir")
        hit = match_dir(ix, a.model_dir, a.arch, a.engine_version, a.tier)
        if not hit:
            print("goethe-engine: no pack for this model, arch and engine version", file=sys.stderr)
            return 3
        print(json.dumps(hit[0], indent=1, sort_keys=True))
        return 0
    except (IndexError_, pack.PackError, OSError) as e:
        print(f"goethe-engine: index REFUSED: {e}", file=sys.stderr)
        return 1
