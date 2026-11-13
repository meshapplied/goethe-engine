"""Pack index: which signed pack serves which model on which arch. Verify and match only; offline, fail-closed.

index.json (signed by index.sig, sign.py scheme, detached over the exact bytes):
  {"schema": "gc-pack-index", "version": 3, "serial": int >= 1 (monotonic), "issued": "YYYY-MM-DDTHH:MM:SSZ",
   "expires": same format,
   "entries": [{"model": {"hf_repo": str, "revision": str, "format": "gguf" | "safetensors",    # hf_repo, revision: informational
                          "structure": [{"variant": str, "sha256": hex, "keys": [str]}]},       # one per variant in the pack (structure.py)
                "arch": "sm_120a", "abi_major": 3, "runtime": {"kind": str, "version": int}, "tier": str,
                "file": "<model>.gcpack",                             # the asset name (ASCII slug), one file per model
                "manifest_sha256": hex,                               # sha256 of the pack's manifest bytes: what `sig` signs (with the
                "sig": {gc-sig-1 object},                             #   domain prefix of pack.py) and what a range fetch is checked against
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

SCHEMA, VERSION = "gc-pack-index", 3
HEX = re.compile(r"^[0-9a-f]{64}$")
SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.gcpack$")
RT_PREF = {"gc-c": 0}   # when several entries match: ranked by runtime, then tier, then manifest hash


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
    need(m.get("format") in ("gguf", "safetensors"), f"{w}: model format")
    st = m.get("structure")
    need(isinstance(st, list) and st and all(isinstance(x, dict) and isinstance(x.get("variant"), str) and HEX.match(str(x.get("sha256")))
                                             and isinstance(x.get("keys"), list) and all(isinstance(k, str) for k in x["keys"]) for x in st),
         f"{w}: model.structure is a list of {{variant, sha256, keys}}")
    need(isinstance(e.get("arch"), str) and re.fullmatch(r"sm_\d+[a-z]?", e["arch"]), f"{w}: arch is not sm_XX")
    need(e.get("abi_major") == 3, f"{w}: abi_major")
    r = e.get("runtime")
    need(isinstance(r, dict) and isinstance(r.get("kind"), str) and isinstance(r.get("version"), int), f"{w}: runtime")
    need(isinstance(e.get("tier"), str) and re.fullmatch(r"[a-z0-9_-]+", e["tier"]), f"{w}: tier")
    need(isinstance(e.get("file"), str) and SLUG.match(e["file"]), f"{w}: file is not a <model>.gcpack asset name")
    need(HEX.match(str(e.get("manifest_sha256"))), f"{w}: manifest_sha256")
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


def match_dir(ix, model_dir, arch, engine_version=__version__, tier=None):
    """The entries for this arch, ABI, runtime and engine version one of whose variant structures a file set of model_dir has
    (header-only), best first (see RT_PREF); each carries "variant": the matching variant id."""
    ev, la = vtuple(engine_version, "engine version"), pack.semver(pack.ABI)[0]
    memo, sets, hit = {}, model_sets(model_dir), []
    for e in ix["entries"]:
        if not (e["arch"] == arch and e["abi_major"] == la and e["runtime"]["version"] in pack.RUNTIMES.get(e["runtime"]["kind"], ())
                and vtuple(e["min_engine"], "min_engine") <= ev and (tier is None or e["tier"] == tier)):
            continue
        found = None
        for x in e["model"]["structure"]:
            for fs in sets:
                k = (tuple(x["keys"]), tuple(fs))
                if k not in memo:
                    try:
                        memo[k] = structure.structure(fs, x["keys"])["sha256"]
                    except (structure.StructureError, OSError, ValueError):
                        memo[k] = None
                if memo[k] == x["sha256"]:
                    found = x["variant"]
                    break
            if found:
                break
        if found:
            hit.append(dict(e, variant=found))
    return sorted(hit, key=lambda e: (RT_PREF.get(e["runtime"]["kind"], 9), e["tier"], e["manifest_sha256"]))


def main(argv, trusted=None):
    """`index verify` and `pack match`: exit 0 ok, 1 refused (signature, key, schema, rollback, expiry), 3 no pack."""
    ap = argparse.ArgumentParser(prog="goethe-engine", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__ + """
Commands (cmd = verify for `index verify`, match for `pack match`); offline, JSON on stdout.
  index verify  -> {"serial", "expires", "keyid", "entries"}
  pack match    -> the best entry one of whose model.structure variants (tensor table and the metadata keys the pack was compiled
                   with, read from headers; structure.py) a file set of --model-dir has, for --arch, ABI major 3, a runtime this
                   engine runs and min_engine <= --engine-version; then tier, then manifest_sha256. The output is the entry plus
                   "variant". Entry fields: model, arch, abi_major, runtime, tier, file, manifest_sha256, size, min_engine, sig
                   (gc-sig-1 object: write it to FILE.gcpack.sig).
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
