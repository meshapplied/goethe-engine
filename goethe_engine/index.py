"""Pack index: which signed pack serves which model on which arch. Verify and match only; offline, fail-closed.

index.json (signed by index.sig, sign.py scheme, detached over the exact bytes):
  {"schema": "gc-pack-index", "version": 1, "serial": int >= 1 (monotonic), "issued": "YYYY-MM-DDTHH:MM:SSZ",
   "expires": same format,
   "entries": [{"model": {"hf_repo": str, "revision": str, "format": "gguf" | "safetensors",
                          "files": {file name: sha256 hex}},          # the model's whole per-file sha256 set
                "arch": "sm_120a", "abi_major": int, "runtime": {"kind": str, "version": int}, "tier": str,
                "manifest_sha256": hex, "manifest_sig_sha256": hex,   # blobs in the store, named by sha256
                "size": int bytes of all pack blobs, "min_engine": "MAJOR.MINOR.PATCH"}]}
Unknown extra keys are ignored; a higher "version" is refused."""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

from . import __version__, pack, sign

SCHEMA, VERSION = "gc-pack-index", 1
HEX = re.compile(r"^[0-9a-f]{64}$")
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
    need(isinstance(m.get("hf_repo"), str) and m["hf_repo"] and isinstance(m.get("revision"), str) and m["revision"],
         f"{w}: model needs hf_repo and revision")
    need(m.get("format") in ("gguf", "safetensors"), f"{w}: model format")
    f = m.get("files")
    need(isinstance(f, dict) and f and all(isinstance(k, str) and k and HEX.match(str(v)) for k, v in f.items()),
         f"{w}: model files must map names to sha256")
    need(isinstance(e.get("arch"), str) and re.fullmatch(r"sm_\d+[a-z]?", e["arch"]), f"{w}: arch is not sm_XX")
    need(isinstance(e.get("abi_major"), int) and e["abi_major"] >= 0, f"{w}: abi_major")
    r = e.get("runtime")
    need(isinstance(r, dict) and isinstance(r.get("kind"), str) and isinstance(r.get("version"), int), f"{w}: runtime")
    need(isinstance(e.get("tier"), str) and re.fullmatch(r"[a-z0-9_-]+", e["tier"]), f"{w}: tier")
    need(HEX.match(str(e.get("manifest_sha256"))) and HEX.match(str(e.get("manifest_sig_sha256"))),
         f"{w}: manifest hashes")
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
    need(isinstance(ix.get("version"), int) and ix["version"] <= VERSION, f"index version {ix.get('version')!r} unknown")
    need(isinstance(ix.get("serial"), int) and ix["serial"] >= 1, "serial must be an integer >= 1")
    iss, exp = when(ix.get("issued"), "issued"), when(ix.get("expires"), "expires")
    need(iss <= exp, "issued after expires")
    need(isinstance(ix.get("entries"), list), "entries must be a list")
    for i, e in enumerate(ix["entries"]):
        validate_entry(e, i)
    need(ix["serial"] >= last_serial, f"serial {ix['serial']} < last seen {last_serial}: rollback (refused)")
    need((now or datetime.now(timezone.utc)) < exp, f"index expired {ix['expires']} (refused)")
    return ix, kid


def match(ix, files, arch, engine_version=__version__, hf_repo=None, revision=None, tier=None):
    """Entries serving this model identity (files: {name: sha256}, exact set; hf_repo / revision if given must equal)
    on arch, whose ABI major, runtime and min_engine this engine supports; best first."""
    ev, la = vtuple(engine_version, "engine version"), pack.semver(pack.ABI)[0]
    hit = [e for e in ix["entries"]
           if e["model"]["files"] == files and e["arch"] == arch and e["abi_major"] == la
           and e["runtime"]["version"] in pack.RUNTIMES.get(e["runtime"]["kind"], ())
           and vtuple(e["min_engine"], "min_engine") <= ev
           and (hf_repo is None or e["model"]["hf_repo"] == hf_repo)
           and (revision is None or e["model"]["revision"] == revision) and (tier is None or e["tier"] == tier)]
    return sorted(hit, key=lambda e: (RT_PREF.get(e["runtime"]["kind"], 9), e["tier"], e["manifest_sha256"]))


def main(argv, trusted=None):
    ap = argparse.ArgumentParser(prog="goethe-engine index")
    ap.add_argument("cmd", choices=("verify", "match"))
    ap.add_argument("--index", required=True)
    ap.add_argument("--sig", required=True)
    ap.add_argument("--last-serial", type=int, default=0)
    ap.add_argument("--now", help="RFC3339 UTC (tests)")
    ap.add_argument("--arch")
    ap.add_argument("--file", action="append", default=[], metavar="NAME=SHA256")
    ap.add_argument("--hf-repo")
    ap.add_argument("--revision")
    ap.add_argument("--tier")
    ap.add_argument("--engine-version", default=__version__)
    ap.add_argument("--all", action="store_true", help="print every match as a JSON list")
    a = ap.parse_args(argv)
    try:
        ix, kid = verify(pack.read_capped(a.index), pack.read_capped(a.sig, sign.MAX_SIG) if os.path.lexists(a.sig) else None,
                         a.last_serial, when(a.now, "--now") if a.now else None, trusted)
        if a.cmd == "verify":
            print(json.dumps(dict(serial=ix["serial"], expires=ix["expires"], keyid=kid, entries=len(ix["entries"]))))
            return 0
        need(a.arch and a.file, "match needs --arch and at least one --file NAME=SHA256")
        files = dict(f.split("=", 1) for f in a.file if "=" in f)
        need(len(files) == len(a.file), "--file takes NAME=SHA256")
        hit = match(ix, files, a.arch, a.engine_version, a.hf_repo, a.revision, a.tier)
        if not hit:
            print("goethe-engine: no pack for this model, arch and engine version", file=sys.stderr)
            return 3
        print(json.dumps(hit if a.all else hit[0], indent=1, sort_keys=True))
        return 0
    except (IndexError_, pack.PackError, OSError) as e:
        print(f"goethe-engine: index REFUSED: {e}", file=sys.stderr)
        return 1

