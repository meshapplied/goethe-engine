"""Pack index: which signed pack serves which model on which arch. Build, verify and match; offline, fail-closed.

index.json (signed by index.sig, sign.py scheme, detached over the exact bytes):
  {"schema": "gc-pack-index", "version": 3, "serial": int >= 1 (monotonic), "issued": "YYYY-MM-DDTHH:MM:SSZ",
   "expires": same format,
   "entries": [{"model": {"hf_repo": str, "revision": str, "format": "gguf" | "safetensors",    # hf_repo, revision: informational
                          "structure": [{"variant": str, "sha256": hex, "keys": [str]}],        # one per variant in the pack (structure.py)
                          "files": [{"name": str, "bytes": int, "sha256": hex}],                # optional, informational: the model files the pack was
                                                                                                #   verified on (what a client downloads); never used to match
                          "with": [{"role": "drafter", "variant": str, "hf_repo": str,        # optional: files a client downloads with the target
                                    "revision": str, "files": [...]}]},                       #   (its drafter, possibly from another repo)
                "arch": "sm_120a", "abi_major": 3, "runtime": {"kind": str, "version": int}, "tier": str,
                "file": "<model>.gcpack",                             # the asset name (ASCII slug), one file per model
                "manifest_sha256": hex,                               # sha256 of the pack's manifest bytes: what `sig` signs (with the
                "sig": {gc-sig-1 object},                             #   domain prefix of pack.py) and what a range fetch is checked against
                "size": int bytes of the file, "min_engine": "MAJOR.MINOR.PATCH"}]}
Unknown extra keys are ignored; any other "version" is refused."""
import argparse
import glob
import hashlib
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
    fl = m.get("files")
    need(fl is None or (isinstance(fl, list) and all(isinstance(x, dict) and isinstance(x.get("name"), str) and x["name"] and "/" not in x["name"]
                                                     and isinstance(x.get("bytes"), int) and x["bytes"] >= 0 and HEX.match(str(x.get("sha256")))
                                                     for x in fl)), f"{w}: model.files is a list of {{name, bytes, sha256}}")
    wl = m.get("with")
    need(wl is None or (isinstance(wl, list) and all(isinstance(x, dict) and x.get("role") == "drafter" and isinstance(x.get("variant"), str)
                                                     and isinstance(x.get("hf_repo"), str) and x["hf_repo"] and isinstance(x.get("revision", ""), str)
                                                     and isinstance(x.get("files"), list) and all(isinstance(f, dict) and isinstance(f.get("name"), str)
                                                     and "/" not in f["name"] and HEX.match(str(f.get("sha256"))) for f in x["files"]) for x in wl)),
         f"{w}: model.with is a list of {{role drafter, variant, hf_repo, revision, files}}")
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


def source(m, vid):
    """{hf_repo, revision, format, files [{name, bytes, sha256}]} of variant vid of manifest m: what a client downloads for it."""
    v = m["variants"][vid]
    need(isinstance(v.get("source"), dict) and v["source"].get("hf_repo"), f"variant {vid}: no source (pack abi minor 2 or newer)")
    fs = [m["model"]["weights"][i] for i in v["weights"]]
    need(all(HEX.match(str(f.get("sha256"))) and isinstance(f.get("bytes"), int) for f in fs), f"variant {vid}: weights without bytes and sha256 (rebuild with pack build)")
    kinds = {os.path.splitext(f["file"])[1] for f in fs}
    need(kinds in ({".gguf"}, {".safetensors"}), f"variant {vid}: files {sorted(kinds)} are not one format")
    return dict(hf_repo=v["source"]["hf_repo"], revision=v["source"].get("revision", ""), format=kinds.pop()[1:],
                files=[dict(name=f["file"], bytes=f["bytes"], sha256=f["sha256"]) for f in fs])


def entries(path, tier="base", min_engine=__version__, trusted=None, left=None, untested=False):
    """The index entries of the pack at path, its signature (PATH.sig) checked against trusted: one per (arch, target variant), the model
    being that variant's source, with the drafters its speculative pairs name under model.with. Left out, each named in left: a variant
    whose sources pin no revision (nothing a client could download), a paired target on an arch without a passing pack check (with
    untested, listed at tier "untested" instead, so a client can say so). A pack left with no entry is refused. O(variants x buckets)."""
    f, _, _, size, raw, _, v = pack._open_signed(path, None, None, trusted)
    f.close()
    need(v == 3, f"{path}: pack {v}: this index lists pack 3")
    m = pack.check_manifest(raw)
    sig = json.loads(pack.read_capped(path + ".sig", sign.MAX_SIG))
    by, out, ok = {b["id"]: b for b in m["buckets"]}, [], pack.checks(path, hashlib.sha256(raw).hexdigest())
    left = [] if left is None else left
    for vid in sorted({b["variant"] for b in m["buckets"] if b["role"] == "target"}):
        drafters = sorted({by[s["drafter"]]["variant"] for s in m.get("spec", []) if s.get("target") in by and s.get("drafter") in by
                           and by[s["target"]]["variant"] == vid})
        src, st = source(m, vid), m["variants"][vid]["structure"]
        unpinned = [s["hf_repo"] for s in [src] + [source(m, d) for d in drafters] if not s["revision"]]
        if unpinned:
            left.append(f"{vid}: {', '.join(unpinned)} pins no revision")
            continue
        model = dict(hf_repo=src["hf_repo"], revision=src["revision"], format=src["format"], files=src["files"],
                     structure=[dict(variant=vid, sha256=st["sha256"], keys=st["keys"])],
                     **({"with": [dict(role="drafter", variant=d, **{k: x for k, x in source(m, d).items() if k != "format"}) for d in drafters]}
                        if drafters else {}))
        for arch in sorted({b["requires"]["arch"] for b in m["buckets"] if b["variant"] == vid}):
            c = ok.get(arch)
            checked = not drafters or (c and c["pairs"] and not c["failures"])
            if not checked and (not untested or (c and c["failures"])):   # a failed check is never listed
                left.append(f"{vid} on {arch}: no passing pack check")
                continue
            out.append(dict(model=model, arch=arch, abi_major=3, runtime=m["runtime"], tier=tier if checked else "untested", file=os.path.basename(path),
                            manifest_sha256=hashlib.sha256(raw).hexdigest(), sig=sig, size=size, min_engine=min_engine))
    need(out, f"{path}: nothing to list ({'; '.join(left)}): pin the sources' revisions, or run pack check on a device of an arch")
    return out


def build(packs, serial, expires, key, out, tier="base", min_engine=__version__, now=None, trusted=None, base=None, drop=(), left=None, untested=False):
    """Write out/index.json and out/index.sig for the packs (each signed by a key in trusted), signed with key; with base (the current
    index bytes and signature, verified against trusted) its entries for other pack files are kept and serial must be above its own.
    Checked by verify before anything is written; drop: pack files whose kept entries are removed. Returns the index dict."""
    now = now or datetime.now(timezone.utc)
    new = [e for p in packs for e in entries(p, tier, min_engine, trusted, left, untested)]
    kept = []
    need(not drop or base, "--drop removes entries of the current index: pass --index and --sig")
    if base:
        old, _ = verify(*base, now=now, trusted=trusted)
        need(serial > old["serial"], f"serial {serial} is not above the current index's {old['serial']}")
        need(set(drop) <= {e["file"] for e in old["entries"]}, f"--drop {sorted(set(drop) - {e['file'] for e in old['entries']})}: not in the current index")
        kept = [e for e in old["entries"] if e["file"] not in {x["file"] for x in new} | set(drop)]
    ix = dict(schema=SCHEMA, version=VERSION, serial=serial, issued=now.strftime("%Y-%m-%dT%H:%M:%SZ"), expires=expires, entries=kept + new)
    raw = (json.dumps(ix, indent=1, sort_keys=True) + "\n").encode()
    sigb = sign.sign_digest(hashlib.sha256(raw).hexdigest(), key)
    verify(raw, sigb, now=now, trusted=(json.loads(sigb)["keyid"],))
    os.makedirs(out, exist_ok=True)
    pack._atomic(os.path.join(out, "index.json"), [raw])
    pack._atomic(os.path.join(out, "index.sig"), [sigb])
    return ix


SHARD = re.compile(r"-\d+-of-\d+\.(gguf|safetensors)$")


def model_sets(model_dir):
    """The file sets of model_dir that can be one model or one drafter (header-only readers take these, first match wins): each
    safetensors shard group (*-0000k-of-0000n, or one file) alone, so a drafter such as mtp.safetensors beside its checkpoint is a set
    of its own, then all safetensors together; each GGUF with its shards alone, then with each projector GGUF (general.architecture
    clip) in both orders. O(groups x projectors) sets."""
    def groups(ext):
        g = {}
        for p in sorted(glob.glob(os.path.join(model_dir, "*." + ext))):
            g.setdefault(SHARD.sub(r".\1", p), []).append(p)
        return list(g.values())
    st, gg = groups("safetensors"), groups("gguf")
    proj = [g for g in gg if projector(g[0])]
    return (st + ([sum(st, [])] if len(st) > 1 else []) + gg
            + [s for g in gg if g not in proj for p in proj for s in (g + p, p + g)])


def projector(path):
    """Whether a GGUF is a vision or audio projector (an mmproj file: general.architecture clip); unreadable files are not."""
    try:
        return structure.gguf_header(path)[0].get("general.architecture") == "clip"
    except (structure.StructureError, OSError, ValueError):
        return False


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


def parser():
    ap = argparse.ArgumentParser(prog="goethe-engine", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__ + """
Commands (cmd = verify for `index verify`, match for `pack match`); offline, JSON on stdout.
  index verify  -> {"serial", "expires", "keyid", "entries"}
  pack match    -> the best entry one of whose model.structure variants (tensor table and the metadata keys the pack was compiled
                   with, read from headers; structure.py) a file set of --model-dir has, for --arch, ABI major 3, a runtime this
                   engine runs and min_engine <= --engine-version; then tier, then manifest_sha256. The output is the entry plus
                   "variant" and echoes model.files. Entry fields: model, arch, abi_major, runtime, tier, file, manifest_sha256, size, min_engine, sig
                   (gc-sig-1 object: write it to FILE.gcpack.sig).
Exit: 0 ok; 1 refused (bad signature, key id not pinned, schema, serial < --last-serial, expired), reason on stderr;
      3 pack match only: valid index, no entry fits; 2 usage.
""")
    ap.add_argument("cmd", choices=("verify", "match", "build"))
    ap.add_argument("--index")
    ap.add_argument("--sig")
    ap.add_argument("--pack", action="append", default=[], help="build: a signed pack (PACK.sig beside it); repeatable")
    ap.add_argument("--drop", action="append", default=[], help="build: a pack file of the current index whose entries are removed; repeatable")
    ap.add_argument("--untested", action="store_true", help="build: list a paired target on an arch without a pack check at tier untested")
    ap.add_argument("--serial", type=int, help="build: the index serial (above the last published one)")
    ap.add_argument("--expires", help="build: YYYY-MM-DDTHH:MM:SSZ")
    ap.add_argument("--key", help="build: the private key PEM that signs the index")
    ap.add_argument("--out", help="build: the directory for index.json and index.sig")
    ap.add_argument("--min-engine", default=__version__, help="build: the oldest engine the packs run on")
    ap.add_argument("--last-serial", type=int, default=0)
    ap.add_argument("--now", help="RFC3339 UTC (tests)")
    ap.add_argument("--arch")
    ap.add_argument("--model-dir")
    ap.add_argument("--tier")
    ap.add_argument("--engine-version", default=__version__)
    return ap


def main(argv, trusted=None):
    """`index verify` and `pack match`: exit 0 ok, 1 refused (signature, key, schema, rollback, expiry), 3 no pack."""
    a = parser().parse_args(argv)
    try:
        if a.cmd == "build":
            need(a.pack and a.serial and a.expires and a.key and a.out, "build needs --pack, --serial, --expires, --key and --out")
            base = (pack.read_capped(a.index), pack.read_capped(a.sig, sign.MAX_SIG)) if a.index else None
            left = []
            ix = build(a.pack, a.serial, a.expires, a.key, a.out, a.tier or "base", a.min_engine, trusted=trusted, base=base, drop=a.drop, left=left, untested=a.untested)
            print(json.dumps(dict(serial=ix["serial"], expires=ix["expires"], entries=len(ix["entries"]), out=a.out,
                                  left_out=left)))
            return 0
        need(a.index and a.sig, f"{a.cmd} needs --index and --sig")
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
