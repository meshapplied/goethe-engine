#!/usr/bin/env python3
r"""Kernel pack reader and writer (pack ABI 3 and 4). A pack is ONE file, NAME.gcpack, for one model; its buckets may target several GPU
architectures (3.1 arches). write() is the one way to make a pack: it validates the manifest and frames it. Layout (stdlib framing):

  b"GCPACK3\n"  u64le N  header (N bytes of JSON {"manifest": [off, len], "blobs": {sha256: [off, len] or [off, len, "xz"]}})  payload
  offsets count from the end of the header; manifest = manifest.json, blobs = every bucket file, named by the sha256 of its raw bytes;
  len = stored bytes, "xz" = the blob is one xz stream (the only codec; read with a 64 MiB decoder memory limit and a raw size cap)

The signature (gc-sig-1, detached, PACK.sig, or the index entry's) is ECDSA over sha256(b"gc-pack-3\n" + manifest bytes): the manifest
names every blob by its sha256, and a blob is hash-checked when it is extracted, so a file with holes is as safe as a whole one for
the blobs actually used. Weights are not in the pack: a manifest "variant" names the structure of the model files its buckets were
compiled for (structure.py: tensor table and metadata keys, no weight bytes); the loader checks the files it is given against it.

Manifest: model.weights (informational: the files the gates ran on), variants {id: {structure, weights: [model.weights indices]}},
buckets[] each with variant, role, sku, contract, ncol, maxctx, files, lib, requires, numerics axes, step_us.

Axes (3.1, additive; derived when absent, so a 3.0 manifest reads unchanged): bucket "platform" (cuda), "kind" (decode: a batch
ladder bucket | stream: one sequence a token a launch | prefill: a prompt chunk a launch | verify: a speculative verify), "mesh"
({axis: size} over dp tp pp ep sp; absent or {} = one device); manifest "arches" {arch: {...}}, model "type" (decoder).
Overlay: a container of the same framing whose manifest has schema "gc-pack-overlay" and "base" = sha256 of the base manifest bytes;
signed on its own (sha256(b"gc-overlay-3\n" + overlay manifest bytes)), it adds buckets, variants, weights, hw, blob_raw, spec and
arches (a new GPU arch or quant format) without touching or re-signing the base. merge() joins them fail-closed.

Pack 4 (b"GCPACK4\n", manifest abi 4.x, signed over sha256(b"gc-pack-4\n" + manifest bytes)): the same container with the blobs outside
it, in a content-addressed store (Store: open(sha256) -> the stored bytes as a file and their entry; DirStore: a directory of <sha256>.xz
or <sha256>, a download cache; a network transport is another Store). Every blob a v4 bucket names has its raw
size pinned in the signed manifest; a store blob is checked exactly as an embedded one (pin, xz limits, hash). The magic fixes the major:
a v4 file with an abi 3 manifest, or the reverse, is refused, so a signature never carries across.

  python -m goethe_engine.pack verify FILE [--sig F] [--arch sm_XX] [--partial [--model-dir D] [--sku S]]   check, discard
  python -m goethe_engine.pack show FILE [--sig F]                   one line per bucket
  python -m goethe_engine.pack need PREFIX --model-dir D [--arch A --sku S --numerics ..]  the byte ranges a download must fetch
"""
import atexit, hashlib, json, lzma, os, re, shutil, stat, subprocess, sys, tempfile

from . import numerics_kw as NK, sign, structure
from .errors import OutOfMemory, Refused

ABI = "3.0.0"                       # pack ABI semver: additive within a major
MAGIC = b"GCPACK3\n"
DOMAIN = b"gc-pack-3\n"             # signed digest = sha256(DOMAIN + manifest bytes)
ABI4, MAGIC4, DOMAIN4 = "4.0.0", b"GCPACK4\n", b"gc-pack-4\n"   # pack 4: the same framing, blobs by hash from a store
MAX_HEADER = 64 << 20               # bytes of container header JSON
SCHEMA = "gc-pack"
RUNTIMES = {"gc-c": (1,)}           # runtime kinds this loader runs, with their versions
MAX_RAW = 4 << 30                   # bytes: no blob inflates past this (xz bombs are refused as the output grows)
XZ_MEM = 64 << 20                   # bytes: xz decoder memory limit
MAX_JSON = 16 << 20                 # bytes: larger manifests are refused before parsing
ROLES = ("target", "drafter")
POLICIES = {"ar": ("K",), "walk": ("gate", "succ", "pred", "feats"),   # speculative round kinds and their fields
            "mtp": ("K", "feats", "tap", "own", "w_in", "w_own")}
HEX = re.compile(r"^[0-9a-f]{64}$")
OVERLAY_SCHEMA = "gc-pack-overlay"
OVERLAY_DOMAIN = b"gc-overlay-3\n"     # signed digest of an overlay = sha256(OVERLAY_DOMAIN + overlay manifest bytes)
PLATFORM = "cuda"                       # the platform of a bucket without the field (every 3.0 pack)
KINDS = ("decode", "stream", "prefill", "verify", "encode")   # encode: a prefill-only graph (embedding, an encoder); never a decode bucket
MESH_AXES = ("dp", "tp", "pp", "ep", "sp")  # data, tensor, pipeline, expert, sequence / context parallel
MODEL_TYPE = "decoder"                  # model.type of a manifest without the field


class PackError(Refused):
    """Any mismatch between a pack, its blobs, the weights and the device: the pack is not loaded."""


def semver(s):
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", s or "")
    if not m:
        raise PackError(f"abi {s!r} is not MAJOR.MINOR.PATCH")
    return tuple(int(x) for x in m.groups())


def need(c, msg):
    if not c:
        raise PackError(msg)


def safe_rel(rel):
    """A file name inside a bucket: relative, no empty, dot or dotdot component, no backslash or NUL."""
    return (isinstance(rel, str) and rel and not rel.startswith("/") and "\\" not in rel and "\0" not in rel
            and all(c not in ("", ".", "..") for c in rel.split("/")))


def validate(m, loader_abi=ABI):
    """Schema and ABI of a manifest dict; raises PackError. The major must be the loader's; any minor reads (fields it
    does not know are ignored; buckets with an enum value it does not know are skipped by select)."""
    need(isinstance(m, dict) and m.get("schema") == SCHEMA,
         f"not a pack manifest (schema {m.get('schema') if isinstance(m, dict) else None!r})")
    pv, la = semver(m.get("abi")), semver(loader_abi)
    need(pv[0] == la[0], f"pack abi {m['abi']} is major {pv[0]}; this loader ({loader_abi}) reads major {la[0]} (refused)")
    for k in ("compiler", "model", "arch", "cuda", "runtime", "contracts", "buckets"):
        need(k in m, f"manifest lacks {k}")
    ws = m["model"].get("weights")
    need(isinstance(ws, list) and ws and all(isinstance(w, dict) and isinstance(w.get("file"), str) for w in ws),
         "model.weights: the list of weight files the gates ran on")
    need(pv[1] < 1 or isinstance(m["model"].get("name"), str) and m["model"]["name"],
         f"pack abi {m['abi']}: model.name (the model's identity) is required from minor 1 (3.1, 4.1)")
    lim = m["model"].get("limits", [])
    need(isinstance(lim, list) and all(isinstance(x, dict) and isinstance(x.get("maxctx"), int) and isinstance(x.get("why"), str) for x in lim),
         "model.limits: [{maxctx, why}]")
    vs = m.get("variants")
    need(isinstance(vs, dict) and vs, "manifest lacks variants")
    for vid, v in vs.items():
        st = v.get("structure") if isinstance(v, dict) else None
        need(isinstance(st, dict) and HEX.match(st.get("sha256", "")) and isinstance(st.get("keys"), list)
             and isinstance(st.get("kv"), dict) and isinstance(st.get("tensors"), list), f"variant {vid}: structure needs sha256, keys, kv and tensors")
        need(isinstance(v.get("weights"), list) and v["weights"] and all(isinstance(i, int) and 0 <= i < len(ws) for i in v["weights"]),
             f"variant {vid}: weights indices")
        src = v.get("source")
        need(pv[1] < 2 or isinstance((v.get("desc") or {}).get("quant"), str) and v["desc"]["quant"] and isinstance(src, dict)
             and isinstance(src.get("hf_repo"), str) and src["hf_repo"] and isinstance(src.get("revision", ""), str),
             f"pack abi {m['abi']}: variant {vid} needs desc.quant and source {{hf_repo, revision}} from minor 2 (3.2, 4.2)")
    need(isinstance(m.get("hw", {}), dict) and all(HEX.match(h or "") for h in m.get("hw", {}).values()), "hw record hashes must be sha256")
    raw = m.get("blob_raw", {})
    need(isinstance(raw, dict) and all(HEX.match(h) and isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= MAX_RAW
                                       for h, n in raw.items()), "manifest blob_raw: {sha256: raw bytes <= 4 GiB}")
    ids = set()
    for b in m["buckets"]:
        for k in ("id", "role", "sku", "contract", "ncol", "maxctx", "files", "lib", "requires"):
            need(k in b, f"bucket {b.get('id')} lacks {k}")
        need(b["id"] not in ids, f"bucket id {b['id']} twice")
        ids.add(b["id"])
        need("step_us" not in b or b["sku"] in m.get("hw", {}), f"bucket {b['id']}: a measured step_us needs the hw record hash of sku {b['sku']}")
        need(b["lib"] in b["files"] and b.get("plan", b["lib"]) in b["files"], f"bucket {b['id']}: lib / plan not in its files")
        need(b["requires"]["arch"] in arches(m), f"bucket {b['id']}: arch {b['requires']['arch']} not in {sorted(arches(m))}")
        need(isinstance(b.get("bodies", {}), dict), f"bucket {b['id']}: bodies is not a map")
        need(isinstance(b.get("platform", PLATFORM), str) and isinstance(b.get("kind", ""), str), f"bucket {b['id']}: platform / kind")
        mesh_key(b.get("mesh"), b["id"])
        for rel, h in b["files"].items():
            need(HEX.match(h) and safe_rel(rel), f"bucket {b['id']}: bad file entry {rel!r}")
    for b in m["buckets"]:
        need(all(v in ids for v in b.get("variants", [])), f"bucket {b['id']}: unknown variant")
        need(0 <= b.get("weights", 0) < len(m["model"]["weights"]), f"bucket {b['id']}: weights index")
        need(b.get("variant") in m["variants"], f"bucket {b['id']}: variant {b.get('variant')!r} is not in the manifest")
    by = {b["id"]: b for b in m["buckets"]}
    for s in m.get("spec", []):
        t, d, pol = by.get(s.get("target")), by.get(s.get("drafter")), s.get("policy") or {}
        if pol.get("kind") not in POLICIES:
            continue                                   # a policy this loader does not know: spec_pair skips it
        need(t and t["role"] == "target" and ((t.get("io") or {}).get("accept") or (t.get("spec_caps") or {}).get("accept")),
             f"spec {s}: target is not a target bucket with device acceptance (io.accept, or spec_caps.accept for gc-c)")
        need(d and d["role"] == "drafter" and d["sku"] == t["sku"], f"spec {s}: drafter is not a drafter bucket of "
             "the target's SKU")
        need(all(isinstance(pol.get(k), str) for k in POLICIES[pol["kind"]]),
             f"spec {s}: policy {pol['kind']} needs string fields {POLICIES[pol['kind']]}")
        need(pol["kind"] not in ("ar", "mtp") or 1 <= int(pol.get("K", 0)) < t["ncol"],
             f"spec {s}: {pol['kind']} K outside 1 .. ncol - 1")
    return m


def arches(m):
    """The GPU arches a manifest's buckets may name: its "arch" and the keys of "arches" (3.1)."""
    a = m.get("arches") or {}
    need(isinstance(a, dict), "manifest arches must be a map")
    return {m["arch"], *a}


def mesh_key(mesh, what="mesh"):
    """The canonical form of a mesh {axis: size}: the sorted (axis, size) pairs with size > 1; () is one device."""
    mesh = mesh or {}
    need(isinstance(mesh, dict) and all(k in MESH_AXES and isinstance(v, int) and not isinstance(v, bool) and v >= 1
                                        for k, v in mesh.items()), f"{what}: mesh {mesh!r} must map {MESH_AXES} to sizes >= 1")
    return tuple(sorted((k, v) for k, v in mesh.items() if v > 1))


def axes(b):
    """(platform, kind, mesh key) of a bucket: its fields, else derived (a 3.0 bucket: cuda, one device, kind from batch / ncol /
    prefill)."""
    kind = b.get("kind") or ("prefill" if b.get("prefill") else "decode" if b.get("batch") else "stream" if b["ncol"] == 1
                             else "verify")
    return b.get("platform", PLATFORM), kind, mesh_key(b.get("mesh"), b["id"])


def restrict(m, *, platform=PLATFORM, mesh=None):
    """The manifest restricted to the buckets of one platform and device mesh (default one device): what a backend plugin opens."""
    mk = mesh_key(mesh)
    bs = [b for b in m["buckets"] if axes(b)[0] == platform and axes(b)[2] == mk]
    ids = {b["id"] for b in bs}
    need(bs, f"no bucket for platform {platform}, mesh {dict(mk) or 'one device'}; the pack has "
         f"{sorted({(axes(b)[0], str(dict(axes(b)[2]) or 'one device')) for b in m['buckets']})}")
    return dict(m, buckets=bs, spec=[s for s in m.get("spec", []) if s.get("target") in ids and s.get("drafter") in ids],
                contracts=[c for c in m["contracts"] if any(b["contract"] == c for b in bs)])


def merge(m, raw, overlays):
    """The manifest of base m (its bytes raw) joined with overlay manifests (dicts; schema gc-pack-overlay, base = sha256(raw)), applied
    in the order of their own manifest digests. Additive only: a bucket id, sku hash, blob size, arch or a different variant under an
    existing id that collides is refused; an overlay's weights indices >= len(its base's model.weights) name its own weights list."""
    if not overlays:
        return m
    out, h = json.loads(raw), hashlib.sha256(raw).hexdigest()
    nb, ws = len(out["model"]["weights"]), out["model"]["weights"]
    ids = {b["id"] for b in out["buckets"]}
    for o in sorted(overlays, key=lambda o: hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()):
        need(isinstance(o, dict) and o.get("schema") == OVERLAY_SCHEMA, "not a pack overlay manifest")
        need(o.get("base") == h, f"overlay is for base manifest {o.get('base')}, this pack's is {h} (refused)")
        need(semver(o.get("abi"))[0] == semver(out["abi"])[0], f"overlay abi {o.get('abi')} vs pack abi {out['abi']}")
        own = o.get("weights", [])
        need(isinstance(own, list) and all(isinstance(w, dict) and isinstance(w.get("file"), str) for w in own), "overlay weights")
        at = {i: i for i in range(nb)}
        for i, w in enumerate(own):            # a file the merged list already names is shared, else appended
            same = [j for j, x in enumerate(ws) if x.get("sha256") and x.get("sha256") == w.get("sha256")]
            at[nb + i] = same[0] if same else len(ws)
            if not same:
                ws.append(w)
        remap = lambda i: (need(i in at, f"overlay weights index {i}"), at[i])[1]
        for k in ("hw", "blob_raw", "arches"):
            for x, v in (o.get(k) or {}).items():
                need(out.setdefault(k, {}).get(x, v) == v, f"overlay {k} {x} differs from the pack's (refused)")
                out[k][x] = v
        for vid, v in (o.get("variants") or {}).items():
            v = dict(v, weights=[remap(i) for i in v.get("weights", [])])
            need(vid not in out["variants"] or out["variants"][vid] == v, f"overlay variant {vid} differs from the pack's (refused)")
            out["variants"][vid] = v
        for b in o.get("buckets", []):
            need(isinstance(b, dict) and b.get("id") not in ids, f"overlay bucket {b.get('id') if isinstance(b, dict) else b} "
                 "is already in the pack (refused)")
            ids.add(b["id"])
            b = dict(b, weights=remap(b.get("weights", 0)))
            if o.get("runtime", out["runtime"]) != out["runtime"]:
                b.setdefault("runtime", o["runtime"])
            out["buckets"].append(b)
        out["contracts"] += [c for c in o.get("contracts", []) if c not in out["contracts"]]
        out["spec"] = out.get("spec", []) + list(o.get("spec", []))
    return validate(out, out["abi"])


def read_capped(path, cap=None):
    """The bytes of a regular file, never through a symlink, refused above cap bytes."""
    cap = cap or MAX_JSON
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        raise PackError(f"{path}: {e.strerror} (symlinks are refused)")
    with os.fdopen(fd, "rb") as f:
        need(stat.S_ISREG(os.fstat(fd).st_mode), f"{path}: not a regular file")
        data = f.read(cap + 1)
    need(len(data) <= cap, f"{path}: larger than {cap} bytes (refused)")
    return data


def check_manifest(raw, loader_abi=ABI, arch=None):
    """The manifest dict of these bytes: schema, ABI major, runtime, arch."""
    try:
        m = json.loads(raw)
    except ValueError as e:
        raise PackError(f"manifest.json: {e}")
    validate(m, loader_abi)
    rt = m["runtime"]
    need(rt.get("kind") in RUNTIMES and rt.get("version") in RUNTIMES[rt["kind"]],
         f"runtime {rt.get('kind')} {rt.get('version')} is not one this engine runs {RUNTIMES}")
    need(arch is None or arch in arches(m), f"pack arches {sorted(arches(m))} lack {arch}")
    return m


def blob_hashes(buckets):
    return sorted({x for b in buckets for x in b["files"].values()})


class Staged:
    """A verified pack file: manifest (verified by the signature), the signer's key id and a private directory dir (mode 0700, ours
    alone; manifest.json, manifest.sha256, blobs/). Blobs are extracted on demand by ensure(), each hash-checked against the manifest
    as it is read, so a pack with holes serves every branch whose blobs are present. close() removes dir."""

    def __init__(self):
        self.dir, self.manifest, self.keyid, self.manifest_sha256 = tempfile.mkdtemp(prefix="goethe-engine-"), None, None, None
        self.f = self.blobs = self.base = self.total = None
        self.src, self.files, self.overlays = {}, [], []    # blob hash -> (file, base, total, entry); open containers; overlay digests
        atexit.register(self.close)

    store = None                     # a Store blobs missing from the containers are read from (pack 4)

    def add(self, f, blobs, base, total):
        """A container whose blobs this stage reads (the base pack first, then each overlay); a hash already present keeps its source."""
        self.files.append(f)
        for bh, e in blobs.items():
            self.src.setdefault(bh, (f, base, total, e))

    def close(self):
        for f in self.files:
            f.close()
        self.files, self.f = [], None
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None

    def ensure(self, buckets):
        """Extract the blobs of buckets (default: all) into dir/blobs; refuses a missing, truncated or altered blob."""
        for bh in blob_hashes(self.manifest["buckets"] if buckets is None else buckets):
            dst = os.path.join(self.dir, "blobs", bh)
            if os.path.exists(dst):
                continue
            if bh not in self.src and self.store is not None:
                got = self.store.open(bh)
                if got:
                    self.files.append(got[0])
                    self.src[bh] = (got[0], 0, got[1][1], got[1])
            need(bh in self.src, f"blob {bh} missing from the pack" + (" and the store" if self.store is not None else ""))
            f, base, total, e = self.src[bh]
            off, ln, codec = blob_entry(e, bh)
            _section(off, ln, total - base, f"blob {bh}")
            f.seek(base + off)
            hh, left = hashlib.sha256(), ln
            dec = lzma.LZMADecompressor(lzma.FORMAT_XZ, memlimit=XZ_MEM) if codec else None
            pin = self.manifest.get("blob_raw", {}).get(bh)       # the signed raw size of the blob (absent: only MAX_RAW bounds it)
            need(pin is not None or not codec, f"blob {bh}: xz without a raw size pinned in the signed manifest (refused)")
            cap = MAX_RAW if pin is None else min(pin, MAX_RAW)
            out = 0

            def put(o, b):
                nonlocal out
                out += len(b)
                need(out <= cap, f"blob {bh}: inflates past {cap} bytes (refused)")
                hh.update(b)
                o.write(b)
            try:
                with _create(dst) as o:
                    while left:
                        chunk = f.read(min(left, 4 << 20))
                        need(chunk, f"blob {bh}: pack truncated or a hole")
                        left -= len(chunk)
                        if not dec:
                            put(o, chunk)
                            continue
                        try:
                            b = dec.decompress(chunk, min(1 << 20, cap + 1 - out))
                            while True:
                                put(o, b)
                                if dec.needs_input or dec.eof:
                                    break
                                b = dec.decompress(b"", min(1 << 20, cap + 1 - out))
                        except (lzma.LZMAError, EOFError) as e:
                            raise PackError(f"blob {bh}: xz: {e}")
                    need(not dec or (dec.eof and not dec.unused_data and not left), f"blob {bh}: xz stream truncated or followed by data")
                need(pin is None or out == pin, f"blob {bh}: {out} raw bytes, the manifest pins {pin}")
            except PackError:
                if os.path.exists(dst):
                    os.unlink(dst)
                raise
            if hh.hexdigest() != bh:
                os.unlink(dst)
                raise PackError(f"blob {bh} does not hash to its name")


def blob_entry(e, what):
    """(offset, stored length, codec) of a header blob entry [off, len] or [off, len, \"xz\"]; any other codec is refused."""
    need(isinstance(e, list) and len(e) in (2, 3) and (len(e) == 2 or e[2] == "xz"), f"blob {what}: container entry {e!r} (codecs: stored, xz)")
    return e[0], e[1], e[2] if len(e) == 3 else None


def _create(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb")


def _section(off, ln, total, what):
    need(isinstance(off, int) and isinstance(ln, int) and 0 <= off and 0 <= ln and off + ln <= total, f"{what}: section out of range")


class DirStore:
    """A content-addressed blob store in a directory: <sha256>.xz (xz stream) or <sha256> (raw bytes), regular files, no symlinks."""

    def __init__(self, path):
        self.path = path

    def open(self, h):
        """(open file, container entry [0, stored length] or [0, stored length, "xz"]) of blob h, or None when the store lacks it."""
        need(HEX.match(h), f"blob name {h!r}")
        for name, codec in ((h + ".xz", ["xz"]), (h, [])):
            try:
                fd = os.open(os.path.join(self.path, name), os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError:
                continue
            except OSError as e:
                raise PackError(f"store blob {name}: {e.strerror} (symlinks are refused)")
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise PackError(f"store blob {name}: not a regular file")
            return os.fdopen(fd, "rb"), [0, os.fstat(fd).st_size, *codec]
        return None


def open_store(store):
    """A Store from an object with open(), a directory path, or None ($GOETHE_BLOB_STORE, else no store)."""
    store = store if store is not None else os.environ.get("GOETHE_BLOB_STORE")
    return store if store is None or hasattr(store, "open") else DirStore(store)


def read_container(f, total):
    """(header blobs, base offset, manifest bytes, offset just past the manifest) of an open pack file (or of a prefix holding its header and
    manifest); pack 3 or 4 framing (read_container_v also returns the major its magic fixes)."""
    return read_container_v(f, total)[1:]


def read_container_v(f, total):
    """(3 or 4, header blobs, base offset, manifest bytes, end of manifest) of an open pack file or prefix."""
    mg = f.read(len(MAGIC))
    need(mg in (MAGIC, MAGIC4), "not a gcpack file (bad magic)")
    n = int.from_bytes(f.read(8), "little")
    need(0 < n <= MAX_HEADER, "container header size")
    try:
        hdr = json.loads(f.read(n))
        base = len(MAGIC) + 8 + n
        mo, ml = hdr["manifest"]
        blobs = hdr["blobs"]
        need(isinstance(blobs, dict), "container header blobs")
    except (ValueError, KeyError, TypeError) as e:
        raise PackError(f"container header: {e}")
    _section(mo, ml, total - base, "manifest")
    need(ml <= MAX_JSON, "manifest.json larger than the cap (refused)")
    f.seek(base + mo)
    raw = f.read(ml)
    need(len(raw) == ml, "manifest truncated")
    return (4 if mg == MAGIC4 else 3), blobs, base, raw, base + mo + ml


def frame(path, raw, blobs, major=3):
    """Write a container whole or not at all: the magic of major, header {manifest, blobs}, manifest bytes raw, then the stored blobs
    ({sha256: stored bytes, or (stored bytes, "xz")}; {} for pack 4, whose blobs live in a store). Checks nothing: write() is the checked
    path, tests use this for packs write() refuses. O(total bytes) time, O(1) extra space."""
    hdr, off, parts = {"manifest": [0, len(raw)], "blobs": {}}, len(raw), [raw]
    for h, b in blobs.items():
        b, *codec = b if isinstance(b, tuple) else (b,)
        hdr["blobs"][h] = [off, len(b), *codec]
        off += len(b)
        parts.append(b)
    hb = json.dumps(hdr).encode()
    _atomic(path, [(MAGIC4 if major == 4 else MAGIC) + len(hb).to_bytes(8, "little") + hb, *parts])


def _atomic(path, chunks):
    """path holds chunks, whole or not at all (a temp file beside it, then rename)."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), prefix=".gcpack.")
    try:
        with os.fdopen(fd, "wb") as f:
            for c in chunks:
                f.write(c)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def write(path, manifest, blobs, store=None):
    """Make a pack: manifest (a dict) validated for its own abi major (3 or 4); blobs (bytes, named here by sha256) exactly the ones its
    buckets name; each stored xz-compressed when that is smaller, every raw size pinned (blob_raw). Pack 3 embeds the blobs; pack 4 puts
    them in the store directory as <sha256>[.xz]. Returns the manifest bytes (what sign_pack signs). O(total blob bytes) time."""
    by = {hashlib.sha256(b).hexdigest(): b for b in blobs}
    major = semver(manifest.get("abi"))[0]
    need(major in (3, 4), f"pack abi {manifest.get('abi')}: write makes pack 3 and 4")
    m = validate(dict(manifest), manifest["abi"])
    named = set(blob_hashes(m["buckets"]))
    need(named <= set(by), f"blobs missing: {sorted(named - set(by))}")
    need(set(by) <= named, f"blobs no bucket names: {sorted(set(by) - named)}")
    need(major == 3 or store, "pack 4 puts its blobs in a store: pass store")
    m = validate(dict(m, blob_raw={h: len(by[h]) for h in sorted(named)}), m["abi"])   # the reader checks xz and store blobs against these
    raw = (json.dumps(m, indent=1, sort_keys=True) + "\n").encode()
    stored = {}
    for h in sorted(named):
        z = lzma.compress(by[h])
        stored[h] = (z, "xz") if len(z) < len(by[h]) else by[h]
    if major == 3:
        frame(path, raw, stored)
        return raw
    os.makedirs(store, exist_ok=True)
    for h, b in stored.items():
        name = os.path.join(store, h + (".xz" if isinstance(b, tuple) else ""))
        if not os.path.exists(name):                   # content-addressed: the same name is the same bytes, hash-checked on read
            _atomic(name, [b[0] if isinstance(b, tuple) else b])
    frame(path, raw, {}, major=4)
    return raw


def _open_signed(path, sig, domain, trusted):
    """(open file, header blobs, base offset, total size, manifest bytes, signer key id, major) of a container whose manifest is signed over
    sha256(domain + manifest bytes) by the detached signature sig (default PATH.sig); domain None: the pack domain its magic fixes."""
    sp = sig or path + ".sig"
    sigb = read_capped(sp, sign.MAX_SIG) if os.path.lexists(sp) else None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as e:
        raise PackError(f"{path}: {e.strerror} (symlinks are refused)")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise PackError(f"{path}: not a regular file")
    f = os.fdopen(fd, "rb")
    try:
        total = os.fstat(fd).st_size
        v, blobs, base, raw, _ = read_container_v(f, total)
        try:
            kid = sign.verify(None, sigb, trusted, digest=hashlib.sha256((domain or (DOMAIN4 if v == 4 else DOMAIN)) + raw).hexdigest())
        except sign.SignatureError as e:
            raise PackError(str(e))
        return f, blobs, base, total, raw, kid, v
    except BaseException:
        f.close()
        raise


def sign_pack(path, priv_path, out=None):
    """Write out (default PATH.sig): the signature over sha256(domain + manifest bytes) of the pack at path (its magic fixes the domain);
    returns the signer's key id."""
    with open(path, "rb") as f:
        v, _, _, raw, _ = read_container_v(f, os.path.getsize(path))
    sig = sign.sign_digest(hashlib.sha256((DOMAIN4 if v == 4 else DOMAIN) + raw).hexdigest(), priv_path)
    with open(out or path + ".sig", "wb") as f:
        f.write(sig)
    return json.loads(sig)["keyid"]


def stage(pack, sig=None, loader_abi=ABI, trusted=None, arch=None, defer=False, overlays=(), store=None):
    """Verify a .gcpack: the signature (sig: path of the detached signature, default PACK.sig) over sha256(DOMAIN + manifest bytes), then
    the manifest. Blobs are extracted (hash-checked) all at once, or with defer=True only when Staged.ensure asks. overlays: overlay files
    (each signed, OVERLAY_DOMAIN, signature at FILE.sig), merged into the manifest (merge); pack may also be [base, overlay, ...]. The
    staged manifest.json is the base's bytes, or with overlays the merged manifest (manifest_sha256 stays the base's: the index entry's).
    store: the Store (or directory) of pack 4 blobs (open_store; default $GOETHE_BLOB_STORE); a blob embedded in a container is used first."""
    if isinstance(pack, (list, tuple)):
        pack, overlays = pack[0], list(pack[1:]) + list(overlays)
    st = Staged()
    try:
        st.f, st.blobs, st.base, st.total, raw, st.keyid, v = _open_signed(pack, sig, None, trusted)
        st.add(st.f, st.blobs, st.base, st.total)
        st.store = open_store(store)
        st.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        st.manifest = check_manifest(raw, ABI4 if v == 4 else loader_abi, None)     # the magic fixes the major
        ovs = []
        for op in overlays:
            f, blobs, base, total, oraw, _, ov = _open_signed(op, None, OVERLAY_DOMAIN, trusted)
            need(ov == v, f"{op}: a pack {ov} overlay of a pack {v} file (refused)")
            st.add(f, blobs, base, total)
            st.overlays.append(hashlib.sha256(oraw).hexdigest())
            try:
                ovs.append(json.loads(oraw))
            except ValueError as e:
                raise PackError(f"{op}: overlay manifest: {e}")
        if ovs:
            st.manifest = merge(st.manifest, raw, ovs)
            raw = (json.dumps(st.manifest, indent=1, sort_keys=True) + "\n").encode()
        if v == 4:
            pins = st.manifest.get("blob_raw", {})
            need(all(h in pins for h in blob_hashes(st.manifest["buckets"])), "pack 4: a blob's raw size is not pinned in the signed manifest")
        need(arch is None or arch in arches(st.manifest), f"pack arches {sorted(arches(st.manifest))} lack {arch}")
        with _create(os.path.join(st.dir, "manifest.json")) as o:
            o.write(raw)
        with _create(os.path.join(st.dir, "manifest.sha256")) as o:
            o.write((hashlib.sha256(raw).hexdigest() + "\n").encode())
        os.mkdir(os.path.join(st.dir, "blobs"), 0o700)
        if not defer:
            st.ensure(None)
        return st
    except BaseException as e:
        st.close()
        raise PackError(f"{pack}: {e}") if isinstance(e, PackError) else e


def verify(pack, sig=None, loader_abi=ABI, trusted=None, arch=None, store=None):
    """Stage everything, check it, discard the copy; returns the manifest."""
    st = stage(pack, sig, loader_abi, trusted, arch, store=store)
    st.close()
    return st.manifest


def arch_runs(arch, cc):
    """Why code of target arch cannot run on compute capability cc ("" if it can), by the CUDA compatibility rules (CUDA C
    Programming Guide, Compute Capabilities: binary compatibility, architecture- and family-specific features): sm_XYa SASS only
    on cc X.Y; sm_XYf and sm_XY SASS on cc X.Z, Z >= Y; compute_XY PTX (driver JIT) on any cc >= X.Y."""
    g = re.fullmatch(r"(sm|compute)_(\d+)(\d)([af]?)", arch or "")
    if not g:
        return f"bucket arch {arch!r} is not sm_XY[a|f] or compute_XY"
    want, have = (int(g[2]), int(g[3])), tuple(int(x) for x in cc.split("."))
    if g[1] == "compute":
        return "" if have >= want else f"PTX {arch} needs cc >= {want[0]}.{want[1]}"
    if g[4] == "a":
        return "" if have == want else f"{arch} SASS runs only on cc {want[0]}.{want[1]} (arch-specific targets are not forward compatible)"
    return "" if have[0] == want[0] and have[1] >= want[1] else f"{arch} SASS runs only on cc {want[0]}.Z with Z >= {want[1]}"


def arch_distance(arch, cc):
    """How far code for arch (one arch_runs accepts on cc) is from the device: 0 native SASS, the minor-version gap for older SASS of
    the family, 100 + the gap for PTX (driver JIT); 0 for a target that is not cuda's. O(1)."""
    g = re.fullmatch(r"(sm|compute)_(\d+)(\d)([af]?)", arch or "")
    if not g or not re.fullmatch(r"\d+\.\d+", cc or ""):
        return 0
    have = tuple(int(x) for x in cc.split("."))
    gap = (have[0] - int(g[2])) * 10 + have[1] - int(g[3])
    return gap if g[1] == "sm" else 100 + gap


def cuda_arch_ok(dev, req, m):
    """Why a cuda device cannot run a bucket that requires req in manifest m ("" if it can): the bucket's arch on the device's compute
    capability (arch_runs), the driver's CUDA version against the one the libraries were built with."""
    why = arch_runs(req["arch"], dev["cc"])
    if why:
        return f"device cc {dev['cc']}: {why}"
    if not (dev.get("cuda") and ver(dev["cuda"]) >= ver(m["cuda"])):
        return f"driver CUDA {dev.get('cuda')} < the libraries' CUDA {m['cuda']}"
    return ""


def fits(dev, req, m):
    """Why device dev cannot run a bucket that requires req ("" if it can): the platform's arch rule, then the plan's structure: its
    persistent grid of req.sms CTAs must be co-resident (device SMs >= req.sms) and its shared memory per block must fit."""
    why = dev["arch_ok"](dev, req, m)
    if why:
        return why
    if req["sms"] > dev["sms"]:
        return f"the plan's grid needs {req['sms']} SMs co-resident, the device has {dev['sms']}"
    if dev.get("smem_block") is not None and req.get("smem_block", 0) > dev["smem_block"]:
        return f"the plan needs {req['smem_block']} B shared memory per block, the device allows {dev['smem_block']}"
    return ""


def pick_sku(m, dev, sku=None):
    """The SKU (variant) of the pack to run on dev: among the SKUs whose requires fit dev (fits), the exact (name, SMs) one, else the
    nearest arch (arch_distance: native SASS, then older SASS of the family, then PTX), then the least dev["price"](sku, dev) (the pack's
    gc_probe / gc_price cost model, lib_price; an unpriced SKU ranks after the priced), then the most SMs and smem (fewest idle). No fit:
    refused with every SKU's reason and what to do instead."""
    req = {}
    for b in m["buckets"]:
        req.setdefault(b["sku"], b["requires"])
    why = {s: fits(dev, r, m) for s, r in req.items()}
    if sku is not None:
        need(sku in req, f"--sku {sku} is not in this pack ({sorted(req)})")
        need(not why[sku], f"--sku {sku} does not run on device {dev['name']}: {why[sku]}")
        return sku
    ok = [s for s in req if not why[s]]
    arch = "sm_" + dev.get("cc", "?").replace(".", "")
    need(ok, f"device {dev['name']} cc {dev.get('cc')} ({dev['sms']} SMs, smem per block {dev.get('smem_block')}) runs no variant "
         f"of this pack: " + "; ".join(f"{s} ({req[s]['arch']}, {req[s]['sms']} SMs): {why[s]}" for s in sorted(req)) +
         f". Fallback: serve this model with llama.cpp; or have a variant built for {arch} and pass it with --overlay")
    exact = sorted(s for s in ok if req[s]["name"] == dev["name"] and req[s]["sms"] == dev["sms"])
    if exact:
        return exact[0]
    price = dev.get("price")
    us = {s: price(s, dev) if price else None for s in ok}
    near = {s: arch_distance(req[s]["arch"], dev.get("cc")) for s in ok}
    return min(ok, key=lambda s: (near[s], us[s] is None, us[s] or 0, -req[s]["sms"], -req[s].get("smem_block", 0), s))


def device(idx=None):
    """The visible cuda GPU: {platform, name, cc, sms, smem_block (opt-in shared memory per block), uuid, cuda (driver's CUDA version),
    arch_ok} from nvidia-smi and the driver API (the cuda platform plugin's device query, registry.PLATFORMS)."""
    import ctypes as C
    vis = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    # nvidia-smi ignores CUDA_VISIBLE_DEVICES: it is asked for the visible device's own index or UUID
    i = vis[idx or 0] if vis else idx if idx is not None else os.environ.get("GC_GPU", "0")
    q = subprocess.run(["nvidia-smi", f"--id={i}", "--query-gpu=name,compute_cap,uuid", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True).stdout.strip().split(", ")
    hdr = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    cv = re.search(r"CUDA Version:\s*([\d.]+)", hdr)
    cu, d, v = C.CDLL("libcuda.so.1"), C.c_int(), C.c_int()
    need(cu.cuInit(0) == 0 and cu.cuDeviceGet(C.byref(d), (idx or 0) if vis else int(i)) == 0,
         f"driver API: no CUDA device {i}")
    attr = lambda a: v.value if cu.cuDeviceGetAttribute(C.byref(v), a, d) == 0 else None   # CUdevice_attribute values (cuda.h)
    return dict(platform="cuda", name=q[0], cc=q[1], uuid=q[2], sms=attr(16), smem_block=attr(97), cuda=cv[1] if cv else None,
                arch_ok=cuda_arch_ok)


def ver(s):
    return tuple(int(x) for x in s.split("."))


def select(m, *, sku=None, dev=None, contract=None, B=1, ctx=None, role="target", verify=False, want=None, kind=None):
    """The bucket for (sku or device, contract, B columns, ctx positions, role): the smallest ncol >= B, then the
    smallest maxctx >= ctx, among nominal (non-variant) buckets. Fail-closed on any device mismatch: dev is a platform plugin's
    device record (name, sms, smem_block, arch_ok(dev, requires, m) -> reason, optional price); its SKU is pick_sku's. kind "encode": only encode buckets; otherwise never one (an
    encode graph serves no decode, prefill or verify)."""
    if dev is not None:
        need(callable(dev.get("arch_ok")), f"device record {dev.get('name')!r} has no arch_ok: not a platform plugin's device (refused)")
        sku = pick_sku(m, dev, sku)
    need(sku is not None, "no SKU: pass a device or a SKU")
    ok = want or (lambda b: contract is not None or b.get("determinism") != "tolerance")   # tolerance buckets only when asked for
    # opaque labels: the pack lists its preference order; the first one with a bucket that satisfies the request (a tolerance
    # request: a contract with tolerance buckets first)
    pre = getattr(ok, "prefer", None)
    order = sorted(m["contracts"], key=lambda x: not any(b["contract"] == x and ok(b) and pre and b.get("determinism") == pre
                                                         for b in m["buckets"]))
    c = contract or next((x for x in order if any(b["contract"] == x and ok(b) for b in m["buckets"])), None) \
        or next((x for x in m["contracts"] if any(b["contract"] == x and ok(b) for b in m["buckets"])), None)
    skip = {b["id"]: usable(m, b) for b in m["buckets"] if usable(m, b)}
    cand = [b for b in m["buckets"] if b["sku"] == sku and b["contract"] == c and b["role"] == role
            and b["id"] not in skip and ok(b)
            and not b.get("variant_of") and b["ncol"] >= B and (ctx is None or b["maxctx"] >= ctx)
            and (axes(b)[1] == "encode") == (kind == "encode")
            and (kind == "encode" or verify or role != "target" or b["batch"] or b["ncol"] == 1)]   # a spec verify serves no batch
    need(cand, f"no {role} bucket for sku {sku}, contract {c}, B {B}, ctx {ctx} (skipped, unknown to this loader: "
         f"{skip or 'none'}; have "
         f"{sorted((b['sku'], b['contract'], b['ncol'], b['maxctx']) for b in m['buckets'] if b['role'] == role)})")
    top = max(rev(b) for b in cand)       # the newest revision of the contract present
    return min((b for b in cand if rev(b) == top), key=lambda b: (b["ncol"], b["maxctx"], b["id"]))


def rev(b):
    """A bucket's contract revision (manifest contract_rev; absent = 1, packs before revisions): rows of two revisions differ."""
    return b.get("contract_rev", 1)


def single_stream(m, *, sku=None, dev=None, contract=None, ctx=None, want=None):
    """The bucket a handle that serves one sequence runs: among the buckets `select` returns for B 1 (one per contract, or just
    `contract`), the one with the least predicted step (bucket field step_us, the pack builder's price); a bucket without a price is
    not ranked, and without any price the 1 column non-batch bucket if the pack has one, else None (the caller takes the batch
    ladder). Contract names stay opaque; a handle keeps the one contract of its bucket, so a bucket that is not batch invariant
    (batch_invariant false) is only ever chosen here, for a handle of one sequence."""
    cs = []
    for c in ([contract] if contract else m["contracts"]):
        try:
            cs.append(select(m, sku=sku, dev=dev, contract=c, B=1, ctx=ctx, want=want))
        except PackError:
            continue
    priced = [b for b in cs if "step_us" in b]
    if priced:
        return min(priced, key=lambda b: b["step_us"])
    return next((b for b in cs if b["ncol"] == 1 and not b["batch"]), None)


DETERMINISM = ("exact", "batch_invariant", "tolerance")   # buckets[].determinism


def feature(b, k):
    """A bucket's numerics axis k ("determinism", "act", "kv"): its record."""
    return b[k]


def wanted(numerics="invariant", act="auto", kv="auto", single=False):
    """The bucket predicate of a request: numerics exact | deterministic (batch_invariant; exact also serves a handle of one
    sequence, which has no batch to depend on) | fast (tolerance buckets preferred by select, else deterministic), act auto
    (a16 where a bucket has it: the pack's contract order) | a8 | a16 (exact buckets carry no act), kv auto (f16) | f16 | fp8."""
    numerics, act = NK.resolve(numerics, act)
    det = {"exact": {"exact"}, "deterministic": {"batch_invariant"} | ({"exact"} if single else set()),
           "fast": {"tolerance", "batch_invariant"} | ({"exact"} if single else set())}
    if numerics not in det or act not in ("auto", "a8", "a16") or kv not in ("auto", "f16", "fp8"):
        raise PackError(f"numerics {numerics!r} act {act!r} kv {kv!r}: numerics reference | invariant | invariant-hp | tolerance, act auto | a8 | "
                        "a16, kv auto | f16 | fp8")
    # an exact bucket that declares batch_invariant (one sequence per graph) also serves a deterministic request
    f = lambda b: ((feature(b, "determinism") in det[numerics] or (numerics == "deterministic" and feature(b, "determinism") == "exact"
                                                                   and b.get("batch_invariant") is True))
                   and (act == "auto" or feature(b, "act") in (None, act)) and feature(b, "kv") == ("f16" if kv == "auto" else kv))
    f.prefer = "tolerance" if numerics == "fast" else None
    return f


def need_bucket(m, sku, want, what):
    """PackError naming the missing capability unless some target bucket of the pack satisfies want."""
    have = sorted({(feature(b, "determinism"), feature(b, "act"), feature(b, "kv")) for b in m["buckets"] if b["role"] == "target"
                   and (sku is None or b["sku"] == sku)})
    need(any(want(b) for b in m["buckets"] if b["role"] == "target"),
         f"no bucket satisfies {what}; the pack has (determinism, act, kv): {have}")


def usable(m, b):
    """Why this loader cannot run bucket b ("" if it can): a runtime or role it does not know. Contract and body labels
    are opaque strings the pack declares: matched by exact string, never enumerated here."""
    rt = b.get("runtime") or m["runtime"]
    if rt.get("kind") not in RUNTIMES or rt.get("version") not in RUNTIMES[rt.get("kind")]:
        return f"runtime {rt.get('kind')} {rt.get('version')}"
    if b["role"] not in ROLES:
        return f"role {b['role']}"
    return ""


def spec_pairs(m, *, sku=None, dev=None, contract=None, ctx=None, want=None):
    """[(target, drafter, policy, name)] of the pack's speculative strategies this loader runs for the device or SKU, contract, ctx
    and numerics predicate, in preference order (tolerance first when want.prefer == 'tolerance', else manifest order); name = the
    entry's optional display label, else the policy kind."""
    ok = want or (lambda b: contract is not None or b.get("determinism") != "tolerance")
    if dev is not None:
        sku = select(m, dev=dev, contract=contract, verify=True, want=ok)["sku"]
    by = {b["id"]: b for b in m["buckets"]}
    pairs = [(by[s["target"]], by[s["drafter"]], s["policy"], s.get("name") or s["policy"].get("kind")) for s in m.get("spec", [])
             if s["policy"].get("kind") in POLICIES and not usable(m, by[s["target"]]) and not usable(m, by[s["drafter"]])
             and "encode" not in (axes(by[s["target"]])[1], axes(by[s["drafter"]])[1])
             and (rev(by[s["target"]]) == rev(by[s["drafter"]]) or by[s["target"]]["contract"] != by[s["drafter"]]["contract"])
             and (sku is None or by[s["target"]]["sku"] == sku) and (contract is None or by[s["target"]]["contract"] == contract)
             and (ctx is None or by[s["target"]]["maxctx"] >= ctx)
             and ok(by[s["target"]])]
    pre = getattr(ok, "prefer", None)
    if pre:
        pairs.sort(key=lambda p: p[0].get("determinism") != pre)
    return pairs


def spec_pair(m, *, sku=None, dev=None, contract=None, ctx=None, want=None):
    """(target, drafter, policy) of the pack's preferred speculative pair for the device or SKU, contract and ctx."""
    ps = spec_pairs(m, sku=sku, dev=dev, contract=contract, ctx=ctx, want=want)
    need(ps, f"no speculative pair for sku {sku}, contract {contract}, ctx {ctx}")
    t, d, pol, _ = ps[0]
    return t, d, pol


def ladder(m, b):
    """The nominal buckets a process serving up to b's columns loads beside b (same SKU, contract, role, maxctx; ncol
    <= b's, same contract revision), smallest first (a shrinking batch steps down through them). Only batch buckets join a ladder: a
    one-sequence stream bucket (kind stream, a linear program the runtime opens alone) is b's ladder only when it is b."""
    return sorted((x for x in m["buckets"] if not x.get("variant_of") and x["ncol"] <= b["ncol"]
                   and (x["batch"] or x["id"] == b["id"]) and axes(x)[1] != "encode"
                   and all(x[k] == b[k] for k in ("sku", "contract", "role", "maxctx")) and rev(x) == rev(b)),
                  key=lambda x: x["ncol"])


MIB = 1 << 20


def device_free(idx=None):
    """Free bytes of CUDA device idx (default: the current one), through the driver API; None without a driver."""
    import ctypes as C
    try:
        cu = C.CDLL("libcuda.so.1")
        ctx, free, tot = C.c_void_p(), C.c_size_t(), C.c_size_t()
        if cu.cuInit(0) or cu.cuDevicePrimaryCtxRetain(C.byref(ctx), max(idx or 0, 0)) or cu.cuCtxSetCurrent(ctx) or cu.cuMemGetInfo_v2(C.byref(free), C.byref(tot)):
            return None
        return free.value
    except OSError:
        return None


def overhead(m, sku):
    """Device bytes a handle needs beyond its declared parts on this SKU: the manifest's measured runtime.overhead, else 1 GiB."""
    return ((m.get("runtime") or {}).get("overhead") or {}).get(sku, 1024 * MIB)


def fit(lad, state_pages, free, pool_blocks=None, margin=1024 * MIB, what="pack"):
    """(pool_blocks or None, resident bytes) for opening the ladder lad on a device with free bytes, keeping margin bytes (overhead(m, sku)):
    resident = the union of the ladder's weight parts + fixed regions + slot bytes * (slots + state_pages) + block bytes * pool_blocks, from each bucket's manifest `resident`.
    pool_blocks None: the largest pool that fits (None when the pack's full pool does, or the pack has no paged KV). Refuses, with
    the numbers, a pack whose weights and smallest pool (one full context) do not fit, or an explicit pool_blocks that does not.
    free None or a bucket without `resident` (an older pack): no check."""
    rs = [b.get("resident") for b in lad]
    if free is None or not all(rs):
        return pool_blocks, None
    i = max((j for j, r in enumerate(rs) if r["block"]), default=len(rs) - 1)   # the pool's owner: the last paged bucket (a prefill bucket after it pages nothing)
    top = rs[i]
    # a handle relays each weight part once across its buckets: the union of the buckets' parts (resident.wparts), else per weight set
    weights = sum({k: v for r in rs for k, v in (r.get("wparts") or {r["wset"]: r["weights"]}).items()}.values())
    # scratch: a bucket's launch workspace when the manifest declares it (runtime.overhead covers it otherwise)
    base = weights + sum(r["fixed"] + r.get("scratch", 0) for r in rs) + top["slot"] * (top["slots"] + state_pages)
    blk, rows = top["block"], top["block_rows"]
    per_seq = lad[i]["maxctx"] // rows if rows else 0
    full = top["slots"] * per_seq
    budget = free - margin

    def gb(x):
        return f"{x / 1e9:.2f} GB"
    if blk == 0:
        need(base <= budget, f"{what}: needs {gb(base)} on the device (weights {gb(weights)}, {top['slots']} sequence slots), "
                             f"{gb(max(free, 0))} free with a {gb(margin)} margin kept; fewer columns or a smaller context")
        return pool_blocks, base
    least = base + blk * per_seq
    need(least <= budget, f"{what}: needs {gb(least)} on the device (weights {gb(weights)}, {top['slots']} sequence slots {gb(top['slot'] * (top['slots'] + state_pages))}, "
                          f"one full context of KV {gb(blk * per_seq)}), {gb(max(free, 0))} free with a {gb(margin)} margin kept; "
                          f"open a ladder with fewer columns or a smaller maxctx")
    if pool_blocks:
        need(base + blk * pool_blocks <= budget, f"{what}: pool_blocks {pool_blocks} needs {gb(base + blk * pool_blocks)}, {gb(max(free, 0))} free "
                                                  f"with a {gb(margin)} margin kept (at most {(budget - base) // blk} blocks fit)")
        return pool_blocks, base + blk * pool_blocks
    fits = int((budget - base) // blk)
    return (None, base + blk * full) if fits >= full else (fits, base + blk * fits)


def open_ctx(pick, m, ctx, fits):
    """(bucket, ladder, prefill bucket) of pick(ctx). ctx None: the context is sized at load, the largest maxctx of the pack whose
    pick fits the device (fits(buckets) raises PackError when not); a pack without resident records fits at its largest maxctx."""
    if ctx is not None:
        b, lad, pf = pick(ctx)
        fits(lad + ([pf] if pf else []))
        return b, lad, pf
    err = None
    for cx in sorted({b["maxctx"] for b in m["buckets"] if b["role"] == "target"}, reverse=True):
        try:
            b, lad, pf = pick(cx)
            fits(lad + ([pf] if pf else []))
            return b, lad, pf
        except PackError as e:
            err = e                 # the smallest context's reason is the one to report
    raise err or PackError("no target bucket")


def open_fit(pick, m, ctx, fits, widths):
    """open_ctx for the widest of widths (batch columns, descending) whose ladder fits the device: a batch the card cannot hold
    steps down to the next narrower ladder bucket instead of failing at launch; none fits: refused with every width's reason."""
    why = []
    for w in widths:
        try:
            return open_ctx(lambda cx: pick(cx, w), m, ctx, fits)
        except PackError as e:
            why.append(f"B {w}: {e}")
    raise OutOfMemory("; ".join(why) or "no target bucket")


def prefill_us(x, L, pos=0):
    """S2 price in us of prefilling L tokens from pos with prefill bucket x: per chunk of W columns the step_us_kv entry of the
    smallest extent >= the chunk's KV extent (pos + rows through it), else step_us; None when unpriced."""
    W, kv = x["prefill"], sorted(x.get("step_us_kv") or [])
    tot = 0.0
    for c in range(-(-L // W)):
        e = pos + min((c + 1) * W, L)
        us = next((u for r, u in kv if e <= r), x.get("step_us"))
        if us is None:
            return None
        tot += us
    return tot


def prefill(m, b, plen=None):
    """The prefill bucket (manifest "prefill": W columns, a non-batch build that prefills a prompt chunk of one sequence a launch
    beside b's ladder; form chain or seq) with b's SKU, contract, revision, role and maxctx, or None; None for a non-batch b (a stream
    bucket opens alone: the runtime takes a prefill bucket only beside batch buckets). Among several: the least total price over the
    chunks of a plen-token prompt (the length the handle is opened for: gc_open takes one prefill bucket); the widest when plen is None
    or any candidate is unpriced."""
    if not b["batch"]:
        return None
    cs = [x for x in m["buckets"] if x.get("prefill") and not x["batch"]
          and all(x[k] == b[k] for k in ("sku", "contract", "role", "maxctx")) and rev(x) == rev(b)]
    if plen is not None and cs:
        us = {x["id"]: prefill_us(x, min(plen, b["maxctx"])) for x in cs}
        if None not in us.values():
            return min(cs, key=lambda x: (us[x["id"]], -x["prefill"], x["id"]))
    return max(cs, key=lambda x: x["prefill"], default=None)


def with_prefill(m, b, *, sku=None, dev=None, contract=None, ctx=None, want=None):
    """B 1 prompts: a 1 column non-batch stream bucket b has no prompt chain, so when the pack has a batch ladder bucket (same request) with a
    prefill bucket, that bucket (its contract's numerics, token-per-launch decode through its 2 column ladder step); else b."""
    if b is None or P_has(m, b) or b["ncol"] != 1 or b["batch"]:
        return b
    try:
        b2 = select(m, sku=sku, dev=dev, contract=contract, B=2, ctx=ctx, want=want)
    except PackError:
        return b
    return b2 if prefill(m, b2) else b


def P_has(m, b):
    return prefill(m, b) is not None


def desc(v):
    """A variant's description for messages (manifest variants[id].desc {format, quant, registry_key, broken?}): 'format quant (key)'."""
    d = v.get("desc") or {}
    return " ".join(str(d[k]) for k in ("format", "quant") if d.get(k)) + (f" ({d['registry_key']})" if d.get("registry_key") else "")


def pick_variant(m, files, role="target", allow_broken=False):
    """The id of the variant of role whose structure the model files have (header-only reads). Refuses naming why each variant
    differs (with its desc) and where other weights go; a matched variant whose desc marks it broken is refused with that text
    unless allow_broken."""
    ids = sorted({b["variant"] for b in m["buckets"] if b["role"] == role})
    why = []
    for v in ids:
        try:
            structure.check(m["variants"][v]["structure"], files)
        except structure.StructureError as e:
            why.append(f"variant {v} [{desc(m['variants'][v]) or 'no desc'}]: {e}")
            continue
        broken = (m["variants"][v].get("desc") or {}).get("broken")
        need(not broken or allow_broken, f"variant {v} [{desc(m['variants'][v])}] is marked broken: {broken} (refused; allow_broken "
                                         "opens it for debugging)")
        return v
    raise PackError(f"{files}: not a model this pack was compiled for ({'; '.join(why) or 'no ' + role + ' buckets'}). Other weights "
                    "of this architecture: build a variant for them, then pass that pack "
                    "or overlay")


def plan(m, dev=None):
    """Per variant: {id, role, quant, source, files [{name, bytes}], max_ctx, runs, why}: the files a download of that variant fetches,
    its longest context, and whether dev (a device record; None: not asked) runs one of its SKUs. Manifest only. O(variants x buckets)."""
    roles, lim, out = {b["variant"]: b["role"] for b in m["buckets"]}, ctx_limit(m), []
    for vid in sorted(roles):
        v, bs = m["variants"][vid], [b for b in m["buckets"] if b["variant"] == vid]
        runs = why = None
        if dev is not None and roles[vid] == "target":
            try:
                pick_sku(dict(m, buckets=bs), dev)
                runs = True
            except PackError as e:
                runs, why = False, str(e)
        out.append(dict(id=vid, role=roles[vid], quant=(v.get("desc") or {}).get("quant"), source=v.get("source"),
                        files=[dict(name=m["model"]["weights"][i]["file"], bytes=m["model"]["weights"][i].get("bytes")) for i in v["weights"]],
                        max_ctx=min([max(b["maxctx"] for b in bs)] + ([lim[0]] if lim else [])), runs=runs, why=why))
    return out


def ctx_limit(m, ctx=None):
    """(maxctx, why) of the model's tightest limit (model.limits), or None; refuses a requested ctx beyond it with its why text."""
    lim = min(m["model"].get("limits") or [], key=lambda x: x["maxctx"], default=None)
    if lim and ctx is not None:
        need(ctx <= lim["maxctx"], f"context {ctx} > {lim['maxctx']}: {lim['why']} (refused)")
    return (lim["maxctx"], lim["why"]) if lim else None


def for_variant(m, target, drafter=None):
    """The manifest restricted to the buckets of the target variant (and, for a speculative pair, the drafter variant)."""
    bs = [b for b in m["buckets"] if b["variant"] == (target if b["role"] == "target" else drafter)]
    ids = {b["id"] for b in bs}
    return dict(m, buckets=bs, spec=[s for s in m.get("spec", []) if s.get("target") in ids and s.get("drafter") in ids],
                contracts=[c for c in m["contracts"] if any(b["contract"] == c for b in bs)])


def check_weights(m, target_files, drafter_files=None):
    """The files of each role (as m, restricted by for_variant, uses them) are regular, owned by this user or root, not world-writable, and
    have the structure of the variant their buckets were compiled for (tensor table and metadata keys; header reads only, no hashing). The
    number, order and sharding of a role's files do not matter: the runtime finds every tensor by name."""
    role = {b["variant"]: b["role"] for b in m["buckets"]}
    for vid in sorted(role):
        fs = target_files if role[vid] == "target" else drafter_files
        need(fs, f"variant {vid}: {role[vid]} weight files are needed")
        for p in fs:
            need(os.path.exists(p), f"weights {p} missing")
            st = os.stat(p)
            need(stat.S_ISREG(st.st_mode) and st.st_uid in (0, os.getuid()) and not st.st_mode & 0o002,
                 f"weights {p}: must be a regular file of this user or root and not world-writable")
        try:
            structure.check(m["variants"][vid]["structure"], fs)
        except structure.StructureError as e:
            raise PackError(f"weights {fs}: not the structure variant {vid} was compiled for: {e}")


def branch(m, model_files, *, sku=None, numerics="invariant", act="auto", kv="auto"):
    """The buckets of the branch a model's files choose: its target variant, for sku (the pack's only SKU if none), satisfying the
    numerics request. Returns (variant id, [buckets])."""
    v = pick_variant(m, model_files)
    mv = for_variant(m, v)
    skus = sorted({b["sku"] for b in mv["buckets"]})
    need(sku is None or sku in skus, f"sku {sku} is not in this pack ({skus})")
    sku = sku or (skus[0] if len(skus) == 1 else None)
    need(sku is not None, f"several SKUs in this pack {skus}: name one")
    want = wanted(numerics, act, kv)
    bs = [b for b in mv["buckets"] if b["sku"] == sku and want(b)]
    need(bs, f"no bucket of variant {v} for sku {sku} with numerics={numerics} act={act} kv={kv}")
    return v, bs


def drafter_branch(m, sets, taken, skus, *, numerics="invariant", act="auto", kv="auto"):
    """(drafter files, buckets) of the drafter variant a file set without the target's files (taken) has, for skus; ([], []) when the
    pack has no drafter or the folder holds none. O(sets) header reads."""
    if not any(b["role"] == "drafter" for b in m["buckets"]):
        return [], []
    want = wanted(numerics, act, kv)
    for fs in sets:
        if taken.isdisjoint(fs):
            try:
                v = pick_variant(m, fs, "drafter")
            except PackError:
                continue
            return fs, [b for b in m["buckets"] if b["role"] == "drafter" and b["variant"] == v and b["sku"] in skus and want(b)]
    return [], []


PRICE_CACHE = {}       # (device uuid, driver, manifest sha, library sha) -> cost constants blob (gc_probe)


def lib_price(st, m, ordinal=-1):
    """price(sku, dev) -> predicted step_us of the sku's one-sequence bucket from the pack's own cost model (gc.h gc_probe /
    gc_price in that bucket's library), or None when the library has no such symbols or declines (pick_sku then ranks by SMs).
    Libraries are read and loaded only for the SKUs pick_sku asks about; the probe runs once per device and library."""
    import ctypes as C

    def price(sku, dev):
        bs = [b for b in m["buckets"] if b["sku"] == sku and b["role"] == "target" and axes(b)[1] != "encode"]
        if not bs:
            return None
        b = min(bs, key=lambda x: (x["ncol"], x["maxctx"], x["id"]))
        st.ensure([b])
        path = os.path.join(materialize(st, b), b["lib"])
        try:
            L = C.CDLL(path)
            probe, pr = L.gc_probe, L.gc_price
        except (OSError, AttributeError):
            return None
        probe.restype = pr.restype = C.c_int
        probe.argtypes = [C.c_int, C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64)]
        pr.argtypes = [C.c_void_p, C.c_uint64, C.c_char_p, C.POINTER(C.c_double)]
        key = (dev.get("uuid") or dev.get("name"), dev.get("cuda"), st.manifest_sha256, b["files"].get(b["lib"]))
        if key not in PRICE_CACHE:
            n = C.c_uint64()
            rc = probe(max(ordinal, 0), None, 0, C.byref(n))
            if rc not in (0, -7):
                return None
            buf = C.create_string_buffer(max(n.value, 1))
            if probe(max(ordinal, 0), buf, n.value, C.byref(n)):
                return None
            PRICE_CACHE[key] = buf.raw[:n.value]
        blob, us = PRICE_CACHE[key], C.c_double()
        return us.value if pr(blob, len(blob), b["id"].encode(), C.byref(us)) == 0 else None
    return price


class Branch:
    """What every backend opens: the manifest restricted to the weights' variant(s) on this platform, the device record, the chosen
    bucket, the buckets whose blobs are read (ensured) and the materialized library path of the last of them."""

    def __init__(self, m, dev, bucket, load, lib):
        self.m, self.dev, self.bucket, self.load, self.lib = m, dev, bucket, load, lib


def open_branch(st, files, pick, *, drafter_files=None, sku=None, ordinal=-1, price=None, allow_broken=False):
    """The one open sequence of every model type: the weights pick their variant by structure (header reads), the manifest is
    restricted to it, pick(m, dev) -> (bucket, [buckets to load], lib bucket) chooses on the device (dev None when sku names the
    SKU), the weights are checked against the variant, only the loaded buckets' blobs are read, and the lib bucket's files are
    materialized. price(sku, dev) -> predicted cost: pick_sku's ranking among fitting SKUs (the probe-lite hook; None: SM count)."""
    m0 = st.manifest
    m = restrict(for_variant(m0, pick_variant(m0, files, allow_broken=allow_broken),
                             pick_variant(m0, drafter_files, "drafter", allow_broken) if drafter_files else None))
    dev = None if sku else dict(device(None if ordinal < 0 else ordinal), price=price or lib_price(st, m, ordinal))
    b, load, top = pick(m, dev)
    check_weights(m, files, drafter_files)
    st.ensure(load)
    return Branch(m, dev, b, load, os.path.join(materialize(st, top), top["lib"]))


def materialize(st, b, out=None):
    """Bucket b's files under their paths in out (default: inside the staged dir), hard-linked from the staged blobs."""
    out = out or os.path.join(st.dir, "run", b["id"])
    for rel, h in b["files"].items():
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), mode=0o700, exist_ok=True)
        if not os.path.exists(dst):
            os.link(os.path.join(st.dir, "blobs", h), dst)
    return out


def parser():
    import argparse
    ap = argparse.ArgumentParser(prog="goethe-engine pack", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__ + """
verify prints {"ok": true, "manifest_sha256", "keyid", "abi", "arch", "runtime": {"kind", "version"}, "buckets", "variants": [ids]}, and with
--partial "weights": the names of the model files of --model-dir the branch matched (the --weights of serve).
Without --partial every blob of every bucket is extracted and hash-checked; with --partial only the manifest signature and the blobs of the
branch --model-dir chooses (a variant by structure, --sku else every SKU, the numerics flags). --sig defaults to FILE.sig.
need reads a PREFIX file (the first bytes of a pack: 16 bytes, the header, the manifest) and prints {"manifest_sha256", "prefix": {off, len},
"ranges": [{off, len, sha256}]}: the byte ranges of exactly the blobs the branch needs; the manifest's signature is checked by `verify`
once the ranges are fetched, and the caller compares manifest_sha256 with the index entry's.
Exit 0 ok; 1 refused (signature missing or bad, key id not pinned, container malformed, a needed blob missing or not hashing to its name,
ABI major not 3, runtime not run here, --arch differs, no variant has the model's structure), reason on stderr.
runs (signature and manifest only: no blob, no weights, no model load) asks whether the visible device runs one of the pack's SKUs and
prints {"runs": true|false, "sku": the SKU it would run or null, "device": name, "sms": N, "why": the reason or null}: exit 0 runs, 3 not.
build reads a pack SPEC (JSON, data only; goethe_engine/build.py) and writes the pack to --out, signed with --key when given; it prints
{"manifest_sha256", "out", "variants", "buckets"}.
plan reads a PREFIX file and prints {"manifest_sha256", "variants": [{id, role, quant, source, files [{name, bytes}], max_ctx, runs, why}]}:
what each variant downloads and whether the visible device runs it (runs null without a visible device); no weights needed.
""")
    ap.add_argument("cmd", choices=("verify", "show", "need", "runs", "plan", "build"))
    ap.add_argument("file")
    ap.add_argument("--sig")
    ap.add_argument("--store", help="pack 4 blob store directory (default $GOETHE_BLOB_STORE)")
    ap.add_argument("--overlay", action="append", default=[], help="a signed overlay merged into the pack (repeatable)")
    ap.add_argument("--arch")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--model-dir")
    ap.add_argument("--sku")
    ap.add_argument("--numerics", default="invariant", choices=NK.CHOICES)
    ap.add_argument("--act", default="auto")
    ap.add_argument("--kv", default="auto")
    ap.add_argument("--out", help="build: the pack file to write")
    ap.add_argument("--key", help="build: the private key PEM to sign it with")
    return ap


def main(a):
    o = parser().parse_args(a)

    def files_of(d):
        from .index import model_sets
        sets = model_sets(d)
        need(sets, f"{d}: no model files (*.safetensors, *.gguf)")
        for fs in sets:
            bs = []
            for s in [o.sku] if o.sku else sorted({b["sku"] for b in m["buckets"]}):   # no --sku: every SKU's branch (no device here)
                try:
                    bs += branch(m, fs, sku=s, numerics=o.numerics, act=o.act, kv=o.kv)[1]
                except PackError as e:
                    last = e
            if bs:
                chosen[:] = [os.path.basename(f) for f in fs]
                dfs, dbs = drafter_branch(m, sets, set(fs), {b["sku"] for b in bs}, numerics=o.numerics, act=o.act, kv=o.kv)
                drafter[:] = [os.path.basename(f) for f in dfs]
                return bs + dbs
        raise last
    chosen, drafter = [], []
    try:
        if o.cmd == "build":
            from .build import build
            need(o.out, "build takes --out")
            raw = build(o.file, o.out, o.key, store=o.store)
            m = json.loads(raw)
            print(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), out=o.out, variants=sorted(m["variants"]), buckets=len(m["buckets"]))))
            return
        if o.cmd == "plan":
            with open(o.file, "rb") as f:
                v, _, _, raw, _ = read_container_v(f, os.fstat(f.fileno()).st_size + (1 << 40))   # a prefix: sections are not range-checked
            m = check_manifest(raw, ABI4 if v == 4 else ABI, o.arch)
            try:
                dev = device()
            except (OSError, subprocess.CalledProcessError, PackError):
                dev = None
            print(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), variants=plan(m, dev))))
            return
        if o.cmd == "need":
            with open(o.file, "rb") as f:
                v, blobs, base, raw, end = read_container_v(f, os.fstat(f.fileno()).st_size + (1 << 40))   # a prefix: sections are not range-checked
            m = check_manifest(raw, ABI4 if v == 4 else ABI, o.arch)
            need(o.model_dir, "need takes --model-dir")
            bs = files_of(o.model_dir)
            hs = blob_hashes(bs)
            need(v == 4 or all(h in blobs for h in hs), "a needed blob is missing from the container header")
            print(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), prefix=dict(off=0, len=end),
                                  ranges=[dict(off=base + blob_entry(blobs[h], h)[0], len=blob_entry(blobs[h], h)[1], sha256=h,
                                               **({"codec": "xz"} if blob_entry(blobs[h], h)[2] else {})) for h in hs if h in blobs],
                                  **({"store": [h for h in hs if h not in blobs]} if v == 4 else {}))))   # pack 4: fetch these by hash
            return
        st = stage(o.file, o.sig, arch=o.arch, defer=True, overlays=o.overlay, store=o.store)
        if o.cmd == "runs":
            m = st.manifest
            st.close()
            dev = device()
    except (PackError, OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"pack: REFUSED: {e}")
    if o.cmd == "runs":
        try:
            sku, why = pick_sku(m, dev, o.sku), None
        except PackError as e:
            sku, why = None, str(e)
        print(json.dumps(dict(runs=sku is not None, sku=sku, device=dev["name"], sms=dev["sms"], why=why)))
        sys.exit(0 if sku else 3)
    try:
        m = st.manifest
        if o.cmd == "verify":
            if o.partial:
                need(o.model_dir, "--partial takes --model-dir")
                st.ensure(files_of(o.model_dir))
            else:
                st.ensure(None)
            print(json.dumps(dict(ok=True, manifest_sha256=st.manifest_sha256, keyid=st.keyid, abi=m["abi"], arch=m["arch"], runtime=m["runtime"],
                                  **({"overlays": st.overlays, "arches": sorted(arches(m))} if st.overlays else {}),
                                  buckets=len(m["buckets"]), variants=sorted(m["variants"]), **({"weights": chosen} if chosen else {}),
                                  **({"drafter": drafter} if drafter else {}))))
        else:
            for b in m["buckets"]:
                print(f"{b['id']:40s} {b['variant']:10s} {b['contract']:4s} ncol {b['ncol']:3d} maxctx {b['maxctx']:6d} {len(b['files'])} files")
    except PackError as e:
        sys.exit(f"pack: REFUSED: {e}")
    finally:
        st.close()


if __name__ == "__main__":
    main(sys.argv[1:])
