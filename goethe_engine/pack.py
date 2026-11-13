#!/usr/bin/env python3
"""Kernel pack reader (pack ABI 3). A pack is ONE file, NAME.gcpack, for one model and GPU architecture. Layout (stdlib framing):

  b"GCPACK3\n"  u64le N  header (N bytes of JSON {"manifest": [off, len], "blobs": {sha256: [off, len] or [off, len, "xz"]}})  payload
  offsets count from the end of the header; manifest = manifest.json, blobs = every bucket file, named by the sha256 of its raw bytes;
  len = stored bytes, "xz" = the blob is one xz stream (the only codec; read with a 64 MiB decoder memory limit and a raw size cap)

The signature (gc-sig-1, detached, PACK.sig, or the index entry's) is ECDSA over sha256(b"gc-pack-3\n" + manifest bytes): the manifest
names every blob by its sha256, and a blob is hash-checked when it is extracted, so a file with holes is as safe as a whole one for
the blobs actually used. Weights are not in the pack: a manifest "variant" names the structure of the model files its buckets were
compiled for (structure.py: tensor table and metadata keys, no weight bytes); the loader checks the files it is given against it.

Manifest: model.weights (informational: the files the gates ran on), variants {id: {structure, weights: [model.weights indices]}},
buckets[] each with variant, role, sku, contract, ncol, maxctx, files, lib, requires, numerics axes, step_us.

  python -m goethe_engine.pack verify FILE [--sig F] [--arch sm_XX] [--partial [--model-dir D] [--sku S]]   check, discard
  python -m goethe_engine.pack show FILE [--sig F]                   one line per bucket
  python -m goethe_engine.pack need PREFIX --model-dir D [--arch A --sku S --numerics ..]  the byte ranges a download must fetch
"""
import atexit, hashlib, json, lzma, os, re, shutil, stat, subprocess, sys, tempfile

from . import sign, structure

ABI = "3.0.0"                       # pack ABI semver: additive within a major
MAGIC = b"GCPACK3\n"
DOMAIN = b"gc-pack-3\n"             # signed digest = sha256(DOMAIN + manifest bytes)
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


class PackError(Exception):
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
    for k in ("compiler", "model", "arch", "cuda", "hw", "runtime", "contracts", "buckets"):
        need(k in m, f"manifest lacks {k}")
    ws = m["model"].get("weights")
    need(isinstance(ws, list) and ws and all(isinstance(w, dict) and isinstance(w.get("file"), str) for w in ws),
         "model.weights: the list of weight files the gates ran on")
    vs = m.get("variants")
    need(isinstance(vs, dict) and vs, "manifest lacks variants")
    for vid, v in vs.items():
        st = v.get("structure") if isinstance(v, dict) else None
        need(isinstance(st, dict) and HEX.match(st.get("sha256", "")) and isinstance(st.get("keys"), list)
             and isinstance(st.get("kv"), dict) and isinstance(st.get("tensors"), list), f"variant {vid}: structure needs sha256, keys, kv and tensors")
        need(isinstance(v.get("weights"), list) and v["weights"] and all(isinstance(i, int) and 0 <= i < len(ws) for i in v["weights"]),
             f"variant {vid}: weights indices")
    need(all(HEX.match(h or "") for h in m["hw"].values()), "hw record hashes must be sha256")
    raw = m.get("blob_raw", {})
    need(isinstance(raw, dict) and all(HEX.match(h) and isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= MAX_RAW
                                       for h, n in raw.items()), "manifest blob_raw: {sha256: raw bytes <= 4 GiB}")
    ids = set()
    for b in m["buckets"]:
        for k in ("id", "role", "sku", "contract", "ncol", "maxctx", "files", "lib", "requires"):
            need(k in b, f"bucket {b.get('id')} lacks {k}")
        need(b["id"] not in ids, f"bucket id {b['id']} twice")
        ids.add(b["id"])
        need(b["sku"] in m["hw"], f"bucket {b['id']}: sku {b['sku']} has no hw record hash")
        need(b["lib"] in b["files"] and b.get("plan", b["lib"]) in b["files"], f"bucket {b['id']}: lib / plan not in its files")
        need(b["requires"]["arch"] == m["arch"], f"bucket {b['id']}: arch {b['requires']['arch']} != {m['arch']}")
        need(isinstance(b.get("bodies", {}), dict), f"bucket {b['id']}: bodies is not a map")
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
    need(arch is None or m["arch"] == arch, f"pack arch {m['arch']} != {arch}")
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
        atexit.register(self.close)

    def close(self):
        if self.f:
            self.f.close()
            self.f = None
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None

    def ensure(self, buckets):
        """Extract the blobs of buckets (default: all) into dir/blobs; refuses a missing, truncated or altered blob."""
        for bh in blob_hashes(self.manifest["buckets"] if buckets is None else buckets):
            dst = os.path.join(self.dir, "blobs", bh)
            if os.path.exists(dst):
                continue
            need(bh in self.blobs, f"blob {bh} missing from the pack")
            off, ln, codec = blob_entry(self.blobs[bh], bh)
            _section(off, ln, self.total - self.base, f"blob {bh}")
            self.f.seek(self.base + off)
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
                        chunk = self.f.read(min(left, 4 << 20))
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


def read_container(f, total):
    """(header blobs, base offset, manifest bytes, offset just past the manifest) of an open pack file (or of a prefix holding its header and
    manifest)."""
    need(f.read(len(MAGIC)) == MAGIC, "not a gcpack3 file (bad magic)")
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
    return blobs, base, raw, base + mo + ml


def stage(pack, sig=None, loader_abi=ABI, trusted=None, arch=None, defer=False):
    """Verify a .gcpack: the signature (sig: path of the detached signature, default PACK.sig) over sha256(DOMAIN + manifest bytes), then
    the manifest. Blobs are extracted (hash-checked) all at once, or with defer=True only when Staged.ensure asks."""
    st = Staged()
    try:
        sp = sig or pack + ".sig"
        sigb = read_capped(sp, sign.MAX_SIG) if os.path.lexists(sp) else None
        try:
            fd = os.open(pack, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as e:
            raise PackError(f"{pack}: {e.strerror} (symlinks are refused)")
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise PackError(f"{pack}: not a regular file")
        st.f = os.fdopen(fd, "rb")
        st.total = os.fstat(fd).st_size
        st.blobs, st.base, raw, _ = read_container(st.f, st.total)
        st.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            st.keyid = sign.verify(None, sigb, trusted, digest=hashlib.sha256(DOMAIN + raw).hexdigest())
        except sign.SignatureError as e:
            raise PackError(str(e))
        st.manifest = check_manifest(raw, loader_abi, arch)
        with _create(os.path.join(st.dir, "manifest.json")) as o:
            o.write(raw)
        with _create(os.path.join(st.dir, "manifest.sha256")) as o:
            o.write((st.manifest_sha256 + "\n").encode())
        os.mkdir(os.path.join(st.dir, "blobs"), 0o700)
        if not defer:
            st.ensure(None)
        return st
    except BaseException as e:
        st.close()
        raise PackError(f"{pack}: {e}") if isinstance(e, PackError) else e


def verify(pack, sig=None, loader_abi=ABI, trusted=None, arch=None):
    """Stage everything, check it, discard the copy; returns the manifest."""
    st = stage(pack, sig, loader_abi, trusted, arch)
    st.close()
    return st.manifest


def device(idx=None):
    """The visible GPU: {name, cc, sms, uuid, cuda (driver's CUDA version)} from nvidia-smi and the CUDA runtime."""
    i = idx if idx is not None else os.environ.get("GC_GPU", "0")
    q = subprocess.run(["nvidia-smi", f"--id={i}", "--query-gpu=name,compute_cap,uuid", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True).stdout.strip().split(", ")
    hdr = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    cv = re.search(r"CUDA Version:\s*([\d.]+)", hdr)
    import torch                                    # the SM count: no nvidia-smi field has it
    p = torch.cuda.get_device_properties(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else int(i))
    return dict(name=q[0], cc=q[1], uuid=q[2], sms=p.multi_processor_count, cuda=cv[1] if cv else None)


def ver(s):
    return tuple(int(x) for x in s.split("."))


def select(m, *, sku=None, dev=None, contract=None, B=1, ctx=None, role="target", verify=False, want=None):
    """The bucket for (sku or device, contract, B columns, ctx positions, role): the smallest ncol >= B, then the
    smallest maxctx >= ctx, among nominal (non-variant) buckets. Fail-closed on any device mismatch."""
    if dev is not None:
        hit = [b for b in m["buckets"] if b["requires"]["name"] == dev["name"] and b["requires"]["sms"] == dev["sms"]]
        need(hit, f"device {dev['name']} ({dev['sms']} SMs) matches no bucket's SKU in this pack "
             f"({sorted({b['sku'] for b in m['buckets']})})")
        s = {b["sku"] for b in hit}
        need(sku is None or sku in s, f"--sku {sku} is not the device's SKU {sorted(s)}")
        sku = sku or s.pop()
        need(dev["cc"] == hit[0]["requires"]["cc"], f"device cc {dev['cc']} != pack cc {hit[0]['requires']['cc']}")
        need(m["arch"].startswith("sm_" + dev["cc"].replace(".", "")), f"device cc {dev['cc']} vs pack arch {m['arch']}")
        need(dev.get("cuda") and ver(dev["cuda"]) >= ver(m["cuda"]),
             f"driver CUDA {dev.get('cuda')} < the libraries' CUDA {m['cuda']}")
    need(sku is not None, "no SKU: pass a device or a SKU")
    ok = want or (lambda b: True)
    # opaque labels: the pack lists its preference order; the first one with a bucket that satisfies the request
    c = contract or next((x for x in m["contracts"] if any(b["contract"] == x and ok(b) for b in m["buckets"])), None) \
        or next(iter(m["contracts"]), None)
    skip = {b["id"]: usable(m, b) for b in m["buckets"] if usable(m, b)}
    cand = [b for b in m["buckets"] if b["sku"] == sku and b["contract"] == c and b["role"] == role
            and b["id"] not in skip and ok(b)
            and not b.get("variant_of") and b["ncol"] >= B and (ctx is None or b["maxctx"] >= ctx)
            and (verify or role != "target" or b["batch"] or b["ncol"] == 1)]   # a spec verify serves no batch
    need(cand, f"no {role} bucket for sku {sku}, contract {c}, B {B}, ctx {ctx} (skipped, unknown to this loader: "
         f"{skip or 'none'}; have "
         f"{sorted((b['sku'], b['contract'], b['ncol'], b['maxctx']) for b in m['buckets'] if b['role'] == role)})")
    return min(cand, key=lambda b: (b["ncol"], b["maxctx"], b["id"]))


def single_stream(m, *, sku=None, dev=None, contract=None, ctx=None, want=None):
    """The bucket a handle that serves one sequence runs: among the buckets `select` returns for B 1 (one per contract, or just
    `contract`), the one with the least predicted step (bucket field step_us, the compiler's S2 price); a bucket without a price is
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


def wanted(numerics="deterministic", act="auto", kv="auto", single=False):
    """The bucket predicate of a request: numerics exact | deterministic (batch_invariant; exact also serves a handle of one
    sequence, which has no batch to depend on) | fast (tolerance, else deterministic: no tolerance bucket exists yet), act auto
    (a16 where a bucket has it: the pack's contract order) | a8 | a16 (exact buckets carry no act), kv auto (f16) | f16 | fp8."""
    det = {"exact": {"exact"}, "deterministic": {"batch_invariant"} | ({"exact"} if single else set()),
           "fast": {"tolerance", "batch_invariant"} | ({"exact"} if single else set())}
    if numerics not in det or act not in ("auto", "a8", "a16") or kv not in ("auto", "f16", "fp8"):
        raise PackError(f"numerics {numerics!r} act {act!r} kv {kv!r}: numerics exact | deterministic | fast, act auto | a8 | "
                        "a16, kv auto | f16 | fp8")
    return lambda b: (feature(b, "determinism") in det[numerics] and (act == "auto" or feature(b, "act") in (None, act))
                      and feature(b, "kv") == ("f16" if kv == "auto" else kv))


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


def spec_pair(m, *, sku=None, dev=None, contract=None, ctx=None):
    """(target, drafter, policy) of the pack's speculative pair for the device or SKU and contract."""
    if dev is not None:
        sku = select(m, dev=dev, contract=contract, verify=True)["sku"]
    by = {b["id"]: b for b in m["buckets"]}
    ps = [(by[s["target"]], by[s["drafter"]], s["policy"]) for s in m.get("spec", [])
          if s["policy"].get("kind") in POLICIES and not usable(m, by[s["target"]]) and not usable(m, by[s["drafter"]])
          and (sku is None or by[s["target"]]["sku"] == sku) and (contract is None or by[s["target"]]["contract"]
                                                                 == contract)
          and (ctx is None or by[s["target"]]["maxctx"] >= ctx)]
    need(ps, f"no speculative pair for sku {sku}, contract {contract}, ctx {ctx}")
    return min(ps, key=lambda x: (x[0]["maxctx"], x[0]["id"]))


def ladder(m, b):
    """The nominal buckets a process serving up to b's columns loads beside b (same SKU, contract, role, maxctx; ncol
    <= b's), smallest first (a shrinking batch steps down through them)."""
    return sorted((x for x in m["buckets"] if not x.get("variant_of") and x["ncol"] <= b["ncol"]
                   and (x["batch"] or x["ncol"] == 1)
                   and all(x[k] == b[k] for k in ("sku", "contract", "role", "maxctx"))), key=lambda x: x["ncol"])


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


def fit(lad, state_pages, free, pool_blocks=None, margin=1024 * MIB, what="pack"):
    """(pool_blocks or None, resident bytes) for opening the ladder lad on a device with free bytes: resident = the ladder's distinct weight
    sets + fixed regions + slot bytes * (slots + state_pages) + block bytes * pool_blocks, from each bucket's manifest `resident`.
    pool_blocks None: the largest pool that fits (None when the pack's full pool does, or the pack has no paged KV). Refuses, with
    the numbers, a pack whose weights and smallest pool (one full context) do not fit, or an explicit pool_blocks that does not.
    free None or a bucket without `resident` (an older pack): no check."""
    rs = [b.get("resident") for b in lad]
    if free is None or not all(rs):
        return pool_blocks, None
    top = rs[-1]
    weights = sum({r["wset"]: r["weights"] for r in rs}.values())
    base = weights + sum(r["fixed"] for r in rs) + top["slot"] * (top["slots"] + state_pages)
    blk, rows = top["block"], top["block_rows"]
    per_seq = lad[-1]["maxctx"] // rows if rows else 0
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


def prefill(m, b):
    """The widest prefill chain bucket (manifest "prefill": W columns, a non-batch build that prefills a prompt chunk of one
    sequence a launch beside b's ladder) with b's SKU, contract, role and maxctx, or None."""
    return max((x for x in m["buckets"] if x.get("prefill") and not x["batch"]
                and all(x[k] == b[k] for k in ("sku", "contract", "role", "maxctx"))), key=lambda x: x["prefill"], default=None)


def pick_variant(m, files, role="target"):
    """The id of the variant of role whose structure the model files have (header-only reads); refuses naming why each variant differs."""
    ids = sorted({b["variant"] for b in m["buckets"] if b["role"] == role})
    why = []
    for v in ids:
        try:
            structure.check(m["variants"][v]["structure"], files)
            return v
        except structure.StructureError as e:
            why.append(f"variant {v}: {e}")
    raise PackError(f"{files}: not a model this pack was compiled for ({'; '.join(why) or 'no ' + role + ' buckets'})")


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


def branch(m, model_files, *, sku=None, numerics="deterministic", act="auto", kv="auto"):
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


def materialize(st, b, out=None):
    """Bucket b's files under their paths in out (default: inside the staged dir), hard-linked from the staged blobs."""
    out = out or os.path.join(st.dir, "run", b["id"])
    for rel, h in b["files"].items():
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), mode=0o700, exist_ok=True)
        if not os.path.exists(dst):
            os.link(os.path.join(st.dir, "blobs", h), dst)
    return out


def main(a):
    import argparse
    ap = argparse.ArgumentParser(prog="goethe-engine pack", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__ + """
verify prints {"ok": true, "manifest_sha256", "keyid", "abi", "arch", "runtime": {"kind", "version"}, "buckets", "variants": [ids]}, and with
--partial "weights": the names of the model files of --model-dir the branch matched (the --weights of serve).
Without --partial every blob of every bucket is extracted and hash-checked; with --partial only the manifest signature and the blobs of the
branch --model-dir chooses (a variant by structure, --sku, the numerics flags). --sig defaults to FILE.sig.
need reads a PREFIX file (the first bytes of a pack: 16 bytes, the header, the manifest) and prints {"manifest_sha256", "prefix": {off, len},
"ranges": [{off, len, sha256}]}: the byte ranges of exactly the blobs the branch needs; the manifest's signature is checked by `verify`
once the ranges are fetched, and the caller compares manifest_sha256 with the index entry's.
Exit 0 ok; 1 refused (signature missing or bad, key id not pinned, container malformed, a needed blob missing or not hashing to its name,
ABI major not 3, runtime not run here, --arch differs, no variant has the model's structure), reason on stderr.
""")
    ap.add_argument("cmd", choices=("verify", "show", "need"))
    ap.add_argument("file")
    ap.add_argument("--sig")
    ap.add_argument("--arch")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--model-dir")
    ap.add_argument("--sku")
    ap.add_argument("--numerics", default="deterministic")
    ap.add_argument("--act", default="auto")
    ap.add_argument("--kv", default="auto")
    o = ap.parse_args(a)

    def files_of(d):
        from .index import model_sets
        sets = model_sets(d)
        need(sets, f"{d}: no model files (*.safetensors, *.gguf)")
        for fs in sets:
            try:
                bs = branch(m, fs, sku=o.sku, numerics=o.numerics, act=o.act, kv=o.kv)[1]
                chosen[:] = [os.path.basename(f) for f in fs]
                return bs
            except PackError as e:
                last = e
        raise last
    chosen = []
    try:
        if o.cmd == "need":
            with open(o.file, "rb") as f:
                blobs, base, raw, end = read_container(f, os.fstat(f.fileno()).st_size + (1 << 40))   # a prefix: sections are not range-checked
            m = check_manifest(raw, ABI, o.arch)
            need(o.model_dir, "need takes --model-dir")
            bs = files_of(o.model_dir)
            hs = blob_hashes(bs)
            need(all(h in blobs for h in hs), "a needed blob is missing from the container header")
            print(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), prefix=dict(off=0, len=end),
                                  ranges=[dict(off=base + blob_entry(blobs[h], h)[0], len=blob_entry(blobs[h], h)[1], sha256=h,
                                               **({"codec": "xz"} if blob_entry(blobs[h], h)[2] else {})) for h in hs])))
            return
        st = stage(o.file, o.sig, arch=o.arch, defer=True)
    except PackError as e:
        sys.exit(f"pack: REFUSED: {e}")
    try:
        m = st.manifest
        if o.cmd == "verify":
            if o.partial:
                need(o.model_dir, "--partial takes --model-dir")
                st.ensure(files_of(o.model_dir))
            else:
                st.ensure(None)
            print(json.dumps(dict(ok=True, manifest_sha256=st.manifest_sha256, keyid=st.keyid, abi=m["abi"], arch=m["arch"], runtime=m["runtime"],
                                  buckets=len(m["buckets"]), variants=sorted(m["variants"]), **({"weights": chosen} if chosen else {}))))
        else:
            for b in m["buckets"]:
                print(f"{b['id']:40s} {b['variant']:10s} {b['contract']:4s} ncol {b['ncol']:3d} maxctx {b['maxctx']:6d} {len(b['files'])} files")
    except PackError as e:
        sys.exit(f"pack: REFUSED: {e}")
    finally:
        st.close()


if __name__ == "__main__":
    main(sys.argv[1:])
