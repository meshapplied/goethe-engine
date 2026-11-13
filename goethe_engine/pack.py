#!/usr/bin/env python3
"""Kernel pack reader. A pack is ONE file, NAME.gcpack, for one model and GPU architecture, with a detached signature
NAME.gcpack.sig (gc-sig-1, over the exact file bytes). Layout (stdlib framing):

  b"GCPACK2\n"  u64le N  header (N bytes of JSON {"manifest": [off, len], "blobs": {sha256: [off, len]}})  payload
  offsets count from the end of the header; manifest = manifest.json, blobs = every bucket file, named by its sha256
Weights are not in the pack: the manifest names them by sha256 and the loader checks the file it is given
(a checkpoint's later shards name its first file in "of" and are checked with it).

stage() copies the file into a private 0700 directory while hashing it, verifies the signature on that copy, extracts the
manifest and blobs there (each blob re-hashed), checks ABI major, arch and runtime, and drops the copy; everything loaded
afterwards comes from the extracted directory. Any failure raises PackError.

  python -m goethe_engine.pack verify FILE [--sig F] [--arch sm_XX]  stage and verify everything, then discard
  python -m goethe_engine.pack show FILE [--sig F]                   one line per bucket
"""
import atexit, hashlib, json, os, re, shutil, stat, subprocess, sys, tempfile

from . import sign, structure

ABI = "2.0.0"                       # pack ABI semver: additive within a major
MAGIC = b"GCPACK2\n"
MAX_HEADER = 64 << 20               # bytes of container header JSON
SCHEMA = "gc-pack"
RUNTIMES = {"gc-c": (1,)}           # runtime kinds this loader runs, with their versions
MAX_JSON = 16 << 20                 # bytes: larger manifests are refused before parsing
ROLES = ("target", "drafter")
POLICIES = {"ar": ("K",), "walk": ("gate", "succ", "pred", "feats"),   # speculative round kinds and their fields
            "mtp": ("K", "feats", "tap", "own", "w_in", "w_own")}
HEX = re.compile(r"^[0-9a-f]{64}$")


class PackError(Exception):
    """Any mismatch between a pack, its blobs, the weights and the device: the pack is not loaded."""


def memo_dir():
    """Per-user cache of weight file hashes, keyed by path, inode, size, mtime and ctime."""
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "goethe-engine")


def sha256_file(p, memo=None):
    p = os.path.realpath(p)
    st = os.stat(p)
    sid = f"{p}\t{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}"
    mf = memo and os.path.join(memo, "sha256.tsv")
    if mf and os.path.exists(mf):
        for ln in open(mf):
            if ln.startswith(sid + "\t"):
                return ln.rstrip("\n").rsplit("\t", 1)[1]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    if mf:
        os.makedirs(memo, mode=0o700, exist_ok=True)
        with open(mf, "a") as f:
            f.write(f"{sid}\t{h.hexdigest()}\n")
    return h.hexdigest()


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
    need(m["model"].get("weights") and all(HEX.match(w.get("sha256", "")) for w in m["model"]["weights"]),
         "model weights must be named by sha256")
    need(all(w.get("of") is None or (isinstance(w["of"], int) and 0 <= w["of"] < j) for j, w in enumerate(m["model"]["weights"])),
         'a weights entry "of" must name an earlier entry')
    st = m.get("structure")
    need(st is None or (isinstance(st, dict) and HEX.match(st.get("sha256", "")) and isinstance(st.get("keys"), list)
                        and isinstance(st.get("kv"), dict) and isinstance(st.get("tensors"), list)),
         "structure needs sha256, keys, kv and tensors")
    need(all(HEX.match(h or "") for h in m["hw"].values()), "hw record hashes must be sha256")
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


class Staged:
    """A verified private extraction of a pack: dir (mode 0700, ours alone; manifest.json, manifest.sha256, blobs/),
    manifest, sha256 of the pack file and the signer's key id. close() removes it."""

    def __init__(self):
        self.dir, self.manifest, self.sha256, self.keyid = tempfile.mkdtemp(prefix="goethe-engine-"), None, None, None
        atexit.register(self.close)

    def close(self):
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None


def _create(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb")


def _section(f, off, ln, total, what):
    need(isinstance(off, int) and isinstance(ln, int) and 0 <= off and 0 <= ln and off + ln <= total, f"{what}: section out of range")


def stage(pack, sig=None, loader_abi=ABI, trusted=None, arch=None):
    """Verify a .gcpack and extract it into a private dir. sig: path of the detached signature (default PACK.sig).
    The file is copied once (hashed on the way), the signature is checked on the copy, and what is loaded later comes
    from Staged.dir, so nothing verified can change underneath."""
    st = Staged()
    try:
        sp = sig or pack + ".sig"
        sigb = read_capped(sp, sign.MAX_SIG) if os.path.lexists(sp) else None
        try:
            fd = os.open(pack, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as e:
            raise PackError(f"{pack}: {e.strerror} (symlinks are refused)")
        cp, h = os.path.join(st.dir, "pack.gcpack"), hashlib.sha256()
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise PackError(f"{pack}: not a regular file")
        with os.fdopen(fd, "rb") as f, _create(cp) as o:
            for chunk in iter(lambda: f.read(4 << 20), b""):
                h.update(chunk)
                o.write(chunk)
        st.sha256 = h.hexdigest()
        try:
            st.keyid = sign.verify(cp, sigb, trusted)
        except sign.SignatureError as e:
            raise PackError(str(e))
        total = os.path.getsize(cp)
        with open(cp, "rb") as f:
            need(f.read(len(MAGIC)) == MAGIC, "not a gcpack file (bad magic)")
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
            _section(f, mo, ml, total - base, "manifest")
            need(ml <= MAX_JSON, "manifest.json larger than the cap (refused)")
            f.seek(base + mo)
            raw = f.read(ml)
            m = st.manifest = check_manifest(raw, loader_abi, arch)
            with _create(os.path.join(st.dir, "manifest.json")) as o:
                o.write(raw)
            with _create(os.path.join(st.dir, "manifest.sha256")) as o:
                o.write((hashlib.sha256(raw).hexdigest() + "\n").encode())
            os.mkdir(os.path.join(st.dir, "blobs"), 0o700)
            for bh in sorted({x for b in m["buckets"] for x in b["files"].values()}):
                need(bh in blobs, f"blob {bh} missing from the pack")
                off, ln = blobs[bh]
                _section(f, off, ln, total - base, f"blob {bh}")
                f.seek(base + off)
                hh, left = hashlib.sha256(), ln
                with _create(os.path.join(st.dir, "blobs", bh)) as o:
                    while left:
                        chunk = f.read(min(left, 4 << 20))
                        need(chunk, f"blob {bh}: pack truncated")
                        hh.update(chunk)
                        o.write(chunk)
                        left -= len(chunk)
                need(hh.hexdigest() == bh, f"blob {bh} does not hash to its name")
        os.unlink(cp)
        return st
    except BaseException as e:
        st.close()
        raise PackError(f"{pack}: {e}") if isinstance(e, PackError) else e


def verify(pack, sig=None, loader_abi=ABI, trusted=None, arch=None):
    """Stage everything, check it, discard the copy; returns the Staged record's manifest, file sha256 and key id."""
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


DETERMINISM = ("exact", "batch_invariant", "tolerance")   # buckets[].determinism (pack ABI 2.2)


def feature(b, k):
    """A bucket's numerics axis k ("determinism", "act", "kv"): its record, or for packs before ABI 2.2 what they imply
    (batch_invariant true or absent: batch_invariant, else exact; act and kv unknown, kv f16)."""
    if k in b:
        return b[k]
    return {"determinism": "batch_invariant" if b.get("batch_invariant", True) else "exact", "act": None, "kv": "f16"}[k]


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


def prefill(m, b):
    """The widest prefill chain bucket (manifest "prefill": W columns, a non-batch build that prefills a prompt chunk of one
    sequence a launch beside b's ladder) with b's SKU, contract, role and maxctx, or None."""
    return max((x for x in m["buckets"] if x.get("prefill") and not x["batch"]
                and all(x[k] == b[k] for k in ("sku", "contract", "role", "maxctx"))), key=lambda x: x["prefill"], default=None)


def weight_span(m, i):
    """Manifest weights indices of the file set that starts at index i: the entry itself and the later shards that name it in "of"."""
    ws = m["model"]["weights"]
    return [i] + [j for j, w in enumerate(ws) if j != i and w.get("of") == i]


def check_weights(m, weights, memo=None, used=None):
    """The weights files given (in model.weights order) hash to the manifest's sha256 values; used: the indices the
    loaded buckets read (default all), each of which must be given. A file must belong to this user or root and be
    not world-writable (the runtime reads it again after this check)."""
    ws = m["model"]["weights"]
    used = sorted({j for i in (range(len(ws)) if used is None else used) for j in weight_span(m, i)})
    need(len(weights) > max(used), f"weights {[ws[i]['file'] for i in used]} needed, {len(weights)} files given")
    # a pack that names its model's structure (the first file set): those weights hashes are what the gates ran on; a drafter's stay exact
    tset = set(weight_span(m, 0)) if m.get("structure") is not None else set()
    for i in used:
        w, p = ws[i], weights[i]
        need(os.path.exists(p), f"weights {p} missing")
        st = os.stat(p)
        need(stat.S_ISREG(st.st_mode) and st.st_uid in (0, os.getuid()) and not st.st_mode & 0o002,
             f"weights {p}: must be a regular file of this user or root and not world-writable")
        if i not in tset:
            need(st.st_size == w["bytes"], f"weights {p}: {st.st_size} bytes, manifest {w['bytes']}")
            need(sha256_file(p, memo) == w["sha256"], f"weights {p}: sha256 differs from the manifest's {w['file']}")
    if tset:     # header reads only: tensor names, shapes and types and the metadata the pack was compiled with
        fs = [weights[i] for i in used if i in tset]
        try:
            structure.check(m["structure"], fs)
        except structure.StructureError as e:
            raise PackError(f"weights {fs}: not the structure this pack was compiled for: {e}")


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
verify prints {"ok": true, "sha256", "keyid", "abi", "arch", "runtime": {"kind", "version"}, "buckets", "weights": [file names in
manifest order: the --weights argument of serve]}. --sig defaults to FILE.sig. Exit 0 ok; 1 refused (signature missing or bad,
key id not pinned, container malformed, a blob not hashing to its name, ABI major not 2, runtime not run here, --arch differs),
reason on stderr.
""")
    ap.add_argument("cmd", choices=("verify", "show"))
    ap.add_argument("file")
    ap.add_argument("--sig")
    ap.add_argument("--arch")
    o = ap.parse_args(a)
    try:
        st = stage(o.file, o.sig, arch=o.arch)
    except PackError as e:
        sys.exit(f"pack: REFUSED: {e}")
    try:
        m = st.manifest
        if o.cmd == "verify":
            print(json.dumps(dict(ok=True, sha256=st.sha256, keyid=st.keyid, abi=m["abi"], arch=m["arch"], runtime=m["runtime"],
                                  buckets=len(m["buckets"]), weights=[w["file"] for w in m["model"]["weights"]])))
        else:
            for b in m["buckets"]:
                print(f"{b['id']:40s} {b['contract']:4s} ncol {b['ncol']:3d} maxctx {b['maxctx']:6d} {len(b['files'])} files")
    finally:
        st.close()


if __name__ == "__main__":
    main(sys.argv[1:])
