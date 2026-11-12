#!/usr/bin/env python3
"""Kernel pack reader. A pack is a directory for one model and GPU architecture:

  manifest.json   the manifest;  manifest.sha256  its sha256 (hex, one line)
  manifest.sig    detached signature over the exact bytes of manifest.json (see sign.py)
  blobs/<sha256>  every file of every bucket, named by the sha256 of its bytes
Weights are not in the pack: the manifest names them by sha256 and the loader checks the file it is given.

stage() copies the pack into a private 0700 directory and verifies the signature, the hashes, the ABI major, the arch
and the runtime on those copies; everything that is loaded afterwards comes from there. Any failure raises PackError.

  python -m goethe_engine.pack verify DIR [--arch sm_XX]   stage and verify everything, then discard
  python -m goethe_engine.pack show DIR                    one line per bucket
  python -m goethe_engine.pack blobs DIR                   blob hashes the signed manifest names that DIR lacks
"""
import atexit, hashlib, json, os, re, shutil, stat, subprocess, sys, tempfile

from . import sign

ABI = "1.1.0"                       # pack ABI semver: additive within a major
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


def read_manifest_files(d):
    """(manifest.json bytes, signature bytes or None, manifest.sha256 text) of a pack dir."""
    need(os.path.exists(os.path.join(d, "manifest.json")) and os.path.exists(os.path.join(d, "manifest.sha256")),
         f"{d}: no manifest.json / manifest.sha256")
    sp = os.path.join(d, "manifest.sig")
    return (read_capped(os.path.join(d, "manifest.json")), read_capped(sp, sign.MAX_SIG) if os.path.lexists(sp) else None,
            read_capped(os.path.join(d, "manifest.sha256"), 1024).decode("ascii", "replace").strip())


def check_manifest(raw, sig, digest, loader_abi=ABI, trusted=None, arch=None):
    """The manifest dict of these bytes: signature by a pinned key, sha256, schema, ABI major, runtime, arch."""
    try:
        sign.verify(raw, sig, trusted)
    except sign.SignatureError as e:
        raise PackError(str(e))
    need(hashlib.sha256(raw).hexdigest() == digest, "manifest.json does not hash to manifest.sha256")
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


def manifest(pack, loader_abi=ABI, trusted=None, arch=None):
    """The checked manifest of a pack dir, without staging it (for listing; load through stage)."""
    try:
        return check_manifest(*read_manifest_files(pack), loader_abi, trusted, arch)
    except PackError as e:
        raise PackError(f"{pack}: {e}")


class Staged:
    """A verified private copy of a pack: dir (mode 0700, ours alone) and manifest. close() removes it."""

    def __init__(self):
        self.dir, self.manifest = tempfile.mkdtemp(prefix="goethe-engine-"), None
        atexit.register(self.close)

    def close(self):
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None


def _put(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def stage(pack, loader_abi=ABI, trusted=None, arch=None):
    """Copy the pack into a private dir, hashing each blob as it is written, and verify the manifest from the copies.
    What is loaded later comes from Staged.dir, so nothing verified can change underneath."""
    st = Staged()
    try:
        raw, sig, digest = read_manifest_files(pack)
        _put(os.path.join(st.dir, "manifest.json"), raw)
        _put(os.path.join(st.dir, "manifest.sha256"), (digest + "\n").encode())
        if sig is not None:
            _put(os.path.join(st.dir, "manifest.sig"), sig)
        st.manifest = m = check_manifest(*read_manifest_files(st.dir), loader_abi, trusted, arch)
        need(not os.path.islink(os.path.join(pack, "blobs")), "blobs is a symlink (refused)")
        os.mkdir(os.path.join(st.dir, "blobs"), 0o700)
        for h in sorted({h for b in m["buckets"] for h in b["files"].values()}):
            src = os.path.join(pack, "blobs", h)
            try:
                fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError as e:
                raise PackError(f"blob {h}: {e.strerror}")
            with os.fdopen(fd, "rb") as f:
                need(stat.S_ISREG(os.fstat(fd).st_mode), f"blob {h}: not a regular file")
                dst, hh = os.open(os.path.join(st.dir, "blobs", h), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), hashlib.sha256()
                with os.fdopen(dst, "wb") as o:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        hh.update(chunk)
                        o.write(chunk)
            need(hh.hexdigest() == h, f"blob {h} does not hash to its name")
        return st
    except BaseException as e:
        st.close()
        raise PackError(f"{pack}: {e}") if isinstance(e, PackError) else e


def verify(pack, loader_abi=ABI, trusted=None, arch=None):
    """Stage everything, check it, discard the copy; returns the manifest."""
    st = stage(pack, loader_abi, trusted, arch)
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


def select(m, *, sku=None, dev=None, contract=None, B=1, ctx=None, role="target", verify=False):
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
    c = contract or next(iter(m["contracts"]), None)   # opaque labels: the pack lists its preference order
    skip = {b["id"]: usable(m, b) for b in m["buckets"] if usable(m, b)}
    cand = [b for b in m["buckets"] if b["sku"] == sku and b["contract"] == c and b["role"] == role
            and b["id"] not in skip
            and not b.get("variant_of") and b["ncol"] >= B and (ctx is None or b["maxctx"] >= ctx)
            and (verify or role != "target" or b["batch"] or b["ncol"] == 1)]   # a spec verify serves no batch
    need(cand, f"no {role} bucket for sku {sku}, contract {c}, B {B}, ctx {ctx} (skipped, unknown to this loader: "
         f"{skip or 'none'}; have "
         f"{sorted((b['sku'], b['contract'], b['ncol'], b['maxctx']) for b in m['buckets'] if b['role'] == role)})")
    return min(cand, key=lambda b: (b["ncol"], b["maxctx"], b["id"]))


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


def check_weights(m, weights, memo=None, used=None):
    """The weights files given (in model.weights order) hash to the manifest's sha256 values; used: the indices the
    loaded buckets read (default all), each of which must be given. A file must belong to this user or root and be
    not world-writable (the runtime reads it again after this check)."""
    ws = m["model"]["weights"]
    used = sorted(set(range(len(ws)) if used is None else used))
    need(len(weights) > max(used), f"weights {[ws[i]['file'] for i in used]} needed, {len(weights)} files given")
    for i in used:
        w, p = ws[i], weights[i]
        need(os.path.exists(p), f"weights {p} missing")
        st = os.stat(p)
        need(stat.S_ISREG(st.st_mode) and st.st_uid in (0, os.getuid()) and not st.st_mode & 0o002,
             f"weights {p}: must be a regular file of this user or root and not world-writable")
        need(st.st_size == w["bytes"], f"weights {p}: {st.st_size} bytes, manifest {w['bytes']}")
        need(sha256_file(p, memo) == w["sha256"], f"weights {p}: sha256 differs from the manifest's {w['file']}")


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
    if len(a) < 2 or a[0] not in ("verify", "show", "blobs"):
        sys.exit(__doc__)
    try:
        if a[0] == "verify":
            m = verify(a[1], arch=a[a.index("--arch") + 1] if "--arch" in a else None)
            print(f"pack {a[1]}: abi {m['abi']}, {len(m['buckets'])} buckets, every blob verified")
        elif a[0] == "blobs":
            m = manifest(a[1])
            for h in sorted({h for b in m["buckets"] for h in b["files"].values()}):
                if not os.path.exists(os.path.join(a[1], "blobs", h)):
                    print(h)
        else:
            for b in manifest(a[1])["buckets"]:
                print(f"{b['id']:40s} {b['contract']:4s} ncol {b['ncol']:3d} maxctx {b['maxctx']:6d} {len(b['files'])} files")
    except PackError as e:
        sys.exit(f"pack: REFUSED: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
