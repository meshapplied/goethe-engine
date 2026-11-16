"""Plugins: the platform backends a pack runs on, the model types the engine serves, and the device mesh a backend opens on.

platform  a device family and its runtime (Platform: backend factory, device query with its arch rule, free memory, availability,
          the most devices a mesh may span): cuda (gc-c, abi/gc.h) now; rocm, vulkan, metal are registered stubs that refuse
model type  how a model is served: decoder (Engine: prefill, decode, speculative rounds), embedding (EmbedRunner: one pooled encoder
          pass per sequence, no decode loop), encoder-decoder (Seq2SeqRunner: the encoder once per sequence into a cross KV pool, then
          the Engine loop; whisper-style ASR, nemotron speech), media (MediaRunner: prompt chunks, media items encoded into rows by the
          pack's media library, then the decode)
mesh      devices and parallel axes (dp tp pp ep sp); the API takes a mesh everywhere, one device is the only size served now

A new platform, model type or parallel axis is one entry here plus its module; nothing else names them."""
import ctypes
import json
import math
import os
from dataclasses import dataclass, field

from . import pack as P


def _gc_c(pack, weights, *, mesh, model_type="decoder", **kw):
    if model_type == "embedding":
        from .embed import GcEmbed
        return GcEmbed(pack, weights, device=mesh.devices[0], **kw)
    if model_type == "media":
        from .media import GcMedia
        return GcMedia(pack, weights, device=mesh.devices[0], **kw)
    if model_type == "encoder-decoder":
        from .seq2seq import GcSeq2Seq
        return GcSeq2Seq(pack, weights, device=mesh.devices[0], **kw)
    from .gc_c import GcBackend
    return GcBackend(pack, weights, device=mesh.devices[0], **kw)


def _cuda_available():
    try:
        cu, n = ctypes.CDLL("libcuda.so.1"), ctypes.c_int()
        return cu.cuInit(0) == 0 and cu.cuDeviceGetCount(ctypes.byref(n)) == 0 and n.value > 0
    except OSError:
        return False


@dataclass(frozen=True)
class Platform:
    backend: object         # (pack, weights, *, mesh, model_type, **kw) -> a backend of that model type
    device: object          # ordinal (None: the default) -> device record {platform, name, sms, ..., arch_ok(dev, requires, m) -> reason}
    free: object            # ordinal -> free device bytes, or None
    available: object       # () -> True when a device of this platform is visible
    max_devices: int = 1    # the most devices one mesh may span on this platform


def _decoder(backend, cfg, **kw):
    from .engine import Engine
    return Engine(backend, cfg, **kw)


def _embedding(backend, cfg, **kw):
    from .embed import EmbedRunner
    return EmbedRunner(backend, cfg)


def _seq2seq(backend, cfg, **kw):
    from .seq2seq import Seq2SeqRunner
    return Seq2SeqRunner(backend, cfg, **kw)


def _media(backend, cfg, **kw):
    from .media import MediaRunner
    return MediaRunner(backend, cfg)


PLATFORMS = {"cuda": Platform(_gc_c, P.device, P.device_free, _cuda_available), "rocm": None, "vulkan": None, "metal": None}  # None: planned
MODEL_TYPES = {"decoder": _decoder, "encoder-decoder": _seq2seq, "embedding": _embedding, "media": _media}  # model.type -> runner factory (None: planned)


@dataclass(frozen=True)
class Mesh:
    """Devices (ordinals; -1 = the current device) and the parallel axes {dp|tp|pp|ep|sp: size} laid over them (product = devices)."""
    devices: tuple = (-1,)
    axes: dict = field(default_factory=dict)

    def __post_init__(self):
        P.mesh_key(self.axes)
        if math.prod(self.axes.values()) != len(self.devices):
            raise ValueError(f"mesh axes {self.axes} span {math.prod(self.axes.values())} devices, {len(self.devices)} given")

    @classmethod
    def single(cls, device=-1):
        return cls((device,), {})

    @property
    def size(self):
        return len(self.devices)

    @property
    def key(self):
        """The manifest form (bucket "mesh"): {axis: size} of the axes above 1."""
        return dict(P.mesh_key(self.axes))


def _planned(kind, name, have):
    return NotImplementedError(f"{kind} {name}: not implemented yet (goethe-engine serves {kind} {', '.join(have)}); "
                               f"registered so a plugin can be added in goethe_engine/registry.py")


def pack_type(pack):
    """model.type of a pack file's manifest (unverified: it only picks the backend class, which verifies the pack and refuses
    what is not one; an unreadable file reads as the default type so that refusal is the backend's)."""
    m = _manifest(pack)
    return model_type(m) if isinstance(m, dict) else P.MODEL_TYPE


def _manifest(pack):
    path = pack[0] if isinstance(pack, (list, tuple)) else pack
    try:
        with open(path, "rb") as f:
            return json.loads(P.read_container(f, os.path.getsize(path))[2])
    except (OSError, ValueError, P.PackError):
        return None


def platform_of(pack):
    """The platform to open pack on: the one registered platform its buckets name; among several, the one whose devices are visible;
    several visible (or an unreadable pack with several visible platforms) fail closed: pass platform= explicitly."""
    m = _manifest(pack)
    named = sorted({P.axes(b)[0] for b in m["buckets"]}) if m and isinstance(m.get("buckets"), list) else []
    have = [p for p in named if PLATFORMS.get(p)]
    if len(have) == 1:
        return have[0]
    seen = [p for p in (have or [k for k, v in PLATFORMS.items() if v]) if PLATFORMS[p].available()]
    if len(seen) == 1:
        return seen[0]
    raise P.PackError(f"platform: the pack names {named or 'none readable'}, visible plugins {seen or 'none'}; pass platform= explicitly")


def open_backend(pack, weights, *, mesh=None, platform=None, model_type=None, **kw):
    """The backend of pack on mesh (default one device) for platform (default: platform_of the pack) and model type (default: the
    manifest's): the one entry serve and callers use."""
    mesh = mesh or Mesh.single()
    platform = platform or platform_of(pack)
    if platform not in PLATFORMS:
        raise ValueError(f"platform {platform!r}: one of {sorted(PLATFORMS)}")
    if PLATFORMS[platform] is None:
        raise _planned("platform", platform, [k for k, v in PLATFORMS.items() if v])
    if mesh.size > PLATFORMS[platform].max_devices:
        raise NotImplementedError(f"mesh {mesh.axes} over {mesh.size} devices: multi-device parallelism ({' '.join(P.MESH_AXES)}) "
                                  f"is not implemented on {platform} yet; open on Mesh.single(device)")
    t = model_type or pack_type(pack)
    if MODEL_TYPES.get(t) is None:
        raise _planned("model type", t, [k for k, v in MODEL_TYPES.items() if v])
    return PLATFORMS[platform].backend(pack, weights, mesh=mesh, model_type=t, **kw)


def model_type(m):
    """The model type of a manifest (model.type, default decoder)."""
    t = ((m or {}).get("model") or {}).get("type", P.MODEL_TYPE)
    if t not in MODEL_TYPES:
        raise P.PackError(f"model type {t!r}: this engine knows {sorted(MODEL_TYPES)}")
    return t


def runner(backend, cfg, **kw):
    """The serving loop of the backend's model type (a backend without a manifest, the reference model, is a decoder)."""
    m = getattr(backend, "m", None)          # GcBackend: its manifest; the reference backend: a model object
    t = "embedding" if hasattr(backend, "embed") else "media" if hasattr(backend, "encode_audio") else \
        "encoder-decoder" if hasattr(backend, "encode") else \
        model_type(m if isinstance(m, dict) else None)
    if MODEL_TYPES[t] is None:
        raise _planned("model type", t, [k for k, v in MODEL_TYPES.items() if v])
    return MODEL_TYPES[t](backend, cfg, **kw)
