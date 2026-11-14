"""Plugins: the platform backends a pack runs on, the model types the engine serves, and the device mesh a backend opens on.

platform  a device family and its runtime: cuda (gc-c, abi/gc.h) now; rocm, vulkan, metal are registered stubs that refuse
model type  how a model is served: decoder (Engine: prefill, decode, speculative rounds) now; encoder-decoder (speech, whisper-style
          and nemotron speech) and embedding (pooling head, no decode loop) are registered stubs that refuse
mesh      devices and parallel axes (dp tp pp ep sp); the API takes a mesh everywhere, one device is the only size served now

A new platform, model type or parallel axis is one entry here plus its module; nothing else names them."""
import math
from dataclasses import dataclass, field

from . import pack as P


def _gc_c(pack, weights, *, device, **kw):
    from .gc_c import GcBackend
    return GcBackend(pack, weights, device=device, **kw)


def _decoder(backend, cfg, **kw):
    from .engine import Engine
    return Engine(backend, cfg, **kw)


PLATFORMS = {"cuda": _gc_c, "rocm": None, "vulkan": None, "metal": None}     # platform -> backend factory (None: planned)
MODEL_TYPES = {"decoder": _decoder, "encoder-decoder": None, "embedding": None}  # model.type -> runner factory (None: planned)


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


def open_backend(pack, weights, *, mesh=None, platform=P.PLATFORM, **kw):
    """The backend of pack on mesh (default one device) for platform: the one entry serve and callers use."""
    mesh = mesh or Mesh.single()
    if platform not in PLATFORMS:
        raise ValueError(f"platform {platform!r}: one of {sorted(PLATFORMS)}")
    if PLATFORMS[platform] is None:
        raise _planned("platform", platform, [k for k, v in PLATFORMS.items() if v])
    if mesh.size > 1:
        raise NotImplementedError(f"mesh {mesh.axes} over {mesh.size} devices: multi-device parallelism ({' '.join(P.MESH_AXES)}) "
                                  "is not implemented yet; open on Mesh.single(device)")
    return PLATFORMS[platform](pack, weights, device=mesh.devices[0], **kw)


def model_type(m):
    """The model type of a manifest (model.type, default decoder)."""
    t = ((m or {}).get("model") or {}).get("type", P.MODEL_TYPE)
    if t not in MODEL_TYPES:
        raise P.PackError(f"model type {t!r}: this engine knows {sorted(MODEL_TYPES)}")
    return t


def runner(backend, cfg, **kw):
    """The serving loop of the backend's model type (a backend without a manifest, the reference model, is a decoder)."""
    m = getattr(backend, "m", None)          # GcBackend: its manifest; the reference backend: a model object
    t = model_type(m if isinstance(m, dict) else None)
    if MODEL_TYPES[t] is None:
        raise _planned("model type", t, [k for k, v in MODEL_TYPES.items() if v])
    return MODEL_TYPES[t](backend, cfg, **kw)
