"""The platform and mesh seam: a fake platform (no device code) registered beside cuda is chosen from the pack, selects by its own
arch rule and by mesh axes, and no cuda query (nvidia-smi, libcuda, torch.cuda) runs on its path."""
import ctypes
import subprocess

import pytest
import torch

from goethe_engine import pack as P, registry as R
from toypack import toy_pack


def fake_arch_ok(dev, req, m):
    return "" if req["arch"].startswith("fk_") else f"fake device vs bucket arch {req['arch']}"


FAKE_DEV = dict(platform="fake", name="toy", sms=170, cc="1.0", arch_ok=fake_arch_ok)


class FakeBackend:
    """Opens the pack's buckets of platform fake on the mesh's axes and selects by the fake device record: the seam, no runtime."""

    def __init__(self, pack, weights, *, mesh, model_type, trusted=None, **_):
        st = P.stage(pack, trusted=trusted, defer=True)
        m = P.restrict(st.manifest, platform="fake", mesh=mesh.key)
        self.bucket = P.select(m, dev=R.PLATFORMS["fake"].device(mesh.devices[0]), B=2)
        self.mesh, self.model_type = mesh, model_type
        st.close()


@pytest.fixture
def fake(monkeypatch):
    def no_cuda(*a, **k):
        raise AssertionError(f"cuda query on the fake platform's path: {a[:1]}")
    real_run, real_cdll = subprocess.run, ctypes.CDLL
    monkeypatch.setattr(subprocess, "run", lambda a, *r, **k: no_cuda(a) if "nvidia-smi" in str(a) else real_run(a, *r, **k))
    monkeypatch.setattr(ctypes, "CDLL", lambda n, *r, **k: no_cuda(n) if "libcuda" in str(n) else real_cdll(n, *r, **k))
    monkeypatch.setattr(torch.cuda, "get_device_properties", no_cuda)
    monkeypatch.setitem(R.PLATFORMS, "fake", R.Platform(FakeBackend, lambda i: dict(FAKE_DEV), lambda i: 1 << 40, lambda: True,
                                                        max_devices=2))


def fake_pack(tmp_path, key, lib, both=False):
    """Toy pack whose buckets are platform fake, arch fk_1: ncol 2 and 8 on one device and on a tp 2 mesh (and the cuda ones if both)."""
    def mut(m):
        cuda = [dict(b) for b in m["buckets"]]
        fk = []
        for b in cuda:
            for mesh in ({}, {"tp": 2}):
                fk.append(dict(b, id=b["id"] + ".fake" + "".join(f".{k}{v}" for k, v in mesh.items()), platform="fake",
                               requires=dict(b["requires"], arch="fk_1", cc="1.0"), **({"mesh": mesh} if mesh else {})))
        m["buckets"] = fk + (cuda if both else [])
        m["arches"] = {"fk_1": dict(platform="fake", cc="1.0")}
    return toy_pack(str(tmp_path), key, lib, mutate=mut)


def test_fake_platform_is_chosen_from_the_pack_and_selects_by_mesh(fake, tmp_path, key, mocklib):
    pack, w, _ = fake_pack(tmp_path, key, mocklib)
    assert R.platform_of(pack) == "fake"
    one = R.open_backend(pack, w, trusted=(key.kid,))
    assert isinstance(one, FakeBackend) and one.bucket["id"].endswith(".fake") and one.model_type == "decoder"
    tp = R.open_backend(pack, w, mesh=R.Mesh((0, 1), {"tp": 2}), trusted=(key.kid,))
    assert tp.bucket["mesh"] == {"tp": 2} and tp.bucket["platform"] == "fake"
    with pytest.raises(NotImplementedError, match="not implemented on fake"):
        R.open_backend(pack, w, mesh=R.Mesh((0, 1, 2, 3), {"dp": 2, "tp": 2}), trusted=(key.kid,))
    with pytest.raises(P.PackError, match="no bucket for platform fake, mesh {'pp': 2}"):
        R.open_backend(pack, w, mesh=R.Mesh((0, 1), {"pp": 2}), trusted=(key.kid,))


def test_two_visible_platforms_fail_closed(fake, tmp_path, key, mocklib, monkeypatch):
    pack, w, _ = fake_pack(tmp_path, key, mocklib, both=True)
    monkeypatch.setitem(R.PLATFORMS, "cuda", R.Platform(R.PLATFORMS["cuda"].backend, P.device, P.device_free, lambda: True))
    with pytest.raises(P.PackError, match="pass platform= explicitly"):
        R.platform_of(pack)
    assert R.open_backend(pack, w, platform="fake", trusted=(key.kid,)).bucket["platform"] == "fake"
    monkeypatch.setitem(R.PLATFORMS, "cuda", R.Platform(R.PLATFORMS["cuda"].backend, P.device, P.device_free, lambda: False))
    assert R.platform_of(pack) == "fake"


def test_selector_takes_the_platform_arch_rule(fake, tmp_path, key, mocklib):
    pack, w, _ = fake_pack(tmp_path, key, mocklib, both=True)
    m = P.stage(pack, trusted=(key.kid,), defer=True).manifest
    with pytest.raises(P.PackError, match="has no arch_ok"):
        P.select(P.restrict(m, platform="fake"), dev=dict(FAKE_DEV, arch_ok=None))
    with pytest.raises(P.PackError, match="fake device vs bucket arch sm_120a"):
        P.select(P.restrict(m), dev=dict(FAKE_DEV, cc="12.0"))
    cuda_dev = dict(platform="cuda", name="toy", sms=170, cc="12.0", cuda="12.8", arch_ok=P.cuda_arch_ok)
    assert P.select(P.restrict(m), dev=cuda_dev, B=2)["requires"]["arch"] == "sm_120a"
    with pytest.raises(P.PackError, match="driver CUDA 12.0"):
        P.select(P.restrict(m), dev=dict(cuda_dev, cuda="12.0"), B=2)
