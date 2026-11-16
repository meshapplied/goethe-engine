"""One step of each task kind on an opened api.Model (torch free: a fresh interpreter runs it against pinned bodies)."""
import numpy as np


def run(m, kind):
    if kind == "generate":
        be = m.backend
        from goethe_engine.backend import Work
        bl = list(range(-(-40 // be.geometry.block_size))) if be.paged else [0]
        r = np.asarray(be.step([Work(1, [5, 7, 11, 13], 0, bl, 1)]))
        be.release(1)
        return r
    if kind == "embed":
        return m.embed([[5, 7, 11, 13], [3]], normalize=False)
    be = m.backend
    n = max(400, min((be.rate or 0) // 10, 4 * (be.bucket["maxctx"] - 16)))   # 100 ms at the bucket rate inside the context: real encoders refuse 25 ms
    return np.asarray(m.transcribe([np.linspace(-0.5, 0.5, n).astype(np.float32)], be.marker, 8))
