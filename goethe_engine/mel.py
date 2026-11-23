"""Log-mel audio front end from a pack's numbers (a media encode bucket's io.mel), llama.cpp mtmd's order of operations:
preemphasis on the raw samples, zero padding of n_fft / 2 each side, the window centered in an n_fft frame, a DFT with the window
folded in (f32 operands), the power spectrum, the mel filterbank, ln(mel + floor), the first n_samples / hop frames kept.
norm: per mel row over those frames (mean, unbiased std + eps), the input an ASR encoder takes. The engine knows these fields,
never a model; checked against llama.cpp's own feature bytes (an encoder record's uploads: relative 3e-6 raw, 9e-6 normalized)."""
import numpy as np


def log_mel(pcm, cfg):
    """(raw [n_mel][n_frames], normalized [n_mel][n_frames]) float32 of float32 mono pcm at cfg["sample_rate"]. cfg: n_fft, hop,
    window (float32, at most n_fft), filters (float32 [n_mel][n_fft / 2 + 1]), preemph, floor, norm_eps"""
    nfft, hop = int(cfg["n_fft"]), int(cfg["hop"])
    window = np.asarray(cfg["window"], np.float32)
    filters = np.asarray(cfg["filters"], np.float32)
    x = np.asarray(pcm, np.float32)
    y = x.copy()
    if cfg.get("preemph"):
        y[1:] = x[1:] - np.float32(cfg["preemph"]) * x[:-1]
    xp = np.zeros(len(y) + nfft, np.float32)
    xp[nfft // 2:nfft // 2 + len(y)] = y
    w = np.zeros(nfft, np.float32)
    w[(nfft - len(window)) // 2:(nfft - len(window)) // 2 + len(window)] = window
    n_len = (len(xp) - nfft) // hop + 1
    frames = np.lib.stride_tricks.as_strided(xp, (n_len, nfft), (hop * 4, 4))
    i = np.arange(nfft)
    th = 2 * np.pi * i / nfft
    nb = nfft // 2 + 1
    q = (np.arange(nb)[:, None] * i[None, :]) % nfft
    dft = np.concatenate([w * np.cos(th).astype(np.float32)[q], w * np.sin(th).astype(np.float32)[q]]).astype(np.float32)
    spec = (frames.astype(np.float64) @ dft.T.astype(np.float64)).astype(np.float32)
    power = spec[:, :nb] * spec[:, :nb] + spec[:, nb:] * spec[:, nb:]
    mel = (power.astype(np.float64) @ filters.T.astype(np.float64)).astype(np.float32)
    raw = np.log(mel + np.float32(cfg["floor"])).T[:, :len(x) // hop].copy()
    n = raw.shape[1]
    mean = raw.astype(np.float64).mean(1, keepdims=True)
    den = np.sqrt(((raw - mean) ** 2).sum(1, keepdims=True) / max(n - 1.0, 1.0)) + float(cfg.get("norm_eps", 1e-5))
    return raw, ((raw - mean) / den).astype(np.float32)


if __name__ == "__main__":       # self-check: a pure tone peaks in the mel row holding its frequency; shapes; normalization
    sr, nfft = 16000, 512
    freqs = np.linspace(0, sr / 2, nfft // 2 + 1)
    edges = np.linspace(0, sr / 2, 34)
    fb = np.array([np.clip(1 - np.abs(freqs - c) / (edges[1] - edges[0]), 0, None) for c in edges[1:-1]], np.float32)
    cfg = dict(n_fft=nfft, hop=160, window=np.hanning(400).astype(np.float32), filters=fb, preemph=0.97, floor=2 ** -24)
    t = np.arange(sr) / sr
    raw, norm = log_mel((0.1 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32), cfg)
    assert raw.shape == norm.shape == (32, 100)
    assert int(np.argmax(raw.mean(1))) == int(np.argmin(np.abs(edges[1:-1] - 2000)))
    assert np.allclose(norm.mean(1), 0, atol=1e-4)
    print("mel ok")
