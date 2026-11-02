"""Machine-specific measurement/structure overlay. Not a SPHARE-BD model input."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


def _kernel(fs: float, f_res: float, zeta: float, n_k: int) -> np.ndarray:
    t = np.arange(n_k, dtype=np.float64) / float(fs)
    w = 2.0 * np.pi * float(f_res)
    k = np.exp(-float(zeta) * w * t) * np.sin(w * t)
    peak = float(np.max(np.abs(k))) if n_k else 1.0
    return k / peak if peak >= 1e-12 else k


def apply_machine_frf(x: np.ndarray, fs: float, profile: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply a linear structural/sensor transfer unique to one machine."""
    y = np.asarray(x, dtype=np.float64)
    if not profile.get("enabled", True):
        return y.astype(np.float32)
    n_k = int(max(32, round(float(fs) * 0.02)))
    k = _kernel(fs, profile["resonance_hz"], profile["zeta"], n_k)
    n = int(len(y) + len(k) - 1)
    spec = np.fft.rfft(y, n=n) * np.fft.rfft(k, n=n)
    y = np.fft.irfft(spec, n=n)[: len(x)]
    nyq = 0.5 * float(fs)
    hp = min(max(float(profile["highpass_hz"]) / nyq, 1e-4), 0.2)
    lp = min(max(float(profile["lowpass_hz"]) / nyq, hp + 0.05), 0.99)
    b_hp, a_hp = butter(1, hp, btype="highpass")
    b_lp, a_lp = butter(2, lp, btype="lowpass")
    y = lfilter(b_hp, a_hp, y)
    y = lfilter(b_lp, a_lp, y)
    y = float(profile["gain"]) * y
    noise = float(profile.get("extra_noise", 0.0))
    if noise > 0:
        y = y + noise * rng.normal(size=len(y))
    return np.asarray(y, dtype=np.float32)
