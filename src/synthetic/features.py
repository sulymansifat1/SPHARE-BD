"""Dataset-validation features. These are not used to train a diagnostic model."""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def _local_ratio(freqs: np.ndarray, spec: np.ndarray, f0: float) -> float:
    if f0 is None or not np.isfinite(f0) or f0 <= 0:
        return 0.0
    half = max(0.7, 0.006 * float(f0))
    m = np.abs(freqs - float(f0)) <= half
    ann = (np.abs(freqs - float(f0)) > 3.0 * half) & (np.abs(freqs - float(f0)) < 12.0 * half) & (freqs > 0.4)
    if not np.any(m) or not np.any(ann):
        return 0.0
    return float(np.max(spec[m]) / max(float(np.median(spec[ann])), 1e-12))


def _band(freqs: np.ndarray, spec: np.ndarray, f0: float) -> float:
    if f0 is None or not np.isfinite(f0) or f0 <= 0:
        return 0.0
    half = max(1.5, 0.012 * float(f0))
    m = np.abs(freqs - float(f0)) <= half
    if not np.any(m):
        return 0.0
    return float(np.sum(spec[m]))


def describe(x: np.ndarray, fs: float, nominal: dict, primary_hz: float | None) -> dict:
    z = np.asarray(x, dtype=np.float64)
    z = z - np.mean(z)
    rms = float(np.sqrt(np.mean(z ** 2)))
    std = float(np.std(z))
    peak = float(np.max(np.abs(z)))
    kurt = float(np.mean(z ** 4) / max(std ** 4, 1e-18))
    spec = np.abs(np.fft.rfft(z)) ** 2
    freqs = np.fft.rfftfreq(len(z), d=1.0 / float(fs))
    ps = spec / max(float(np.sum(spec)), 1e-18)
    centroid = float(np.sum(freqs * ps))
    bandwidth = float(np.sqrt(np.sum((freqs - centroid) ** 2 * ps)))
    env = np.abs(hilbert(z))
    env = env - np.mean(env)
    espec = np.abs(np.fft.rfft(env))
    efreqs = np.fft.rfftfreq(len(env), d=1.0 / float(fs))
    observed = np.nan
    if primary_hz is not None and primary_hz > 0:
        half = max(1.5, 0.012 * float(primary_hz))
        m = np.abs(efreqs - float(primary_hz)) <= half
        if np.any(m):
            local_i = int(np.argmax(espec[m]))
            observed = float(efreqs[m][local_i])
    return {
        "rms": rms,
        "std": std,
        "kurtosis": kurt,
        "crest_factor": float(peak / max(rms, 1e-12)),
        "peak_to_rms": float(peak / max(rms, 1e-12)),
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "envelope_energy": float(np.sum(espec)),
        "order_shaft": _band(efreqs, espec, nominal.get("shaft_hz")),
        "order_ftf": _band(efreqs, espec, nominal.get("FTF")),
        "order_bpfo": _band(efreqs, espec, nominal.get("BPFO")),
        "order_bpfi": _band(efreqs, espec, nominal.get("BPFI")),
        "order_bsf2": _band(efreqs, espec, nominal.get("BSF2")),
        "ratio_ftf": _local_ratio(efreqs, espec, nominal.get("FTF")),
        "ratio_bpfo": _local_ratio(efreqs, espec, nominal.get("BPFO")),
        "ratio_bpfi": _local_ratio(efreqs, espec, nominal.get("BPFI")),
        "ratio_bsf2": _local_ratio(efreqs, espec, nominal.get("BSF2")),
        "total_spectral_energy": float(np.sum(spec)),
        "low_frequency_energy": float(np.sum(spec[freqs <= 200.0])),
        "broadband_energy": float(np.sum(spec[(freqs > 200.0) & (freqs <= 4000.0)])),
        "observed_primary_hz": observed,
    }
