"""Adaptive spectral features: order domain if shaft rate exists, else Nyquist-normalized."""
from __future__ import annotations

import numpy as np


def shaft_rate_hz_from_speed(speed: float | None, rev_per_minute_factor: float) -> float | None:
    if speed is None:
        return None
    return float(speed) / float(rev_per_minute_factor)


def estimate_shaft_rate_hz(window: np.ndarray, fs: float) -> float:
    x = np.asarray(window, dtype=np.float64)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    n = max(len(spec) // 4, 2)
    hps = spec[:n].copy()
    for h in (2, 3):
        down = spec[::h][:n]
        hps[: len(down)] *= down
    peak = int(np.argmax(hps[1:]) + 1)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return float(freqs[min(peak, len(freqs) - 1)])


def _log_energy(power: np.ndarray, mask: np.ndarray, eps: float) -> float:
    return float(np.log(max(float(np.sum(power[mask])), eps)))


def window_spectral_features(
    window: np.ndarray,
    fs: float,
    shaft_hz: float | None,
    order_max: float | None,
    n_bins: int,
    low_order_cut: float | None,
    eps: float,
    nyquist_fraction: float,
) -> dict[str, float]:
    x = np.asarray(window, dtype=np.float64)
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    nyq = float(fs) * float(nyquist_fraction)
    out = {
        "log_total_energy": float(np.log(max(float(np.sum(power)), eps))),
        "log_rms": float(np.log(max(float(np.sqrt(np.mean(x**2))), eps))),
    }
    if shaft_hz is not None and shaft_hz > eps and order_max is not None:
        orders = freqs / shaft_hz
        hi = float(order_max)
        edges = np.linspace(0.0, hi, n_bins + 1)
        for i in range(n_bins):
            m = (orders >= edges[i]) & (orders < edges[i + 1])
            out[f"order_bin_{i}_logE"] = _log_energy(power, m, eps)
        if low_order_cut is not None:
            out["low_order_logE"] = _log_energy(power, (orders >= 0.0) & (orders < float(low_order_cut)), eps)
    else:
        fn = freqs / max(nyq, eps)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        for i in range(n_bins):
            m = (fn >= edges[i]) & (fn < edges[i + 1])
            out[f"normf_bin_{i}_logE"] = _log_energy(power, m, eps)
        out["low_normf_logE"] = _log_energy(power, fn < (1.0 / n_bins), eps)
    return out


def recording_feature_median(windows: np.ndarray, fs: float, shaft_hz: float | None, spec_cfg: dict, eps: float) -> dict[str, float]:
    feats = [
        window_spectral_features(
            w,
            fs,
            shaft_hz,
            spec_cfg.get("order_max"),
            int(spec_cfg["n_bins"]),
            spec_cfg.get("low_order_cut"),
            eps,
            float(spec_cfg["nyquist_fraction"]),
        )
        for w in windows
    ]
    keys = feats[0].keys()
    return {k: float(np.median([f[k] for f in feats])) for k in keys}


def source_order_limits(source_shaft_hz: list[float], fs: float, energy_mass: float, n_bins: int, nyquist_fraction: float) -> dict:
    med_fr = float(np.median(source_shaft_hz))
    nyq = float(fs) * float(nyquist_fraction)
    order_max = float(nyq / max(med_fr, float(1e-12)))
    low_cut = float(order_max * energy_mass)
    return {
        "order_max": order_max,
        "low_order_cut": low_cut,
        "n_bins": int(n_bins),
        "median_source_shaft_hz": med_fr,
        "nyquist_fraction": float(nyquist_fraction),
        "representation": "rotational_order",
    }
