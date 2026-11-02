"""Adaptive windows from sampling rate and a generic duration."""
from __future__ import annotations

import numpy as np


def derived_window_params(fs: float, window_duration_sec: float, hop_fraction: float) -> dict:
    n = int(round(float(fs) * float(window_duration_sec)))
    hop = max(1, int(round(n * float(hop_fraction))))
    return {
        "sampling_rate_hz": float(fs),
        "window_duration_sec": float(window_duration_sec),
        "window_length_samples": n,
        "hop_samples": hop,
    }


def extract_windows(signal: np.ndarray, n: int, hop: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < n:
        return np.empty((0, n), dtype=np.float64)
    starts = range(0, len(x) - n + 1, hop)
    return np.stack([x[s : s + n] for s in starts], axis=0)


def select_support_windows(windows: np.ndarray, k: int) -> np.ndarray:
    if len(windows) == 0:
        return windows
    if len(windows) <= k:
        return windows
    idx = np.linspace(0, len(windows) - 1, k, dtype=int)
    return windows[idx]
