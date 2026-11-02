"""Source-only normalization."""
from __future__ import annotations

import numpy as np


def dc_remove(windows: np.ndarray) -> np.ndarray:
    w = np.asarray(windows, dtype=np.float64)
    return w - w.mean(axis=-1, keepdims=True)


def fit_scale(windows: np.ndarray, eps: float) -> float:
    w = dc_remove(windows)
    return float(max(np.std(w), eps))


def apply_scale(windows: np.ndarray, scale: float) -> np.ndarray:
    w = dc_remove(windows)
    return w / float(scale)
