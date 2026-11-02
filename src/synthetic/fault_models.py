"""Impulsive fault excitation at kinematic frequencies."""
from __future__ import annotations

import numpy as np


MECHANISM_FREQ = {
    "INNER_RACE_FAULT": "BPFI",
    "OUTER_RACE_FAULT": "BPFO",
    "BALL_FAULT": "BSF2",
    "CAGE_FAULT": "FTF",
}


def mechanism_plan(fault_class: str, rng: np.random.Generator) -> list[str]:
    if fault_class == "HEALTHY":
        return []
    if fault_class == "MULTI_FAULT":
        pool = ["INNER_RACE_FAULT", "OUTER_RACE_FAULT", "BALL_FAULT"]
        pick = rng.choice(pool, size=2, replace=False)
        return [str(p) for p in pick]
    return [fault_class]


def place_events_variable(
    n: int,
    fs: float,
    rate_hz: np.ndarray,
    rng: np.random.Generator,
    amplitude: float,
    jitter_frac: float,
    dropout_p: float,
    modulation: np.ndarray | None,
) -> np.ndarray:
    """Place impulses by integrating a time-varying event rate."""
    train = np.zeros(n, dtype=np.float64)
    if amplitude == 0.0:
        return train
    rate = np.maximum(np.asarray(rate_hz, dtype=np.float64), 1e-9)
    mean_rate = float(np.mean(rate))
    phase0 = float(rng.uniform(0.0, 1.0))
    cum = np.cumsum(rate / float(fs)) + phase0
    k_max = int(np.floor(cum[-1]))
    if k_max < 1:
        return train
    targets = np.arange(1, k_max + 1, dtype=np.float64)
    idx = np.searchsorted(cum, targets)
    jitter = (rng.uniform(-jitter_frac, jitter_frac, size=idx.shape) * (float(fs) / mean_rate)).astype(np.int64)
    idx = np.clip(idx + jitter, 0, n - 1)
    keep = rng.random(size=idx.shape) >= dropout_p
    amps = amplitude * rng.uniform(0.7, 1.35, size=idx.shape)
    if modulation is not None:
        amps = amps * modulation[idx]
    for i, a, k in zip(idx, amps, keep):
        if k:
            train[int(i)] += float(a)
    return train
