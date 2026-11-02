"""Source-only local manifolds."""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def discover_regions(C: np.ndarray, min_count: int) -> np.ndarray:
    X = np.asarray(C, dtype=np.float64)
    scale = np.maximum(np.std(X, axis=0), 1.0)
    key = np.round(X / scale, 1)
    uniq, labels = np.unique(key, axis=0, return_inverse=True)
    counts = np.bincount(labels)
    if int(np.min(counts)) >= min_count and len(uniq) >= 1:
        return labels
    return np.zeros(len(X), dtype=int)


def fit_manifolds(features: np.ndarray, labels: np.ndarray, percentile: float, min_count: int) -> dict:
    F = np.asarray(features, dtype=np.float64)
    labs = np.asarray(labels)
    regions = {}
    for lab in sorted(set(labs.tolist())):
        idx = np.where(labs == lab)[0]
        X = F[idx]
        if len(X) < min_count:
            continue
        lw = LedoitWolf().fit(X)
        diff = X - lw.location_
        dist = np.sqrt(np.maximum(np.sum(diff @ lw.precision_ * diff, axis=1), 0.0))
        tau = float(np.percentile(dist, percentile))
        regions[str(int(lab))] = {
            "location": lw.location_.tolist(),
            "precision": lw.precision_.tolist(),
            "tau": tau,
            "n": int(len(X)),
            "centroid_C": np.mean(X, axis=0).tolist(),
        }
    return regions


def assign_region(c_vec: np.ndarray, region_centroids: np.ndarray) -> int:
    d = np.sum((region_centroids - c_vec) ** 2, axis=1)
    return int(np.argmin(d))


def manifold_decide(feat: np.ndarray, region: dict) -> dict:
    loc = np.asarray(region["location"], dtype=np.float64)
    prec = np.asarray(region["precision"], dtype=np.float64)
    tau = float(region["tau"])
    diff = feat - loc
    dist = float(np.sqrt(max(float(diff @ prec @ diff), 0.0)))
    ok = dist <= tau
    return {
        "pass": bool(ok),
        "score": float(tau - dist),
        "threshold": tau,
        "distance": dist,
        "decision": "ACCEPT" if ok else "ABSTAIN",
        "reason": "ACCEPT" if ok else "LOCAL_MANIFOLD_MISMATCH",
    }
