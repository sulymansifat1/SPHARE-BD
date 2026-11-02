"""Source-only axis + joint (convex-hull) condition support."""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def _eta2(values: np.ndarray, labels: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    labs = np.asarray(labels)
    if len(vals) < 2:
        return 0.0
    grand = float(np.mean(vals))
    ss_tot = float(np.sum((vals - grand) ** 2))
    if ss_tot <= 0:
        return 0.0
    ss_b = 0.0
    for lab in np.unique(labs):
        sub = vals[labs == lab]
        ss_b += len(sub) * (float(np.mean(sub)) - grand) ** 2
    return float(ss_b / ss_tot)


def _unique_cells(X: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(X, dtype=np.float64), axis=0)


def _normalize(pts: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float) -> np.ndarray:
    span = np.maximum(hi - lo, eps)
    return (pts - lo) / span


class SourceSupportEstimator:
    def fit(self, C: np.ndarray, names: list[str], cfg: dict) -> dict:
        X = np.asarray(C, dtype=np.float64)
        n, d = X.shape
        eps = float(cfg["epsilon"])
        axes = []
        for j, name in enumerate(names):
            col = X[:, j]
            finite = col[np.isfinite(col)]
            lo_q = float(np.quantile(finite, cfg["support_quantile_low"]))
            hi_q = float(np.quantile(finite, cfg["support_quantile_high"]))
            span = hi_q - lo_q
            if span <= eps:
                slack = max(abs(hi_q) * cfg["degenerate_rel_slack"], cfg["support_min_buffer"])
            else:
                slack = max(span * cfg["support_range_buffer_frac"], cfg["support_min_buffer"])
            axes.append(
                {
                    "name": name,
                    "lo": lo_q - slack,
                    "hi": hi_q + slack,
                    "source_min": float(np.min(finite)),
                    "source_max": float(np.max(finite)),
                    "slack": slack,
                    "n": int(len(finite)),
                }
            )
        cells = _unique_cells(X)
        lo = np.array([ax["source_min"] for ax in axes], dtype=np.float64)
        hi = np.array([ax["source_max"] for ax in axes], dtype=np.float64)
        cells_n = _normalize(cells, lo, hi, eps)
        hull_ok = False
        hull_vertices = cells.tolist()
        delaunay_pts = cells_n.tolist()
        if len(cells) >= 3 and d == 2:
            try:
                hull = ConvexHull(cells_n)
                hull_vertices = cells[hull.vertices].tolist()
                tri = Delaunay(cells_n)
                delaunay_pts = cells_n.tolist()
                hull_ok = True
            except Exception:
                hull_ok = False
                tri = None
        else:
            tri = None
        source_inside = []
        for row in X:
            source_inside.append(self._point_in_support(row, axes, lo, hi, tri, hull_ok, eps))
        coverage = float(np.mean(source_inside)) if source_inside else 0.0
        return {
            "n_source": int(n),
            "n_dim": int(d),
            "condition_dimension_names": names,
            "axes": axes,
            "source_cells": cells.tolist(),
            "n_source_cells": int(len(cells)),
            "source_min": lo.tolist(),
            "source_max": hi.tolist(),
            "hull_vertices": hull_vertices,
            "delaunay_points": delaunay_pts,
            "joint_geometry": "convex_hull_of_source_cells" if hull_ok else "axis_box_fallback",
            "hull_available": hull_ok,
            "source_coverage": coverage,
            "source_rejection_rate": float(1.0 - coverage),
            "calibration_method": "axis_quantiles_plus_source_convex_hull",
            "epsilon": eps,
        }

    def _axis_ok(self, c: dict[str, float], axes: list[dict]) -> tuple[bool, dict, bool]:
        scores = {}
        ok = True
        missing = False
        for ax in axes:
            val = c.get(ax["name"])
            if val is None or not np.isfinite(val):
                missing = True
                scores[ax["name"]] = None
                ok = False
                continue
            val = float(val)
            width = max(ax["hi"] - ax["lo"], 1e-12)
            if val < ax["lo"]:
                s = (val - ax["lo"]) / width
                ok = False
            elif val > ax["hi"]:
                s = (ax["hi"] - val) / width
                ok = False
            else:
                s = min(val - ax["lo"], ax["hi"] - val) / width
            scores[ax["name"]] = s
        return ok, scores, missing

    def _in_hull(self, vec: np.ndarray, lo: np.ndarray, hi: np.ndarray, tri, hull_ok: bool, eps: float) -> bool:
        if not hull_ok or tri is None:
            return True
        pn = _normalize(vec.reshape(1, -1), lo, hi, eps)[0]
        return int(tri.find_simplex(pn)) >= 0

    def _point_in_support(self, vec, axes, lo, hi, tri, hull_ok, eps) -> bool:
        c = {ax["name"]: float(vec[j]) for j, ax in enumerate(axes)}
        axis_ok, _, missing = self._axis_ok(c, axes)
        if missing or not axis_ok:
            return False
        return self._in_hull(np.asarray(vec, dtype=np.float64), lo, hi, tri, hull_ok, eps)

    def decide(self, c: dict[str, float], fitted: dict) -> dict:
        axes = fitted["axes"]
        axis_ok, scores, missing = self._axis_ok(c, axes)
        lo = np.asarray(fitted["source_min"], dtype=np.float64)
        hi = np.asarray(fitted["source_max"], dtype=np.float64)
        vec = np.array([float(c.get(ax["name"], np.nan)) for ax in axes], dtype=np.float64)
        cells = np.asarray(fitted["source_cells"], dtype=np.float64)
        if len(cells) and np.all(np.isfinite(vec)):
            dist = float(np.min(np.linalg.norm(cells - vec, axis=1)))
        else:
            dist = float("inf")
        hull_ok = bool(fitted.get("hull_available"))
        tri = None
        if hull_ok:
            pts = np.asarray(fitted["delaunay_points"], dtype=np.float64)
            try:
                tri = Delaunay(pts)
            except Exception:
                hull_ok = False
        in_hull = self._in_hull(vec, lo, hi, tri, hull_ok, float(fitted.get("epsilon", 1e-12)))
        joint_ok = bool(axis_ok and in_hull and not missing)
        if missing:
            reason = "INSUFFICIENT_SOURCE_SUPPORT"
        elif not axis_ok:
            reason = "CONDITION_AXIS_OUT_OF_SUPPORT"
        elif not in_hull:
            reason = "CONDITION_JOINT_OUT_OF_HULL"
        else:
            reason = "ACCEPT"
        score = float(np.min([v for v in scores.values() if v is not None])) if any(v is not None for v in scores.values()) else 0.0
        return {
            "pass": joint_ok,
            "score": score,
            "threshold": 0.0,
            "decision": "ACCEPT" if joint_ok else "ABSTAIN",
            "reason": reason,
            "scores": scores,
            "axis_membership": bool(axis_ok and not missing),
            "joint_membership": bool(in_hull),
            "distance_to_source_cells": dist,
        }


def screen_features(rows: list[dict], feat_keys: list[str], cfg: dict) -> list[dict]:
    out = []
    speeds = np.array([r["speed_est"] for r in rows], dtype=np.float64)
    faults = np.array([r.get("y") if r.get("y") is not None else -1 for r in rows])
    for k in feat_keys:
        vals = np.array([r["features"][k] for r in rows], dtype=np.float64)
        eta_c = _eta2(vals, np.round(speeds, 0).astype(int))
        eta_f = _eta2(vals, faults) if len(set(faults)) > 1 else 0.0
        selected = (eta_c >= cfg["feature_condition_min_eta2"]) and (eta_f <= cfg["feature_fault_max_eta2"])
        out.append(
            {
                "feature": k,
                "source_mean": float(np.mean(vals)),
                "source_std": float(np.std(vals)),
                "eta2_condition": eta_c,
                "eta2_fault": eta_f,
                "n": int(len(vals)),
                "selected": bool(selected),
                "rejection_reason": None if selected else "weak_condition_or_fault_confound",
            }
        )
    if not any(r["selected"] for r in out):
        best = max(out, key=lambda r: r["eta2_condition"])
        best["selected"] = True
        best["rejection_reason"] = None
        best["note"] = "fallback_highest_condition_eta2"
    return out
