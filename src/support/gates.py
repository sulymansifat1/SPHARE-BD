"""AND hierarchical support decision for Stage 1 + Stage 2 only."""
from __future__ import annotations


def combine_stages(s1: dict, s2: dict, s3: dict | None = None) -> dict:
    """Combines Stage 1 and Stage 2 decisions.

    Stage 3 is completely excised from the active decision pipeline.
    Acceptance requires both Stage 1 and Stage 2 to pass.
    """
    accept = bool(s1["pass"] and s2["pass"])
    if accept:
        reason = "ACCEPT"
        first = None
    elif not s1["pass"]:
        reason = s1["reason"]
        first = "Stage1"
    else:
        reason = s2["reason"]
        first = "Stage2"
    return {
        "accept": accept,
        "abstain": not accept,
        "reason": reason,
        "first_rejection_stage": first,
        "stage1": s1,
        "stage2": s2,
    }


def envelope_decide(feat_val: float, lo: float, hi: float) -> dict:
    ok = lo <= feat_val <= hi
    width = max(hi - lo, 1e-8)
    if feat_val < lo:
        score = (feat_val - lo) / width
    elif feat_val > hi:
        score = (hi - feat_val) / width
    else:
        score = min(feat_val - lo, hi - feat_val) / width
    return {
        "pass": bool(ok),
        "score": float(score),
        "threshold": [lo, hi],
        "decision": "ACCEPT" if ok else "ABSTAIN",
        "reason": "ACCEPT" if ok else "PHYSICAL_INCONSISTENCY",
        "value": float(feat_val),
    }
