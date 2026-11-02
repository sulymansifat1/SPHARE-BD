"""Unit test: Verify Stage 1 and Stage 2 support gates and short-circuit conjunction logic."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support.gates import combine_stages, envelope_decide
from support.source_support import SourceSupportEstimator


def test_stage1_bounding_box():
    freeze_path = REPO_ROOT / "configs" / "freeze" / "EXPERIMENT_A_FROZEN_seed42.json"
    assert freeze_path.exists(), f"Freeze file missing: {freeze_path}"

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    est = SourceSupportEstimator()
    support = freeze["support"]

    # In-distribution test: 2000 RPM, 0.0 kN
    res_in = est.decide({"speed": 2000.0, "load": 0.0}, support)
    assert res_in["pass"] is True, f"Expected 2000 RPM, 0 kN to pass Stage 1: {res_in}"

    # Out-of-support speed: 1000 RPM, 0.0 kN
    res_out = est.decide({"speed": 1000.0, "load": 0.0}, support)
    assert res_out["pass"] is False, f"Expected 1000 RPM to fail Stage 1: {res_out}"
    assert res_out["reason"] == "CONDITION_AXIS_OUT_OF_SUPPORT"

    # Unsupported load shift: 2000 RPM, 20.0 kN
    res_load = est.decide({"speed": 2000.0, "load": 20.0}, support)
    assert res_load["pass"] is False, f"Expected 20 kN load to fail Stage 1: {res_load}"
    assert res_load["reason"] == "CONDITION_AXIS_OUT_OF_SUPPORT"
    print("Stage 1 bounding box tests passed.")


def test_combine_stages_short_circuit():
    # Both pass
    s1_pass = {"pass": True, "reason": "ACCEPT"}
    s2_pass = {"pass": True, "reason": "ACCEPT"}
    comb_pass = combine_stages(s1_pass, s2_pass)
    assert comb_pass["accept"] is True
    assert comb_pass["reason"] == "ACCEPT"
    assert comb_pass["first_rejection_stage"] is None

    # Stage 1 fails -> early exit
    s1_fail = {"pass": False, "reason": "CONDITION_AXIS_OUT_OF_SUPPORT"}
    comb_s1 = combine_stages(s1_fail, s2_pass)
    assert comb_s1["accept"] is False
    assert comb_s1["reason"] == "CONDITION_AXIS_OUT_OF_SUPPORT"
    assert comb_s1["first_rejection_stage"] == "Stage1"

    # Stage 1 passes, Stage 2 fails -> early exit
    s2_fail = {"pass": False, "reason": "PHYSICAL_INCONSISTENCY"}
    comb_s2 = combine_stages(s1_pass, s2_fail)
    assert comb_s2["accept"] is False
    assert comb_s2["reason"] == "PHYSICAL_INCONSISTENCY"
    assert comb_s2["first_rejection_stage"] == "Stage2"
    print("Combine stages short-circuit tests passed.")


if __name__ == "__main__":
    test_stage1_bounding_box()
    test_combine_stages_short_circuit()
