"""Virtual-bearing geometry and standard rolling-element kinematic frequencies."""
from __future__ import annotations

import numpy as np


def characteristic_frequencies(
    shaft_hz: float,
    n_balls: int,
    ball_diameter_mm: float,
    pitch_diameter_mm: float,
    contact_angle_deg: float,
) -> dict[str, float]:
    """Harris / Randall kinematic frequencies for a deep-groove ball bearing.

    FTF  = (fr/2) * (1 - (d/D) cos α)
    BPFO = (n fr/2) * (1 - (d/D) cos α)
    BPFI = (n fr/2) * (1 + (d/D) cos α)
    BSF  = (D/(2d)) fr * (1 - ((d/D) cos α)^2)

    A rolling-element defect strikes both raceways, so the impact rate used
    by the generator is 2*BSF. That choice is documented in the math report.
    """
    d = float(ball_diameter_mm)
    D = float(pitch_diameter_mm)
    if d <= 0 or D <= 0 or D <= d:
        raise ValueError("bearing diameters are not physically ordered")
    alpha = np.deg2rad(float(contact_angle_deg))
    ratio = (d / D) * float(np.cos(alpha))
    if ratio >= 1.0:
        raise ValueError("d/D cos(alpha) must be < 1")
    fr = float(shaft_hz)
    n = int(n_balls)
    ftf = 0.5 * fr * (1.0 - ratio)
    bpfo = (n * fr / 2.0) * (1.0 - ratio)
    bpfi = (n * fr / 2.0) * (1.0 + ratio)
    bsf = (D / (2.0 * d)) * fr * (1.0 - ratio ** 2)
    return {
        "shaft_hz": fr,
        "FTF": float(ftf),
        "BPFO": float(bpfo),
        "BPFI": float(bpfi),
        "BSF": float(bsf),
        "BSF2": float(2.0 * bsf),
    }


def sample_bearing_population(master_seed: int, n_bearings: int, ranges: dict) -> list[dict]:
    bearings = []
    balls = list(ranges["n_balls"])
    for i in range(int(n_bearings)):
        rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), 1009, i]))
        d_mm = float(rng.uniform(*ranges["ball_diameter_mm"]))
        D_mm = float(rng.uniform(*ranges["pitch_diameter_mm"]))
        if D_mm < d_mm * 3.5:
            D_mm = d_mm * 3.5 + float(rng.uniform(0.0, 4.0))
        bearings.append(
            {
                "virtual_bearing_id": i,
                "n_balls": int(balls[int(rng.integers(0, len(balls)))]),
                "ball_diameter_mm": d_mm,
                "pitch_diameter_mm": D_mm,
                "contact_angle_deg": float(rng.uniform(*ranges["contact_angle_deg"])),
                "radial_clearance_um": float(rng.uniform(*ranges["radial_clearance_um"])),
                "resonance_hz": float(rng.uniform(*ranges["resonance_hz"])),
                "damping_ratio": float(rng.uniform(*ranges["damping_ratio"])),
                "nominal_gain": float(rng.uniform(*ranges["nominal_gain"])),
                "baseline_level": float(rng.uniform(*ranges["baseline_level"])),
                "structural_coupling": float(rng.uniform(*ranges["structural_coupling"])),
            }
        )
    return bearings


def formula_unit_check() -> dict:
    """Fixed numerical check of the kinematic identities. Not a fitted dataset."""
    f = characteristic_frequencies(10.0, 8, 8.0, 40.0, 0.0)
    expected = {"FTF": 4.0, "BPFO": 32.0, "BPFI": 48.0, "BSF": 24.0, "BSF2": 48.0}
    ok = True
    for k, v in expected.items():
        if abs(f[k] - v) > 1e-9:
            ok = False
    return {"pass": ok, "computed": f, "expected": expected}
