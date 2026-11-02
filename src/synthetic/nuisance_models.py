"""Recording-level nuisance draws. Streams are independent of the fault stream."""
from __future__ import annotations

import numpy as np


def _uniform(rng: np.random.Generator, span) -> float:
    return float(rng.uniform(float(span[0]), float(span[1])))


def sample_nuisance(rng: np.random.Generator, cfg: dict, gen_shift: bool) -> dict:
    nu = cfg["nuisance"]
    if gen_shift:
        gs = cfg["gen_shift"]
        gain_span = gs["sensor_gain"]
        add_span = gs["additive_noise_level"]
        col_span = gs["colored_noise_level"]
        res_span = gs["resonance_shift_frac"]
        damp_span = gs["damping_shift_frac"]
    else:
        gain_span = nu["sensor_gain"]
        add_span = nu["additive_noise_level"]
        col_span = nu["colored_noise_level"]
        res_span = nu["resonance_shift_frac"]
        damp_span = nu["damping_shift_frac"]
    exp_span = nu["colored_noise_exponent"]
    if not isinstance(exp_span, (list, tuple)):
        exp_span = [float(exp_span), float(exp_span)]
    return {
        "sensor_gain": _uniform(rng, gain_span),
        "additive_noise_level": _uniform(rng, add_span),
        "colored_noise_level": _uniform(rng, col_span),
        "colored_noise_exponent": _uniform(rng, exp_span),
        "resonance_shift_frac": _uniform(rng, res_span),
        "damping_shift_frac": _uniform(rng, damp_span),
        "shaft_jitter_frac": _uniform(rng, nu["shaft_jitter_frac"]),
        "speed_drift_frac": _uniform(rng, nu["speed_drift_frac"]),
        "load_fluct_frac": _uniform(rng, nu["load_fluct_frac"]),
        "baseline_vibration_scale": _uniform(rng, nu["baseline_vibration_scale"]),
        "contact_stiffness_scale": _uniform(rng, nu["contact_stiffness_scale"]),
        "phase_shift": float(rng.uniform(0.0, 2.0 * np.pi)),
        "sensor_cutoff_hz": _uniform(rng, nu["sensor_cutoff_hz"]),
        "sensor_highpass_hz": _uniform(rng, nu["sensor_highpass_hz"]),
        "offset": float(rng.normal(0.0, float(nu["offset_std"]))),
        "clip_margin": _uniform(rng, nu["clip_margin"]),
    }


def sample_severity(rng: np.random.Generator, cfg: dict, gen_shift: bool) -> float:
    spec = cfg["gen_shift"]["severity"] if gen_shift else cfg["severity"]
    return float(rng.uniform(float(spec["low"]), float(spec["high"])))
