"""Compose one recording from independent physical components (dataset v1.1)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

from .bearing_physics import characteristic_frequencies
from .fault_models import MECHANISM_FREQ, mechanism_plan, place_events_variable
from .nuisance_models import sample_nuisance, sample_severity


def _stream(base: list[int], tag: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([*base, int(tag)]))


def _kernel(fs: float, f_res: float, zeta: float, n_k: int) -> np.ndarray:
    t = np.arange(n_k, dtype=np.float64) / float(fs)
    w = 2.0 * np.pi * float(f_res)
    k = np.exp(-float(zeta) * w * t) * np.sin(w * t)
    peak = float(np.max(np.abs(k))) if n_k else 1.0
    if peak < 1e-12:
        return k
    return k / peak


def _excite(train: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n = int(len(train) + len(kernel) - 1)
    spec = np.fft.rfft(train, n=n) * np.fft.rfft(kernel, n=n)
    return np.fft.irfft(spec, n=n)[: len(train)]


def _colored_noise(rng: np.random.Generator, n: int, fs: float, exponent: float) -> np.ndarray:
    white = rng.normal(size=n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    scale = np.ones_like(freqs)
    scale[0] = 0.0
    scale[1:] = freqs[1:] ** (-0.5 * float(exponent))
    spec = spec * scale
    x = np.fft.irfft(spec, n=n)
    sd = float(np.std(x))
    return x / sd if sd > 0 else x


def _sensor(x: np.ndarray, fs: float, hp_hz: float, lp_hz: float) -> np.ndarray:
    nyq = 0.5 * float(fs)
    hp = min(max(hp_hz / nyq, 1e-4), 0.2)
    lp = min(max(lp_hz / nyq, hp + 0.05), 0.99)
    b_hp, a_hp = butter(1, hp, btype="highpass")
    b_lp, a_lp = butter(2, lp, btype="lowpass")
    y = lfilter(b_hp, a_hp, x)
    return lfilter(b_lp, a_lp, y)


def hertz_contact(lam: np.ndarray | float, sat: float) -> np.ndarray | float:
    """Saturating Hertz-like contact: λ^{3/2} / (sat + λ^{3/2})."""
    x = np.asarray(lam, dtype=np.float64)
    p = np.power(np.clip(x, 0.0, 1.5), 1.5)
    return p / (float(sat) + p)


def load_path(lam_t: np.ndarray, rec_scale: float, cfg: dict, hetero: float) -> dict:
    m = cfg.get("load_model", {})
    sat = m.get("hertz_sat", 0.3)
    c = hertz_contact(lam_t, sat) * rec_scale
    impulse = m.get("impulse_floor", 0.5) + m.get("impulse_span", 0.5) * c * (1.0 + hetero)
    damping = 1.0 + m.get("damping_gain", 0.2) * lam_t
    modulation = m.get("modulation_floor", 0.2) + m.get("modulation_span", 0.3) * c
    lf = m.get("lf_floor", 0.05) + m.get("lf_span", 0.1) * (lam_t ** 2) * rec_scale
    stiffness = 1.0 + m.get("stiffness_harm_gain", 0.15) * c
    return {
        "contact": c,
        "impulse": impulse,
        "damping": damping,
        "modulation": modulation,
        "lf": lf,
        "stiffness": stiffness,
    }


def synthesize(
    bearing: dict,
    speed_rpm: float,
    load_N: float,
    fault_class: str,
    fault_code: int,
    recording_index: int,
    master_seed: int,
    variant: int,
    cfg: dict,
    load_min: float,
    load_max: float,
    gen_shift: bool = False,
) -> tuple[np.ndarray, dict]:
    tags = cfg["seed_tags"]
    base = [
        int(master_seed),
        int(bearing["virtual_bearing_id"]),
        int(speed_rpm),
        int(load_N),
        int(fault_code),
        int(recording_index),
        int(variant),
    ]
    rng_n = _stream(base, tags["nuisance"])
    rng_s = _stream(base, tags["shaft"])
    rng_h = _stream(base, tags["harmonics"])
    rng_f = _stream(base, tags["fault"])
    rng_b = _stream(base, tags["background"])
    rng_sensor = _stream(base, tags["sensor"])
    rng_d = _stream(base, tags["disturbance"])

    nu = sample_nuisance(rng_n, cfg, gen_shift)
    fs = float(cfg["sampling_rate_hz"])
    t_keep = float(cfg["duration_sec"])
    t_int = float(cfg.get("internal_horizon_sec", t_keep))
    n_int = int(round(fs * t_int))
    n = int(round(fs * t_keep))
    crop = n_int - n
    t = np.arange(n_int, dtype=np.float64) / fs
    fr0 = float(speed_rpm) / 60.0
    span_L = max(float(load_max) - float(load_min), 1e-9)
    lam0 = float(np.clip((float(load_N) - float(load_min)) / span_L, 0.0, 1.0))

    # Instantaneous speed: slow drift + sinusoidal jitter + colored wander.
    wander = _colored_noise(rng_s, n_int, fs, 1.8)
    wander = wander / max(float(np.max(np.abs(wander))), 1e-9)
    jit_f = float(rng_s.uniform(0.35, 1.8))
    inst_fr = fr0 * (
        1.0
        + nu["speed_drift_frac"] * (t / t_int - 0.5)
        + nu["shaft_jitter_frac"] * np.sin(2.0 * np.pi * jit_f * t + nu["phase_shift"])
        + 0.004 * wander
    )
    inst_fr = np.clip(inst_fr, 0.4 * fr0, 1.6 * fr0)
    phase = 2.0 * np.pi * np.cumsum(inst_fr) / fs

    # Instantaneous load around the declared cell; label stays nominal.
    load_w = _colored_noise(rng_n, n_int, fs, 2.2)
    load_w = load_w / max(float(np.max(np.abs(load_w))), 1e-9)
    load_t = float(load_N) * (1.0 + nu["load_fluct_frac"] * load_w)
    lam_t = np.clip((load_t - float(load_min)) / span_L, 0.0, 1.4)
    hetero = float(cfg["load_model"]["hetero_frac"]) * lam0 * float(rng_n.uniform(-1.0, 1.0))
    rec_scale = nu["contact_stiffness_scale"] * float(bearing.get("structural_coupling", 1.0))
    lp = load_path(lam_t, rec_scale, cfg, hetero)

    freqs = characteristic_frequencies(
        fr0,
        bearing["n_balls"],
        bearing["ball_diameter_mm"],
        bearing["pitch_diameter_mm"],
        bearing["contact_angle_deg"],
    )
    f_res = float(bearing["resonance_hz"]) * (1.0 + nu["resonance_shift_frac"]) * (1.0 + 0.04 * (lam0 - 0.5))
    zeta0 = float(bearing["damping_ratio"]) * (1.0 + nu["damping_shift_frac"])
    zeta = float(np.clip(zeta0 * float(np.mean(lp["damping"])), 0.012, 0.22))

    am_f = float(rng_s.uniform(0.15, 2.4))
    am_p = float(rng_s.uniform(0.0, 2.0 * np.pi))
    shaft_scale = nu["baseline_vibration_scale"] * bearing["baseline_level"]
    x_shaft = shaft_scale * (1.0 + 0.08 * np.sin(2.0 * np.pi * am_f * t + am_p)) * np.sin(phase)

    x_harm = np.zeros(n_int, dtype=np.float64)
    hcfg = cfg["harmonics"]
    for k in range(2, int(hcfg["max_order"]) + 1):
        ak = float(rng_h.uniform(hcfg["amp_min"], hcfg["amp_max"])) / (k ** float(rng_h.uniform(0.7, 1.3)))
        if k in (2, 3):
            ak *= float(np.mean(lp["stiffness"]))
        ph = float(rng_h.uniform(0.0, 2.0 * np.pi))
        x_harm += shaft_scale * ak * np.sin(k * phase + ph)

    mechanisms = mechanism_plan(fault_class, rng_f)
    severities = [sample_severity(rng_f, cfg, gen_shift) for _ in mechanisms]
    slip = float(cfg["nuisance"]["fault_slip_frac"])
    x_fault = np.zeros(n_int, dtype=np.float64)
    realized = {}
    rate_scale = inst_fr / fr0
    for mech, sev in zip(mechanisms, severities):
        key = MECHANISM_FREQ[mech]
        f_nom = float(freqs[key]) * (1.0 + float(rng_f.uniform(-slip, slip)))
        realized[key] = f_nom
        ksec = float(rng_f.uniform(cfg["fault"]["kernel_sec_min"], cfg["fault"]["kernel_sec_max"]))
        ksec *= 1.15 - 0.3 * sev
        kern = _kernel(fs, f_res * (0.95 + 0.1 * sev), zeta * (1.0 + 0.15 * (1.0 - sev)), int(round(fs * ksec)))
        dropout = float(cfg["fault"]["dropout_scale"]) * (1.15 - sev)
        amp0 = float(cfg["fault"]["impulse_amp"]) * (0.35 + 0.9 * sev)
        if mech == "CAGE_FAULT":
            amp0 *= float(cfg["fault"]["cage_amp_scale"]) * (0.6 + 0.8 * sev)
        amp_t = amp0 * lp["impulse"]
        if mech == "INNER_RACE_FAULT":
            amp_mod = amp_t * (1.0 + lp["modulation"] * np.sin(phase))
        elif mech == "BALL_FAULT":
            amp_mod = amp_t * (1.0 + 0.55 * lp["modulation"] * np.sin(2.0 * np.pi * freqs["FTF"] * np.cumsum(rate_scale) / fs))
        else:
            amp_mod = amp_t
        train = place_events_variable(
            n_int,
            fs,
            f_nom * rate_scale,
            rng_f,
            1.0,
            float(cfg["nuisance"]["event_jitter_frac"]) * (1.2 - 0.4 * sev),
            dropout,
            amp_mod,
        )
        x_fault += _excite(train, kern)

    if "CAGE_FAULT" in mechanisms:
        sev_c = float(severities[mechanisms.index("CAGE_FAULT")])
        ftf = float(realized.get("FTF", freqs["FTF"]))
        depth = (0.25 + 0.55 * sev_c) * float(np.mean(lp["modulation"])) / max(float(np.mean(lp["modulation"])), 1e-6)
        depth = 0.25 + 0.55 * sev_c
        cage_mod = 1.0 + depth * np.cos(2.0 * np.pi * np.cumsum(ftf * rate_scale) / fs + float(rng_f.uniform(0.0, 2.0 * np.pi)))
        x_shaft = x_shaft * cage_mod
        x_harm = x_harm * cage_mod

    bg = cfg["background"]
    rate = float(rng_b.uniform(*bg["poisson_rate_hz"])) * (1.0 + 0.25 * lam0)
    n_events = int(rng_b.poisson(rate * t_int))
    bg_train = np.zeros(n_int, dtype=np.float64)
    if n_events:
        idx = rng_b.integers(0, n_int, size=n_events)
        amps = rng_b.uniform(bg["poisson_amp"][0], bg["poisson_amp"][1], size=n_events) * (1.0 + 0.2 * lam0)
        np.add.at(bg_train, idx, amps)
    kern_bg = _kernel(fs, f_res, zeta, int(round(fs * 0.02)))
    x_res = lp["lf"] * _excite(bg_train, kern_bg)
    hum_f = float(rng_b.uniform(*bg["hum_freq_hz"]))
    hum_a = float(rng_b.uniform(*bg["hum_amp"]))
    x_bg = hum_a * np.sin(2.0 * np.pi * hum_f * t + float(rng_b.uniform(0.0, 2.0 * np.pi)))

    x_col = _colored_noise(rng_b, n_int, fs, nu["colored_noise_exponent"])
    x_white = rng_b.normal(size=n_int)

    dist = cfg["disturbances"]
    x_dist = np.zeros(n_int, dtype=np.float64)
    has_burst = float(rng_d.random()) < float(dist["impulse_burst_p"])
    has_lf = float(rng_d.random()) < float(dist["lf_burst_p"])
    has_nb = float(rng_d.random()) < float(dist["narrowband_p"])
    has_glitch = float(rng_d.random()) < float(dist["sensor_glitch_p"])
    has_artifact = float(rng_d.random()) < float(dist["artifact_p"])
    if has_burst:
        n_imp = int(rng_d.integers(2, 8))
        loc = rng_d.integers(0, n_int, size=n_imp)
        x_dist[loc] += rng_d.uniform(0.4, 1.6, size=n_imp)
        x_dist = _excite(x_dist, kern_bg)
    if has_lf:
        a0 = int(rng_d.integers(0, max(n_int // 2, 1)))
        a1 = min(n_int, a0 + int(rng_d.integers(int(0.05 * fs), int(0.25 * fs))))
        x_dist[a0:a1] += 0.35 * np.sin(2.0 * np.pi * float(rng_d.uniform(4.0, 18.0)) * t[a0:a1])
    if has_nb:
        fn = float(rng_d.uniform(120.0, 900.0))
        x_dist += 0.08 * np.sin(2.0 * np.pi * fn * t + float(rng_d.uniform(0.0, 2.0 * np.pi)))
    if has_glitch:
        g0 = int(rng_d.integers(0, n_int - int(0.01 * fs)))
        g1 = g0 + int(rng_d.integers(int(0.002 * fs), int(0.02 * fs)))
        x_dist[g0:g1] += float(rng_d.uniform(-0.6, 0.6))

    mech = x_shaft + x_harm + x_fault + x_res + x_bg + 0.45 * x_dist
    x = mech + nu["colored_noise_level"] * x_col * (1.0 + 0.35 * lam0) + nu["additive_noise_level"] * x_white
    x = _sensor(x, fs, nu["sensor_highpass_hz"], nu["sensor_cutoff_hz"])
    x = nu["sensor_gain"] * bearing["nominal_gain"] * x + nu["offset"]
    if has_artifact:
        x = x + 0.02 * np.sin(2.0 * np.pi * 50.0 * t)
        peak = float(np.max(np.abs(x)))
        lim = nu["clip_margin"] * float(np.std(x) + 1e-9)
        if peak > lim:
            x = np.clip(x, -lim, lim)
    x = x[crop:]

    load_keep = load_t[crop:]
    info = {
        "virtual_bearing_id": int(bearing["virtual_bearing_id"]),
        "fault_class": fault_class,
        "fault_severity": float(severities[0]) if severities else 0.0,
        "fault_severity_2": float(severities[1]) if len(severities) > 1 else 0.0,
        "fault_mechanisms": "+".join(mechanisms) if mechanisms else "NONE",
        "sensor_gain": nu["sensor_gain"],
        "noise_level": nu["additive_noise_level"],
        "colored_noise_level": nu["colored_noise_level"],
        "resonance_shift": nu["resonance_shift_frac"],
        "damping_ratio": zeta,
        "shaft_jitter": nu["shaft_jitter_frac"],
        "baseline_vibration_scale": nu["baseline_vibration_scale"],
        "phase_shift": nu["phase_shift"],
        "effective_resonance_hz": f_res,
        "nominal_frequencies_hz": freqs,
        "realized_fault_hz": realized,
        "load_lambda": lam0,
        "nominal_load": float(load_N),
        "actual_mean_load": float(np.mean(load_keep)),
        "actual_load_std": float(np.std(load_keep)),
        "mean_inst_speed_rpm": float(np.mean(inst_fr[crop:]) * 60.0),
        "sensor_artifact": int(has_artifact),
        "disturbance_burst": int(has_burst),
        "disturbance_lf": int(has_lf),
        "disturbance_narrowband": int(has_nb),
        "disturbance_glitch": int(has_glitch),
        "structural_coupling": float(bearing.get("structural_coupling", 1.0)),
        "contact_stiffness_scale": nu["contact_stiffness_scale"],
        "generator_seed": int(np.random.SeedSequence(base).generate_state(1)[0]),
    }
    return np.asarray(x, dtype=np.float32), info
