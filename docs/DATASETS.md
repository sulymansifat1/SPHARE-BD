# Dataset Specifications & Acquisition Guide

This document describes the benchmark datasets evaluated in SPHARE-BD, their physical testbed configurations, anti-leakage temporal slicing protocols, and acquisition procedures.

---

## 1. Benchmark Overview

| Dataset | Machine / Testbed | Role in Study | Sampling Rate ($f_s$) | Speed Range | Static Load Range | Coarse Classes | Split Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SDUST** | Shandong Univ. of Science & Tech | Primary In-Domain Benchmark (Exp A) | 25,600 Hz | 1000–3000 RPM | 0.0 & 20.0 kN | NC, IRF, ORF, REF (4 classes) | Non-overlapping temporal intervals |
| **CWRU** | Case Western Reserve Univ. | Target Foreign Testbed (Exp A) | 12,000 Hz | 1730–1797 RPM | 0–3 HP | NC, IRF, ORF, REF (4 classes) | 16 held-out recordings |
| **Synthetic Bearing** | Physics-Informed Simulator | 2D Geometry Benchmark (Exp B) | 10,000 Hz | 800–2400 RPM | 500–1500 N | 6 Fault Categories | 96 grid cells |
| **Synthetic Cross-Machine** | Physics-Informed Simulator | Structural Transfer Benchmark (Exp C) | 10,000 Hz | 800–2400 RPM | 500–1500 N | 6 Fault Categories | Virtual testbed (+7% gain/res) |

---

## 2. SDUST Benchmark Dataset

### 2.1 Testbed Operating Regimes
- **Source Regimes ($\mathcal{C}_{\mathrm{src}}$):** Rotational speeds 1500, 2000, 2500 RPM under constant 0.0 kN static radial load (30 physical recordings).
- **Supported Test Evaluation ($\mathcal{D}_{\mathrm{supp}}$):**
  - `IN`: 2000 RPM, 0.0 kN (nominal interior).
  - `EDGE-1500`: 1500 RPM, 0.0 kN (lower speed boundary).
  - `EDGE-2500`: 2500 RPM, 0.0 kN (upper speed boundary).
- **Unsupported Shifts ($\mathcal{D}_{\mathrm{unsupp}}$):**
  - `OUT`: 1000 RPM, 0.0 kN (speed extrapolation).
  - `NEW_AXIS`: 1500, 2000, 2500 RPM at 20.0 kN load (**unsupported load-axis shift**; sensor channel `zcjshang:+X` remained fixed).
  - `COMPOUND`: 1000 RPM at 20.0 kN load (**compound speed extrapolation and load shift**).

### 2.2 Anti-Leakage Temporal Slicing Protocol
To guarantee zero data contamination between training, validation, and evaluation:
- `source_train`: Samples $[0, 626688)$ across 30 source files (306 windows of 2048 samples; 32 drawn uniformly per recording during training).
- *Guard Gap 1:* Samples $[626688, 628736)$ (2,048 samples = 1 full window length discarded).
- `source_val`: Samples $[628736, 837632)$ (102 windows of 2048 samples; used strictly for threshold calibration).
- *Guard Gap 2:* Samples $[837632, 839680)$ (2,048 samples discarded).
- `eval` (Supported & Unsupported): Samples $[839680, 1048576)$ (102 windows of 2048 samples; evaluated strictly after model freeze).

---

## 3. CWRU Cross-Machine Testbed

- **Purpose:** Evaluate target-free selective abstention under foreign machine physics without adaptation.
- **Operating Conditions:** 1730 to 1797 RPM, motor loads 0, 1, 2, 3 HP.
- **Unit Mismatch Stress Test:** CWRU motor loads in HP are evaluated against SDUST radial force bounds in kN without unit conversion, evaluating robustness to uncalibrated telemetry.
- **Automatic Acquisition:** `run_all.py` automatically downloads the 16 required `.mat` files from Case Western Reserve University into `data/cwru/`.

---

## 4. Dataset Directory Layout

Place dataset files according to the following layout:

```text
data/
├── sdust/
│   ├── IF0.2/
│   ├── IF0.4/
│   ├── IF0.6/
│   ├── NC/
│   ├── OF0.2/
│   ├── OF0.4/
│   ├── OF0.6/
│   ├── RF0.2/
│   ├── RF0.4/
│   └── RF0.6/
└── cwru/
    ├── 97.mat
    ├── 98.mat
    ├── ...
    └── 133.mat
```
Alternatively, set the environment variable:
```bash
export SPHARE_DATASET_ROOT=/path/to/datasets
```
(On Windows: `set SPHARE_DATASET_ROOT=C:\path\to\datasets`).
