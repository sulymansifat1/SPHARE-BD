# SPHARE-BD: Hierarchical Support Assessment for Selective Bearing Fault Diagnosis Under Operating-Condition Shifts

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Venue: IEEE SPL](https://img.shields.io/badge/IEEE-Signal%20Processing%20Letters-00629B.svg)](https://signalprocessingsociety.org/publications-resources/ieee-signal-processing-letters)

Official research code and reproducibility repository for:

> **"Hierarchical Support Assessment for Selective Bearing Fault Diagnosis Under Operating-Condition Shifts"**  
> *Md. Sulyman Islam Sifat and Md Alamgir Kabir (Senior Member, IEEE)*  
> **IEEE Signal Processing Letters (2026)**  
> Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh.

---

## Table of Contents
- [1. Overview & Research Postulate](#1-overview--research-postulate)
- [2. Method Architecture](#2-method-architecture)
- [3. Key Experimental Findings](#3-key-experimental-findings)
- [4. Repository Structure](#4-repository-structure)
- [5. System Requirements](#5-system-requirements)
- [6. One-Command Reproduction](#6-one-command-reproduction)
- [7. Modular Execution Workflow](#7-modular-execution-workflow)
- [8. Benchmark Datasets & Acquisition](#8-benchmark-datasets--acquisition)
- [9. Output Locations & Verification Checklist](#9-output-locations--verification-checklist)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Citation](#11-citation)
- [12. License & Data Use](#12-license--data-use)

---

## 1. Overview & Research Postulate

Standard deep learning models for vibration-based bearing fault diagnosis operate predominantly under a **closed-set, forced-classification paradigm**: given an arbitrary vibration waveform $x$, the network is compelled to emit an argmax prediction over training fault classes, even when the input reflects unlearned rotational speeds, secondary load variations, or foreign machine mechanics. Under domain shift, such models output catastrophically confident but physically erroneous fault labels.

**SPHARE-BD** reformulates bearing diagnostics into a target-free, source-only selective classification problem:
$$\hat{y}(x, c) = \begin{cases} f_\theta(x), & \text{if } A(x, c) = 1, \\ \bot, & \text{if } A(x, c) = 0 \quad (\text{Safe Abstention}), \end{cases}$$
where a fault diagnosis is issued **if and only if** the recording is proven to be physically and statistically supported by the source operating regime; otherwise, the framework withholds diagnosis ($\bot$).

---

## 2. Method Architecture

SPHARE-BD implements a **Two-Stage Hierarchical Support-Assessment Architecture** ($A(x, c) = g_1(c) \cdot g_2(x, c)$) with strict sequential early-exit short-circuiting:

```text
[Raw Input: Vibration Signal x[n] + Operational Telemetry c = [s, L]^T]
                                  |
                                  v
            [Stage 1: Operating-Condition Support Gate g1(c)]
              - Screens telemetry c = [s, L]^T against source support geometry
              - Delaunay 2D convex hull (planar) or collinear bounding box (SDUST)
              - Active SDUST bounds: s in [1400, 2600] RPM, L in [-0.15, 0.15] kN
              - If Fail: SHORT-CIRCUIT EARLY EXIT (0.14 ms CPU latency)
                Decision: ABSTAIN (BOT), Reason: CONDITION_AXIS_OUT_OF_SUPPORT
                                  |
                                  v (Pass: g1(c) = 1)
            [Conditional Preprocessing & Windowing]
              - Window length: Lw = round(fs * 0.08 s) = 2,048 samples (at 25.6 kHz)
              - Non-overlapping extraction (100% hop)
              - Demeaning: w_zero = w - mean(w)
              - Global scaling: w_norm = w_zero / sigma_train (sigma = 3.89934)
              - K = 16 support windows subsampled uniformly
                                  |
                                  v
            [Stage 2: Kinematic Order-Domain Spectral Envelope Gate g2(x, c)]
              - Tracks rotational speed: fr = s / 60
              - Order mapping: o_k = f_k / fr across orders [0, 384]
              - 11 physical spectral descriptors (8 order bins + log-total + log-RMS + low-order)
              - Screened regional tolerance envelopes: lo_k <= phi_k <= hi_k
              - If Fail: SHORT-CIRCUIT EARLY EXIT (5.40 ms CPU latency)
                Decision: ABSTAIN (BOT), Reason: PHYSICAL_INCONSISTENCY
                                  |
                                  v (Pass: g2(x, c) = 1; Both Gates Pass)
            [Frozen 1D-CNN Fault Classifier f_theta(x)]
              - Evaluates all W = 102 normalized windows (8.16 s duration)
              - Window-level majority voting aggregation
              - Latency: 22.06 ms CPU
              - Emits Fault Diagnosis y_hat in {0: IRF, 1: NC, 2: ORF, 3: REF}
```

> [!NOTE]
> **Stage 3 Excision:** The historical three-stage prototype included a deep feature manifold gate (Stage 3). Rigorous empirical auditing demonstrated that Stage 3 provided 0.0% unique OOD rejection benefit on unsupported shifts while depressing supported coverage from 90.0% down to 66.7% (falsely rejecting all REF recordings in nominal interior stratum `IN`). Stage 3 is therefore excised from active inference and retained strictly as an ablation baseline.

---

## 3. Key Experimental Findings

### 3.1 Main Results (SDUST & CWRU Benchmarks)
Evaluated across three random seeds (42, 123, 2024; 288 total evaluations):

| Method / Gate Configuration | Supported Coverage (%) $\uparrow$ | Accepted Diagnostic Accuracy (%) $\uparrow$ | SDUST Unsupported FAR (%) $\downarrow$ | CWRU Cross-Machine FAR (%) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **Forced Classifiers (Ungated)** | | | | |
| 1D-CNN (Kiranyaz et al. 2021) | 100.0 | 100.0 | 100.0 | 100.0 |
| LiConvFormer (Ding et al. 2024) | 100.0 | 100.0 | 100.0 | 100.0 |
| BDC (Zhang et al. 2024) | 100.0 | 98.1 | 100.0 | --- |
| SPHARE-BD backbone (ungated) | 100.0 | 100.0 | 100.0 | 100.0 |
| **Selective Classification Baseline** | | | | |
| CNN-1D Confidence (MSP @ 5th percentile) | 90.0 | 100.0 | 58.0 | 50.0 |
| **Coverage-Matched Post-Hoc OOD Baselines ($\approx 90\%$ Coverage)** | | | | |
| KNN-OOD ($k=50$, Sun et al. 2022) | 90.0 | 100.0 | 54.0 | 0.0 |
| Deep Mahalanobis (Ren et al. 2021) | 91.1 | 100.0 | 54.7 | 91.7 |
| Energy-OOD (Liu et al. 2020) | 91.1 | 100.0 | 70.7 | 16.7 |
| ReAct-Energy (Sun et al. 2021) | 90.0 | 100.0 | 74.0 | 33.3 |
| **Proposed SPHARE-BD (Stages 1+2)** | **90.0** | **100.0** | **0.00** | **0.00** |

### 3.2 Sequential Gate Ablation
| Configuration | Supported Coverage (%) | SDUST FAR (%) | CWRU FAR (%) | Safety Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Classifier only** | 100.0 | 100.0 | 100.0 | Zero safety; 100% false acceptance on all shifts |
| **Stage 1 only** | 100.0 | 0.0 | 25.0 | Telemetry bounds eliminate SDUST FAR; admits 4 CWRU files @ 0 HP |
| **Stages 1+2 (Proposed)** | **90.0** | **0.0** | **0.0** | Order spectral envelope eliminates foreign CWRU; optimal safety |
| **Full SPHARE-BD (Hist. 3-Stage)**| 66.7 | 0.0 | 0.0 | Stage 3 adds 7 false rejections per seed without extra OOD benefit |

### 3.3 Computational Latency (Intel Core i3-1115G4 CPU)
- **Stage 1 Support Gate:** $0.14\text{ ms}$ median ($>45,000\times$ faster than real-time)
- **Stage 2 Spectral Envelope Gate:** $5.40\text{ ms}$ median ($>1,400\times$ faster than real-time)
- **Frozen 1D-CNN (102 windows, 8.16 s):** $22.06\text{ ms}$ median ($>310\times$ faster than real-time)
- **Total In-Memory Decision:** $27.60\text{ ms}$ median ($>295\times$ faster than real-time)

---

## 4. Repository Structure

```text
SPHARE-BD-GitHub/
├── run_all.py                   <- Central one-command research pipeline launcher
├── RUN_ALL.bat                  <- Windows one-click execution launcher
├── requirements.txt             <- Verified Python dependencies
├── pyproject.toml               <- Standard packaging metadata
├── LICENSE                      <- MIT Open-Source License
├── CITATION.cff                 <- Machine-readable citation metadata
├── .gitignore                   <- Robust Git ignore rules
│
├── configs/
│   ├── experiment_a.json        <- Experiment A configuration
│   ├── synthetic_signal_config.json <- Synthetic signal generation configuration
│   └── freeze/                  <- Frozen support envelopes, scale, and ANOVA features
│       ├── EXPERIMENT_A_FROZEN_seed42.json
│       ├── EXPERIMENT_A_FROZEN_seed123.json
│       └── EXPERIMENT_A_FROZEN_seed2024.json
│
├── src/
│   ├── model.py                 <- Frozen 1D-CNN architecture (51,028 parameters)
│   ├── inference_engine.py      <- SPHARE-BD sequential short-circuit engine
│   ├── plot_figures.py          <- Generates Figs 1-4 publication charts
│   ├── generate_ieee_figures.py <- Generates IEEE manuscript Figs 1 & 2
│   ├── datasets/                <- Data adapters, slice loader, and DataManager
│   ├── preprocessing/           <- Demeaning, scaling (s=3.89934), windowing
│   ├── support/                 <- Stage 1, Stage 2, and short-circuit conjunction
│   ├── baselines/               <- Energy, KNN, Mahalanobis, and ReAct baselines
│   └── synthetic/               <- Synthetic physics-informed signal generators
│
├── experiments/
│   ├── run_experiment_a.py      <- Master Experiment A script (SDUST & CWRU)
│   ├── run_ablation.py          <- Sequential gate ablation
│   ├── run_baselines.py         <- Coverage-matched OOD baseline recalibration
│   ├── run_synthetic.py         <- Controlled synthetic benchmarks (Exp B & C)
│   └── run_latency_benchmark.py <- Computational latency benchmarking
│
├── checkpoints/
│   ├── classifier_seed42.pt     <- Frozen 1D-CNN weights (Seed 42)
│   ├── classifier_seed123.pt    <- Frozen 1D-CNN weights (Seed 123)
│   └── classifier_seed2024.pt   <- Frozen 1D-CNN weights (Seed 2024)
│
├── data/
│   ├── README.md                <- Dataset specifications and placement guide
│   ├── manifests/
│   │   ├── sdust/               <- Verified SDUST split manifests
│   │   └── cwru/                <- CWRU cross-machine evaluation manifest
│   └── baselines/
│       └── risk_coverage_sweep.csv <- Authoritative baseline sweep cache
│
├── results/
│   ├── csv/                     <- Clean CSV deliverables
│   ├── figures/                 <- Publication figures (PDF & PNG 300 DPI)
│   ├── raw/                     <- Raw recording predictions (expA_predictions.csv)
│   ├── logs/                    <- Complete execution logs
│   └── reports/                 <- Master reproducibility & verification reports
│
├── tests/                       <- Unit test suite (imports, model, gates, synthetic)
└── docs/                        <- Technical documentation (Architecture, Datasets, Reproducibility)
```

---

## 5. System Requirements

- **Python Version:** Python 3.9, 3.10, 3.11, or 3.12 (Tested on Python 3.11.9 64-bit).
- **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS.
- **Hardware Requirements:**
  - Minimum: Standard x86_64 CPU (e.g., Intel Core i3 / AMD Ryzen 3), 4 GB RAM.
  - CUDA GPU: Supported but not required (all benchmark inference runs on CPU in < 50 seconds).

---

## 6. One-Command Reproduction

### Windows One-Click
Double-click `RUN_ALL.bat` in the repository root.

### Terminal / Command Prompt
```bash
python run_all.py
```

### What Happens Automatically:
1. **Environment Check:** Verifies Python 3.9+, initializes an isolated `.venv`, and installs dependencies.
2. **Dataset Check:** Verifies manifests, links existing datasets, and automatically downloads public CWRU files.
3. **Model Verification:** Validates frozen checkpoints and exact parameter counts.
4. **Experiment A:** Runs SDUST in-domain and foreign CWRU evaluations across seeds 42, 123, 2024.
5. **OOD Baselines:** Recalibrates post-hoc OOD detectors at matched ~90% supported coverage.
6. **Ablation & Benchmarks:** Runs gate ablations, synthetic experiments B & C, and latency measurements.
7. **Figures & Reports:** Renders 300 DPI publication vector graphics and compiles `REPRODUCIBILITY_REPORT.md`.

---

## 7. Modular Execution Workflow

Researchers can also invoke modular experiment scripts independently:

```bash
# 1. Run Main Experiment A (SDUST & CWRU)
python experiments/run_experiment_a.py

# 2. Run Gate Ablation Analysis
python experiments/run_ablation.py

# 3. Recalibrate OOD Baselines
python experiments/run_baselines.py

# 4. Run Controlled Synthetic Benchmarks
python experiments/run_synthetic.py

# 5. Measure Computational Latency
python experiments/run_latency_benchmark.py

# 6. Render Publication Figures
python src/plot_figures.py
python src/generate_ieee_figures.py

# 7. Run Unit Tests
python -m unittest discover tests/
```

---

## 8. Benchmark Datasets & Acquisition

### SDUST Dataset (Shandong University of Science and Technology)
- Comprises vibration acceleration recorded from deep groove ball bearings (N205EU) under speeds 1000–3000 RPM and radial loads 0.0 & 20.0 kN.
- SDUST raw files are subject to institutional data-use conditions. All split manifests and temporal indices are included in `data/manifests/sdust/`.
- If you have downloaded SDUST `.mat` files, place them in `data/sdust/` or set the environment variable:
  ```bash
  set SPHARE_DATASET_ROOT=C:\path\to\datasets
  ```

### CWRU Dataset (Case Western Reserve University)
- 16 held-out 12 kHz drive-end recordings across 0–3 HP motor loads.
- Downloaded automatically by `run_all.py` from the official [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter).

---

## 9. Output Locations & Verification Checklist

All generated outputs are saved to structured subdirectories inside `results/`:
- `results/csv/`:
  - `per_recording_outcomes.csv`: Individual predictions for all 288 evaluations across 3 seeds.
  - `stratum_coverage_rejection.csv`: Per-stratum acceptance, rejection, and reason codes.
  - `per_seed_metrics.csv` & `seed_aggregated_metrics.csv`: Diagnostic accuracy, coverage, and FAR.
  - `baseline_comparison_coverage_matched_90pct.csv`: Coverage-matched baseline metrics.
  - `gate_ablation_comparison.csv`: Sequential gate ablation results.
  - `computational_latency.csv`: Execution latency breakdown.
- `results/figures/`: Publication vector PDFs and 300 DPI PNGs.
- `results/reports/`:
  - `REPRODUCIBILITY_REPORT.md`: Comprehensive markdown report.
  - `VERIFICATION_REPORT.json`: Machine-readable audit against all 10 sanity checks.

---

## 10. Troubleshooting

- **Python Not Found on Windows:** Ensure Python is added to your system `PATH`, or run `py -3 run_all.py`.
- **Missing SDUST Files:** Ensure SDUST `.mat` files are placed under `data/sdust/<fault_type>/` (e.g. `data/sdust/IF0.2/IF0.2 2000 0.mat`) or configure `SPHARE_DATASET_ROOT`.
- **Virtual Environment Permission Issues:** Pass `--no-venv` to run using your currently active Python environment:
  ```bash
  python run_all.py --no-venv
  ```

---

## 11. Citation

If you use SPHARE-BD in your research, please cite our IEEE Signal Processing Letters paper:

```bibtex
@article{sifat2026sphare,
  author    = {Sifat, Md. Sulyman Islam and Kabir, Md Alamgir},
  title     = {Hierarchical Support Assessment for Selective Bearing Fault Diagnosis Under Operating-Condition Shifts},
  journal   = {IEEE Signal Processing Letters},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## 12. License & Data Use

- **Software Code:** Licensed under the [MIT License](LICENSE).
- **Checkpoints & Manifests:** Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Third-Party Data:** CWRU and SDUST datasets remain subject to their respective institutional terms of use.
