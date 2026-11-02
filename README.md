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
- [1. Overview](#1-overview)
- [2. Method Architecture](#2-method-architecture)
- [3. Repository Structure](#3-repository-structure)
- [4. System Requirements & Dependencies](#4-system-requirements--dependencies)
- [5. Quick Start (One-Command Execution)](#5-quick-start-one-command-execution)
- [6. Modular Execution Workflow](#6-modular-execution-workflow)
- [7. Benchmark Datasets & Setup](#7-benchmark-datasets--setup)
- [8. Output Artifacts & Reports](#8-output-artifacts--reports)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Citation](#10-citation)
- [11. License](#11-license)

---

## 1. Overview

Standard deep learning models for vibration-based bearing fault diagnosis operate predominantly under a closed-set, forced-classification paradigm: given an arbitrary vibration waveform $x$, the network is compelled to emit an argmax prediction over training fault classes, even when the input reflects unlearned rotational speeds, secondary load variations, or foreign machine mechanics. Under domain shift, such models output catastrophically confident but physically erroneous fault labels.

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
              - If Fail: SHORT-CIRCUIT EARLY EXIT
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
              - If Fail: SHORT-CIRCUIT EARLY EXIT
                Decision: ABSTAIN (BOT), Reason: PHYSICAL_INCONSISTENCY
                                  |
                                  v (Pass: g2(x, c) = 1; Both Gates Pass)
            [Frozen 1D-CNN Fault Classifier f_theta(x)]
              - Evaluates all W = 102 normalized windows (8.16 s duration)
              - Window-level majority voting aggregation
              - Emits Fault Diagnosis y_hat in {0: IRF, 1: NC, 2: ORF, 3: REF}
```

> [!NOTE]
> **Sequential Early-Exit Execution:** If telemetry falls outside known operating limits in Stage 1, the pipeline immediately triggers early exit and abstains from issuing a diagnosis, skipping subsequent signal windowing, spectral transformations, and neural network inference.

---

## 3. Repository Structure

```text
SPHARE-BD-GitHub/
├── run_all.py                   <- Central one-command research pipeline launcher
├── RUN_ALL.bat                  <- Windows one-click execution launcher
├── requirements.txt             <- Verified Python dependencies
├── pyproject.toml               <- Standard packaging metadata
├── LICENSE                      <- MIT Open-Source License
├── CITATION.cff                 <- Machine-readable citation metadata
├── .gitignore                   <- Git ignore rules
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
├── results/                     <- Directory where pipeline outputs are written
│   ├── csv/                     <- Clean CSV deliverables
│   ├── figures/                 <- Generated publication figures (PDF & PNG 300 DPI)
│   ├── raw/                     <- Raw recording predictions (expA_predictions.csv)
│   ├── logs/                    <- Complete execution logs
│   └── reports/                 <- Master reproducibility & verification reports
│
├── tests/                       <- Unit test suite (imports, model, gates, synthetic)
└── docs/                        <- Technical documentation (Architecture, Datasets, Reproducibility)
```

---

## 4. System Requirements & Dependencies

- **Python Version:** Python 3.9, 3.10, 3.11, or 3.12 (Tested on Python 3.11.9 64-bit).
- **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS.
- **Hardware Requirements:**
  - Standard x86_64 CPU (e.g., Intel Core i3 / AMD Ryzen 3), 4 GB RAM.
  - CUDA GPU: Supported but not required (all benchmark inference runs on CPU in < 60 seconds).

### Dependency Installation

You can install dependencies directly into your environment:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `torch >= 2.0.0`
- `numpy >= 1.22.0`
- `scipy >= 1.9.0`
- `scikit-learn >= 1.1.0`
- `pandas >= 1.4.0`
- `matplotlib >= 3.5.0`

---

## 5. Quick Start (One-Command Execution)

### Windows One-Click
Double-click `RUN_ALL.bat` in the repository root folder.

### Terminal / Command Prompt
```bash
python run_all.py
```

### Useful CLI Options
- **Use existing active Python environment (bypass virtual environment creation):**
  ```bash
  python run_all.py --no-venv
  ```
- **Run for specific random seeds:**
  ```bash
  python run_all.py --seeds 42
  ```
- **Skip pre-flight unit tests:**
  ```bash
  python run_all.py --skip-tests
  ```

### What `run_all.py` Executes Automatically:
1. **Environment Check:** Verifies Python 3.9+, initializes an isolated virtual environment (`.venv`), and installs dependencies.
2. **Dataset Check:** Verifies manifests, links existing datasets, and downloads public CWRU files if missing.
3. **Pre-flight Tests:** Executes unit tests across modules.
4. **Experiment A:** Runs SDUST in-domain and foreign CWRU evaluations across random seeds.
5. **OOD Baselines:** Recalibrates post-hoc OOD detectors (Energy, ReAct, Mahalanobis, k-NN) at matched coverage.
6. **Ablation & Benchmarks:** Runs gate ablations, synthetic experiments B & C, and latency measurements.
7. **Figures & Reports:** Renders high-resolution vector and raster publication figures and compiles reproducibility reports into `results/`.

---

## 6. Modular Execution Workflow

Each component can also be executed individually:

```bash
# 1. Run Unit Tests
python -m unittest discover tests/

# 2. Run Main Experiment A (SDUST & CWRU)
python experiments/run_experiment_a.py

# 3. Run Sequential Gate Ablation Analysis
python experiments/run_ablation.py

# 4. Recalibrate Post-Hoc OOD Baselines
python experiments/run_baselines.py

# 5. Run Controlled Synthetic Benchmarks (Exp B & C)
python experiments/run_synthetic.py

# 6. Benchmark Computational Latency
python experiments/run_latency_benchmark.py

# 7. Render Publication Figures
python src/plot_figures.py
python src/generate_ieee_figures.py
```

---

## 7. Benchmark Datasets & Setup

Detailed dataset instructions and file schemas are documented in [docs/DATASETS.md](docs/DATASETS.md) and [data/README.md](data/README.md).

### SDUST Dataset (Shandong University of Science and Technology)
- Comprises vibration acceleration recorded from deep groove ball bearings (N205EU) under speeds 1000–3000 RPM and radial loads 0.0 & 20.0 kN.
- SDUST raw files are subject to institutional data-use conditions. All split manifests and temporal indices are included in `data/manifests/sdust/`.
- If you have downloaded SDUST `.mat` files, place them in `data/sdust/` or set the environment variable:
  ```bash
  # Windows CMD
  set SPHARE_DATASET_ROOT=C:\path\to\datasets

  # Windows PowerShell
  $env:SPHARE_DATASET_ROOT="C:\path\to\datasets"

  # Linux / macOS
  export SPHARE_DATASET_ROOT="/path/to/datasets"
  ```

### CWRU Dataset (Case Western Reserve University)
- 16 held-out 12 kHz drive-end recordings across 0–3 HP motor loads.
- Downloaded automatically by `run_all.py` from the official [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter) if not already present.

---

## 8. Output Artifacts & Reports

When the pipeline finishes running, all generated artifacts are written to `results/`:

- **CSV Tables (`results/csv/`):**
  - `per_recording_outcomes.csv`: Individual predictions for each recording evaluation.
  - `stratum_coverage_rejection.csv`: Per-stratum acceptance, rejection, and decision reason codes.
  - `per_seed_metrics.csv` & `seed_aggregated_metrics.csv`: Diagnostic accuracy, coverage, and false acceptance rate summary metrics.
  - `baseline_comparison_coverage_matched_90pct.csv`: Coverage-matched baseline comparison.
  - `gate_ablation_comparison.csv`: Sequential gate ablation results.
  - `computational_latency.csv`: Execution latency breakdown.
  - `expB_synthetic_stage_ablation.csv`: Synthetic Exp B results.
  - `expC_cross_machine_comparison.csv`: Synthetic Exp C results.

- **Publication Figures (`results/figures/`):**
  - `fig1_coverage_vs_far.png` / `.pdf`: Risk-coverage curve comparison.
  - `fig1_framework.png` / `.pdf`: Architectural framework diagram.
  - `fig2_stratum_acceptance.png` / `.pdf` & `fig2_three_panel.png` / `.pdf`: Stratum acceptance distributions.
  - `fig3_gate_ablation.png` / `.pdf`: Sequential ablation breakdown.
  - `fig4_baseline_comparison_90pct.png` / `.pdf`: Matched baseline comparison.

- **Reports & Verification (`results/reports/`):**
  - `REPRODUCIBILITY_REPORT.md`: Comprehensive markdown report generated after execution.
  - `VERIFICATION_REPORT.json`: Machine-readable audit output checking all sanity checks.
  - `MASTER_REPORT.txt`: Full textual execution log.

---

## 9. Troubleshooting

- **Python Command Not Found on Windows:**
  Ensure Python is added to your Windows `PATH`, or invoke the Python launcher:
  ```cmd
  py -3 run_all.py
  ```

- **SDUST Raw Data Placement:**
  Ensure SDUST `.mat` files are organized under `data/sdust/<fault_type>/` (e.g. `data/sdust/IF0.2/IF0.2 2000 0.mat`) or specify the directory using `SPHARE_DATASET_ROOT`.

- **Virtual Environment Permission Issues:**
  Pass `--no-venv` to run using your currently active Python environment:
  ```bash
  python run_all.py --no-venv
  ```

---

## 10. Citation

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

## 11. License

- **Software Code:** Licensed under the [MIT License](LICENSE).
- **Checkpoints & Manifests:** Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Third-Party Datasets:** CWRU and SDUST datasets remain subject to their respective institutional terms of use.
