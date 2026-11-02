# SPHARE-BD: Hierarchical Support Assessment for Selective Bearing Fault Diagnosis Under Operating-Condition Shifts

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official research code and reproducibility repository for:

> **"Hierarchical Support Assessment for Selective Bearing Fault Diagnosis Under Operating-Condition Shifts"**  
> *Md. Sulyman Islam Sifat and Md Alamgir Kabir*  
> Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh.

---

## Architecture

SPHARE-BD introduces a hierarchical support-assessment architecture ($A(x, c) = g_1(c) \cdot g_2(x, c)$) with sequential early-exit short-circuiting for selective vibration-based bearing fault diagnosis:

<p align="center">
  <img src="docs/figures/SPHARE-BD_FRAMEWORK.png" alt="SPHARE-BD Two-Stage Architecture" width="100%">
</p>

<p align="center">
  <a href="docs/figures/SPHARE-BD_FRAMEWORK.pdf"><b>📄 Download High-Resolution Architecture Vector PDF</b></a>
</p>

### Sequential Decision Pipeline
1. **Stage 1 (Operating-Condition Support Gate $g_1(c)$):** Screens operational telemetry $c = [s, L]^T$ (speed $s$, load $L$) against the calibrated convex support boundary. Inputs failing Stage 1 exit immediately with early abstention ($\bot$).
2. **Conditional Signal Preprocessing & Windowing:** Upon passing Stage 1, vibration signal $x[n]$ is segmented into standardized non-overlapping windows and normalized using training scale statistics.
3. **Stage 2 (Order-Domain Envelope Gate $g_2(x, c)$):** Maps vibration signals to kinematic order spectra and checks physical spectral consistency against regional tolerance envelopes. Inconsistent signatures trigger early abstention ($\bot$).
4. **Frozen 1D-CNN Fault Classifier ($f_\theta(x)$):** Only inputs that pass both physical support stages are evaluated by the classifier, with window-level majority voting issuing the final health state diagnosis.

---

## Supported Environments & Platforms

- **Operating Systems:** Windows 10/11, Linux (Ubuntu 20.04/22.04 LTS), macOS.
- **Python Environments:** Python 3.9, 3.10, 3.11, or 3.12 (Tested on Python 3.11.9 64-bit).
- **IDEs & Notebooks:** VS Code, Jupyter Lab, Jupyter Notebook, or standard terminal / shell.
- **Hardware:** Standard x86_64 CPU (GPU supported but not required; full evaluation runs in seconds on CPU).

---

## Quick Start (One-Command Execution)

Clone the repository:
```bash
git clone https://github.com/sulymansifat1/SPHARE-BD.git
cd SPHARE-BD
```

Install requirements:
```bash
pip install -r requirements.txt
```

### Run the Full Pipeline
- **Windows (One-Click):** Double-click `RUN_ALL.bat` or run `RUN_ALL.bat` in CMD.
- **Terminal (Linux / macOS / Windows):**
  ```bash
  python run_all.py
  ```

---

## Reviewer Guide: Running Individual Experiments

Reviewers and researchers can run any specific experiment from the manuscript with a single dedicated command:

| Experiment in Manuscript | Single Execution Command | CLI Pipeline Alternative |
| :--- | :--- | :--- |
| **1. Proposed SPHARE-BD Model** <br> *(Evaluates SDUST in-domain, shifted strata, and foreign CWRU)* | `python experiments/run_proposed_model.py` | `python run_all.py --only model` |
| **2. Sequential Gate Ablation** <br> *(Classifier-only, Stage 1 only, Stages 1+2)* | `python experiments/run_ablation.py` | `python run_all.py --only ablation` |
| **3. Post-Hoc OOD Baselines** <br> *(Recalibrates Energy, KNN, Mahalanobis, ReAct, MSP @ 90% cov.)* | `python experiments/run_baselines.py` | `python run_all.py --only baselines` |
| **4. Controlled Synthetic Benchmarks** <br> *(Exp B: condition shifts, Exp C: machine parameter transfer)* | `python experiments/run_synthetic.py` | `python run_all.py --only synthetic` |
| **5. Computational Latency** <br> *(Per-stage CPU execution time and real-time factor)* | `python experiments/run_latency.py` | `python run_all.py --only latency` |

---

## Interactive Jupyter Notebook (VS Code / Jupyter)

For an interactive, cell-by-cell walkthrough, open [`quickstart.ipynb`](quickstart.ipynb) in **VS Code** or **Jupyter Notebook**:
1. Open the repository folder in VS Code (`File > Open Folder`).
2. Open `quickstart.ipynb`.
3. Select your Python interpreter and click **Run All** (or run cells step-by-step).

---

## Dataset Organization

Detailed instructions and checksums are provided in [data/README.md](data/README.md) and [docs/DATASETS.md](docs/DATASETS.md).

- **SDUST Dataset:** Deep groove ball bearing vibration recordings across speeds (1000–3000 RPM) and radial loads (0.0 & 20.0 kN). All split manifests are pre-indexed in `data/manifests/sdust/`. Place raw `.mat` files in `data/sdust/` or export the dataset directory:
  ```bash
  # Windows CMD:
  set SPHARE_DATASET_ROOT=C:\path\to\datasets
  # PowerShell:
  $env:SPHARE_DATASET_ROOT="C:\path\to\datasets"
  # Linux / macOS:
  export SPHARE_DATASET_ROOT="/path/to/datasets"
  ```
- **CWRU Dataset:** 16 cross-machine evaluation recordings (12 kHz drive-end). Downloaded automatically by `run_all.py` from the Case Western Reserve University Bearing Data Center if not locally present.

---

## Repository Structure

```text
SPHARE-BD/
├── run_all.py                   # Central pipeline launcher
├── RUN_ALL.bat                  # Windows one-click launcher
├── quickstart.ipynb             # Interactive Jupyter / VS Code notebook
├── requirements.txt             # Verified Python dependencies
├── pyproject.toml               # Package build configuration
├── LICENSE                      # MIT Open-Source License
├── CITATION.cff                 # Citation metadata
├── .gitignore                   # Git ignore specifications
│
├── checkpoints/                 # Verified frozen model checkpoints
│   ├── classifier_seed42.pt
│   ├── classifier_seed123.pt
│   └── classifier_seed2024.pt
│
├── configs/                     # Experiment configurations
│   ├── experiment_a.json
│   ├── synthetic_signal_config.json
│   └── freeze/                  # Frozen support envelopes & scale parameters
│
├── data/
│   ├── README.md                # Dataset placement and schema guide
│   ├── manifests/               # Deterministic split manifests (SDUST & CWRU)
│   └── baselines/               # Baseline sweep cache
│
├── docs/                        # Technical documentation
│   ├── figures/                 # Architecture diagram (PNG & vector PDF)
│   ├── ARCHITECTURE.md          # Theoretical derivation of support gates
│   ├── DATASETS.md              # Dataset specifications & acquisition
│   └── REPRODUCIBILITY.md       # Validation protocol and sanity checks
│
├── experiments/                 # Modular single-command experiment scripts
│   ├── run_proposed_model.py    # Proposed SPHARE-BD evaluation
│   ├── run_ablation.py          # Sequential gate ablation
│   ├── run_baselines.py         # Coverage-matched OOD baselines
│   ├── run_synthetic.py         # Synthetic benchmarks (Exp B & C)
│   └── run_latency.py           # CPU latency profiling
│
├── src/                         # Core library
│   ├── model.py                 # Frozen 1D-CNN backbone
│   ├── inference_engine.py      # Sequential short-circuiting inference engine
│   ├── plot_figures.py          # Figure rendering routines
│   ├── generate_ieee_figures.py # Manuscript figures
│   ├── datasets/                # Dataset adapters and slice loaders
│   ├── preprocessing/           # Signal windowing & normalization
│   ├── support/                 # Stage 1 and Stage 2 support gates
│   ├── baselines/               # Post-hoc OOD baseline detectors
│   └── synthetic/               # Physics-informed synthetic signal generators
│
└── tests/                       # Unit test suite
```

---

## Citation

```bibtex
@article{sifat2026sphare,
  author    = {Sifat, Md. Sulyman Islam and Kabir, Md Alamgir},
  title     = {Hierarchical Support Assessment for Selective Bearing Fault Diagnosis Under Operating-Condition Shifts},
  year      = {2026}
}
```

---

## License

- **Source Code:** [MIT License](LICENSE)
- **Checkpoints & Manifests:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Datasets:** SDUST and CWRU datasets are governed by their respective institutional terms.
