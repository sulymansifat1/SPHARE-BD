================================================================================
SPHARE-BD: RESEARCH CODE & ARTIFACT REPRODUCIBILITY GUIDE
================================================================================

1. PAPER TITLE AND PROJECT DESCRIPTION
--------------------------------------------------------------------------------
Title:
  Hierarchical Support Assessment for Selective Bearing Fault Diagnosis Under
  Operating-Condition Shifts

Authors:
  Md. Sulyman Islam Sifat and Md Alamgir Kabir
  Department of Computer Science and Engineering,
  Daffodil International University, Dhaka, Bangladesh.

Abstract & Project Description:
  Deep learning models for vibration-based bearing fault diagnosis typically
  operate under a closed-set, forced-classification assumption. When deployed
  under unlearned operating conditions (such as unobserved rotational speeds,
  load variations, or foreign machine mechanics), these models often output
  catastrophically confident but erroneous diagnostic predictions.

  This project provides the official implementation of SPHARE-BD, a source-only
  framework that reformulates bearing diagnostics into a selective decision
  problem. SPHARE-BD screens inputs through a two-stage hierarchical support
  assessment before invoking a neural classifier:
    1. Stage 1 (Operating-Condition Support Gate): Screens operational telemetry
       (rotational speed and radial load) against the calibrated convex source
       boundary to prevent operational extrapolation.
    2. Stage 2 (Order-Domain Spectral Envelope Gate): Verifies whether the
       vibration signal's rotational order spectrum conforms to calibrated
       kinematic tolerance envelopes.

  A frozen 1D-CNN classifier evaluates a recording if and only if both support
  stages pass; otherwise, the framework triggers an early-exit safe abstention
  (bot), suppressing false acceptances on unsupported operational regimes.


2. GITHUB REPOSITORY URL
--------------------------------------------------------------------------------
Repository URL:
  https://github.com/sulymansifat1/SPHARE-BD

Clone Command:
  git clone https://github.com/sulymansifat1/SPHARE-BD.git
  cd SPHARE-BD


3. REPOSITORY STRUCTURE
--------------------------------------------------------------------------------
SPHARE-BD/
|-- run_all.py                   # Central automated pipeline orchestrator
|-- RUN_ALL.bat                  # Windows one-click desktop launcher
|-- quickstart.ipynb             # Interactive Jupyter / VS Code notebook
|-- requirements.txt             # Verified Python package dependencies
|-- pyproject.toml               # Standard packaging and build configuration
|-- LICENSE                      # MIT Open-Source License
|-- CITATION.cff                 # Machine-readable citation metadata
|-- .gitignore                   # Version control ignore specifications
|
|-- checkpoints/                 # Verified frozen model checkpoints
|   |-- classifier_seed42.pt     # Frozen 1D-CNN backbone (Seed 42)
|   |-- classifier_seed123.pt    # Frozen 1D-CNN backbone (Seed 123)
|   `-- classifier_seed2024.pt   # Frozen 1D-CNN backbone (Seed 2024)
|
|-- configs/                     # Immutable configuration files
|   |-- experiment_a.json        # Main experiment parameters & hyperparameters
|   |-- synthetic_signal_config.json # Synthetic benchmark signal definitions
|   `-- freeze/                  # Frozen support envelopes and scale statistics
|       |-- EXPERIMENT_A_FROZEN_seed42.json
|       |-- EXPERIMENT_A_FROZEN_seed123.json
|       `-- EXPERIMENT_A_FROZEN_seed2024.json
|
|-- data/
|   |-- README.md                # Dataset placement and indexing guidelines
|   |-- manifests/               # Deterministic CSV slice manifests
|   |   |-- sdust/               # SDUST split manifests (eval and source)
|   |   `-- cwru/                # CWRU cross-machine evaluation manifest
|   |-- baselines/               # Precomputed baseline sweep cache
|   |   `-- risk_coverage_sweep.csv
|   `-- synthetic/               # Synthetic benchmark reference datasets
|       |-- expB_synthetic_stage_ablation.csv
|       `-- expC_cross_machine_comparison.csv
|
|-- docs/                        # Theoretical and technical documentation
|   |-- ARCHITECTURE.md          # Formal derivation of sequential gates
|   |-- DATASETS.md              # Dataset specifications and protocols
|   |-- REPRODUCIBILITY.md       # Reproducibility guidelines & verification
|   `-- figures/                 # Architecture diagram (PNG & vector PDF)
|       |-- SPHARE-BD_FRAMEWORK.pdf
|       `-- SPHARE-BD_FRAMEWORK.png
|
|-- experiments/                 # Modular single-command experiment scripts
|   |-- run_proposed_model.py    # Proposed SPHARE-BD evaluation (SDUST & CWRU)
|   |-- run_ablation.py          # Sequential gate ablation
|   |-- run_baselines.py         # Coverage-matched post-hoc OOD baselines
|   |-- run_synthetic.py         # Controlled synthetic benchmarks (Exp B & C)
|   `-- run_latency.py           # CPU latency and computational efficiency
|
|-- src/                         # Modular framework source code
|   |-- model.py                 # Frozen 1D-CNN fault classification backbone
|   |-- inference_engine.py      # Sequential short-circuiting inference engine
|   |-- plot_figures.py          # Publication figure generation routines
|   |-- generate_ieee_figures.py # Vector framework and summary figure scripts
|   |-- datasets/                # Adapters for SDUST, CWRU, HDF5, and data manager
|   |-- preprocessing/           # Demeaning, global scaling, and windowing
|   |-- support/                 # Convex hull and order spectral envelope gates
|   |-- baselines/               # Energy, KNN, Mahalanobis, and ReAct detectors
|   `-- synthetic/               # Synthetic signal generation modules
|
`-- tests/                       # Unit test suite verifying modules and logic


4. PYTHON & DEPENDENCY REQUIREMENTS
--------------------------------------------------------------------------------
Compatible Operating Systems:
  - Windows 10/11 (64-bit)
  - Linux (Ubuntu 20.04 / 22.04 LTS or compatible distribution)
  - macOS (Intel & Apple Silicon)

Supported Python Versions:
  - Python 3.9, 3.10, 3.11, or 3.12 (Tested on Python 3.11.9 64-bit)

Hardware Requirements:
  - Minimum: Standard x86_64 CPU (e.g., Intel Core i3 / AMD Ryzen 3), 4 GB RAM.
  - GPU: CUDA-enabled GPU supported but optional (CPU decision latency is ~27 ms).

Installation:
  1. Create and activate a clean virtual environment (recommended):
       python -m venv .venv
       # Windows:
       .venv\Scripts\activate
       # Linux / macOS:
       source .venv/bin/activate

  2. Install required dependencies:
       pip install -r requirements.txt

Core Package Dependencies:
  - torch >= 2.0.0
  - numpy >= 1.23.0
  - scipy >= 1.9.0
  - scikit-learn >= 1.2.0
  - pandas >= 1.5.0
  - matplotlib >= 3.6.0
  - h5py >= 3.8.0


5. DATASET DOWNLOAD AND ACCESS INSTRUCTIONS
--------------------------------------------------------------------------------
1. SDUST Bearing Dataset (Shandong University of Science and Technology):
   - Scope: Deep groove ball bearings (N205EU) under operating speeds from
     1000 to 3000 RPM and radial static loads of 0.0 and 20.0 kN.
   - Access: Redistribution of full raw industrial vibration recordings is
     subject to institutional terms. All deterministic evaluation split
     manifests and temporal slice indices are pre-packaged in
     `data/manifests/sdust/`.
   - Setup: Place raw `.mat` files in `data/sdust/<fault_type>/` (e.g.,
     `data/sdust/IF0.2/IF0.2 2000 0.mat`) or specify an external directory:
       Windows (CMD):        set SPHARE_DATASET_ROOT=C:\path\to\datasets
       Windows (PowerShell): $env:SPHARE_DATASET_ROOT="C:\path\to\datasets"
       Linux / macOS:        export SPHARE_DATASET_ROOT="/path/to/datasets"

2. CWRU Bearing Dataset (Case Western Reserve University):
   - Scope: 16 cross-machine evaluation recordings (12 kHz drive-end) across
     0 to 3 HP motor loads.
   - Access: Publicly available from the CWRU Bearing Data Center.
   - Automatic Acquisition: `run_all.py` automatically downloads the necessary
     `.mat` files directly from the official repository if not present locally.

3. Synthetic Benchmark Datasets:
   - Scope: Exp B (physics-informed operating condition shifts, 15,876 cases)
     and Exp C (cross-machine parameter transfer, 9,990 cases).
   - Setup: Pre-computed reference evaluation metrics are packaged directly
     in `data/synthetic/`.


6. EXECUTION INSTRUCTIONS
--------------------------------------------------------------------------------
Option A: One-Command Automated Execution (Full Research Pipeline)
  - Windows:
      Double-click `RUN_ALL.bat` or run in Command Prompt:
      RUN_ALL.bat

  - Linux / macOS / Terminal:
      python run_all.py

  This automatically checks the environment, verifies checkpoints, runs
  Experiment A, executes gate ablations, recalibrates OOD baselines, verifies
  synthetic benchmarks, measures latency, and generates all publication figures
  and summary tables into the local `results/` folder.

Option B: Interactive Notebook Execution (VS Code / Jupyter)
  1. Open the repository root folder in VS Code or Jupyter Lab.
  2. Open `quickstart.ipynb`.
  3. Select your Python interpreter and click "Run All" (or execute step-by-step).

Option C: Modular Reviewer Execution (Single Dedicated Command per Experiment)
  Reviewers can execute individual experiments with a single command:

  1. Proposed SPHARE-BD Model (SDUST & CWRU across seeds):
       python experiments/run_proposed_model.py
       # (Alternative: python run_all.py --only model)

  2. Sequential Gate Ablation (Classifier-only, Stage 1, Stages 1+2):
       python experiments/run_ablation.py
       # (Alternative: python run_all.py --only ablation)

  3. Coverage-Matched OOD Baselines (Energy, KNN, Mahalanobis, ReAct, MSP):
       python experiments/run_baselines.py
       # (Alternative: python run_all.py --only baselines)

  4. Controlled Synthetic Benchmarks (Exp B & Exp C):
       python experiments/run_synthetic.py
       # (Alternative: python run_all.py --only synthetic)

  5. Computational Latency Profiling (CPU decision execution times):
       python experiments/run_latency.py
       # (Alternative: python run_all.py --only latency)


7. EXPECTED OUTPUTS AND GENERATED FIGURES
--------------------------------------------------------------------------------
When executed, output files are generated in structured subdirectories inside
`results/`:

CSV Result Tables (`results/csv/`):
  - `per_recording_outcomes.csv`: Individual predictions for all 288 evaluations
    across 3 random seeds (42, 123, 2024).
  - `stratum_coverage_rejection.csv`: Per-stratum acceptance rates, rejection
    fractions, and exit stage distributions.
  - `per_seed_metrics.csv`: Supported coverage, diagnostic accuracy, and FAR
    per random seed.
  - `seed_aggregated_metrics.csv`: Mean and standard deviation across seeds.
  - `baseline_comparison_coverage_matched_90pct.csv`: Comparative performance
    against post-hoc OOD baselines at matched ~90% supported coverage.
  - `gate_ablation_comparison.csv`: Impact of bypassing Stage 1 or Stage 2.
  - `computational_latency.csv`: Execution latency and real-time speed factors.
  - `expB_synthetic_stage_ablation.csv`: Synthetic Exp B results.
  - `expC_cross_machine_comparison.csv`: Synthetic Exp C results.

Publication Figures (`results/figures/`):
  - `fig1_coverage_vs_far.png` / `.pdf`: Risk-coverage tradeoff curves.
  - `fig1_framework.png` / `.pdf`: Architectural workflow diagram.
  - `fig2_stratum_acceptance.png` / `.pdf` & `fig2_three_panel.png` / `.pdf`:
    Stratum-wise acceptance, baseline comparison, and ablation summary.
  - `fig3_gate_ablation.png` / `.pdf`: Sequential gate ablation bar plot.
  - `fig4_baseline_comparison_90pct.png` / `.pdf`: Matched baseline bar chart.

Audit & Reproducibility Reports (`results/reports/`):
  - `VERIFICATION_REPORT.json`: Machine-readable verification audit evaluating
    10 strict sanity criteria against the manuscript numbers.
  - `MASTER_REPORT.txt` & `REPRODUCIBILITY_REPORT.md`: Comprehensive text and
    markdown execution summaries.


8. REPRODUCIBILITY & CONTACT INFORMATION
--------------------------------------------------------------------------------
Reproducibility Guarantee:
  - Evaluation uses fixed random seeds (42, 123, 2024).
  - All frozen model weights (51,028 parameters each), calibrated support
    bounds, and ANOVA feature subsets are version-controlled in `checkpoints/`
    and `configs/`.
  - The pipeline does not require out-of-distribution training data or stochastic
    retraining, ensuring exact numerical reproducibility.

Contact Information:
  - Md. Sulyman Islam Sifat: sifat15-4004@diu.edu.bd
  - Md Alamgir Kabir:        kabir.cse@diu.edu.bd
  Department of Computer Science and Engineering,
  Daffodil International University, Dhaka, Bangladesh.

License:
  - Software code is distributed under the MIT Open-Source License.
  - Model checkpoints, configurations, and manifests are licensed under CC BY 4.0.
================================================================================
