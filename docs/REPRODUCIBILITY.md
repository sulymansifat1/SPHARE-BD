# SPHARE-BD Step-by-Step Reproduction Guide

This guide describes how to reproduce the complete experimental findings reported in our IEEE Signal Processing Letters manuscript.

---

## 1. Quick Start: One-Command Reproduction

### On Windows:
Double-click:
```cmd
RUN_ALL.bat
```
or run in terminal:
```cmd
python run_all.py
```

### On Linux / macOS:
```bash
python3 run_all.py
```

The script will automatically:
1. Verify Python 3.9+ compatibility.
2. Initialize an isolated virtual environment (`.venv`) and install dependencies from `requirements.txt`.
3. Verify dataset availability and download public CWRU files.
4. Verify frozen classifier weights (`classifier_seed{42,123,2024}.pt`).
5. Execute Experiment A across 3 seeds (288 evaluations total).
6. Recalibrate post-hoc OOD baselines at matched ~90% supported coverage.
7. Run gate ablation, synthetic benchmarks, and latency profiling.
8. Regenerate all publication figures (PDF & PNG 300 DPI).
9. Output a master reproducibility report in `results/reports/REPRODUCIBILITY_REPORT.md`.

---

## 2. Granular Step-by-Step Execution

Researchers seeking to execute individual pipeline stages independently can use the modular experiment scripts:

### Step 2.1: Run Main Experiment A (SDUST & CWRU)
```bash
python experiments/run_experiment_a.py
```
Outputs:
- `results/raw/expA_predictions.csv`
- `results/csv/per_recording_outcomes.csv`
- `results/csv/per_seed_metrics.csv`
- `results/csv/stratum_coverage_rejection.csv`
- `results/csv/seed_aggregated_metrics.csv`

### Step 2.2: Run Gate Ablation
```bash
python experiments/run_ablation.py
```
Outputs:
- `results/csv/gate_ablation_comparison.csv`

### Step 2.3: Run Coverage-Matched OOD Baselines
```bash
python experiments/run_baselines.py
```
Outputs:
- `results/csv/baseline_comparison_coverage_matched_90pct.csv`

### Step 2.4: Run Controlled Synthetic Benchmarks
```bash
python experiments/run_synthetic.py
```
Outputs:
- Verified summaries for Experiments B and C

### Step 2.5: Benchmark Computational Latency
```bash
python experiments/run_latency_benchmark.py
```
Outputs:
- `results/csv/computational_latency.csv`

### Step 2.6: Generate Figures
```bash
python src/plot_figures.py
python src/generate_ieee_figures.py
```
Outputs:
- `results/figures/fig1_coverage_vs_far.pdf`
- `results/figures/fig2_stratum_acceptance.pdf`
- `results/figures/fig3_gate_ablation.pdf`
- `results/figures/fig4_baseline_comparison_90pct.pdf`
- `results/figures/fig1_framework.pdf`
- `results/figures/fig2_three_panel.pdf`

---

## 3. Automated Verification Checklist

All outputs are automatically verified against 10 strict sanity checks recorded in `results/reports/VERIFICATION_REPORT.json`:

| Check ID | Verification Rule | Target |
| :--- | :--- | :---: |
| **SC-01** | Stage 3 is not called by the active inference pipeline | PASS |
| **SC-02** | Stage 3 cannot affect the final acceptance mask ($A = g_1 \land g_2$) | PASS |
| **SC-03** | Stage 2 executes strictly on Stage 1-passing inputs | PASS |
| **SC-04** | Frozen classifier executes strictly on inputs passing both gates | PASS |
| **SC-05** | Rejected inputs emit $\bot$ and valid first rejection stage | PASS |
| **SC-06** | Checkpoints, splits, normalization ($s=3.89934$), windowing ($2048$) match frozen protocol | PASS |
| **SC-07** | Metrics recomputed strictly from regenerated execution outputs | PASS |
| **SC-08** | Seed evaluations aggregated with $\text{ddof}=0$ | PASS |
| **SC-09** | All figures rendered from verified results | PASS |
| **SC-10** | Zero missing dependencies, zero mock values, 100% execution success | PASS |
