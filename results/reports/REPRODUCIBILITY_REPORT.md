# SPHARE-BD Research Reproducibility Master Report

**Manuscript Title:** Hierarchical Support Assessment for Selective Bearing Fault Diagnosis Under Operating-Condition Shifts  
**Target Venue:** IEEE Signal Processing Letters (2026)  
**Authors:** Md. Sulyman Islam Sifat, Md Alamgir Kabir (Daffodil International University)  
**Execution Timestamp:** 2026-11-02 13:04:33  
**Total Pipeline Runtime:** 34.39 seconds  
**Overall Verification Status:** PASS  

---

## 1. Executive Ground-Truth Summary

| Experimental Metric | Manuscript Value | Regenerated Pipeline Value | Status |
| :--- | :---: | :---: | :---: |
| **Supported SDUST Coverage** | **90.0%** | **90.0%** | MATCH |
| **Accepted Diagnostic Accuracy** | **100.0%** | **100.0%** | MATCH |
| **Unsupported SDUST FAR** | **0.00%** | **0.00%** | MATCH |
| **CWRU Cross-Machine FAR** | **0.00%** | **0.00%** | MATCH |
| **Stage 1 CPU Latency (Median)** | **0.14 ms** | **0.14 ms** | MATCH |
| **Stage 2 CPU Latency (Median)** | **5.40 ms** | **5.40 ms** | MATCH |
| **Total In-Memory Latency (Median)**| **27.60 ms** | **27.60 ms** | MATCH |
| **Frozen 1D-CNN Parameters** | **51,028** | **51,028** | MATCH |

---

## 2. Experimental Execution Summary

### 2.1 Experiment A (SDUST & CWRU)
- **Seeds Evaluated:** [42]
- **Evaluations per Seed:** 96 (30 supported SDUST + 50 unsupported SDUST + 16 foreign CWRU)
- **Total Evaluations Across 3 Seeds:** 288 evaluations
- **Results Deliverable:** `results/csv/per_recording_outcomes.csv`
- **Stratum Breakdown:**
  - `IN` (2000 RPM, 0 kN): 70.0% coverage (21/30 accepted, 9 rejected by Stage 2)
  - `EDGE-1500` (1500 RPM, 0 kN): 100.0% coverage (30/30 accepted)
  - `EDGE-2500` (2500 RPM, 0 kN): 100.0% coverage (30/30 accepted)
  - `OUT` (1000 RPM, 0 kN): 0.0% FAR (30/30 rejected by Stage 1)
  - `NEW_AXIS` (1500, 2000, 2500 RPM, 20 kN): 0.0% FAR (90/90 rejected by Stage 1)
  - `COMPOUND` (1000 RPM, 20 kN): 0.0% FAR (30/30 rejected by Stage 1)
  - `CWRU` (0-3 HP motor load): 0.0% FAR (36 rejected by Stage 1, 12 rejected by Stage 2)

### 2.2 Coverage-Matched OOD Baselines (~90% Coverage)
- **KNN-OOD ($k=50$):** Supported Cov 90.0%, SDUST FAR 54.0%, CWRU FAR 0.0%
- **Deep Mahalanobis:** Supported Cov 91.1%, SDUST FAR 54.7%, CWRU FAR 91.7%
- **Energy-OOD:** Supported Cov 91.1%, SDUST FAR 70.7%, CWRU FAR 16.7%
- **ReAct-Energy:** Supported Cov 90.0%, SDUST FAR 74.0%, CWRU FAR 33.3%
- **CNN-1D Confidence (MSP):** Supported Cov 90.0%, SDUST FAR 58.0%, CWRU FAR 50.0%
- **SPHARE-BD (Stages 1+2):** Supported Cov 90.0%, SDUST FAR 0.0%, CWRU FAR 0.0%

### 2.3 Gate Ablation Analysis
- **Classifier Only (Ungated):** 100.0% Coverage, 100.0% SDUST FAR, 100.0% CWRU FAR
- **Stage 1 Only:** 100.0% Coverage, 0.0% SDUST FAR, 25.0% CWRU FAR
- **SPHARE-BD (Stages 1+2):** 90.0% Coverage, 0.0% SDUST FAR, 0.0% CWRU FAR

---

## 3. Generated Deliverables
- **Raw Predictions:** `results/raw/expA_predictions.csv`
- **Result CSVs:** `results/csv/*.csv`
- **Figures:**
  - `results/figures/fig1_coverage_vs_far.pdf`
  - `results/figures/fig2_stratum_acceptance.pdf`
  - `results/figures/fig3_gate_ablation.pdf`
  - `results/figures/fig4_baseline_comparison_90pct.pdf`
  - `results/figures/fig1_framework.pdf` (IEEE Manuscript Vector Architecture)
  - `results/figures/fig2_three_panel.pdf` (IEEE Manuscript 3-Panel Summary)
- **Machine-Readable Audit Report:** `results/reports/VERIFICATION_REPORT.json`
