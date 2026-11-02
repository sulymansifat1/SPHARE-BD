# SPHARE-BD Benchmark Datasets & Manifests

This directory contains dataset manifests, baseline validation cache, and data layout specifications for the SPHARE-BD evaluation suite.

---

## 1. Directory Structure

```text
data/
├── README.md                          <- This guide
├── manifests/
│   ├── sdust/                         <- Official SDUST split manifests (temporal holdouts)
│   │   ├── eval_IN_manifest.csv       <- Nominal interior stratum (2000 RPM, 0 kN)
│   │   ├── eval_EDGE_manifest.csv     <- Boundary edge strata (1500 & 2500 RPM, 0 kN)
│   │   ├── eval_OUT_manifest.csv      <- Speed extrapolation stratum (1000 RPM, 0 kN)
│   │   ├── eval_NEWAXIS_manifest.csv  <- Unsupported load-axis shift (1500, 2000, 2500 RPM, 20 kN)
│   │   ├── eval_COMPOUND_manifest.csv <- Compound speed & load shift (1000 RPM, 20 kN)
│   │   ├── source_train_manifest.csv  <- Training split slices
│   │   ├── source_val_manifest.csv    <- Validation calibration slices
│   │   ├── dataset_manifest.csv       <- Complete inventory
│   │   └── recording_manifest.csv     <- Physical recording catalog
│   └── cwru/
│       └── eval_CROSSMACHINE_manifest.csv <- CWRU cross-machine evaluation manifest (16 files)
├── baselines/
│   └── risk_coverage_sweep.csv        <- Authoritative baseline validation sweep cache
├── sdust/                             <- (Optional) Local SDUST .mat files
└── cwru/                              <- (Optional / Auto-downloaded) CWRU 12k .mat files
```

---

## 2. Dataset Specifications

### 2.1 Shandong University of Science and Technology (SDUST) Dataset
- **Role:** Primary in-domain benchmark (Source training, validation, supported test, unsupported shifts).
- **Sampling Rate:** $f_s = 25,600\text{ Hz}$.
- **Health Classes (4 Coarse):**
  - `0`: Inner Ring Fault (IRF, sizes 0.2, 0.4, 0.6 mm)
  - `1`: Normal Condition (NC)
  - `2`: Outer Ring Fault (ORF, sizes 0.2, 0.4, 0.6 mm)
  - `3`: Rolling Element Fault (REF, sizes 0.2, 0.4, 0.6 mm)
- **Source Conditions:** $s \in \{1500, 2000, 2500\}\text{ RPM}$, radial load $L = 0.0\text{ kN}$.
- **Unsupported Shifts:**
  - `OUT`: $1000\text{ RPM}, 0.0\text{ kN}$ (speed extrapolation).
  - `NEW_AXIS`: $s \in \{1500, 2000, 2500\}\text{ RPM}, L = 20.0\text{ kN}$ (unsupported load-axis shift).
  - `COMPOUND`: $1000\text{ RPM}, L = 20.0\text{ kN}$ (compound speed extrapolation and load shift).
- **Anti-Leakage Temporal Slicing:**
  - `source_train`: Samples $[0, 626688)$
  - `source_val`: Samples $[628736, 837632)$
  - `eval`: Samples $[839680, 1048576)$
  - Non-overlapping guard gap: 2,048 samples (1 window length) between splits. Zero sample overlap.
- **Acquisition:** Contact Shandong University of Science and Technology or institutional repositories. Place `.mat` files in `data/sdust/` or point `SPHARE_DATASET_ROOT` to your existing directory.

### 2.2 Case Western Reserve University (CWRU) Dataset
- **Role:** Target-free foreign cross-machine testbed. Zero CWRU files participate in training or calibration.
- **Sampling Rate:** $f_s = 12,000\text{ Hz}$ (drive-end accelerometer).
- **Evaluation Pool:** 16 recordings across 0, 1, 2, and 3 HP motor loads.
- **Automatic Acquisition:** Running `python run_all.py` automatically downloads all 16 required `.mat` files directly from the Case Western Reserve University Bearing Data Center into `data/cwru/`.

### 2.3 Synthetic Benchmarks
- **Experiment B:** Independent Physics-Informed Synthetic Bearing Benchmark (15,876 cases across 3 seeds).
- **Experiment C:** Synthetic Cross-Machine Benchmark (9,990 cases across 3 seeds).
- Evaluated via `experiments/run_synthetic.py` and synthesized programmatically via `src/synthetic/`.
