# SPHARE-BD Architectural Specification
**Two-Stage Hierarchical Support-Assessment Architecture (Stage 1 + Stage 2 Only)**  
*Reference Document for IEEE Signal Processing Letters (2026)*

---

## 1. Selective Diagnosis Formulation

Standard deep learning models for bearing fault diagnosis operate under a closed-set forced-choice paradigm: given an arbitrary vibration input $x$, the network maps it to a softmax distribution and emits an argmax prediction, regardless of whether the operating conditions or kinematics are supported by training physics.

SPHARE-BD reformulates bearing diagnostics as a source-only selective classification problem:
$$\hat{y}(x, c) = \begin{cases} f_\theta(x), & \text{if } A(x, c) = 1, \\ \bot, & \text{if } A(x, c) = 0, \end{cases}$$
where $\bot$ denotes safe abstention, and $A(x, c)$ is the hierarchical acceptance function:
$$A(x, c) = g_1(c) \cdot g_2(x, c)$$

---

## 2. Decision Pipeline & Short-Circuit Hierarchy

```text
[Raw Input: Vibration Signal x[n] + Operational Telemetry c = [s, L]^T]
                                  |
                                  v
            [Stage 1: Operating-Condition Support Gate g1(c)]
              - Telemetry c = [s, L]^T screened against source bounds
              - 2D Delaunay Convex Hull (planar) or Collinear Bounding Box (SDUST)
              - SDUST bounds: s in [1400, 2600] RPM, L in [-0.15, 0.15] kN
              - If Fail: SHORT-CIRCUIT EARLY EXIT (0.14 ms CPU latency)
                Decision: ABSTAIN (BOT), Reason: CONDITION_AXIS_OUT_OF_SUPPORT
                                  |
                                  v (Pass: g1(c) = 1)
            [Conditional Preprocessing & Windowing]
              - Window length: Lw = round(fs * 0.08 s) = 2,048 samples (SDUST)
              - Non-overlapping extraction: Hop = 100% (stride = 2,048)
              - Per-window DC removal: w_zero = w - mean(w)
              - Global source scaling: w_norm = w_zero / sigma_train (sigma = 3.89934)
              - K = 16 support windows subsampled uniformly
                                  |
                                  v
            [Stage 2: Kinematic Order-Domain Spectral Envelope Gate g2(x, c)]
              - Shaft rotational frequency: fr = s / 60
              - Order mapping: o_k = f_k / fr across orders [0, 384]
              - 11 physical spectral descriptors (8 order bins + log-total + log-RMS + low-order)
              - Regional tolerance envelopes: lo_k <= phi_k <= hi_k
              - If Fail: SHORT-CIRCUIT EARLY EXIT (5.40 ms CPU latency)
                Decision: ABSTAIN (BOT), Reason: PHYSICAL_INCONSISTENCY
                                  |
                                  v (Pass: g2(x, c) = 1; Both Gates S1 & S2 Pass)
            [Frozen 1D-CNN Fault Classifier f_theta(x)]
              - Evaluates all W = 102 windows
              - Window-level majority voting
              - Latency: 22.06 ms CPU
              - Total in-memory decision latency: 27.60 ms CPU
              - Emits Fault Diagnosis y_hat in {0, 1, 2, 3}
```

---

## 3. Rationale for Stage 3 Excision

The historical 3-stage prototype included a local source-manifold gate (Stage 3) using Ledoit-Wolf covariance shrinkage in deep feature space. Forensic auditing revealed:
1. **Zero Unique OOD Rejection Benefit:** 100% of unsupported operating-condition shifts and foreign testbeds are already eliminated by Stages 1 and 2.
2. **False Rejection of Supported Inputs:** In the nominal interior stratum (`IN`), Stage 3 falsely rejected all Rolling Element Fault (`REF`) recordings, reducing `IN` 4-class Macro-F1 to 0.7500 and overall supported coverage from 90.0% down to 66.7%.
3. **Definitive Ground Truth:** Excising Stage 3 restores supported coverage to 90.0% with 100.0% accuracy on admitted inputs, while preserving 0.00% FAR across all unsupported regimes.

---

## 4. Frozen 1D-CNN Architecture Specification

- **Input:** Single-channel normalized vibration window $(1, 2048)$.
- **Architecture:** 4 Conv1D-BatchNorm-ReLU blocks + AdaptiveAvgPool1d + Dense Head:
  - Block 1: `Conv1d(1, 16, k=64, s=8, p=28) -> BatchNorm1d(16) -> ReLU -> MaxPool1d(2)` (1,040 params)
  - Block 2: `Conv1d(16, 32, k=3, s=1, p=1) -> BatchNorm1d(32) -> ReLU -> MaxPool1d(2)` (1,568 params)
  - Block 3: `Conv1d(32, 64, k=3, s=1, p=1) -> BatchNorm1d(64) -> ReLU -> MaxPool1d(2)` (6,208 params)
  - Block 4: `Conv1d(64, 128, k=3, s=1, p=1) -> BatchNorm1d(128) -> ReLU -> AdaptiveAvgPool1d(1)` (24,704 params)
  - Head: `Linear(128, 128) -> ReLU -> Linear(128, 4)` (17,028 params)
- **Total Trainable Parameters:** Exactly 51,028 float32 weights.
