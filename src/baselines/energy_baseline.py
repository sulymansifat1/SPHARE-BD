"""Energy-based Out-of-Distribution Detection Baseline (Liu et al., NeurIPS 2020).

Official implementation reference:
https://github.com/wetliu/energy_ood (commit: 77f3c09b788bb5a7bfde6fd3671228320ea0949c)
Paper: Liu et al., 'Energy-based Out-of-distribution Detection', NeurIPS 2020.
DOI: 10.48550/arXiv.2010.03759

Scoring rule:
Negative Helmholtz free energy: S(x) = T * logsumexp(logits / T).
Recording score = mean of window scores across all windows of the recording.
Threshold = 5th percentile of source_val recording scores.
Accept iff score >= threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class EnergyDecision:
    recording_id: str
    score: float
    threshold: float
    decision: str  # "ACCEPT" or "ABSTAIN"
    predicted_class: int | None
    forced_majority: int


def compute_window_energy_scores(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """Compute negative energy score for each window (higher = more likely in-distribution)."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be [windows, classes], got {tuple(logits.shape)}")
    scaled_logits = logits / float(temperature)
    # torch.logsumexp computes log(sum(exp(scaled_logits)))
    neg_energy = float(temperature) * torch.logsumexp(scaled_logits, dim=-1)
    return neg_energy.detach().cpu().numpy()


def compute_recording_energy_score(window_energy_scores: np.ndarray) -> float:
    """Recording-level score is the mean across all non-overlapping windows."""
    vals = np.asarray(window_energy_scores, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        raise ValueError("Cannot compute energy score on empty window sequence")
    return float(vals.mean())


def fit_energy_threshold(source_val_scores: list[float] | np.ndarray, percentile: float = 5.0) -> float:
    """Calibrate rejection threshold on source_val recordings strictly."""
    scores = np.asarray(source_val_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Source-validation scores cannot be empty")
    return float(np.percentile(scores, percentile))


def decide_energy_recording(
    recording_id: str,
    window_classes: np.ndarray,
    window_energy_scores: np.ndarray,
    threshold: float,
) -> EnergyDecision:
    """Make selective decision for a single recording based on frozen energy threshold."""
    score = compute_recording_energy_score(window_energy_scores)
    # Ties in majority vote keep the smallest class index
    classes = np.asarray(window_classes, dtype=int)
    counts = np.bincount(classes)
    forced_majority = int(np.argmax(counts))

    if score < threshold:
        return EnergyDecision(
            recording_id=recording_id,
            score=score,
            threshold=float(threshold),
            decision="ABSTAIN",
            predicted_class=None,
            forced_majority=forced_majority,
        )
    return EnergyDecision(
        recording_id=recording_id,
        score=score,
        threshold=float(threshold),
        decision="ACCEPT",
        predicted_class=forced_majority,
        forced_majority=forced_majority,
    )
