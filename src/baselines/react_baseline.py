"""Rectified Activations (ReAct) Out-of-Distribution Detection Baseline (Sun et al., NeurIPS 2021).

Official implementation reference:
https://github.com/deeplearning-wisc/react (commit: 2aa35d9993ddb00409dc41824dbc008b5cc16e20)
Paper: Sun et al., 'ReAct: Out-of-Distribution Detection with Rectified Activations', NeurIPS 2021.
DOI: 10.48550/arXiv.2111.12797

Scoring rule:
Estimate upper activation bound c as 90th percentile of penultimate activations on source_val.
Rectify penultimate features: h*(x) = min(h(x), c).
Calculate rectified logits: f*(x) = W h*(x) + b.
Score with negative energy: S(x) = T * logsumexp(f*(x) / T).
Recording score = mean of window scores across all windows of the recording.
Threshold = 5th percentile of source_val recording scores.
Accept iff score >= threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ReActDecision:
    recording_id: str
    score: float
    threshold: float
    decision: str  # "ACCEPT" or "ABSTAIN"
    predicted_class: int | None
    forced_majority: int


class ReActDetector:
    def __init__(self, clip_percentile: float = 90.0, temperature: float = 1.0):
        self.clip_percentile = clip_percentile
        self.temperature = temperature
        self.activation_clip_val: float | None = None

    def fit_val_activations(self, val_features: np.ndarray | torch.Tensor) -> float:
        """Calibrate activation clipping value c from source_val activations strictly."""
        if isinstance(val_features, torch.Tensor):
            vals = val_features.detach().cpu().numpy()
        else:
            vals = np.asarray(val_features, dtype=np.float32)
        c = float(np.percentile(vals.flatten(), self.clip_percentile))
        self.activation_clip_val = c
        return c

    def score_windows(self, window_features: torch.Tensor, fc_layer: nn.Linear) -> np.ndarray:
        """Compute ReAct energy score for each window."""
        if self.activation_clip_val is None:
            raise RuntimeError("ReAct detector has not calibrated clipping threshold c on source_val")

        with torch.no_grad():
            # Rectify penultimate activations
            clipped_feat = torch.clamp(window_features, max=float(self.activation_clip_val))
            # Forward through final classifier linear head
            rectified_logits = fc_layer(clipped_feat)
            # Energy calculation
            scaled_logits = rectified_logits / float(self.temperature)
            neg_energy = float(self.temperature) * torch.logsumexp(scaled_logits, dim=-1)
            return neg_energy.detach().cpu().numpy()

    def score_recording(self, window_features: torch.Tensor, fc_layer: nn.Linear) -> float:
        """Recording-level score is the mean across all non-overlapping windows."""
        w_scores = self.score_windows(window_features, fc_layer)
        return float(np.mean(w_scores))


def fit_react_threshold(source_val_scores: list[float] | np.ndarray, percentile: float = 5.0) -> float:
    """Calibrate rejection threshold on source_val recordings strictly."""
    scores = np.asarray(source_val_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Source-validation scores cannot be empty")
    return float(np.percentile(scores, percentile))


def decide_react_recording(
    recording_id: str,
    window_classes: np.ndarray,
    window_features: torch.Tensor,
    fc_layer: nn.Linear,
    detector: ReActDetector,
    threshold: float,
) -> ReActDecision:
    """Make selective decision for a single recording based on frozen ReAct threshold."""
    score = detector.score_recording(window_features, fc_layer)
    classes = np.asarray(window_classes, dtype=int)
    counts = np.bincount(classes)
    forced_majority = int(np.argmax(counts))

    if score < threshold:
        return ReActDecision(
            recording_id=recording_id,
            score=score,
            threshold=float(threshold),
            decision="ABSTAIN",
            predicted_class=None,
            forced_majority=forced_majority,
        )
    return ReActDecision(
        recording_id=recording_id,
        score=score,
        threshold=float(threshold),
        decision="ACCEPT",
        predicted_class=forced_majority,
        forced_majority=forced_majority,
    )
