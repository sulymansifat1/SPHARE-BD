"""Deep Mahalanobis Distance Out-of-Distribution Detection Baseline.

Reference / Provenance:
OpenOOD benchmark (Yang et al., NeurIPS 2022 / TPAMI 2024),
Relative Mahalanobis (Ren et al., NeurIPS 2021),
based on Lee et al., 'A Simple Unified Framework for Detecting Out-of-Distribution Samples...', NeurIPS 2018.

Scoring rule:
Fit class-conditional centroids mu_c on source_train penultimate representations.
Compute tied covariance with Ledoit-Wolf shrinkage to ensure well-conditioned precision P = Sigma^{-1}.
Window in-distribution score: S(x) = - min_c (h(x) - mu_c)^T P (h(x) - mu_c).
Recording score = mean of window scores across all windows of the recording.
Threshold = 5th percentile of source_val recording scores.
Accept iff score >= threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.covariance import LedoitWolf
import torch


@dataclass
class MahalanobisDecision:
    recording_id: str
    score: float
    threshold: float
    decision: str  # "ACCEPT" or "ABSTAIN"
    predicted_class: int | None
    forced_majority: int


class DeepMahalanobisDetector:
    def __init__(self):
        self.class_means: dict[int, np.ndarray] = {}
        self.precision: np.ndarray | None = None
        self.classes: list[int] = []

    def fit_train_features(self, train_features: np.ndarray | torch.Tensor, train_labels: np.ndarray | torch.Tensor) -> None:
        """Fit class centroids and tied Ledoit-Wolf regularized covariance on source_train only."""
        if isinstance(train_features, torch.Tensor):
            X = train_features.detach().cpu().numpy()
        else:
            X = np.asarray(train_features, dtype=np.float64)

        if isinstance(train_labels, torch.Tensor):
            Y = train_labels.detach().cpu().numpy()
        else:
            Y = np.asarray(train_labels, dtype=int)

        self.classes = sorted(list(set(Y.tolist())))
        self.class_means = {}

        # Center features per class
        centered_list = []
        for c in self.classes:
            idx = np.where(Y == c)[0]
            if len(idx) == 0:
                continue
            mu_c = np.mean(X[idx], axis=0)
            self.class_means[c] = mu_c
            centered_list.append(X[idx] - mu_c)

        centered_all = np.concatenate(centered_list, axis=0)
        # Use Ledoit-Wolf shrinkage for regularized tied covariance
        lw = LedoitWolf().fit(centered_all)
        self.precision = np.asarray(lw.precision_, dtype=np.float64)

    def score_windows(self, window_features: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute negative minimum Mahalanobis distance across all classes for each window."""
        if self.precision is None:
            raise RuntimeError("Mahalanobis detector has not been fitted")
        if isinstance(window_features, torch.Tensor):
            X = window_features.detach().cpu().numpy()
        else:
            X = np.asarray(window_features, dtype=np.float64)

        # Compute distance to each class
        dists = []
        for c in self.classes:
            diff = X - self.class_means[c]
            # Mahalanobis distance squared: diff @ precision @ diff^T
            maha_sq = np.sum(diff @ self.precision * diff, axis=1)
            dists.append(maha_sq)

        min_maha_sq = np.min(np.stack(dists, axis=1), axis=1)
        # In-distribution score is negative squared Mahalanobis distance
        return -min_maha_sq

    def score_recording(self, window_features: np.ndarray | torch.Tensor) -> float:
        """Recording-level score is the mean across all non-overlapping windows."""
        w_scores = self.score_windows(window_features)
        return float(np.mean(w_scores))


def fit_mahalanobis_threshold(source_val_scores: list[float] | np.ndarray, percentile: float = 5.0) -> float:
    """Calibrate rejection threshold on source_val recordings strictly."""
    scores = np.asarray(source_val_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Source-validation scores cannot be empty")
    return float(np.percentile(scores, percentile))


def decide_mahalanobis_recording(
    recording_id: str,
    window_classes: np.ndarray,
    window_features: np.ndarray | torch.Tensor,
    detector: DeepMahalanobisDetector,
    threshold: float,
) -> MahalanobisDecision:
    """Make selective decision for a single recording based on frozen Mahalanobis threshold."""
    score = detector.score_recording(window_features)
    classes = np.asarray(window_classes, dtype=int)
    counts = np.bincount(classes)
    forced_majority = int(np.argmax(counts))

    if score < threshold:
        return MahalanobisDecision(
            recording_id=recording_id,
            score=score,
            threshold=float(threshold),
            decision="ABSTAIN",
            predicted_class=None,
            forced_majority=forced_majority,
        )
    return MahalanobisDecision(
        recording_id=recording_id,
        score=score,
        threshold=float(threshold),
        decision="ACCEPT",
        predicted_class=forced_majority,
        forced_majority=forced_majority,
    )
