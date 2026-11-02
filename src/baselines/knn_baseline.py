"""Deep Nearest Neighbors Out-of-Distribution Detection Baseline (Sun et al., ICML 2022).

Official implementation reference:
https://github.com/deeplearning-wisc/knn-ood (commit: 2afb2bbed60a8d69384dc9b28e5637711345222b)
Paper: Sun et al., 'Out-of-Distribution Detection with Deep Nearest Neighbors', ICML 2022.
DOI: 10.48550/arXiv.2204.06507

Scoring rule:
Penultimate features normalized to unit hypersphere: z = h / (||h||_2 + 1e-10).
Negative Euclidean distance to the k-th nearest neighbor in source_train feature bank:
S(x) = - ||z - z_(k)||_2.
Recording score = mean of window scores across all windows of the recording.
Threshold = 5th percentile of source_val recording scores.
Accept iff score >= threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


@dataclass
class KNNDecision:
    recording_id: str
    score: float
    threshold: float
    decision: str  # "ACCEPT" or "ABSTAIN"
    predicted_class: int | None
    forced_majority: int


def normalize_features(features: np.ndarray | torch.Tensor) -> np.ndarray:
    """L2-normalize feature embeddings onto unit hypersphere."""
    if isinstance(features, torch.Tensor):
        feat = features.detach().cpu().numpy()
    else:
        feat = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(feat, ord=2, axis=-1, keepdims=True)
    return (feat / np.maximum(norms, 1e-10)).astype(np.float32)


class KNNOODDetector:
    def __init__(self, k: int = 50):
        self.k = k
        self.nn_model: NearestNeighbors | None = None
        self.train_embeddings: np.ndarray | None = None

    def fit_train_features(self, train_features: np.ndarray | torch.Tensor) -> None:
        """Store normalized source_train feature embeddings and fit NearestNeighbors index."""
        norm_feat = normalize_features(train_features)
        self.train_embeddings = norm_feat
        effective_k = min(self.k, len(norm_feat))
        self.nn_model = NearestNeighbors(n_neighbors=effective_k, algorithm="auto", metric="euclidean")
        self.nn_model.fit(norm_feat)

    def score_windows(self, window_features: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute negative distance to k-th nearest neighbor for each window."""
        if self.nn_model is None:
            raise RuntimeError("KNN detector has not been fitted with train_features")
        norm_feat = normalize_features(window_features)
        distances, _ = self.nn_model.kneighbors(norm_feat)
        # distances is [N_windows, K], so the last column is the k-th nearest neighbor distance
        kth_distance = distances[:, -1]
        # In-distribution score: negative distance (closer to train = higher score)
        return -kth_distance.astype(np.float64)

    def score_recording(self, window_features: np.ndarray | torch.Tensor) -> float:
        """Recording-level score is the mean across all non-overlapping windows."""
        w_scores = self.score_windows(window_features)
        return float(np.mean(w_scores))


def fit_knn_threshold(source_val_scores: list[float] | np.ndarray, percentile: float = 5.0) -> float:
    """Calibrate rejection threshold on source_val recordings strictly."""
    scores = np.asarray(source_val_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Source-validation scores cannot be empty")
    return float(np.percentile(scores, percentile))


def decide_knn_recording(
    recording_id: str,
    window_classes: np.ndarray,
    window_features: np.ndarray | torch.Tensor,
    detector: KNNOODDetector,
    threshold: float,
) -> KNNDecision:
    """Make selective decision for a single recording based on frozen KNN threshold."""
    score = detector.score_recording(window_features)
    classes = np.asarray(window_classes, dtype=int)
    counts = np.bincount(classes)
    forced_majority = int(np.argmax(counts))

    if score < threshold:
        return KNNDecision(
            recording_id=recording_id,
            score=score,
            threshold=float(threshold),
            decision="ABSTAIN",
            predicted_class=None,
            forced_majority=forced_majority,
        )
    return KNNDecision(
        recording_id=recording_id,
        score=score,
        threshold=float(threshold),
        decision="ACCEPT",
        predicted_class=forced_majority,
        forced_majority=forced_majority,
    )
