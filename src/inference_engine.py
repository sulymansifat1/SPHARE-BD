"""SPHARE-BD Final Version: Active Inference Engine.

Implements the strictly sequential Stage 1 + Stage 2 pipeline:
Input (x, c) -> Stage 1: g1(c) -> Stage 2: g2(x, c) -> Frozen 1D-CNN: f(x) -> Prediction / Abstention

Stage 3 is completely excised from active inference.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from datasets.base_adapter import RecordingMeta
from datasets.sdust_adapter import SdustAdapter
from datasets.cwru_adapter import CwruAdapter
from model import FrozenFaultClassifier
from preprocessing.normalize import apply_scale
from preprocessing.windowing import derived_window_params, extract_windows, select_support_windows
from support.gates import envelope_decide
from support.local_manifold import assign_region
from support.physical_envelope import recording_feature_median
from support.source_support import SourceSupportEstimator


class SPHAREBDFinalEngine:
    """Active inference engine for SPHARE-BD Final Version (Stage 1 + Stage 2 only)."""

    def __init__(self, root_dir: Path, seed: int, device: str = "cpu"):
        self.root_dir = root_dir
        self.seed = seed
        self.device = torch.device(device)
        
        # Load freeze configuration
        freeze_file = root_dir / "configs" / "freeze" / f"EXPERIMENT_A_FROZEN_seed{seed}.json"
        if not freeze_file.exists():
            raise FileNotFoundError(f"Freeze file not found: {freeze_file}")
        self.freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
        self.cfg = self.freeze["cfg"]
        
        # Support estimator for Stage 1
        self.est = SourceSupportEstimator()
        self.support = self.freeze["support"]
        self.cond_names = self.freeze["condition_names"]
        
        # Spectral envelope for Stage 2
        self.spec = self.freeze["spec"]
        self.local = self.freeze["local"]
        self.scale = float(self.freeze["scale"])
        self.duration = float(self.cfg["window_duration_sec"])
        self.hop_frac = float(self.cfg["hop_fraction"])
        self.k_supp = int(self.cfg["n_support_windows"])
        
        # Load frozen classifier checkpoint
        ckpt_file = root_dir / "checkpoints" / f"classifier_seed{seed}.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
        
        ckpt = torch.load(ckpt_file, map_location=self.device)
        self.model = FrozenFaultClassifier(n_classes=int(self.freeze["n_classes"]))
        state = ckpt["model"] if "model" in ckpt else (ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        
        # Execution counters for verification audit
        self.calls_stage1 = 0
        self.calls_stage2 = 0
        self.calls_classifier = 0
        self.calls_stage3 = 0  # MUST remain strictly 0

    def shaft_hz(self, speed: Optional[float]) -> Optional[float]:
        if speed is None:
            return None
        return float(speed) / float(self.cfg["rev_per_minute_factor"])

    def evaluate_recording(
        self,
        adapter: Any,
        rec: RecordingMeta,
        protocol_stratum: Optional[str] = None,
        force_classifier_for_audit: bool = True,
    ) -> Dict[str, Any]:
        """Evaluates a single recording through the Stage 1 + Stage 2 active pipeline.
        
        Short-circuits immediately upon any gate failure.
        """
        self.calls_stage1 += 1
        
        # 1. Stage 1: Operating-condition support check
        c_dict = {"speed": rec.speed, "load": rec.load}
        s1 = self.est.decide(c_dict, self.support)
        
        # Prepare basic metadata
        fs = rec.sampling_rate_hz
        sh = self.shaft_hz(rec.speed)
        C_vec = [float(c_dict.get(n, np.nan)) for n in self.cond_names]
        
        out = {
            "recording_id": rec.recording_id,
            "dataset": rec.dataset,
            "split": rec.split,
            "stratum": protocol_stratum or rec.stratum or rec.split,
            "stratum_report": protocol_stratum or rec.stratum or rec.split,
            "seed": self.seed,
            "y": rec.y,
            "speed_meta": rec.speed,
            "load_meta": rec.load,
            "sampling_rate_hz": fs,
            "stage1_pass": bool(s1["pass"]),
            "stage1_score": float(s1.get("score", 0.0)),
            "stage1_reason": str(s1.get("reason", "")),
            "stage2_pass": None,
            "stage2_score": None,
            "stage3_pass": None,  # Stage 3 completely excised
            "stage3_score": None,
            "accept": False,
            "abstain": True,
            "reason": None,
            "first_rejection_stage": None,
            "yhat": None,
            "forced_yhat": None,
        }
        
        # SHORT-CIRCUIT 1: Stage 1 fails
        if not s1["pass"]:
            out["reason"] = s1["reason"]
            out["first_rejection_stage"] = "Stage1"
            
            # Optionally compute forced classification purely for historical / forced macro-F1 comparison
            if force_classifier_for_audit:
                win_p = derived_window_params(fs, self.duration, self.hop_frac)
                sig = adapter.load_sliced_signal(rec) if hasattr(adapter, "load_sliced_signal") else adapter.load_signal(rec)
                wins = extract_windows(sig, win_p["window_length_samples"], win_p["hop_samples"])
                out["forced_yhat"] = self._predict_windows(wins)
                out["n_windows"] = len(wins)
            return out
        
        # 2. Stage 2: Order-domain spectral-consistency check (only if Stage 1 passed)
        self.calls_stage2 += 1
        win_p = derived_window_params(fs, self.duration, self.hop_frac)
        sig = adapter.load_sliced_signal(rec) if hasattr(adapter, "load_sliced_signal") else adapter.load_signal(rec)
        wins = extract_windows(sig, win_p["window_length_samples"], win_p["hop_samples"])
        out["n_windows"] = len(wins)
        out["slice_start"] = rec.extras.get("start_index", 0)
        out["slice_end"] = rec.extras.get("end_index", len(wins) * win_p["hop_samples"])
        
        supp = select_support_windows(wins, self.k_supp)
        supp_n = apply_scale(supp, self.scale)
        feats = recording_feature_median(supp_n, fs, sh, self.spec, self.cfg["epsilon"])
        
        C_arr = np.array(C_vec, dtype=np.float64)
        cents = np.asarray(self.local["centroids"], dtype=np.float64)
        rid = assign_region(C_arr, cents)
        key = str(rid)
        if key not in self.local["regions"]:
            key = list(self.local["regions"].keys())[0]
            
        env = self.local["envelopes"][key]
        selected = self.local["selected_features"]
        
        stage2_ok = True
        s2_scores = []
        for feat in selected:
            dec = envelope_decide(feats[feat], env[feat][0], env[feat][1])
            s2_scores.append(dec["score"])
            stage2_ok = stage2_ok and dec["pass"]
            
        out["stage2_pass"] = bool(stage2_ok)
        out["stage2_score"] = float(np.min(s2_scores)) if s2_scores else 0.0
        
        # SHORT-CIRCUIT 2: Stage 2 fails
        if not stage2_ok:
            out["reason"] = "PHYSICAL_INCONSISTENCY"
            out["first_rejection_stage"] = "Stage2"
            if force_classifier_for_audit:
                out["forced_yhat"] = self._predict_windows(wins)
            return out
        
        # 3. Frozen 1D-CNN Classification (executed strictly and only when Stage 1 and Stage 2 pass)
        self.calls_classifier += 1
        yhat = self._predict_windows(wins)
        
        out["accept"] = True
        out["abstain"] = False
        out["reason"] = "ACCEPT"
        out["first_rejection_stage"] = None
        out["yhat"] = yhat
        out["forced_yhat"] = yhat
        return out

    def _predict_windows(self, wins: np.ndarray) -> int:
        """Inference of frozen 1D-CNN across all windows followed by majority voting."""
        wins_n = apply_scale(wins, self.scale)
        x = torch.tensor(wins_n, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        values, counts = np.unique(preds, return_counts=True)
        order = np.lexsort((values, -counts))
        return int(values[order[0]])
