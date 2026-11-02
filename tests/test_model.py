"""Unit test: Verify Frozen 1D-CNN architecture, parameter count, and forward pass."""
from __future__ import annotations

from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import FrozenFaultClassifier


def test_classifier_structure():
    model = FrozenFaultClassifier(n_classes=4)
    model.eval()

    # Total trainable parameters must match exactly 51,028
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 51028, f"Expected 51,028 parameters, got {n_params}"

    # Verify forward pass on batch of 2 windows of length 2048
    x = torch.randn(2, 1, 2048)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"
    print(f"Classifier test passed: 51,028 parameters, forward pass verified.")


def test_checkpoint_loading():
    ckpts = list((REPO_ROOT / "checkpoints").glob("*.pt"))
    assert len(ckpts) == 3, f"Expected 3 checkpoints, found {len(ckpts)}"

    for ckpt_path in ckpts:
        model = FrozenFaultClassifier(n_classes=4)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else (ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        model.load_state_dict(state)
        model.eval()
        print(f"Verified checkpoint loading: {ckpt_path.name}")


if __name__ == "__main__":
    test_classifier_structure()
    test_checkpoint_loading()
