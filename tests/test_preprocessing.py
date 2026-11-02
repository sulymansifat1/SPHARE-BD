"""Unit test: Verify windowing, normalization, and scale consistency."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing.normalize import apply_scale
from preprocessing.windowing import derived_window_params, extract_windows, select_support_windows


def test_windowing():
    fs = 25600.0
    duration = 0.08
    hop_frac = 1.0

    params = derived_window_params(fs, duration, hop_frac)
    assert params["window_length_samples"] == 2048
    assert params["hop_samples"] == 2048

    # Create dummy signal of 208,896 samples (102 windows)
    sig = np.random.randn(208896)
    wins = extract_windows(sig, params["window_length_samples"], params["hop_samples"])
    assert wins.shape == (102, 2048)

    supp = select_support_windows(wins, 16)
    assert supp.shape == (16, 2048)
    print("Window extraction and support subsampling tests passed.")


def test_normalization():
    scale = 3.8993436671993384
    wins = np.ones((5, 2048)) * 10.0
    normed = apply_scale(wins, scale)

    # Demeaned -> mean across window is 0
    assert np.allclose(normed, 0.0)
    print("Normalization test passed.")


if __name__ == "__main__":
    test_windowing()
    test_normalization()
