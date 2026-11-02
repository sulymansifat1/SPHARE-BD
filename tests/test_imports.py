"""Unit test: Verify that all required dependencies and core modules import cleanly."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_core_imports():
    import numpy as np
    import scipy
    import sklearn
    import torch
    import pandas as pd
    import matplotlib

    assert hasattr(np, "__version__")
    assert hasattr(torch, "__version__")

    from model import FrozenFaultClassifier
    from preprocessing.normalize import apply_scale
    from preprocessing.windowing import extract_windows
    from support.gates import combine_stages
    from support.source_support import SourceSupportEstimator
    from support.physical_envelope import recording_feature_median
    from inference_engine import SPHAREBDFinalEngine
    from datasets.sdust_adapter import SdustAdapter
    from datasets.cwru_adapter import CwruAdapter

    print("All core imports passed successfully.")


if __name__ == "__main__":
    test_core_imports()
